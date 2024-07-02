#coding = utf-8
import os
import warnings
import datetime
import logging
import re
from argparse import Namespace, ArgumentParser

import pytorch_lightning as pl
import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only, parsing
from pytorch_lightning.utilities import argparse as pl_argparse
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything


logging.basicConfig(level=logging.INFO)


class RWKVPEFTArgumentParser(Namespace):
    """
    参数解析及默认值设置
    """
    def __init__(self, **kwargs):
        allowed_arguments= {
            'load_model': ['', str],
            'wandb': ['', str],
            'proj_dir': ['out', str],
            'random_seed': [-1, int],

            'data_file': ['', str],
            'data_type': ['utf-8', str],
            'vocab_size': [0, int],

            'ctx_len': [1024, int],
            'epoch_steps': ['1000', int],
            'epoch_count': ['500', int],
            'epoch_begin': [0, int],
            'epoch_save': [5, int],

            'micro_bsz': [12, int],
            'n_layer': [6, int],
            'n_embd': [512, int],
            'dim_att': [0, int],
            'dim_ffn': [0, int],
            'pre_ffn': [0, int],
            'head_qk': [0, int],
            'tiny_att_dim': [0, int],
            'tiny_att_layer': [-999, int],

            'lr_init': [6e-4, float],
            'lr_final': [1e-5, float],
            'warmup_steps': [-1, int],
            'beta1': [0.9, float],
            'beta2': [0.99, float],
            'adam_eps': [1e-8, float],
            'grad_cp': [0, int],
            'dropout': [0, float],
            'weight_decay': [0, float],
            'weight_decay_final': [-1, float],

            'my_pile_version': [1, int],
            'my_pile_stage': [0, int],
            'my_pile_shift': [-1, int],
            'my_pile_edecay': [0, int],
            'layerwise_lr': [1, int],
            'ds_bucket_mb': [200, int],

            'my_sample_len': [0, int],
            'my_ffn_shift': [1, int],
            'my_att_shift': [1, int],
            'head_size_a': [64, int],
            'head_size_divisor': [8, int],
            'my_pos_emb': [0, int],
            'load_partial': [0, int],
            'magic_prime': [0, int],
            'my_qa_mask': [0, int],
            'my_random_steps': [0, int],
            'my_testing': ['x052', str],
            'my_exit': [99999999, int],
            'my_exit_tokens': [0, int],

            'emb': [False, bool], # 当传了该参数就是True，没传就是False
            'lora': [False, bool],
            'lora_load': ['', str],
            'lora_r': [8, int],
            'lora_alpha': [32, float],
            'lora_dropout': [0.01, float],
            'lora_parts': ['att,ln,time', str],

            'LISA': [False, bool],
            'lisa_r': [2, int],
            'lisa_k': [100, int],

            'PISSA': [False, bool],
            'svd_niter': [4, int],

            'quant': ['none', str],

            'dataload': ['get', str],

            'state_tune': [False, bool],

            'chunk_ctx': [512, int],

            'fla': [False, bool],

            'train_type': ['none', str],


        }

        if pl.__version__[0] == '2':
            allowed_arguments["accelerator"] =  ["gpu", str]
            allowed_arguments["strategy"] = ["auto",str]
            allowed_arguments["devices"] = [1, int]
            allowed_arguments["num_nodes"]= [1, int]
            allowed_arguments["precision"]= ["fp16", str]
            allowed_arguments["accumulate_grad_batches"] = [1, int]
        else:
            parser = ArgumentParser()
            parser = Trainer.add_argparse_args(parser) # 添加解析其它版本参数
            type_map = {'float': float, 'str': str, 'int': int,
                        'str_to_bool_or_str': parsing.str_to_bool_or_str,
                        '_precision_allowed_type': pl_argparse._precision_allowed_type,
                        'str_to_bool': parsing.str_to_bool,
                        'str_to_bool_or_int':parsing.str_to_bool_or_int,
                        '_gpus_allowed_type': pl_argparse._gpus_allowed_type,
                        '_int_or_float_type': pl_argparse._int_or_float_type,
                        }
            for item in parser._actions:
                value_type = type_map.get(getattr(item.type, '__name__', None), None)
                name = item.dest
                default_value = item.default
                allowed_arguments[name] = (default_value, value_type)
        for key, value in kwargs.items():
            values = allowed_arguments.get(key)

            if values and isinstance(values,(list, tuple)):
                value = value or values[0]
                if len(values) == 2:
                    type_name = type(values[1])
                    if (type_name == 'type' and not isinstance(value, values[1])) or type_name == 'function': # 参数校验
                        if value is not None:
                            try:
                                value = values[1](value)
                            except:
                                raise TypeError(f"{key} type error")


            setattr(self, key, value)
        # 设置默认值
        for key, values in allowed_arguments.items():
            if not hasattr(self, key):
                setattr(self, key, values[0])

        super().__init__(**kwargs)



class RWKVPEFTTrainer:
    def __init__(self, **kwargs):
        self.args = RWKVPEFTArgumentParser(**kwargs)

    def run(self):
        rank_zero_info("########## work in progress ##########")
        args = self.args
        if "deepspeed" in args.strategy:
            import deepspeed
        if args.random_seed >= 0:
            print(
                f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
            seed_everything(args.random_seed)

        np.set_printoptions(precision=4, suppress=True, linewidth=200)
        warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
        warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")
        # os.environ["WDS_SHOW_SEED"] = "1"

        args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
        args.enable_checkpointing = False
        args.replace_sampler_ddp = False
        args.logger = False
        args.gradient_clip_val = 1.0
        args.num_sanity_val_steps = 0
        args.check_val_every_n_epoch = int(1e20)
        args.log_every_n_steps = int(1e20)
        args.max_epochs = -1  # continue forever
        if args.dataload != 'get':
            args.max_epochs = args.epoch_count
        args.betas = (args.beta1, args.beta2)
        args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
        os.environ["RWKV_MY_TESTING"] = args.my_testing
        os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
        os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
        ######state tuning
        os.environ["RWKV_TRAIN_TYPE"] = ''
        if args.train_type == 'state':
            os.environ["RWKV_TRAIN_TYPE"] = 'states'
        elif args.train_type == 'infctx':
            os.environ["RWKV_TRAIN_TYPE"] = 'infctx'

        os.environ["WKV"] = 'fla' if args.fla else ''
        if args.dim_att <= 0:
            args.dim_att = args.n_embd
        if args.dim_ffn <= 0:
            args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)  # default = 3.5x emb size

        if args.data_type == "wds_img":
            args.run_name = f"v{args.my_img_version}-{args.my_img_size}-{args.my_img_bit}bit-{args.my_img_clip}x{args.my_img_clip_scale}"
            args.proj_dir = f"{args.proj_dir}-{args.run_name}"
        else:
            args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"
        if not os.path.exists(args.proj_dir):
            os.makedirs(args.proj_dir)

        if args.my_pile_stage > 0:
            magic_prime_bak = args.magic_prime

            if args.my_pile_shift < 0:
                args.my_pile_shift = 0

            if magic_prime_bak > 0:
                args.magic_prime = magic_prime_bak
            if args.my_qa_mask == 2:
                args.epoch_count = 2 * args.magic_prime // 40320
            else:
                args.epoch_count = args.magic_prime // 40320

            args.epoch_steps = 40320 // args.real_bsz
            assert args.epoch_steps * args.real_bsz == 40320
            # if args.my_pile_stage == 2:
            #     assert args.lr_final == args.lr_init
            if args.my_pile_stage >= 2:  # find latest saved model
                list_p = []
                for p in os.listdir(args.proj_dir):
                    if p.startswith("rwkv") and p.endswith(".pth"):
                        p = ((p.split("-"))[1].split("."))[0]
                        if p != "final":
                            if p == "init":
                                p = -1
                            else:
                                p = int(p)
                            list_p += [p]
                list_p.sort()
                max_p = list_p[-1]
                if len(list_p) > 1:
                    args.my_pile_prev_p = list_p[-2]  # in case max_p is corrupted
                if max_p == -1:
                    args.load_model = f"{args.proj_dir}/rwkv-init.pth"
                else:
                    args.load_model = f"{args.proj_dir}/rwkv-{max_p}.pth"
                    if args.warmup_steps < 0:
                        if args.my_pile_stage == 2:
                            args.warmup_steps = 10
                        else:
                            args.warmup_steps = 30
                args.epoch_begin = max_p + 1

        samples_per_epoch = args.epoch_steps * args.real_bsz
        tokens_per_epoch = samples_per_epoch * args.ctx_len
        try:
            deepspeed_version = deepspeed.__version__
        except:
            deepspeed_version = None
        rank_zero_info(
            f"""
        ############################################################################
        #
        # RWKV-5 {args.precision.upper()} on {args.num_nodes}x{args.devices} {args.accelerator.upper()}, bsz {args.num_nodes}x{args.devices}x{args.micro_bsz}={args.real_bsz}, {args.strategy} {'with grad_cp' if args.grad_cp > 0 else ''}
        #
        # Data = {args.data_file} ({args.data_type}), ProjDir = {args.proj_dir}
        #
        # Epoch = {args.epoch_begin} to {args.epoch_begin + args.epoch_count - 1} (will continue afterwards), save every {args.epoch_save} epoch
        #
        # Each "epoch" = {args.epoch_steps} steps, {samples_per_epoch} samples, {tokens_per_epoch} tokens
        #
        # Model = {args.n_layer} n_layer, {args.n_embd} n_embd, {args.ctx_len} ctx_len
        #
        # Adam = lr {args.lr_init} to {args.lr_final}, warmup {args.warmup_steps} steps, beta {args.betas}, eps {args.adam_eps}
        #
        # Found torch {torch.__version__}, recommend 1.13.1+cu117 or newer
        # Found deepspeed {deepspeed_version}, recommend 0.7.0 (faster than newer versions)
        # Found pytorch_lightning {pl.__version__}, recommend 1.9.5
        #
        ############################################################################
        """
        )
        rank_zero_info(str(vars(args)) + "\n")

        assert args.data_type in ["utf-8", "utf-16le", "numpy", "binidx", "dummy", "uint16"]

        if args.lr_final == 0 or args.lr_init == 0:
            rank_zero_info("\n\nNote: lr_final = 0 or lr_init = 0. Using linear LR schedule instead.\n\n")

        assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
        os.environ["RWKV_FLOAT_MODE"] = args.precision
        if args.precision == "fp32":
            for i in range(10):
                rank_zero_info("\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n")
        if args.precision == "fp16":
            rank_zero_info("\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")

        os.environ["RWKV_JIT_ON"] = "0"
        if "deepspeed_stage_3" in args.strategy:
            os.environ["RWKV_JIT_ON"] = "0"

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        if args.precision == "fp32":
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        else:
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True

        if "32" in args.precision:
            args.precision = 32
        elif args.precision == "fp16":
            args.precision = 16
        else:
            args.precision = "bf16"

        ########################################################################################################
        from rwkv_peft.src.trainer import train_callback, generate_init_weight
        from rwkv_peft.src.dataset import MyDataset
        # 该导入包不能写在文件前面

        train_data = MyDataset(args)
        args.vocab_size = train_data.vocab_size

        from rwkv_peft.src.rwkvLinear import LORA_CONFIG, LoraLinear
        from rwkv_peft.src.model import RWKV
        if args.lora:
            assert args.lora_r > 0, "LoRA should have its `r` > 0"
            LORA_CONFIG["r"] = args.lora_r
            LORA_CONFIG["alpha"] = args.lora_alpha
            LORA_CONFIG["dropout"] = args.lora_dropout
            LORA_CONFIG["parts"] = set(str(args.lora_parts).split(','))
            enable_time_finetune = 'time' in LORA_CONFIG["parts"]
            enable_ln_finetune = 'ln' in LORA_CONFIG["parts"]
        if args.quant != 'none':
            LORA_CONFIG["quant"] = True
        model = RWKV(args)
        freeze = False
        if args.lora or args.LISA or args.train_type == 'state':
            model.requires_grad_(False)
            freeze = True

        if args.state_tune or args.train_type == 'state':
            for name, module in model.named_modules():
                for pname, param in module.named_parameters():
                    if 'state' in pname:
                        param.requires_grad = True
                break

        if args.LISA:
            select_layers = np.random.choice(range(args.n_layer), args.lisa_r, replace=False)
            for name, module in model.named_modules():
                for pname, param in module.named_parameters():
                    if 'emb' in pname or 'head' in pname or '.ln' in pname or 'time' in pname:
                        param.requires_grad = True
                    match = re.search(r'\d+', pname)
                    if match:
                        number = int(match.group())
                        if number in select_layers:
                            param.requires_grad = True
                break

        elif args.lora:
            for name, module in model.named_modules():
                if len(args.load_model) == 0:
                    if any(n.startswith("emb.") for n, _ in module.named_parameters()):
                        for pname, param in module.named_parameters():
                            if 'emb.weight' == pname:
                                print(f'  EMB additionally training module {pname}')
                                param.requires_grad = True
                    if any(n.startswith("head.") for n, _ in module.named_parameters()):
                        for pname, param in module.named_parameters():
                            if 'head.weight' == pname:
                                print(f'  head additionally training module {pname}')
                                param.requires_grad = True
                    if 'ln' in name:
                        print(f'  LoRA additionally training module {name}')
                        for param in module.parameters():
                            param.requires_grad = True
                if any(n.startswith("emb.") for n, _ in module.named_parameters()):
                    for pname, param in module.named_parameters():
                        if args.emb and 'emb.weight' == pname:
                            print(f'  EMB additionally training module {pname}')
                            param.requires_grad = True
                if any(n.startswith("head.") for n, _ in module.named_parameters()):
                    for pname, param in module.named_parameters():
                        if args.emb and 'head.weight' == pname:
                            print(f'  head additionally training module {pname}')
                            param.requires_grad = True
                if any(n.startswith("lora_") for n, _ in module.named_parameters()):
                    print(f'  LoRA additionally training module {name}')
                    for pname, param in module.named_parameters():
                        param.requires_grad = 'lora_' in pname
                elif enable_ln_finetune and '.ln' in name:
                    print(f'  LoRA additionally training module {name}')
                    for param in module.parameters():
                        param.requires_grad = True
                elif enable_time_finetune and any(n.startswith("time") for n, _ in module.named_parameters()):
                    for pname, param in module.named_parameters():
                        if pname.startswith("time"):
                            print(f'  LoRA additionally training parameter {pname}')
                            param.requires_grad = True

        if len(args.load_model) == 0 or args.my_pile_stage == 1:  # shall we build the initial weights?
            init_weight_name = f"{args.proj_dir}/rwkv-init.pth"
            generate_init_weight(model, init_weight_name)  # save initial weights
            args.load_model = init_weight_name

        rank_zero_info(f"########## Loading {args.load_model}... ##########")
        model.load_state_dict(torch.load(args.load_model, map_location="cpu"), strict=(not freeze))
        if os.path.isfile(args.lora_load):
            model.load_state_dict(torch.load(args.lora_load, map_location="cpu"),
                                  strict=False)
        if args.PISSA:
            init_dict = {}
            rank_zero_info(f"########## Init PISSA... ##########")
            for name, m in model.named_modules():
                if hasattr(m, "pissa_init") and callable(getattr(m, "pissa_init")):
                    m.pissa_init(args.svd_niter)
                    init_dict[f'{name}.init_lora_A'] = m.lora_A.data
                    init_dict[f'{name}.init_lora_B'] = m.lora_B.data
            torch.save(init_dict, f'{args.proj_dir}/init_lora.pth')

        if args.quant != 'none':
            rank_zero_info(f"########## Quant... ##########")
            for name, m in model.named_modules():
                if hasattr(m, "quant") and callable(getattr(m, "quant")):
                    m.quant(args.quant)

        if pl.__version__[0] == '2':
            trainer = Trainer(accelerator=args.accelerator, strategy=args.strategy, devices=args.devices,
                              num_nodes=args.num_nodes, precision=args.precision,
                              logger=args.logger, callbacks=[train_callback(args)], max_epochs=args.max_epochs,
                              check_val_every_n_epoch=args.check_val_every_n_epoch,
                              num_sanity_val_steps=args.num_sanity_val_steps,
                              log_every_n_steps=args.log_every_n_steps, enable_checkpointing=args.enable_checkpointing,
                              accumulate_grad_batches=args.accumulate_grad_batches,
                              gradient_clip_val=args.gradient_clip_val)
        else:
            trainer = Trainer.from_argparse_args(
                args,
                callbacks=[train_callback(args)],
            )

        if trainer.global_rank == 0:
            for n in model.state_dict():
                shape = model.state_dict()[n].shape
                shape = [i for i in shape if i != 1]
                if len(shape) > 1:
                    print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {n}")
                else:
                    print(f"{str(shape[0]).ljust(5)}       {n}")

        if "deepspeed" in args.strategy:
            trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = args.ds_bucket_mb * 1000 * 1000
            trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = args.ds_bucket_mb * 1000 * 1000

        # must set shuffle=False, persistent_workers=False (because worker is in another thread)
        data_loader = DataLoader(train_data, shuffle=False, pin_memory=True, batch_size=args.micro_bsz, num_workers=1,
                                 persistent_workers=False, drop_last=True)

        trainer.fit(model, data_loader)