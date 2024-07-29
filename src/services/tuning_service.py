#coding=utf-8
"""
微调服务
"""
import os

from configuration import config as project_config
from src.services import AbstractServiceWorker
from src.utils import RWKVPEFTTrainer


class ServiceWorker(AbstractServiceWorker):
    def init_with_config(self, config):
        self.proj_dir = config.get('project_dir')
        self.data_file = config.get('data_file')

    def process(self, cmd: dict):
        received_cmd = cmd.get('cmd')
        # 优先使用前端输入的数据，没有就用配置文件里的数值
        load_model = cmd.get('load_model') or project_config.default_base_model_path
        print("load_model:", load_model, '     yyyyyyyyyyyyyyyyy')
        proj_dir = cmd.get('proj_dir') or self.proj_dir
        data_file = cmd.get('data_file') or self.data_file
        if not load_model:
            raise ValueError("load_model is required")
        if not proj_dir:
            raise ValueError("proj_dir is required")
        if not data_file:
            raise ValueError("data_file is required")
        if not os.path.exists(load_model):
            raise FileNotFoundError(f"{load_model} not exists")
        if not os.path.exists(proj_dir):
            raise NotADirectoryError(f"{proj_dir} not exists")
        if not os.path.exists(data_file + '.idx'):
            raise FileNotFoundError(f"{data_file}.idx/bin not exists")

        quant = cmd.get('quant', 'nf4')
        n_layer = cmd.get('n_layer', 24)
        n_embd = cmd.get('n_embd', 2048)
        ctx_len = cmd.get('ctx_len', 1024)
        data_type = cmd.get('data_type', 'binidx')
        epoch_save = cmd.get('epoch_save', 1)
        vocab_size = cmd.get('vocab_size', 65536)
        epoch_begin = cmd.get('epoch_begin', 0)
        pre_ffn = cmd.get('pre_ffn', 0)
        head_qk = cmd.get('head_qk', 0)
        beta1 = cmd.get('beta1', 0.9)
        beta2 = cmd.get('beta2', 0.99)
        adam_eps = cmd.get('adam_eps', 1e-8)
        my_testing = cmd.get('my_testing', 'x060')
        strategy = cmd.get('strategy', 'deepspeed_stage_1')
        wandb = cmd.get('wandb', '')

        if received_cmd == "LORA":
            lora_r = cmd.get('lora_r', 64)
            lora_alpha = cmd.get('lora_alpha', 128)
            micro_bsz = cmd.get('micro_bsz', 8)
            epoch_steps = cmd.get('epoch_steps', 1000)
            epoch_count = cmd.get('epoch_count', 20)
            lr_init = cmd.get('lr_init', 5e-5)
            lr_final = cmd.get('lr_final', 5e-5)
            warmup_steps = cmd.get('warmup_steps', 0)
            accelerator = cmd.get('accelerator', 'gpu')
            devices = cmd.get('devices', 1)
            precision = cmd.get('precision', 'bf16')
            grad_cp = cmd.get('grad_cp', 1)
            my_testing = cmd.get('my_testing', 'x060')
            lora_load = cmd.get('lora_load', 'rwkv-0')
            lora = cmd.get('lora', True)
            lora_dropout = cmd.get('lora_dropout', 0.01)
            lora_parts = cmd.get('lora_parts', 'att,ffn,time,ln')

            trainer = RWKVPEFTTrainer(load_model=load_model, proj_dir=proj_dir, data_file=data_file, data_type=data_type,
                                      vocab_size=vocab_size, ctx_len=ctx_len, epoch_steps=epoch_steps,
                                      epoch_count=epoch_count,epoch_begin=epoch_begin,epoch_save=epoch_save,
                                      micro_bsz=micro_bsz,n_layer=n_layer,n_embd=n_embd,pre_ffn=pre_ffn,
                                      head_qk=head_qk,lr_init=lr_init,lr_final=lr_final,warmup_steps=warmup_steps,
                                      beta1=beta1,beta2=beta2,adam_eps=adam_eps,accelerator=accelerator,devices=devices,
                                      precision=precision,strategy=strategy,grad_cp=grad_cp,my_testing=my_testing,
                                      lora_load=lora_load,lora=lora,lora_r=lora_r,lora_alpha=lora_alpha,
                                      lora_dropout=lora_dropout,lora_parts=lora_parts,wandb=wandb)
        elif received_cmd == 'STATE_TUNING':
            micro_bsz = cmd.get('micro_bsz', 1)
            epoch_steps = cmd.get('epoch_steps', 800)
            epoch_count = cmd.get('epoch_count', 10)
            lr_init = cmd.get('lr_init', 1)
            lr_final = cmd.get('lr_final', 1e-2)
            warmup_steps = cmd.get('warmup_steps', 10)
            dataload = cmd.get('dataload', 'pad')
            accelerator = cmd.get('accelerator', 'gpu')
            devices = cmd.get('devices', 1)
            precision = cmd.get('precision', 'bf16')
            grad_cp = cmd.get('grad_cp', 1)
            trainer = RWKVPEFTTrainer(load_model=load_model, proj_dir=proj_dir, data_file=data_file, data_type=data_type,
                                      vocab_size= vocab_size, ctx_len=ctx_len, epoch_steps=epoch_steps,
                                      epoch_count=epoch_count,epoch_begin=epoch_begin,epoch_save=epoch_save,
                                      micro_bsz=micro_bsz, n_layer=n_layer, n_embd=n_embd, pre_ffn=pre_ffn,
                                      head_qk=head_qk, lr_init=lr_init, lr_final=lr_final,warmup_steps=warmup_steps,
                                      beta1=beta1, beta2=beta2,adam_eps=adam_eps,
                                      accelerator=accelerator,devices=devices,precision=precision,strategy=strategy,
                                      grad_cp=grad_cp, my_testing=my_testing,
                                      train_type='state', dataload=dataload, quant=quant,wandb=wandb)
        elif received_cmd == 'PISSA':
            svd_niter = cmd.get('svd_niter', 4)
            lora_r = cmd.get('lora_r', 64)
            micro_bsz = cmd.get('micro_bsz', 8)
            epoch_steps = cmd.get('epoch_steps', 1000)
            epoch_count = cmd.get('epoch_count', 1)
            lr_init = cmd.get('lr_init', 5e-5)
            lr_final = cmd.get('lr_final', 5e-5)
            warmup_steps = cmd.get('warmup_steps', 0)
            accelerator = cmd.get('accelerator', 'gpu')
            precision = cmd.get('precision', 'bf16')
            grad_cp = cmd.get('grad_cp', 1)
            devices = cmd.get('devices', 1)
            lora_load = cmd.get('lora_load', 'rwkv-0')
            lora = cmd.get('lora', True)
            lora_alpha = cmd.get('lora_alpha', 128)
            lora_dropout = cmd.get('lora_dropout', 0.01)
            lora_parts = cmd.get('lora_parts', 'att,ffn,time,ln')
            pissa = cmd.get('PISSA', True)
            dataload = cmd.get('dataload', 'pad')
            trainer = RWKVPEFTTrainer(load_model=load_model, proj_dir=proj_dir, data_file=data_file,
                                      data_type=data_type,
                                      vocab_size=vocab_size, ctx_len=ctx_len, epoch_steps=epoch_steps,
                                      epoch_count=epoch_count, epoch_begin=epoch_begin, epoch_save=epoch_save,
                                      micro_bsz=micro_bsz, n_layer=n_layer, n_embd=n_embd, pre_ffn=pre_ffn,
                                      head_qk=head_qk, lr_init=lr_init, lr_final=lr_final, warmup_steps=warmup_steps,
                                      beta1=beta1, beta2=beta2, adam_eps=adam_eps,accelerator=accelerator,
                                      devices=devices,precision=precision,strategy=strategy, my_testing=my_testing,
                                      lora_load=lora_load,lora=lora,lora_r=lora_r,lora_alpha=lora_alpha,
                                      lora_dropout=lora_dropout,lora_parts=lora_parts, PISSA=pissa, svd_niter=svd_niter,
                                      grad_cp=grad_cp,  dataload=dataload, quant=quant, wandb=wandb)


        else:
            return ServiceWorker.UNSUPPORTED_COMMAND
        try:
            trainer.run()
        except:
            import traceback
            traceback.print_exc()