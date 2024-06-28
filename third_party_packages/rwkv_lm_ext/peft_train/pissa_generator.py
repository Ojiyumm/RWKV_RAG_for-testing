ckpt = '/media/yueyulin/KINGSTON/models/rwkv6/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'
lora_file = '/media/yueyulin/data_4t/models/pissa/epoch_6/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth.pth'
tokenizer_file = '/home/yueyulin/github/RWKV_LM_EXT/tokenizer/rwkv_vocab_v20230424.txt'
target_modules = ['emb','ffn.key','ffn.value','ffn.receptance','att.key','att.value','att.receptance']
lora_r = 8
lora_alpha = 32
lora_dropout = 0
is_lora = True
import os
parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.append(parent_path)
print(f'add path: {parent_path} to sys.path')
os.environ['RWKV_JIT_ON'] = '0'
os.environ['RWKV_T_MAX'] = '4096'
os.environ['RWKV_FLOAT_MODE'] = 'bf16'
os.environ['RWKV_HEAD_SIZE_A'] = '64'
os.environ['RWKV_T_MAX'] = '4096'
os.environ["RWKV_MY_TESTING"]='x060'
os.environ['RWKV_CTXLEN'] = '4096'
import torch
from src.model_run import RWKV,PIPELINE_ARGS,create_empty_args,load_embedding_ckpt_and_parse_args,generate
device = 'cuda'
dtype = torch.bfloat16
args = create_empty_args()
w = load_embedding_ckpt_and_parse_args(ckpt, args)
print(args)
model = RWKV(args)
info = model.load_state_dict(w)
model.eval()
print(model)
print(info)
if is_lora:
    from peft import LoraConfig
    lora_config = LoraConfig(
        # init_lora_weights="pissa_niter_4",
        target_modules=target_modules,lora_dropout=lora_dropout)
    from peft import inject_adapter_in_model
    model = inject_adapter_in_model(lora_config,model,adapter_name='sft_lora')
    print(model)
    states = torch.load(lora_file)
    print(states.keys())
    info = model.load_state_dict(states,strict=False)
    print(info)
    model.eval()
states_value = None
gen_args = PIPELINE_ARGS(temperature = 1, top_p = 0.96, top_k = 20, # top_k = 0 then ignore
                        alpha_frequency = 0.25,
                        alpha_presence = 0.25,
                        alpha_decay = 0.996, # gradually decay the penalty
                        token_ban = [], # ban the generation of some tokens
                        token_stop = [0,1], # stop generation whenever you see any token here
                        chunk_len = 512)
cat_char = '🐱'
bot_char = '🤖'
instruction ='根据给定的短文，回答以下问题：黄循财的是哪国人？'
input_text = '黄循财（英语：Lawrence Wong Shyun Tsai，1972年12月18日—），新加坡华裔政治人物，现任新加坡总理兼财政部部长、人民行动党社区基金会主席。他与王乙康和颜金勇共同主持了因应新加坡2019冠状病毒病大流行的多部委工作组。曾任新加坡副总理，教育部、国家发展部、文化、社区及青年部的部长，通讯及新闻部和财政部的第二部长，以及人民行动党副秘书长。[1]黄循财是人民行动党第四代领导层，也是人民行动党中央执行委员会首任副秘书长兼政策论坛顾问。'
ctx = f'{cat_char}:{instruction}\n{input_text}\n{bot_char}:'
print(ctx)
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
tokenizer = TRIE_TOKENIZER(tokenizer_file)
print(tokenizer.encode(ctx))
model = model.to(dtype)
model = model.to(device)
with torch.no_grad():
    with torch.autocast(enabled=True,device_type='cuda',dtype=dtype):
        output = generate(model, ctx,tokenizer, token_count=256, args=gen_args,callback=None,state=None)
    print(output)
