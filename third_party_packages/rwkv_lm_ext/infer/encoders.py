import os
os.environ['RWKV_JIT_ON'] = '0'
os.environ['RWKV_T_MAX'] = '4096'
os.environ['RWKV_FLOAT_MODE'] = 'bf16'
os.environ['RWKV_HEAD_SIZE_A'] = '64'
os.environ['RWKV_T_MAX'] = '4096'
os.environ["RWKV_MY_TESTING"]='x060'
os.environ['RWKV_CTXLEN'] = '4096'

from src.model_run import RWKV,RwkvForClassification,RwkvForSequenceEmbedding,PIPELINE_ARGS,create_empty_args,load_embedding_ckpt_and_parse_args,generate,generate_beamsearch
import torch
from src.layers import inject_lora_adapter_with_state_dict,set_adapter


class BiCrossFusionEncoder:
    """
    This encoder is to fuse the bi-encoder and cross-encoder with the same rwkv base model injected with 2 sets of lora adapters.
    We have 2 assumptions here:
    1. The bi-encoder and cross-encoder share the same base model
    2. The lora types of bi-encoder and cross-encoder are the same
    This class instance is not thread-safe since we need to switch the adapter name before encoding.
    """
    def __init__(self,
                 base_rwkv,
                 bi_lora_path,
                 cross_lora_path,
                 chat_lora_path,
                 tokenizer,
                 ce_lora_r=8,
                 ce_lora_alpha=32,
                 be_lora_r=8,
                 be_lora_alpha=32,
                 chat_lora_r=8,
                 chat_lora_alpha=8,
                 target_ce_modules=['emb','ffn.key','ffn.value','ffn.receptance'],
                 target_be_modules=['emb','ffn.key','ffn.value','ffn.receptance'],
                 target_chat_modules=['att','ffn'],
                 cross_adapter_name='cross_encoder_lora',
                 bi_adapter_name='bi_embedding_lora',
                 chat_adapter_name='chat_lora',
                 sep_token_id = 2,
                 chat_pissa_path = None,
                 device = 'cuda',
                 dtype = torch.bfloat16
                 ) -> None:
        self.base_rwkv = base_rwkv
        args = create_empty_args()
        w = load_embedding_ckpt_and_parse_args(base_rwkv,args)
        rwkv = RWKV(args)
        info = rwkv.load_state_dict(w)
        print(f'load model from {base_rwkv},result is {info}')
        should_delete_head = False
        #load cross encoder and inject cross adapter
        cross_encoder_dict = torch.load(cross_lora_path,map_location='cpu')
        inject_lora_adapter_with_state_dict(
            rwkv,
            cross_adapter_name,
            cross_encoder_dict,
            ce_lora_r,
            ce_lora_alpha,
            targets=target_ce_modules,
        )
        self.cross_encoder = RwkvForClassification(rwkv,should_delete_head=should_delete_head)
        self.cross_encoder.score.weight.data = cross_encoder_dict['score.weight']
        print(f'load model from {cross_lora_path},result is {info}')
        del cross_encoder_dict
        #load bi encoder and inject bi adapter
        bi_encoder_dict = torch.load(bi_lora_path,map_location='cpu')
        inject_lora_adapter_with_state_dict(
            rwkv,
            bi_adapter_name,
            bi_encoder_dict,
            be_lora_r,
            be_lora_alpha,
            targets=target_be_modules,
        )
        add_mlp = 'dense.weight' in bi_encoder_dict
        output_dim = 0
        if add_mlp:
            output_dim = bi_encoder_dict['dense.weight'].shape[0]
        print(f'RWKV Embedding model add_mlp = {add_mlp} output_dim = {output_dim}')
        self.bi_encoder = RwkvForSequenceEmbedding(rwkv,add_mlp=add_mlp,output_dim=output_dim,should_delete_head=should_delete_head)
        if add_mlp:
            self.bi_encoder.dense.weight.data = bi_encoder_dict['dense.weight']
            self.bi_encoder.dense.bias.data = bi_encoder_dict['dense.bias']
        #load chat lora and inject chat adapter
        chat_lora_dict = torch.load(chat_lora_path,map_location='cpu')
        pissa =  torch.load(chat_pissa_path, map_location='cpu') if chat_pissa_path else None
        inject_lora_adapter_with_state_dict(
            rwkv,
            chat_adapter_name,
            chat_lora_dict,
            chat_lora_r,
            chat_lora_alpha,
            targets=target_chat_modules,
            pissa_dict=pissa
        )
        self.tokenizer = tokenizer
        self.sep_token_id = sep_token_id
        self.rwkv = rwkv
        
        #move to device
        self.cross_encoder = self.cross_encoder.to(device=device,dtype=dtype)
        self.bi_encoder = self.bi_encoder.to(device=device,dtype=dtype)
        self.rwkv = self.rwkv.to(device=device,dtype=dtype)
        self.rwkv.eval()
        self.cross_adapter_name = cross_adapter_name
        self.bi_adapter_name = bi_adapter_name
        self.chat_adapter_name = chat_adapter_name
        self.device = device
        self.dtype = dtype

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        set_adapter(model=self.rwkv,adapter_name=adapter_name)

    def encode_text(self,text,chunk_size=1024):
        input_ids =  tokenizer.encode(text)
        input_ids.append(self.bi_encoder.embedding_id)
        state = None
        offset = 0
        while offset < len(input_ids):
            chunk = input_ids[offset:offset+chunk_size]
            with torch.autocast(enabled=True,device_type=self.device,dtype=self.dtype):
                outputs,state = self.bi_encoder(torch.tensor(chunk,dtype=torch.long,device=self.device),state=state)
            offset += len(chunk)

        return outputs.tolist()

    def encode_texts(self,texts):
        self.set_adapter(self.bi_adapter_name)
        return [self.encode_text(text) for text in texts]
    
    def cross_encode_text(self,text_a, text_b):
        text_a_ids = self.tokenizer.encode(text_a)
        text_b_ids = self.tokenizer.encode(text_b)
        input_ids = text_a_ids+[self.sep_token_id]+text_b_ids+[self.cross_encoder.class_id]
        offset = 0
        state = None
        with torch.autocast(enabled=True,device_type=self.device,dtype=self.dtype):
            while offset < len(input_ids):
                chunk = input_ids[offset:offset+1024]
                output,state = self.cross_encoder(torch.tensor(chunk,dtype=torch.long,device=self.device),state=state)
                offset += len(chunk)
        return output.item()

    def cross_encode_texts(self,texts_a, texts_b):
        assert len(texts_a) == len(texts_b)
        self.set_adapter(self.cross_adapter_name)
        outputs = []
        for text_a,text_b in zip(texts_a,texts_b):
            outputs.append(self.cross_encode_text(text_a,text_b))
        return outputs
    
    def beam_generate(self,instruction,input_text,token_count=128,num_beams=5,return_num_sequences=5,num_group=5,do_sample=True,is_sum_logprobs=True,length_penalty=0.6):
        self.set_adapter(self.chat_adapter_name)
        cat_char = '🐱'
        bot_char = '🤖'
        ctx = f'{cat_char}:{instruction}\n{input_text}\n{bot_char}:'
        with torch.no_grad():
            with torch.autocast(enabled=True,device_type=self.device,dtype=self.dtype):
                results = generate_beamsearch(
                    self.rwkv, 
                    ctx,self.tokenizer, 
                    token_count=token_count,
                    num_beams=num_beams,
                    return_num_sequences=return_num_sequences,
                    num_group=num_group,
                    do_sample=do_sample,
                    is_sum_logprobs=is_sum_logprobs,
                    length_penalty=length_penalty)
        import math
        results = [(tokenizer.decode(output.tolist()),math.exp(score.item()),beam_idx) for score, output,beam_idx in results]   
        return results

    def sampling_generate(self,instruction,input_text,token_count=128,
                          temperature=1.0,
                          top_p=0,
                          top_k=0,
                          alpha_frequency=0.25,
                          alpha_presence=0.25,
                          alpha_decay=0.996,
                          token_stop=[0,1]):
        self.set_adapter(self.chat_adapter_name)
        gen_args = PIPELINE_ARGS(temperature = temperature, top_p = top_p, top_k=top_k, # top_k = 0 then ignore
                        alpha_frequency = alpha_frequency,
                        alpha_presence = alpha_presence,
                        alpha_decay = alpha_decay, # gradually decay the penalty
                        token_ban = [], # ban the generation of some tokens
                        token_stop = [0,1], # stop generation whenever you see any token here
                        chunk_len = 512)
        cat_char = '🐱'
        bot_char = '🤖'
        ctx = f'{cat_char}:{instruction}\n{input_text}\n{bot_char}:'
        with torch.no_grad():
            with torch.autocast(enabled=True,device_type=self.device,dtype=self.dtype):
                output = generate(self.rwkv,ctx,self.tokenizer,token_count=token_count,args=gen_args,callback=None)
        return output
