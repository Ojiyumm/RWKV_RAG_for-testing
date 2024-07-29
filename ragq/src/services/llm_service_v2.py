import os
import math

import torch
from FlagEmbedding import FlagReranker, BGEM3FlagModel
from rwkv.model import RWKV as OriginRWKV
from rwkv.utils import PIPELINE

from src.services.helpers import ServiceWorker as _ServiceWorker
from rwkv_lm_ext.src.model_run import (RWKV,RwkvForClassification,RwkvForSequenceEmbedding,
                                       PIPELINE_ARGS,create_empty_args,load_embedding_ckpt_and_parse_args,
                                       generate,generate_beamsearch)
from rwkv_lm_ext.src.layers import inject_lora_adapter_with_state_dict,set_adapter
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER

os.environ['RWKV_JIT_ON'] = '1'
os.environ['RWKV_T_MAX'] = '4096'
os.environ['RWKV_FLOAT_MODE'] = 'bf16'
os.environ['RWKV_HEAD_SIZE_A'] = '64'
os.environ['RWKV_T_MAX'] = '4096'
os.environ["RWKV_MY_TESTING"]='x060'
os.environ['RWKV_CTXLEN'] = '4096'



class LLMService:
    
    def __init__(self,
                 base_rwkv,
                 tokenizer,
                 load_finetuning_type='state',
                 sep_token_id=2,
                 device = 'cuda',
                 dtype = torch.bfloat16,
                 **kwargs
                 ) -> None:
        """
        Args:
            tokenizer:tokenizer.rwkv_tokenizer.TRIE_TOKENIZER
        """
        # TODO Deprecated
        # self.cross_encoder = None
        # self.cross_adapter_name = ''
        # self.bi_encoder = None
        # self.bi_adapter_name = ''
        # self.chat_adapter_name = ''

        self.base_rwkv = base_rwkv
        self._is_state_tuning = load_finetuning_type == 'state'
        self.device = device
        self.dtype = dtype
        self.kwargs = kwargs

        if self._is_state_tuning:
            strategy = kwargs.get('strategy', 'cuda fp16')
            self.model = OriginRWKV(base_rwkv, strategy=strategy)
            info = vars(self.model.args) # types.SimpleNamespace()实例转字典

        else:
            # TODO Deprecated
            args = create_empty_args()
            self.model = RWKV(args)
            w = load_embedding_ckpt_and_parse_args(base_rwkv, args)
            info = self.model.load_state_dict(w)
        print(f'load model from {base_rwkv},result is {info}')

        self.tokenizer = tokenizer
        self.sep_token_id = sep_token_id
        self.states_value = []

        #move to device
        # TODO Deprecated
        # self.cross_encoder = self.cross_encoder.to(device=device, dtype=dtype)
        # self.bi_encoder = self.bi_encoder.to(device=device, dtype=dtype)
        # self.model = self.model.to(device=device, dtype=dtype)
        # self.model.eval()

        self.bgem3 = None
        self.reranker = None

    # def load_cross_encoder(self):
    #     # load cross encoder and inject cross adapter
    #     kwargs = self.kwargs
    #     cross_lora_path = kwargs.get('cross_lora_path')
    #     cross_encoder_dict = torch.load(cross_lora_path, map_location='cpu')
    #     cross_adapter_name = kwargs.get('cross_adapter_name', 'cross_encoder_lora')
    #     ce_lora_r = kwargs.get('cross_lora_r', 8)
    #     ce_lora_alpha = kwargs.get('cross_lora_alpha', 32)
    #     target_ce_modules = kwargs.get('cross_encoder_targets', ['emb', 'ffn.key', 'ffn.value', 'ffn.receptance'])
    #     inject_lora_adapter_with_state_dict(
    #         self.model,
    #         cross_adapter_name,
    #         cross_encoder_dict,
    #         ce_lora_r,
    #         ce_lora_alpha,
    #         targets=target_ce_modules,
    #     )
    #     self.cross_encoder = RwkvForClassification(self.model, should_delete_head=False)
    #     self.cross_encoder.score.weight.data = cross_encoder_dict['score.weight']
    #     self.cross_adapter_name = cross_adapter_name
    #     del cross_encoder_dict
    #     print(f'load model from {cross_lora_path}')


    # def load_bi_encoder(self):
    #     # load bi encoder and inject bi adapter
    #     kwargs = self.kwargs
    #     bi_lora_path = kwargs.get('bi_lora_path')
    #     bi_encoder_dict = torch.load(bi_lora_path, map_location='cpu')
    #     bi_adapter_name = kwargs.get('bi_adapter_name', 'bi_embedding_lora')
    #     be_lora_r = kwargs.get('be_lora_r', 8)
    #     be_lora_alpha = kwargs.get('be_lora_alpha', 32)
    #     target_be_modules = kwargs.get('bi_encoder_targets', ['emb', 'ffn.key', 'ffn.value', 'ffn.receptance'])
    #     inject_lora_adapter_with_state_dict(
    #         self.model,
    #         bi_adapter_name,
    #         bi_encoder_dict,
    #         be_lora_r,
    #         be_lora_alpha,
    #         targets=target_be_modules,
    #     )
    #     add_mlp = 'dense.weight' in bi_encoder_dict
    #     output_dim = 0
    #     if add_mlp:
    #         output_dim = bi_encoder_dict['dense.weight'].shape[0]
    #     print(f'RWKV Embedding model add_mlp = {add_mlp} output_dim = {output_dim}')
    #     self.bi_encoder = RwkvForSequenceEmbedding(self.model, add_mlp=add_mlp, output_dim=output_dim,
    #                                                should_delete_head=False)
    #     if add_mlp:
    #         self.bi_encoder.dense.weight.data = bi_encoder_dict['dense.weight']
    #         self.bi_encoder.dense.bias.data = bi_encoder_dict['dense.bias']
    #     self.bi_adapter_name = bi_adapter_name

    # def load_chat_lora(self):
    #     # load chat lora and inject chat adapter
    #     kwargs = self.kwargs
    #     chat_lora_path = kwargs.get('chat_lora_path')
    #     chat_pissa_path = kwargs.get('chat_pissa_path')
    #     chat_adapter_name = kwargs.get('chat_adapter_name', 'chat_lora')
    #     chat_lora_r = kwargs.get('chat_lora_r', 8)
    #     chat_lora_alpha = kwargs.get('chat_lora_alpha', 8)
    #     target_chat_modules = kwargs.get('chat_targets', ['att','ffn'])
    #     chat_lora_dict = torch.load(chat_lora_path, map_location='cpu')
    #     pissa = torch.load(chat_pissa_path, map_location='cpu') if chat_pissa_path else None
    #     inject_lora_adapter_with_state_dict(
    #         self.model,
    #         chat_adapter_name,
    #         chat_lora_dict,
    #         chat_lora_r,
    #         chat_lora_alpha,
    #         targets=target_chat_modules,
    #         pissa_dict=pissa
    #     )
    #     self.chat_adapter_name = chat_adapter_name

    def load_state_tuning(self, states_file):
        if self._is_state_tuning:
            assert states_file is not None
            states = torch.load(states_file)
            states_value = []
            # n_head = self.model.args.n_head
            # head_size = self.model.args.n_embd // self.model.args.n_head
            for i in range(self.model.args.n_layer):
                key = f'blocks.{i}.att.time_state'
                value = states[key]
                prev_x = torch.zeros(self.model.args.n_embd, device=self.device, dtype=torch.float16) # TODO 这一步精度要与self.dtype保持一致吗
                prev_states = value.clone().detach().to(device=self.device, dtype=torch.float16).transpose(1, 2)
                prev_ffn = torch.zeros(self.model.args.n_embd, device=self.device, dtype=torch.float16)
                states_value.append(prev_x)
                states_value.append(prev_states)
                states_value.append(prev_ffn)
            self.states_value = states_value



    def load_bgem3(self):
        if self.bgem3 is not None:
            return
        kwargs = self.kwargs
        model_path = kwargs.get('bgem3_path')
        assert model_path is not None
        self.bgem3 = BGEM3FlagModel(model_path,use_fp16=True)  # Setting use_fp16 to True speeds up computation with a slight performance degradation

    def load_rerank(self):
        if self.reranker is not None:
            return
        kwargs = self.kwargs
        model_path = kwargs.get('rerank_path')
        assert model_path is not None
        self.reranker = FlagReranker(model_path,use_fp16=True)  # Setting use_fp16 to True speeds

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        set_adapter(model=self.model,adapter_name=adapter_name)

    # def encode_text(self,text,chunk_size=1024):
    #     input_ids = self.tokenizer.encode(text)
    #     # input_ids.append(self.bgem3.embedding_id)
    #     state = None
    #     offset = 0
    #     while offset < len(input_ids):
    #         chunk = input_ids[offset:offset+chunk_size]
    #         with torch.autocast(enabled=True,device_type=self.device,dtype=self.dtype):
    #             outputs,state = self.bgem3(torch.tensor(chunk,dtype=torch.long,device=self.device),state=state)
    #         offset += len(chunk)
    #
    #     return outputs.tolist()

    # def encode_texts(self,texts):
    #     self.set_adapter(self.bi_adapter_name)
    #     return [self.encode_text(text) for text in texts]

    def get_embeddings(self,inputs):
        if isinstance(inputs,str):
            inputs = [inputs]
        print(inputs)
        self.load_bgem3()
        outputs = self.bgem3.encode(inputs, 
                                    batch_size=12, 
                                    max_length=512, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                    )['dense_vecs'].tolist()

        print(outputs)
        return outputs
    def cross_encode_text(self,text_a, text_b):
        self.load_rerank()
        score = self.reranker.compute_score([text_a, text_b])
        return score

    def cross_encode_texts(self,texts_a, texts_b):
        assert len(texts_a) == len(texts_b)
        outputs = []
        for text_a,text_b in zip(texts_a,texts_b):
            outputs.append(self.cross_encode_text(text_a,text_b))
        return outputs
    
    def beam_generate(self,instruction,input_text,token_count=128,num_beams=5,
                      return_num_sequences=5,num_group=5,do_sample=True,
                      is_sum_logprobs=True,length_penalty=0.6):
        #self.set_adapter(self.chat_adapter_name) # TODO ???
        cat_char = '🐱'
        bot_char = '🤖'
        ctx = f'{cat_char}:{instruction}\n{input_text}\n{bot_char}:'
        with torch.no_grad():
            with torch.autocast(enabled=True,device_type=self.device,dtype=self.dtype):
                results = generate_beamsearch(
                    self.model,
                    ctx,self.tokenizer, 
                    token_count=token_count,
                    num_beams=num_beams,
                    return_num_sequences=return_num_sequences,
                    num_group=num_group,
                    do_sample=do_sample,
                    is_sum_logprobs=is_sum_logprobs,
                    length_penalty=length_penalty)
        results = [(self.tokenizer.decode(output.tolist()),math.exp(score.item()),beam_idx) for score, output,beam_idx in results]   
        return results

    def sampling_generate(self,instruction,input_text,state_file,
                          temperature=1.0,
                          top_p=0,
                          top_k=0,
                          alpha_frequency=0.25,
                          alpha_presence=0.25,
                          alpha_decay=0.996,
                         ):
        #self.set_adapter(self.chat_adapter_name)
        try:
            self.load_state_tuning(state_file)
        except:
            print('11111111111111')
            import traceback
            print(traceback.format_exc())
        print('222222222')
        gen_args = PIPELINE_ARGS(temperature = temperature, top_p = top_p, top_k=top_k, # top_k = 0 then ignore
                        alpha_frequency = alpha_frequency,
                        alpha_presence = alpha_presence,
                        alpha_decay = alpha_decay, # gradually decay the penalty
                        token_ban = [0], # ban the generation of some tokens
                        token_stop = [0,1], # stop generation whenever you see any token here
                        chunk_len = 256)
        #cat_char = '🐱'
        #bot_char = '🤖'
        ctx = f'User: 请阅读下文，回答:用文章中详细的数据以及逻辑回答{instruction}\\n{input_text}\\n问题:用文章中详细的数据以及逻辑回答{instruction}\\n\\nAssistant:'
        #ctx = f'{cat_char}:{instruction}\n{input_text}\n{bot_char}:'
        print('prompt=',ctx)
        # with torch.no_grad():
        #     with torch.autocast(enabled=True,device_type=self.device,dtype=self.dtype):
        #         output = generate(self.model,ctx,self.tokenizer,token_count=token_count,args=gen_args,callback=None)
        pipeline = PIPELINE(self.model, "rwkv_vocab_v20230424")
        print('33333333')
        output = pipeline.generate(ctx, token_count=200, args=gen_args, state=self.states_value)
        print('444444444444')
        return output,ctx


class ServiceWorker(_ServiceWorker):
    def init_with_config(self, config):

        base_model_file = config["base_model_file"]
        bgem3_path = config["bgem3_path"]
        rerank_path = config["rerank_path"]

        try:

            tokenizer = TRIE_TOKENIZER()
            print('imported tokenizer')
        except Exception as e:
            raise ValueError('failed to print tokenizer',e)
        self.llm_service = LLMService(base_model_file, tokenizer, bgem3_path=bgem3_path,rerank_path=rerank_path)
    
    def process(self, cmd):
        if cmd['cmd'] == 'GET_EMBEDDINGS':
            texts = cmd["texts"]
            value = self.llm_service.get_embeddings(texts)
            return value
        elif cmd['cmd'] == 'GET_CROSS_SCORES':
            texts_0 = cmd["texts_0"]
            texts_1 = cmd["texts_1"]
            value = self.llm_service.cross_encode_texts(texts_0,texts_1)
            return value
        elif cmd['cmd'] == 'BEAM_GENERATE':
            instruction = cmd["instruction"]
            input_text = cmd["input_text"]
            token_count = cmd.get('token_count', 128)
            num_beams = cmd.get('num_beams', 5)
            return_num_sequences = cmd.get('return_num_sequences', 5)
            num_group = cmd.get('num_group', 5)
            do_sample = cmd.get('do_sample', True)
            is_sum_logprobs = cmd.get('is_sum_logprobs', True)
            length_penalty = cmd.get('length_penalty', 0.6)
            value=self.llm_service.beam_generate(instruction, input_text, token_count, num_beams)
            return value
        elif cmd['cmd'] == 'SAMPLING_GENERATE':
            instruction = cmd["instruction"]
            input_text = cmd["input_text"]
            temperature = cmd.get('temperature', 1.0)
            top_p = cmd.get('top_p', 0)
            state_file = cmd.get('state_file')
            value = self.llm_service.sampling_generate(instruction, input_text, state_file,temperature,top_p)
            return value       
        return ServiceWorker.UNSUPPORTED_COMMAND

