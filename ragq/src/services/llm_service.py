import os
import math

import torch
from FlagEmbedding import FlagReranker, BGEM3FlagModel
from rwkv.model import RWKV as OriginRWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

from src.services.helpers import ServiceWorker as _ServiceWorker
from rwkv_lm_ext.src.model_run import generate_beamsearch
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
                 device = 'cuda',
                 dtype = torch.bfloat16,
                 **kwargs
                 ) -> None:
        """
        Args:
            base_rwkv： str， the path of rwkv model
            tokenizer:tokenizer.rwkv_tokenizer.TRIE_TOKENIZER
        """

        self.base_rwkv = base_rwkv
        self._is_state_tuning = True # 暂支持state，后续支持lora
        self.device = device
        self.dtype = dtype
        self.kwargs = kwargs


        strategy = kwargs.get('strategy', 'cuda fp16')
        self.model = OriginRWKV(base_rwkv, strategy=strategy)
        info = vars(self.model.args)
        print(f'load model from {base_rwkv},result is {info}')
        self.tokenizer = tokenizer

        self.bgem3 = None
        self.reranker = None


    def load_state_tuning(self, states_file):
        assert states_file is not None
        states = torch.load(states_file)
        states_value = []

        for i in range(self.model.args.n_layer):
            key = f'blocks.{i}.att.time_state'
            value = states[key]
            prev_x = torch.zeros(self.model.args.n_embd, device=self.device, dtype=torch.float16)
            prev_states = value.clone().detach().to(device=self.device, dtype=torch.float16).transpose(1, 2)
            prev_ffn = torch.zeros(self.model.args.n_embd, device=self.device, dtype=torch.float16)
            states_value.append(prev_x)
            states_value.append(prev_states)
            states_value.append(prev_ffn)
        return states_value


    def load_bgem3(self):
        if self.bgem3 is not None:
            return
        kwargs = self.kwargs
        model_path = kwargs.get('bgem3_path')
        assert model_path is not None
        self.bgem3 = BGEM3FlagModel(model_path,use_fp16=True)

    def load_rerank(self):
        if self.reranker is not None:
            return
        kwargs = self.kwargs
        model_path = kwargs.get('rerank_path')
        assert model_path is not None
        self.reranker = FlagReranker(model_path,use_fp16=True)  # Setting use_fp16 to True speeds

    def get_embeddings(self,inputs):
        if isinstance(inputs,str):
            inputs = [inputs]
        self.load_bgem3()
        outputs = self.bgem3.encode(inputs, 
                                    batch_size=12, 
                                    max_length=512,
                                    )['dense_vecs'].tolist()

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
                print('33333333')
                try:
                    results = generate_beamsearch(
                    self.model,
                    ctx,self.tokenizer,
                    token_count=token_count,
                    num_beams=num_beams,
                    return_num_sequences=return_num_sequences,
                    num_group=num_group,
                    do_sample=do_sample,
                    is_sum_logprobs=is_sum_logprobs,
                    length_penalty=length_penalty,
                    device=self.device)
                except:
                    import traceback
                    print(traceback.format_exc())
        print('5555555')
        results = [(self.tokenizer.decode(output.tolist()),math.exp(score.item()),beam_idx) for score, output,beam_idx in results]
        print('2222222222')
        return results

    def sampling_generate(self,instruction,input_text,state_file,
                          temperature=1.0,
                          top_p=0,
                          top_k=0,
                          alpha_frequency=0.25,
                          alpha_presence=0.25,
                          alpha_decay=0.996,
                          template_prompt=None,
                         ):

        states_value = self.load_state_tuning(state_file)
        gen_args = PIPELINE_ARGS(temperature = temperature, top_p = top_p, top_k=top_k, # top_k = 0 then ignore
                        alpha_frequency = alpha_frequency,
                        alpha_presence = alpha_presence,
                        alpha_decay = alpha_decay, # gradually decay the penalty
                        token_ban = [0], # ban the generation of some tokens
                        token_stop = [0,1], # stop generation whenever you see any token here
                        chunk_len = 256)
        if not template_prompt:
            ctx = f'User: 请阅读下文，回答:用文章中详细的数据以及逻辑回答{instruction}\\n{input_text}\\n问题:用文章中详细的数据以及逻辑回答{instruction}\\n\\nAssistant:'
        else:
            ctx = template_prompt
        print('prompt=',ctx)
        pipeline = PIPELINE(self.model, "rwkv_vocab_v20230424")
        output = pipeline.generate(ctx, token_count=200, args=gen_args, state=states_value)
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
            value=self.llm_service.beam_generate(instruction, input_text, token_count, num_beams)
            return value
        elif cmd['cmd'] == 'SAMPLING_GENERATE':
            instruction = cmd["instruction"]
            input_text = cmd["input_text"]
            temperature = cmd.get('temperature', 1.0)
            top_p = cmd.get('top_p', 0)
            state_file = cmd.get('state_file')
            template_prompt = cmd.get('template_prompt')
            value = self.llm_service.sampling_generate(instruction, input_text, state_file,temperature,top_p,
                                                       template_prompt=template_prompt)
            return value       
        return ServiceWorker.UNSUPPORTED_COMMAND

