import sys
import os
RWKV_PATH ='/home/rwkv/Peter/RWKV_LM_EXT-main'
sys.path.append(RWKV_PATH)
from helpers import start_proxy, ServiceWorker
from src.model_run import RWKV,create_empty_args,load_embedding_ckpt_and_parse_args,BiCrossFusionEncoder,generate,enable_lora
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
import sqlitedict
import torch
os.environ['RWKV_JIT_ON'] = '0'
os.environ['RWKV_T_MAX'] = '4096'
os.environ['RWKV_FLOAT_MODE'] = 'bf16'
os.environ['RWKV_HEAD_SIZE_A'] = '64'
os.environ['RWKV_T_MAX'] = '4096'
os.environ["RWKV_MY_TESTING"]='x060'
os.environ['RWKV_CTXLEN'] = '4096'
from src.model_run import RWKV,RwkvForClassification,RwkvForSequenceEmbedding,PIPELINE_ARGS,create_empty_args,load_embedding_ckpt_and_parse_args,generate,generate_beamsearch
from torch.cuda import amp
from src.layers import inject_lora_adapter_with_state_dict,set_adapter
from FlagEmbedding import BGEM3FlagModel,FlagReranker


class LLMService:
    
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
        from FlagEmbedding import BGEM3FlagModel
        self.bgem3 = BGEM3FlagModel('/home/rwkv/Peter/model/bi/bge-m31',  
                       use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
        self.reranker = FlagReranker('/home/rwkv/Peter/model/bi/BAAIbge-reranker-v2-m3', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    def set_adapter(self, adapter_name: str | list[str]) -> None:
        set_adapter(model=self.rwkv,adapter_name=adapter_name)

    def encode_text(self,text,chunk_size=1024):
        input_ids = self.tokenizer.encode(text)
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
    def get_embeddings(self,inputs):
        if isinstance(inputs,str):
            inputs = [inputs]
        print(inputs)

        outputs = self.bgem3.encode(inputs, 
                                    batch_size=12, 
                                    max_length=512, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                    )['dense_vecs'].tolist()

        print(outputs)
        return outputs
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
        results = [(self.tokenizer.decode(output.tolist()),math.exp(score.item()),beam_idx) for score, output,beam_idx in results]   
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
        ctx = f'User: 请阅读下文，回答:用文章中详细的数据以及逻辑回答{instruction}\\n{input_text}\\n问题:用文章中详细的数据以及逻辑回答{instruction}\\n\\nAssistant:'
        print('prompt=',ctx)
        with torch.no_grad():
            with torch.autocast(enabled=True,device_type=self.device,dtype=self.dtype):
                output = generate(self.rwkv,ctx,self.tokenizer,token_count=token_count,args=gen_args,callback=None)
        return output,ctx
class ServiceWorker(ServiceWorker):
    def init_with_config(self, config):

        base_model_file = config["base_model_file"]
        bi_lora_path = config["bi_lora_path"]
        cross_lora_path = config["cross_lora_path"]
        chat_lora_path = config["chat_lora_path"]
        tokenizer_file = config["tokenizer_file"]
        from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
        try:

            tokenizer = TRIE_TOKENIZER(tokenizer_file)
            print('imported tokenizer')
        except Exception as e:
            print('failed to print tokenizer',e)    
        device = config.get("device", 'cuda:3')
        chat_lora_r = config["chat_lora_r"]
        chat_lora_alpha = config["chat_lora_alpha"]
        chat_pissa_path = config["chat_pissa_path"]
        self.llm_service=LLMService(
            base_model_file,
            bi_lora_path,
            cross_lora_path,
            chat_lora_path,
            tokenizer,
            chat_lora_r=chat_lora_r,
            chat_lora_alpha=chat_lora_alpha,
            chat_pissa_path=chat_pissa_path)
    
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
            token_count = cmd.get('token_count', 128)  # 如果没有提供，默认值为128
            temperature = cmd.get('temperature', 1.0)
            top_p = cmd.get('top_p', 0)
            value = self.llm_service.sampling_generate(instruction, input_text, token_count, temperature, top_p)    
            return value       
        return ServiceWorker.UNSUPPORTED_COMMAND


if __name__ == '__main__':

    base_rwkv_model = '/home/rwkv/Peter/model/base/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'
    bi_lora_path = '/home/rwkv/Peter/model/bi/RWKV-x060-World-1B6-v2_rwkv_lora.pth'
    cross_lora_path = '/home/rwkv/Peter/model/cross/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth (1).pth'
    tokenizer_file = os.path.join('/home/rwkv/Peter/RWKV_LM_EXT-main/tokenizer/rwkv_vocab_v20230424.txt')
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer = TRIE_TOKENIZER(tokenizer_file)
    chat_lora_path = '/home/rwkv/Peter/model/chat/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth.pth'
    chat_pissa_path = '/home/rwkv/Peter/model/chat/init_pissa.pth'
    chat_lora_r = 64
    chat_lora_alpha = 64
    llm_service=LLMService(
        base_rwkv_model,
        bi_lora_path,
        cross_lora_path,
        chat_lora_path,
        tokenizer,
        chat_lora_r=chat_lora_r,
        chat_lora_alpha=chat_lora_alpha,
        chat_pissa_path=chat_pissa_path)

   
    



    instruction ='养臭水是什么'
    input_text = '“养臭水”近期再次席卷中国校园，许多中小学生热衷钻研各种配方，把包括唾液、牛奶、蟑螂、苍蝇、蚊子、老鼠尾巴、生猪肉、护手霜等各种毫不相干的原料放入饮料瓶，让令人恶心的液体在瓶子里发酵一段时间观察变化，并坐等瓶子炸开臭水喷发，而后在网上分享经验之谈。不少网民对学生养臭水的行为表示不理解，“这有什么好玩儿的？”“现在的小朋友无聊到这种程度了吗？”“作业太少了”“真不明白这些东西到底是谁兴起的，从萝卜刀到烟卡，这会又是生化武器。”评论并指出，养臭水看似是对化学、对科学的热爱，实则跟以求知与创新为宗旨的化学、科学关系不大，化学和科学是在严谨的实验之下，进行有目的的探索，而不是盲目尝试。养臭水的舆论之争，实际上也是教育方针之争，以及团体秩序与个人自由之间的拔河。世界各国伟大著名的科学家在进行旨在推进人类福祉、但有不小风险的试验时，都不会选择在人群密度高的地方进行。中小学生私下养臭水，满足好奇心甚至是求知欲，无可厚非，但如果行为妨碍公共秩序，影响他人健康和自由，也不应被合理化。评论指出，养臭水的行为看似是无聊的游戏或恶作剧，但实际上可能是孩子们寻求关注、发泄情绪或寻找归属感的一种方式。在表面平静的校园里，孩子们面临来自学业、家庭、社交等多方面的压力，而养臭水这种看似不可能完成的任务，正好为他们提供挑战自我、超越极限的机会。他们通过耐心观察、细心呵护、不断尝试和调整，最后体验到成功的喜悦和自豪。教职工群体则发通知提醒：“如果你们班里的小孩，神秘兮兮拿着个瓶子左躲右藏你最好真的别捡。”这是因为他们手里拿着的大概率是小孩圈流行的臭水。'
    output = llm_service.sampling_generate(instruction,input_text)
    print('s=',output)

    #beam_results = llm_service.beam_generate(instruction,input_text)
    #print('b=',result)