from src.clients.llm_client import LLMClient


llm_client = LLMClient("tcp://localhost:7781")

embeddings = llm_client.encode(["大家好", "我来自北京"])
print("Embeddings:", embeddings)

cross_scores = llm_client.cross_encode(["大家好", "我是一个机器人"], ["你好", "小猫咪真可爱"])
print("Cross Scores:", cross_scores)


instruction ='根据给定的短文，回答以下问题：黄循财的是哪国人？'
token_count = 128
num_beams =5
input_text = '黄循财（英语：Lawrence Wong Shyun Tsai，1972年12月18日—），新加坡华裔政治人物，现任新加坡总理兼财政部部长、人民行动党社区基金会主席。他与王乙康和颜金勇共同主持了因应新加坡2019冠状病毒病大流行的多部委工作组。曾任新加坡副总理，教育部、国家发展部、文化、社区及青年部的部长，通讯及新闻部和财政部的第二部长，以及人民行动党副秘书长。[1]黄循财是人民行动党第四代领导层，也是人民行动党中央执行委员会首任副秘书长兼政策论坛顾问.'
try:
    output = llm_client.sampling_generate(instruction,input_text,token_count)
    print('s=',output)
except Exception as e:
    print('s failed',e)
try:
    beam_results = llm_client.beam_generate(instruction,input_text,token_count,num_beams)
    print(beam_results)
except Exception as e:
    print('b failed',beam_results)
    #for result in beam_results:
    #    print('b=',result)