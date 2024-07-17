import os

from src.services.llm_service import LLMService
from src.tokenizer.rwkv_tokenizer import TRIE_TOKENIZER



base_rwkv_model = '/home/rwkv/Peter/model/base/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth'
bi_lora_path = '/home/rwkv/Peter/model/bi/RWKV-x060-World-1B6-v2_rwkv_lora.pth'
cross_lora_path = '/home/rwkv/Peter/model/cross/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth (1).pth'
tokenizer_file = os.path.join('/home/rwkv/Peter/RWKV_LM_EXT-main/tokenizer/rwkv_vocab_v20230424.txt')
tokenizer = TRIE_TOKENIZER(tokenizer_file)
chat_lora_path = '/home/rwkv/Peter/model/chat/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth.pth'
chat_pissa_path = '/home/rwkv/Peter/model/chat/init_pissa.pth'
chat_lora_r = 64
chat_lora_alpha = 64
llm_service = LLMService(
        base_rwkv_model,
        bi_lora_path,
        cross_lora_path,
        chat_lora_path,
        tokenizer,
        chat_lora_r=chat_lora_r,
        chat_lora_alpha=chat_lora_alpha,
        chat_pissa_path=chat_pissa_path)

instruction = '养臭水是什么'
input_text = '“养臭水”近期再次席卷中国校园，许多中小学生热衷钻研各种配方，把包括唾液、牛奶、蟑螂、苍蝇、蚊子、老鼠尾巴、生猪肉、护手霜等各种毫不相干的原料放入饮料瓶，让令人恶心的液体在瓶子里发酵一段时间观察变化，并坐等瓶子炸开臭水喷发，而后在网上分享经验之谈。不少网民对学生养臭水的行为表示不理解，“这有什么好玩儿的？”“现在的小朋友无聊到这种程度了吗？”“作业太少了”“真不明白这些东西到底是谁兴起的，从萝卜刀到烟卡，这会又是生化武器。”评论并指出，养臭水看似是对化学、对科学的热爱，实则跟以求知与创新为宗旨的化学、科学关系不大，化学和科学是在严谨的实验之下，进行有目的的探索，而不是盲目尝试。养臭水的舆论之争，实际上也是教育方针之争，以及团体秩序与个人自由之间的拔河。世界各国伟大著名的科学家在进行旨在推进人类福祉、但有不小风险的试验时，都不会选择在人群密度高的地方进行。中小学生私下养臭水，满足好奇心甚至是求知欲，无可厚非，但如果行为妨碍公共秩序，影响他人健康和自由，也不应被合理化。评论指出，养臭水的行为看似是无聊的游戏或恶作剧，但实际上可能是孩子们寻求关注、发泄情绪或寻找归属感的一种方式。在表面平静的校园里，孩子们面临来自学业、家庭、社交等多方面的压力，而养臭水这种看似不可能完成的任务，正好为他们提供挑战自我、超越极限的机会。他们通过耐心观察、细心呵护、不断尝试和调整，最后体验到成功的喜悦和自豪。教职工群体则发通知提醒：“如果你们班里的小孩，神秘兮兮拿着个瓶子左躲右藏你最好真的别捡。”这是因为他们手里拿着的大概率是小孩圈流行的臭水。'
output = llm_service.sampling_generate(instruction, input_text)
print('s=', output)

    # beam_results = llm_service.beam_generate(instruction,input_text)
    # print('b=',result)