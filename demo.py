import sys
import os

ragq_base_dir = os.path.dirname(os.path.abspath(__file__))
ext_project_package_dir = os.path.join(ragq_base_dir, 'third_party_packages')

sys.path.insert(0, ext_project_package_dir)

from FlagEmbedding import BGEM3FlagModel
import jieba
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
from FlagEmbedding import FlagReranker


if __name__ == '__main__':
    # model = BGEM3FlagModel('/home/rwkv/Peter/model/bi/bge-m31',
    #                        use_fp16=True)
    # sentences_1 = ["What is BGE M3?", "Definition of BM25"]
    # embeddings_1 = model.encode(sentences_1, batch_size=12, max_length=8192)['dense_vecs']
    # print(embeddings_1)
    try:
        reranker = FlagReranker('/home/rwkv/Peter/model/bi/BAAIbge-reranker-v2-m3',
                                use_fp16=True)
        tokenizer = TRIE_TOKENIZER()
        a = tokenizer.encode("我爱自然语言处理")
        b = tokenizer.encode("我爱自然语言处理,它能解决很多问题")
        a_str = [str(i) for i in a]
        b_str = [str(i) for i in b]
        print(a_str)
        print(b_str)
        print([a_str, b_str])
        s = reranker.compute_score([a_str, b_str], normalize=True)
        print(s)
    except Exception as e:
        print('failed to print tokenizer', e)

    # sent = '中文分词是文本处理不可或缺的一步!'
    #
    # seg_list = jieba.cut(sent, cut_all=True)
    #
    # print('全模式：', '/ '.join(seg_list))
    #
    # seg_list = jieba.cut(sent, cut_all=False)
    # print('精确模式：', '/ '.join(seg_list))
    #
    # seg_list = jieba.cut(sent)
    # print('默认精确模式：', '/ '.join(seg_list))
    #
    # seg_list = jieba.cut_for_search(sent)
    # print('搜索引擎模式', '/ '.join(seg_list))

