import os
import asyncio

import streamlit as st

from src.clients.index_client import IndexClient
from src.clients.llm_client import LLMClient
from src.utils.loader import load_and_split_text
from src.utils.internet import search_on_baike
from src.clients.jsonl2binidx_client import Jsonl2BinIdxClient
from src.clients.tuning_client import RWKVPEFTClient



tabs_title = ["知识库管理", "模型微调"]
async def search_and_notify(search_query, output_dir, output_filename):
    # Run the async search function
    await search_on_baike(search_query, output_dir, output_filename)
    return os.path.join(output_dir, output_filename)

    # Run the async function in the event loop
    return asyncio.run(async_search())

#设置页面的全局CSS样式
def set_page_style():
    st.markdown(
        """
        <style>
        /* 调整侧边栏的样式 */
        .st-Sidebar {
            position: absolute; /* 绝对定位 */
            left: 0; /* 左侧对齐 */
            top: 0; /* 上侧对齐 */
            height: 100%; /* 占满整个视口高度 */
            padding: 20px; /* 内边距 */
            box-sizing: border-box; /* 边框计算在宽度内 */
        }
        /* 调整主内容区域的样式 */
        .st-App {
            margin-left: 250px; /* 为侧边栏留出空间 */
        }
        </style>
        """,
        unsafe_allow_html=True)




def knowledge_manager(index_client: IndexClient, llm_client: LLMClient):
    st.title("知识库管理")
    # 显示所有知识库
    if st.button('显示所有知识库'):
        collections = index_client.show_collection()
        print(collections)
        if collections:
            st.write("现有知识库:")
            st.write(collections)
        else:
            st.warning("没有找到任何知识库。")

    new_collection_name = st.text_input("请输入新知识库的名称:")

    if st.button('添加知识库'):
        if new_collection_name:
            try:
                index_client.create_collection(new_collection_name)
                st.success(f"知识库 '{new_collection_name}' 已成功添加。")
            except Exception as e:
                st.error(f"添加知识库时出错: {str(e)}")
        else:
            st.warning("请输入有效的知识库名称。")

    collection_to_delete = st.text_input("选择要删除的知识库:")
    if st.button('删除知识库'):
        if collection_to_delete:
            try:
                index_client.delete_collection(collection_to_delete)
                st.success(f"知识库 '{collection_to_delete}' 已成功删除。")
            except Exception as e:
                st.error(f"删除知识库时出错: {str(e)}")
        else:
            st.warning("请选择一个知识库进行删除。")


    st.title("联网搜索")
    # 搜索查询输入
    search_query = st.text_input("请输入搜索关键词:", "搜索内容", key="query_main")

    # 输出目录输入
    output_dir = st.text_input("请输入输出目录:", "", key="output_dir_main")

    # 输出文件名输入
    output_filename = st.text_input("请输入输出文件名:", "result.txt", key="output_filename_main")

    # 按钮触发搜索并保存
    if st.button("搜索并保存"):
        if not search_query:
            st.error("请提供搜索关键词。")
        elif not output_dir:
            st.error("请提供输出目录。")
        else:
            try:
                if not output_filename:
                    output_filename = f'{search_query}.txt'
                filepath = asyncio.run(search_and_notify(search_query, output_dir, output_filename))

                st.success(f"搜索结果已保存到: {filepath}")
            except Exception as e:
                st.error(f"发生错误: {str(e)}")

    # 询问用户输入payload的方式
    input_method = st.selectbox(
        "请选择输入Payload的方式",
        ["本地文件", "手动输入"],
        index=0
    )

    if input_method == "手动输入":
        payload_input = st.text_area("请输入payload内容（每条文本一行），然后Ctrl+Enter", height=200)
        payload_texts = payload_input.split("\n")
        indexed_texts = [index_client.index_texts([text]) for text in payload_texts]
        st.header("索引结果")
        for idx, result in enumerate(indexed_texts):
            st.write(f"文本 {idx + 1}: {result}")
    elif input_method == "本地文件":
        st.title("知识库内容管理")

        # 使用Session State存储用户输入
        if "input_path" not in st.session_state:
            st.session_state.input_path = ""
        if "output_path" not in st.session_state:
            st.session_state.output_path = ""
        if "chunk_size" not in st.session_state:
            st.session_state.chunk_size = 512
        if "chunk_overlap" not in st.session_state:
            st.session_state.chunk_overlap = 8
        if "chunks" not in st.session_state:
            st.session_state.chunks = []
        if "知识库名称" not in st.session_state:
            st.session_state.kb_name = []

            # 用户输入
        st.write("### 输入参数")
        st.session_state.input_path = st.text_input("请输入输入文件路径或目录路径:", key="input_path_key")
        st.session_state.output_path = st.text_input("请输入输出目录路径:", key="output_path_key")
        st.session_state.chunk_size = st.number_input("请输入块大小（字符数）:", min_value=1, value=512,
                                                      key="chunk_size_key")
        st.session_state.chunk_overlap = st.number_input("请输入块重叠（字符数）:", min_value=1, value=8,
                                                         key="chunk_overlap_key")
        st.seesion_state_kb_name = st.text_input("请输入知识库名称", key="知识库名称")

        # 加载按钮
        load_button = st.button("加载并分割文件")
        if load_button and os.path.exists(st.session_state.input_path) and st.session_state.output_path:
            try:
                # 调用load_and_split_text函数，并根据需要调整参数
                chunks = load_and_split_text(st.session_state.input_path, st.session_state.output_path,
                                             st.session_state.chunk_size, st.session_state.chunk_overlap)
                st.success("文件已加载并分割完成！")
                st.session_state.chunks = chunks  # 存储分割后的文本列表到session_state
                indexed_chunks = [index_client.index_texts([chunk], collection_name=st.seesion_state_kb_name) for chunk
                                  in chunks]
                st.header("索引结果")
                for idx, result in enumerate(indexed_chunks):
                    st.write(f"文本 {idx + 1}: {result}")
                st.session_state.indexed_texts = indexed_chunks
            except Exception as e:
                st.error(f"加载和分割过程中出现错误：{str(e)}")
        elif load_button:
            st.warning("请确保输入路径有效。")

    # 提交payload按钮
    # submit_payload_button = st.button("索引Payload")

    # if submit_payload_button:
    #   if st.session_state.payload_texts:  # 如果用户手动输入了文本
    #      indexed_texts = [index_client.index_texts([text]) for text in st.session_state.payload_texts]
    # elif st.session_state.chunks :  # 如果用户从本地文件加载了文本
    #    indexed_texts = [index_client.index_texts([chunk]) for chunk in st.session_state.chunks]

    # 显示索引结果
    # st.header("索引结果")
    # for idx, result in enumerate(indexed_texts):
    #   st.write(f"文本 {idx+1}: {result}")

    # 用户输入query
    st.title('召回最匹配知识')
    query_input_key = "query_input_key"
    query_input = st.text_input("请输入查询", key=query_input_key)
    recall_button = st.button("召回")

    if recall_button and query_input:
        search_results = index_client.search_nearby(query_input, collection_name=st.seesion_state_kb_name)['value']
        documents = search_results["documents"][0]
        st.write(documents)
        cross_scores = llm_client.cross_encode([query_input for i in range(len(documents))], documents)
        st.header("Cross_score")
        st.write(cross_scores)
        max_score_index = cross_scores["value"].index(max(cross_scores["value"]))
        best_match = documents[max_score_index]
        st.header("最佳匹配")
        st.write(f"Best Match: {best_match}")
        st.session_state.best_match = best_match

    st.title('知识问答')
    instruction_input = st.text_input("请输入指令:", key="instruction_input")
    beam_generate_button = st.button("Beam Generate")
    sampling_generate_button = st.button("Sampling Generate")
    if 'best_match' in st.session_state:
        if beam_generate_button:
            num_beams = st.number_input("束的数量:", min_value=1, value=5)
            token_count = st.number_input("令牌数量:", min_value=1, value=128)
            if num_beams != 0 and token_count != 0:
                beam_results = llm_client.beam_generate(instruction_input, st.session_state.best_match, token_count,
                                                        num_beams)
                st.write("Beam Generate 结果:")
                st.write(beam_results)

        if sampling_generate_button:
            token_count = st.number_input("令牌数量:", min_value=1, value=128)
            if token_count != 0:
                # 使用best_match作为输入文本
                sampling_results = llm_client.sampling_generate(instruction_input, st.session_state.best_match,
                                                                token_count)
                st.write("Sampling Generate 结果:")
                st.write(sampling_results)


def jsonl2binidx_manager(client: Jsonl2BinIdxClient):
    """
    数据转换
    """


    st.title("准备微调数据")
    epoch = st.number_input("Epoch:", min_value=1, value=3, key='tuning_epoch', max_value=10)
    context_len = st.number_input("Context Length:", min_value=1, value=2048, key='tuning_context_len', disabled=False)
    # 输出路径
    output_dir = st.text_input("输出文件路径:", "/home/rwkv/Peter/Data/Telechat5", key="output_dir", disabled=True)
    # 询问用户输入payload的方式
    input_method = st.selectbox(
        "请选择输入数据格式",
        ["本地上传","手动输入"],
        index=0
    )
    payload_input = None
    if input_method == "手动输入":
        payload_input = st.text_area("请输入payload内容（每条文本一行，格式参照https://rwkv.cn/RWKV-Fine-Tuning/FT-Dateset)", height=220)
    else:
        payload_file = st.file_uploader("请上传文件(格式参照https://rwkv.cn/RWKV-Fine-Tuning/FT-Dateset)", type=["jsonl"], key="payload_input")
        if payload_file:
            # TODO 暂时不考虑多用户时，文件名冲突的情况
            # file_name = payload_file.name
            # file_name_prefix = file_name.rsplit('.', 1)[0]
            # output_name_endfix = '%s_%s' % (file_name_prefix, get_random_string(4))
            payload_input = payload_file.read().decode("utf-8", errors='ignore')

    if st.button("提交") and payload_input:
        binidx_path = client.transform(payload_input, epoch, output_dir, context_len, is_str=True)
        st.write(binidx_path)


def tuning_manager(client: RWKVPEFTClient ,app_scenario,):
    """
    模型微调
    """
    st.title("模型微调")
    with st.sidebar:
        if app_scenario == tabs_title[1]:
            tuning_base_model = st.selectbox("Base Model:", ["rwkv6_1.6B"], key="tuning_base_model")
            accelerator = st.text_input("accelerator:", "gpu", key="accelerator", disabled=False)
            precision = st.text_input("precision:",  "bf16", key="precision")
            quant = st.text_input("quant:", 'nf4', key="quant")
            n_layer = st.number_input("n_layer:", min_value=1, value=24, key="n_layer", disabled=False)
            n_embd = st.number_input("n_embd:", min_value=1, value=2048, key="n_embd", disabled=False)
            ctx_len = st.number_input("ctx_len:", min_value=1, value=1024, key="ctx_len", disabled=False)
            data_type = st.selectbox("data_type:", ['binidx'], key="data_type")
            epoch_save = st.number_input("epoch_save:", min_value=1, value=1, key="epoch_save")
            vocab_size = st.number_input("vocab_size:", min_value=1, value=65536, key="vocab_size", disabled=False)
            epoch_begin = st.number_input("epoch_begin:", min_value=0, value=0, key="epoch_begin", disabled=False)
            pre_ffn = st.number_input("pre_ffn:", min_value=0, value=0, key="pre_ffn", disabled=False)
            head_qk = st.number_input("head_qk:", min_value=0, value=0, key="head_qk", disabled=False)
            beta1 = st.number_input("beta1:", min_value=0.0, value=0.9, key="beta1", disabled=False)
            beta2 = st.number_input("beta2:", min_value=0.0, value=0.99, key="beta2", disabled=False)
            adam_eps = st.number_input("adam_eps:", min_value=0.0, value=1e-8, key="adam_eps", disabled=False)
            my_testing = st.text_input("my_testing:", "x060", key="my_testing", disabled=True)
            strategy = st.text_input("strategy:", "deepspeed_stage_1", key="strategy", disabled=False)
            devices = st.number_input("devices:", min_value=1, value=1, key="devices", disabled=False)
            dataload = st.text_input("dataload:", "pad", key="dataload", disabled=False)
            grad_cp = st.number_input("grad_cp:", min_value=1, value=1, key="grad_cp", disabled=False)

    tuning_type = st.selectbox("微调算法:", [ "state", "pissa", "lora"], index=0)
    load_model = st.text_input("基底模型路径:", '/home/rwkv/Peter/model/base/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth', key="load_model", disabled=False)

    proj_dir = st.text_input("输出路径:", "", key="proj_dir")
    data_file = st.text_input("训练数据集的路径:(路径中不需要带 bin 和 idx 后缀，仅需文件名称)", "", key="data_file")
    if tuning_type == 'state':
        micro_bsz = st.number_input("micro_bsz:", min_value=1, value=1, key="micro_bsz")
        epoch_steps = st.number_input("epoch_steps:(如果微调训练数据表较少,建议调小该值)", min_value=1, value=800, key="epoch_steps")
        epoch_count = st.number_input("epoch_count:", min_value=1, value=10, key="epoch_count")
        lr_init = st.number_input("lr_init:", min_value=0.0, value=1.0, key="lr_init")
        lr_final = st.number_input("lr_final:", min_value=0.0, value=1e-2, key="lr_final")
        warmup_steps = st.number_input("warmup_steps:", min_value=0, value=10, key="warmup_steps")
    elif tuning_type == 'pissa':
        svd_niter = st.number_input("svd_niter:", min_value=1, value=4, key="svd_niter")
        lora_r = st.number_input("lora_r:", min_value=1, value=64, key="lora_r")
        micro_bsz = st.number_input("micro_bsz:", min_value=1, value=8, key="micro_bsz")
        epoch_steps = st.number_input("epoch_steps:(如果微调训练数据表较少,建议调小该值)", min_value=1, value=1000, key="epoch_steps")
        epoch_count = st.number_input("epoch_count:", min_value=1, value=1, key="epoch_count",disabled=False)
        lr_init = st.number_input("lr_init:", min_value=0.0, value=5e-5, key="lr_init")
        lr_final = st.number_input("lr_final:", min_value=0.0, value=5e-5, key="lr_final")
        lora_load = st.text_input("lora_load:", "'rwkv-0'", key="lora_load", disabled=True)
        lora_alpha = st.number_input("lora_alpha:", min_value=1, value=128, key="lora_alpha", disabled=False)
        lora_dropout = st.number_input("lora_dropout:", min_value=0.0, value=0.01, key="lora_dropout", disabled=False)
        lora_parts = st.text_input("lora_parts:", "att,ffn,time,ln", key="lora_parts", disabled=False)

    else:
        lora_r = st.number_input("lora_r:", min_value=1, value=64, key="lora_r")
        micro_bsz = st.number_input("micro_bsz:", min_value=1, value=8, key="micro_bsz")
        epoch_steps = st.number_input("epoch_steps:(如果微调训练数据表较少,建议调小该值)", min_value=1, value=1000, key="epoch_steps")
        epoch_count = st.number_input("epoch_count:", min_value=1, value=1, key="epoch_count",disabled=False)
        lr_init = st.number_input("lr_init:", min_value=0.0, value=5e-5, key="lr_init")
        lr_final = st.number_input("lr_final:", min_value=0.0, value=5e-5, key="lr_final")
        lora_alpha = st.number_input("lora_alpha:", min_value=1, value=128, key="lora_alpha", disabled=False)
        warmup_steps = st.number_input("warmup_steps:", min_value=0, value=0, key="warmup_steps", disabled=False)
        lora_load = st.text_input("lora_load:", "'rwkv-0'", key="lora_load", disabled=False)
        lora_dropout = st.number_input("lora_dropout:", min_value=0.0, value=0.01, key="lora_dropout", disabled=False)
        lora_parts = st.text_input("lora_parts:", "att,ffn,time,ln", key="lora_parts", disabled=False)


    if st.button("开始"):
        if not proj_dir or not data_file:
            st.warning("输出路径和训练数据集的路径不能为空")
            return
        if tuning_type == 'state':
            client.state_tuning_train(
                load_model=load_model, proj_dir=proj_dir, data_file=data_file, data_type=data_type,
                vocab_size=vocab_size, ctx_len=ctx_len, epoch_steps=epoch_steps,
                epoch_count=epoch_count, epoch_begin=epoch_begin, epoch_save=epoch_save,
                micro_bsz=micro_bsz, n_layer=n_layer, n_embd=n_embd, pre_ffn=pre_ffn,
                head_qk=head_qk, lr_init=lr_init, lr_final=lr_final, warmup_steps=warmup_steps,
                beta1=beta1, beta2=beta2, adam_eps=adam_eps,
                accelerator=accelerator, devices=devices, precision=precision, strategy=strategy,
                grad_cp=grad_cp, my_testing=my_testing,
                train_type='state', dataload=dataload, quant=quant,
                wandb='statetuning_test'
            )

        elif tuning_type == 'pissa':
            client.pissa_train(load_model=load_model, proj_dir=proj_dir, data_file=data_file,
                                      data_type=data_type,
                                      vocab_size=vocab_size, ctx_len=ctx_len, epoch_steps=epoch_steps,
                                      epoch_count=epoch_count, epoch_begin=epoch_begin, epoch_save=epoch_save,
                                      micro_bsz=micro_bsz, n_layer=n_layer, n_embd=n_embd, pre_ffn=pre_ffn,
                                      head_qk=head_qk, lr_init=lr_init, lr_final=lr_final, warmup_steps=warmup_steps,
                                      beta1=beta1, beta2=beta2, adam_eps=adam_eps,accelerator=accelerator,
                                      devices=devices,precision=precision,strategy=strategy, my_testing=my_testing,
                                      lora_load=lora_load,lora=True,lora_r=lora_r,lora_alpha=lora_alpha,
                                      lora_dropout=lora_dropout,lora_parts=lora_parts, PISSA=True, svd_niter=svd_niter,
                                      grad_cp=grad_cp,  dataload=dataload, quant=quant)
        else:
            client.lora_train(load_model=load_model, proj_dir=proj_dir, data_file=data_file, data_type=data_type,
                                      vocab_size=vocab_size, ctx_len=ctx_len, epoch_steps=epoch_steps,
                                      epoch_count=epoch_count,epoch_begin=epoch_begin,epoch_save=epoch_save,
                                      micro_bsz=micro_bsz,n_layer=n_layer,n_embd=n_embd,pre_ffn=pre_ffn,
                                      head_qk=head_qk,lr_init=lr_init,lr_final=lr_final,warmup_steps=warmup_steps,
                                      beta1=beta1,beta2=beta2,adam_eps=adam_eps,accelerator=accelerator,devices=devices,
                                      precision=precision,strategy=strategy,grad_cp=grad_cp,my_testing=my_testing,
                                      lora_load=lora_load,lora=True,lora_r=lora_r,lora_alpha=lora_alpha,
                                      lora_dropout=lora_dropout,lora_parts=lora_parts)


def main():
    # 初始化客户端
    index_client = IndexClient("tcp://localhost:7783")
    llm_client = LLMClient("tcp://localhost:7781")
    jsonl2binid_client = Jsonl2BinIdxClient('tcp://localhost:7787')
    tuning_client = RWKVPEFTClient('tcp://localhost:7789')

    set_page_style()

    with st.sidebar:
        st.header("RWKV RAGQ")
        st.write('\n')
        # 创建垂直排列的标签页
        app_scenario = st.radio('Select Scenario:', tabs_title)

        st.write('\n')
        st.markdown("""
        <hr style="border: 1px solid #CCCCCC; margin-top: 20px; margin-bottom: 20px;">
        """, unsafe_allow_html=True)
        st.write('\n')
        if app_scenario== tabs_title[0]:
            # 根据选中的标签更新 session_state
            base_model = st.selectbox("Base Model:", ["rwkv6_1.6B"], key="base_model")
            bi_encoder = st.selectbox("BiEncoder:", ["BGEM3"], key="bi_encoder")
            reranker = st.selectbox("Reranker:", ["rwkv6_1.6B_crosslora"], key="reranker")
            infer = st.selectbox("Infer:", ["rwkv6_1.6B_pissa"], key="infer")

        else:
            pass

    if app_scenario== tabs_title[0]:
        knowledge_manager(index_client, llm_client)
    else:
        # 在这里添加微调选项卡的内容
        jsonl2binidx_manager(jsonl2binid_client)
        tuning_manager(tuning_client, app_scenario,)





if __name__ == "__main__":
    main()