import uuid
import json
import os

import streamlit as st

from src.clients.index_client import IndexClient
from src.clients.llm_client import LLMClient
from src.utils.loader import load_and_split_text
from src.utils.internet import search_on_baike
def main():
    # 初始化客户端
    index_client = IndexClient("tcp://localhost:7783")
    llm_client = LLMClient("tcp://localhost:7781")

    # Streamlit UI
    st.write("#RWKV_RAGQ")
    
      # 创建侧边栏
    sidebar = st.sidebar

    # 设置模块
    sidebar.title("设置")

    
    base_model = sidebar.selectbox("Base Model:", ["rwkv6_1.6B"], key="base_model")
    bi_encoder = sidebar.selectbox("BiEncoder:", ["BGEM3"], key="bi_encoder")
    reranker = sidebar.selectbox("Reranker:", ["rwkv6_1.6B_crosslora"], key="reranker")
    infer = sidebar.selectbox("Infer:", ["rwkv6_1.6B_pissa"], key="infer")

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
                filepath = os.path.join(output_dir, output_filename)

                # 调用异步函数并运行
                result = search_on_baike(search_query, output_dir, output_filename)
                st.success(f"搜索结果已保存到: {filepath}")
            except Exception as e:
                st.error(f"发生错误: {str(e)}")

    
     # 询问用户输入payload的方式
    input_method = st.selectbox(
        "请选择输入Payload的方式",
        ["本地文件","手动输入"],
        index=0
    )
    


    if input_method == "手动输入":
        payload_input = st.text_area("请输入payload内容（每条文本一行）", height=200)
        payload_texts = payload_input.split("\n")
        indexed_texts = [index_client.index_texts([text]) for text in payload_texts]
        st.header("索引结果")
        for idx, result in enumerate(indexed_texts):
            st.write(f"文本 {idx+1}: {result}")
    elif input_method == "本地文件":
            st.title("Text Loader and Splitter")

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

            # 用户输入
            st.write("### 输入参数")
            st.session_state.input_path = st.text_input("请输入输入文件路径或目录路径:", key="input_path_key")
            st.session_state.output_path = st.text_input("请输入输出目录路径:", key="output_path_key")
            st.session_state.chunk_size = st.number_input("请输入块大小（字符数）:", min_value=1, value=512, key="chunk_size_key")
            st.session_state.chunk_overlap = st.number_input("请输入块重叠（字符数）:", min_value=1, value=8, key="chunk_overlap_key")

            # 加载按钮
            load_button = st.button("加载并分割文件")
            if load_button and os.path.exists(st.session_state.input_path) and st.session_state.output_path:
                try:
                    # 调用load_and_split_text函数，并根据需要调整参数
                    chunks= load_and_split_text(st.session_state.input_path, st.session_state.output_path,st.session_state.chunk_size, st.session_state.chunk_overlap)
                    st.success("文件已加载并分割完成！")
                    st.session_state.chunks = chunks  # 存储分割后的文本列表到session_state
                    indexed_chunks = [index_client.index_texts([chunk]) for chunk in chunks]
                    st.header("索引结果")
                    for idx, result in enumerate(indexed_chunks):
                         st.write(f"文本 {idx+1}: {result}")
                    st.session_state.indexed_texts = indexed_chunks
                except Exception as e:
                    st.error(f"加载和分割过程中出现错误：{str(e)}")
            elif load_button:
                st.warning("请确保输入路径有效。")

    # 提交payload按钮
    #submit_payload_button = st.button("索引Payload")

    #if submit_payload_button:
     #   if st.session_state.payload_texts:  # 如果用户手动输入了文本
      #      indexed_texts = [index_client.index_texts([text]) for text in st.session_state.payload_texts]
       # elif st.session_state.chunks :  # 如果用户从本地文件加载了文本
        #    indexed_texts = [index_client.index_texts([chunk]) for chunk in st.session_state.chunks]

        # 显示索引结果
        #st.header("索引结果")
        #for idx, result in enumerate(indexed_texts):
         #   st.write(f"文本 {idx+1}: {result}")

    # 用户输入query
    query_input_key = "query_input_key"
    query_input = st.text_input("请输入查询", key=query_input_key)
    recall_button = st.button("召回")
    
    if recall_button and query_input:
        search_results = index_client.search_nearby(query_input)['value']
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



    instruction_input = st.text_input("请输入指令:", key="instruction_input")
    beam_generate_button = st.button("Beam Generate")
    sampling_generate_button = st.button("Sampling Generate")
    if 'best_match' in st.session_state:
        if beam_generate_button:
            num_beams = st.number_input("束的数量:", min_value=1, value=5)
            token_count = st.number_input("令牌数量:", min_value=1, value=128)
            if num_beams != 0 and token_count != 0:
                beam_results = llm_client.beam_generate(instruction_input, st.session_state.best_match, token_count, num_beams)
                st.write("Beam Generate 结果:")
                st.write(beam_results)

        if sampling_generate_button:
            token_count = st.number_input("令牌数量:", min_value=1, value=128)
            if token_count != 0:
                # 使用best_match作为输入文本
                sampling_results = llm_client.sampling_generate(instruction_input, st.session_state.best_match, token_count)
                st.write("Sampling Generate 结果:")
                st.write(sampling_results)

if __name__ == "__main__":
    main()