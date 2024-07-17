import os
import streamlit as st
from src.clients.index_client import IndexClient
from src.clients.llm_client import LLMClient
from src.utils.loader import load_and_split_text

def main():
    # 初始化客户端
    index_client = IndexClient("tcp://localhost:7783")
    llm_client = LLMClient("tcp://localhost:7781")

    # Streamlit UI
    st.write(" RWKV_Chatbot")

    # 创建侧边栏
    sidebar = st.sidebar

    # 设置模块
    sidebar.title("设置")

    # 输入Payload和索引
    sidebar.subheader("新增知识库")

    # 询问用户输入payload的方式
    input_method = sidebar.selectbox(
        "请选择输入Payload的方式",
        ["手动输入", "本地文件"],
        index=0
    )

    if input_method == "手动输入":
        payload_input = sidebar.text_area("请输入payload内容（每条文本一行）", height=200)
        payload_texts = payload_input.split("\n")
        load_button = sidebar.button("加载并分割文件")
        if load_button:
            indexed_texts = [index_client.index_texts([text]) for text in payload_texts]
            sidebar.write("索引结果:")
            for idx, result in enumerate(indexed_texts):
                sidebar.write(f"文本 {idx+1}: {result}")

    elif input_method == "本地文件":
        # 使用Session State存储用户输入
        if "input_path" not in st.session_state:
            st.session_state.input_path = ""
        if "output_path" not in st.session_state:
            st.session_state.output_path = ""
        if "chunk_size" not in st.session_state:
            st.session_state.chunk_size = 512
        if "chunk_overlap" not in st.session_state:
            st.session_state.chunk_overlap = 8

        # 用户输入
        sidebar.write("### 输入参数")
        st.session_state.input_path = sidebar.text_input("请输入输入文件路径或目录路径:", key="input_path_key")
        st.session_state.output_path = sidebar.text_input("请输入输出目录路径:", key="output_path_key")
        st.session_state.chunk_size = sidebar.number_input("请输入块大小（字符数）:", min_value=1, value=512, key="chunk_size_key")
        st.session_state.chunk_overlap = sidebar.number_input("请输入块重叠（字符数）:", min_value=1, value=8, key="chunk_overlap_key")

           # 加载并分割文件按钮
        load_button = sidebar.button("加载并分割文件")
        if load_button and os.path.exists(st.session_state.input_path) and st.session_state.output_path:
            try:
                chunks = load_and_split_text(st.session_state.input_path, st.session_state.output_path, st.session_state.chunk_size, st.session_state.chunk_overlap)
                sidebar.success("文件已加载并分割完成！")
                st.session_state.chunks = chunks
                indexed_chunks = [index_client.index_texts([chunk]) for chunk in chunks]
                sidebar.write("索引结果:")
                for idx, result in enumerate(indexed_chunks):
                    sidebar.write(f"文本 {idx+1}: {result}")
            except Exception as e:
                sidebar.error(f"加载和分割过程中出现错误：{str(e)}")

  
    # 中间部分
    # 中间部分
    st.write("聊天机器人")

    # 初始化聊天历史
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # 用户输入query
    query_input = st.text_input("请输入查询", key="query_input_key")
    submit_button = st.button("提交")

    # 显示聊天记录
    chat_history_str = "\n".join(st.session_state.chat_history)
    st.text_area("聊天记录:", value=chat_history_str, height=400, disabled=True)

    if submit_button and query_input:
        search_results = index_client.search_nearby(query_input)['value']
        documents = search_results["documents"][0]
        cross_scores = llm_client.cross_encode([query_input for _ in range(len(documents))], documents)
        max_score_index = cross_scores["value"].index(max(cross_scores["value"]))
        best_match = documents[max_score_index]
        recent_chat_history = st.session_state.chat_history[-10:] or st.session_state.chat_history
        integrated_best_match = "\n".join(recent_chat_history) + "\n" + str(best_match)
        instruction_input = (
            "您是一位RWKV模型的专家。请根据以下信息回答用户的问题：\n在您回答问题的同时，请您给出原因以及背后的逻辑"
                        
        )
        
        sampling_results = llm_client.sampling_generate(instruction_input, integrated_best_match,token_count=4096)
            
        st.session_state.chat_history.append(f"用户: {query_input}")
        st.session_state.chat_history.append(f"机器人: {sampling_results}")
        chat_history_str = "\n".join(st.session_state.chat_history)  # 获取最新的聊天记录
        
        
        query_input = ""  

if __name__ == "__main__":
    main()