import os
import string
import random

import fitz  # PyMuPDF for PDF handling

def load_and_split_text(file_path, output_dir,chunk_size, chunk_overlap):
    # 获取文件扩展名
    file_ext = os.path.splitext(file_path)[1]

    if file_ext == ".txt":
        # 处理TXT文件
        with open(file_path, 'r', encoding='utf-8') as f:
            corpus = f.read()
    elif file_ext == ".pdf":
        # 处理PDF文件
        doc = fitz.open(file_path)
        corpus = ""
        for page in doc:
            corpus += page.get_text()
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

    chunks = []
    chunk_size = int(chunk_size)
    chunk_overlap = int(chunk_overlap)

    start = 0
    while start < len(corpus):
        start = int(start) 
        end = start + chunk_size
        # 如果下一个chunk会超过文本长度，则取到文本末尾
        if end > len(corpus):
            end = len(corpus)
        # 添加当前chunk到列表
        chunks.append(corpus[start:end])
        # 移动起始位置以考虑重叠
        start += chunk_size - chunk_overlap

    # 先将chunks用换行符连接起来
    splitted_text = "\n".join(chunks)

    # 将分割后的文本写入输出文件
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(output_dir, f"{base_filename}.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(splitted_text)
    return chunks


def get_random_string(length):
    haracters = string.ascii_uppercase + string.digits
    return ''.join(random.choices(haracters, k=length))