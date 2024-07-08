import os
import argparse
import json
import multiprocessing as mp
from langchain.text_splitter import CharacterTextSplitter
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a text file into chunks and save them in a TXT file.")
    parser.add_argument("--input", help="Path to the input text file or directory.")
    parser.add_argument("--output", help="Path to the output directory.")
    parser.add_argument("--num_processes", type=int, default=1, help="Number of processes to use for parallel processing.")
    parser.add_argument("--chunk_size", type=int, help="Size of each chunk in characters.")
    parser.add_argument("--chunk_overlap", type=int, help="Overlap size between chunks in characters.")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        with mp.Pool(args.num_processes) as pool:
            files_to_process = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(('.txt', '.pdf'))]
            for file in files_to_process:
                pool.apply(load_and_split_text, (file, args.output,args.chunk_size, args.chunk_overlap))
    else:
        load_and_split_text(args.input,  args.output,args.chunk_size, args.chunk_overlap)