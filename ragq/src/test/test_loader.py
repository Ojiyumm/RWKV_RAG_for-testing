import os
import argparse
import multiprocessing as mp

from src.utils.loader import load_and_split_text


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