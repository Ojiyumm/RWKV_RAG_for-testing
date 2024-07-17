import pandas as pd
import orjson
def find_uniq_task_name_and_task_type(input_file):
    df = pd.read_parquet(input_file)
    df['task_type'] = df['task_type'].apply(str)
    task_type = df['task_type'].unique()
    task_name = df['task_name_in_eng'].unique()
    return task_name,task_type
def convert_parquet_2_jsonl(input_file,output_jsonl,inquired_task_type='阅读理解'):
    # task_name,task_type = find_uniq_task_name_and_task_type(input_file)
    # file_meta_file_name = output_jsonl.replace('.jsonl','_meta.json')
    # with open(file_meta_file_name,'w',encoding='UTF-8') as f:
    #     meta_data = {
    #         'task_name':task_name.tolist(),
    #         'task_type':task_type.tolist()
    #     }
    #     f.write(orjson.dumps(meta_data).decode('UTF-8'))
    df = pd.read_parquet(input_file)
    from tqdm import tqdm
    progress = tqdm(total=len(df),desc=f'convert {input_file} to {output_jsonl}')
    output_jsonl_file_name = output_jsonl.replace('.jsonl',f'_{inquired_task_type}.jsonl')
    output_fp = None
    for index, row in df.iterrows():
        instructional_data = {
            'input':row['input'],
            'instruction':row["instruction"],
            'output':row['output']
        }
        task_type = row['task_type']
        major = task_type['major']
        minor = task_type['minor']
        matched_task_type = None
        if inquired_task_type in major :
            matched_task_type = major
        elif inquired_task_type in minor:
            matched_task_type = minor

        if matched_task_type is not None:
            if output_fp is None:
                output_fp = open(output_jsonl_file_name,'w',encoding='UTF-8')
            output_fp.write(orjson.dumps(instructional_data).decode('UTF-8'))
            output_fp.write('\n')
        progress.update(1)
    progress.close()
    if output_fp is not None:
        output_fp.close()

            
