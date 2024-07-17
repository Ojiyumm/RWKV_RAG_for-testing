def convert_drcd_file(input_file,output_jsonl):
    import orjson
    with open(input_file,'r',encoding='UTF-8') as f:
        data = orjson.loads(f.read())
    with open(output_jsonl,'w',encoding='UTF-8') as f:
        for item in data['data']:
            for paragraph in item['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    for answer in qa['answers']:
                        answer_text = answer['text']
                        instructional_data = {
                            'input':context,
                            'instruction':f'根据给定短文，回答以下问题：{question}',
                            'output':answer_text
                        }
                        f.write(orjson.dumps(instructional_data).decode('UTF-8')+'\n')
                        

