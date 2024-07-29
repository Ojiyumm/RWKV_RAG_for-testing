from copy import deepcopy
import random
import torch
def whole_word_mask(token_ids, mlm_probability,segments = None):
    if segments is None:
        mask = []
        for token in token_ids:
            if random.random() < mlm_probability:
                mask.append(1)
            else:
                mask.append(0)
        return mask
    else:
        mask = []
        offset = 0
        for i in range(len(segments)):
            if random.random() < mlm_probability:
                mask.extend([1]*len(segments[i]))
            else:
                mask.extend([0]*len(segments[i]))
        return mask

def dup_mae_collator(examples,
                     max_seq_length,
                     encoder_mlm_probability,
                     mask_id = 3,
                     emb_id = 1,
                     vocab_size = 65536):
    batch = {
        'encoder_input_ids': [],
        'encoder_labels': [],
        'decoder_input_ids': [],
        'decoder_labels': [],
        'bag_word_weight': []
    }
    tgt_len = max_seq_length - 1
    for example in examples:
        token_ids = example['token_ids']
        segment_ids = example['segment_ids'] if 'segment_ids' in example else None
        encoder_input_ids = token_ids[:tgt_len]
        encoder_input_ids.append(emb_id)
            
        padding_size = max_seq_length - len(encoder_input_ids)
        mask = whole_word_mask(encoder_input_ids[:-1], encoder_mlm_probability,segments = segment_ids)

        encoder_labels = deepcopy(encoder_input_ids)
        decoder_labels = deepcopy(encoder_input_ids)
        decoder_input_ids = deepcopy(encoder_input_ids)
        for i, m in enumerate(mask):
            if i >= tgt_len:
                break#in case for some extem cases
            if m == 1:
                encoder_input_ids[i] = mask_id
            else:
                encoder_labels[i] = -100

        weight = torch.zeros(size=(vocab_size,))
        for t in example['token_ids'][:tgt_len]:
            weight[t] = 1 / len(example['token_ids'][:tgt_len])
        batch['bag_word_weight'].append(weight.unsqueeze(0))
        encoder_labels[-1] = -100
        decoder_labels[-1] = -100
        batch['encoder_input_ids'].append(encoder_input_ids if padding_size == 0 else encoder_input_ids + [0]*padding_size)
        batch['encoder_labels'].append(encoder_labels if padding_size == 0 else encoder_labels + [-100]*padding_size)
        batch['decoder_input_ids'].append(decoder_input_ids if padding_size == 0 else decoder_input_ids + [0]*padding_size)
        batch['decoder_labels'].append(decoder_labels if padding_size == 0 else decoder_labels + [-100]*padding_size)
    batch['encoder_input_ids'] = torch.tensor(batch['encoder_input_ids'],dtype=torch.long)
    batch['encoder_labels'] = torch.tensor(batch['encoder_labels'],dtype=torch.long)
    batch['decoder_input_ids'] = torch.tensor(batch['decoder_input_ids'],dtype=torch.long)
    batch['decoder_labels'] = torch.tensor(batch['decoder_labels'],dtype=torch.long)
    batch['bag_word_weight'] = torch.cat(batch['bag_word_weight'],dim=0)

    return batch

def mae_collator(examples, 
                 max_seq_length, 
                 encoder_mlm_probability, 
                 mask_id = 3,
                 emb_id = 1,):
    batch = {
        'encoder_input_ids': [],
        'encoder_labels': [],
        'decoder_input_ids': [],
        'decoder_labels': []
    }
    tgt_len = max_seq_length - 1
    for example in examples:
        token_ids = example['token_ids']
        segment_ids = example['segment_ids'] if 'segment_ids' in example else None
        encoder_input_ids = token_ids[:tgt_len]
        encoder_input_ids.append(emb_id)
            
        padding_size = max_seq_length - len(encoder_input_ids)
        mask = whole_word_mask(encoder_input_ids[:-1], encoder_mlm_probability,segments = segment_ids)

        encoder_labels = deepcopy(encoder_input_ids)
        decoder_labels = deepcopy(encoder_input_ids)
        decoder_input_ids = deepcopy(encoder_input_ids)
        for i, m in enumerate(mask):
            if m == 1:
                encoder_input_ids[i] = mask_id
            else:
                encoder_labels[i] = -100
        encoder_labels[-1] = -100
        decoder_labels[-1] = -100
        batch['encoder_input_ids'].append(encoder_input_ids if padding_size == 0 else encoder_input_ids + [0]*padding_size)
        batch['encoder_labels'].append(encoder_labels if padding_size == 0 else encoder_labels + [-100]*padding_size)
        batch['decoder_input_ids'].append(decoder_input_ids if padding_size == 0 else decoder_input_ids + [0]*padding_size)
        batch['decoder_labels'].append(decoder_labels if padding_size == 0 else decoder_labels + [-100]*padding_size)
    batch['encoder_input_ids'] = torch.tensor(batch['encoder_input_ids'],dtype=torch.long)
    batch['encoder_labels'] = torch.tensor(batch['encoder_labels'],dtype=torch.long)
    batch['decoder_input_ids'] = torch.tensor(batch['decoder_input_ids'],dtype=torch.long)
    batch['decoder_labels'] = torch.tensor(batch['decoder_labels'],dtype=torch.long)

    return batch

def mlm_collator(examples, 
                 max_seq_length, 
                 encoder_mlm_probability, 
                 mask_id = 3,
                 emb_id = 1,):
    batch = {
        'encoder_input_ids': [],
        'encoder_labels': []
    }
    tgt_len = max_seq_length - 1
    for example in examples:
        token_ids = example['token_ids']
        segment_ids = example['segment_ids'] if 'segment_ids' in example else None
        encoder_input_ids = token_ids[:tgt_len]
        encoder_input_ids.append(emb_id)
            
        padding_size = max_seq_length - len(encoder_input_ids)
        mask = whole_word_mask(encoder_input_ids[:-1], encoder_mlm_probability,segments = segment_ids)

        encoder_labels = deepcopy(encoder_input_ids)
        for i, m in enumerate(mask):
            if m == 1:
                encoder_input_ids[i] = mask_id
            else:
                encoder_labels[i] = -100
        encoder_labels[-1] = -100
        batch['encoder_input_ids'].append(encoder_input_ids if padding_size == 0 else encoder_input_ids + [0]*padding_size)
        batch['encoder_labels'].append(encoder_labels if padding_size == 0 else encoder_labels + [-100]*padding_size)
    batch['encoder_input_ids'] = torch.tensor(batch['encoder_input_ids'],dtype=torch.long)
    batch['encoder_labels'] = torch.tensor(batch['encoder_labels'],dtype=torch.long)

    return batch
