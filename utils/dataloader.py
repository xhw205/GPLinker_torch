# -*- coding: utf-8 -*-
"""
@Auth: Xhw
@Description: CHIP/CBLUE 医学实体关系抽取，数据来源 https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset

def load_name(filename):
    #{"text": "产后抑郁症@区分产后抑郁症与轻度情绪失调（产后忧郁或“婴儿忧郁”）是重要的，因为轻度情绪失调不需要治疗。", "spo_list": [{"Combined": false, "predicate": "鉴别诊断", "subject": "产后抑郁症", "subject_type": "疾病", "object": {"@value": "轻度情绪失调"}, "object_type": {"@value": "疾病"}}]}
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            D.append({
                "text":line["text"],
                "spo_list":[(spo["subject"], spo["predicate"], spo["object"]["@value"], spo["subject_type"], spo["object_type"]["@value"])
                            for spo in line["spo_list"]]
            })
        return D

def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

class data_generator(Dataset):
    def __init__(self, data, tokenizer, max_len, schema):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.schema = schema #spo
    def __len__(self):
        return len(self.data)

    def encoder(self, item):
        text = item["text"]
        encoder_text = self.tokenizer(text, return_offsets_mapping=True, max_length=self.max_len, truncation=True)
        # 将raw_text的下标 与 token的start和end下标对应
        input_ids = encoder_text["input_ids"]
        token_type_ids = encoder_text["token_type_ids"] #RoBERTa不需要NSP任务
        attention_mask = encoder_text["attention_mask"]
        spoes = set()
        for s, p, o, s_t, o_t in item["spo_list"]:
            s = self.tokenizer.encode(s, add_special_tokens=False)
            p = self.schema[s_t + "_" + p + "_" +o_t]
            o = self.tokenizer.encode(o, add_special_tokens=False)
            sh = search(s, input_ids)
            oh = search(o, input_ids)
            if sh != -1 and oh != -1:
                spoes.add((sh, sh+len(s)-1, p, oh, oh+len(o)-1))
        entity_labels = [set() for i in range(2)]
        head_labels = [set() for i in range(len(self.schema))]
        tail_labels = [set() for i in range(len(self.schema))]
        for sh, st, p, oh, ot in spoes:
            entity_labels[0].add((sh, st)) #实体提取：2个类型，头实体or尾实体
            entity_labels[1].add((oh, ot))
            head_labels[p].add((sh, oh)) #类似TP-Linker
            tail_labels[p].add((st, ot))
        for label in entity_labels+head_labels+tail_labels:
            if not label:
                label.add((0,0))
        # 例如entity = [{(1,3)}, {(4,5), (7,9)}]
        # entity[0]即{(1,3)}代表头实体首尾， entity[1]即{(4,5),{7,9}}代表尾实体首尾
        # 需要标签对齐为 [[[1,3][0,0]] , [[4,5][7,9]]]
        entity_labels = sequence_padding([list(l) for l in entity_labels])
        head_labels = sequence_padding([list(l) for l in head_labels])
        tail_labels = sequence_padding([list(l) for l in tail_labels])
        return text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.encoder(item)

    @staticmethod
    def collate(examples):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_entity_labels, batch_head_labels, batch_tail_labels = [], [], []
        text_list = []
        for item in examples:
            text, entity_labels, head_labels, tail_labels, input_ids, attention_mask, token_type_ids = item
            batch_entity_labels.append(entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            batch_token_ids.append(input_ids)
            batch_mask_ids.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            text_list.append(text)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()#RoBERTa 不需要NSP
        batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2)).long()
        batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
        batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()\

        return text_list, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels


