# -*- coding: utf-8 -*-
"""
@Auth: Xhw
@Description: token-pair范式的实体关系抽取pytorch实现
"""
import torch
import json
import sys
import numpy as np
import torch.nn as nn
from nets.gpNet import RawGlobalPointer, sparse_multilabel_categorical_crossentropy
from transformers import BertTokenizerFast, BertModel
from utils.dataloader import data_generator, load_name
from torch.utils.data import DataLoader
import configparser
from torch.utils.tensorboard import SummaryWriter
from utils.bert_optimization import BertAdam

con = configparser.ConfigParser()
con.read('./config.ini', encoding='utf8')
args_path = dict(dict(con.items('paths')), **dict(con.items("para")))
tokenizer = BertTokenizerFast.from_pretrained(args_path["model_path"], do_lower_case=True)
encoder = BertModel.from_pretrained(args_path["model_path"])

with open(args_path["schema_data"], 'r', encoding='utf-8') as f:
    schema = {}
    for idx, item in enumerate(f):
        item = json.loads(item.rstrip())
        schema[item["subject_type"]+"_"+item["predicate"]+"_"+item["object_type"]] = idx
id2schema = {}
for k,v in schema.items(): id2schema[v]=k
train_data = data_generator(load_name(args_path["train_file"]), tokenizer, max_len=con.getint("para", "maxlen"), schema=schema)
dev_data = data_generator(load_name(args_path["val_file"]), tokenizer, max_len=con.getint("para", "maxlen"), schema=schema)
train_loader = DataLoader(train_data , batch_size=con.getint("para", "batch_size"), shuffle=True, collate_fn=train_data.collate)
dev_loader = DataLoader(dev_data , batch_size=con.getint("para", "batch_size"), shuffle=True, collate_fn=dev_data.collate)

device = torch.device("cuda:1")

mention_detect = RawGlobalPointer(hiddensize=1024, ent_type_size=2, inner_dim=64).to(device)#实体关系抽取任务默认不提取实体类型
s_o_head = RawGlobalPointer(hiddensize=1024, ent_type_size=len(schema), inner_dim=64, RoPE=False, tril_mask=False).to(device)
s_o_tail = RawGlobalPointer(hiddensize=1024, ent_type_size=len(schema), inner_dim=64, RoPE=False, tril_mask=False).to(device)
class ERENet(nn.Module):
    def __init__(self, encoder, a, b, c):
        super(ERENet, self).__init__()
        self.mention_detect = a
        self.s_o_head = b
        self.s_o_tail = c
        self.encoder = encoder

    def forward(self, batch_token_ids, batch_mask_ids, batch_token_type_ids):
        outputs = self.encoder(batch_token_ids, batch_mask_ids, batch_token_type_ids)

        mention_outputs = self.mention_detect(outputs, batch_mask_ids)
        so_head_outputs = self.s_o_head(outputs, batch_mask_ids)
        so_tail_outputs = self.s_o_tail(outputs, batch_mask_ids)
        return mention_outputs, so_head_outputs, so_tail_outputs

net = ERENet(encoder, mention_detect, s_o_head, s_o_tail).to(device)
# optimizer = torch.optim.AdamW(
# 	net.parameters(),
#     lr=1e-5
# )
def set_optimizer(model, train_steps=None):
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=2e-5,
                         warmup=0.1,
                         t_total=train_steps)
    return optimizer

optimizer = set_optimizer(net, train_steps= (int(len(train_data) / con.getint("para", "batch_size")) + 1) * con.getint("para", "epochs"))
total_loss, total_f1 = 0., 0.
for eo in range(con.getint("para", "epochs")):
    for idx, batch in enumerate(train_loader):
        text, batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels = batch
        batch_token_ids, batch_mask_ids, batch_token_type_ids, batch_entity_labels, batch_head_labels, batch_tail_labels = \
            batch_token_ids.to(device), batch_mask_ids.to(device), batch_token_type_ids.to(device), batch_entity_labels.to(device), batch_head_labels.to(device), batch_tail_labels.to(device)
        logits1, logits2, logits3 = net(batch_token_ids, batch_mask_ids, batch_token_type_ids)
        loss1 = sparse_multilabel_categorical_crossentropy(y_true=batch_entity_labels, y_pred=logits1, mask_zero=True)
        loss2 = sparse_multilabel_categorical_crossentropy(y_true=batch_head_labels, y_pred=logits2, mask_zero=True)
        loss3 = sparse_multilabel_categorical_crossentropy(y_true=batch_tail_labels, y_pred=logits3, mask_zero=True)
        loss = sum([loss1, loss2, loss3]) / 3
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        sys.stdout.write("\r [EPOCH %d/%d] [Loss:%f]"%(eo, con.getint("para", "epochs"), loss.item()))

    torch.save(net.state_dict(), './erenet.pth')



