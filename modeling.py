# -*- coding:utf-8 -*-

import os
import argparse
import random
import logging
from tqdm import tqdm

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, TensorDataset, DataLoader
import transformers
from transformers import BertTokenizerFast, BertTokenizer, BertModel, BertForMaskedLM, BertConfig

from utils import load_data

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

class CrossEncoder(torch.nn.Module):
    def __init__(self, bert_name_or_path, pool_type="cls"):
        super().__init__()
        self.config = BertConfig.from_pretrained(bert_name_or_path)
        self.bert = BertModel.from_pretrained(bert_name_or_path, config=self.config)
        self.dropout = torch.nn.Dropout(0.25)
        self.classifier = torch.nn.Linear(self.config.hidden_size, 2, bias=True)
        self.pool_type = pool_type
        for parameter in self.bert.named_parameters():
            # if parameter[0].startswith(("embeddings")):
            #     parameter[1].requires_grad=False
            logger.info(f"{parameter[0]} {parameter[1].shape} {parameter[1].requires_grad}")
        for parameter in self.classifier.named_parameters():
            logger.info(f"{parameter[0]} {parameter[1].shape} {parameter[1].requires_grad}")

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        if self.pool_type=="cls":
            pooler_output = bert_outputs['last_hidden_state'][:, 0]
        elif self.pool_type=="avg":
            hidden_states = bert_outputs["hidden_states"]
            pooler_output = hidden_states[0] + hidden_states[-1]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(pooler_output.size()).float()
            pooler_output = torch.sum(pooler_output * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooler_output = self.dropout(pooler_output)
        logits = self.classifier(pooler_output)
        return logits

class TowerEncoder(torch.nn.Module):
    def __init__(self, bert_name_or_path):
        super().__init__()
        self.config = BertConfig.from_pretrained(bert_name_or_path)
        self.bert = BertModel.from_pretrained(bert_name_or_path, config=self.config)
        self.dropout = torch.nn.Dropout(0.25)
        self.classifier = torch.nn.Linear(3*self.config.hidden_size, 2, bias=True)
        for parameter in self.bert.named_parameters():
            # if parameter[0].startswith(("embeddings")):
            #     parameter[1].requires_grad=False
            logger.info(f"{parameter[0]} {parameter[1].shape} {parameter[1].requires_grad}")
        for parameter in self.classifier.named_parameters():
            logger.info(f"{parameter[0]} {parameter[1].shape} {parameter[1].requires_grad}")

    def forward(self, input_ids1, attention_mask1, token_type_ids1, input_ids2, attention_mask2, token_type_ids2):
        sent_embeddings = []
        for input_ids, attention_mask, token_type_ids in [(input_ids1, attention_mask1, token_type_ids1), (input_ids2, attention_mask2, token_type_ids2)]:
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
            # pooler_output = bert_outputs["pooler_output"]
            hidden_states = bert_outputs["hidden_states"]
            pooler_output = hidden_states[0]+hidden_states[-1]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(pooler_output.size()).float()
            pooler_output = torch.sum(pooler_output * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            sent_embeddings.append(pooler_output)
        pool_embedding = torch.cat([sent_embeddings[0], sent_embeddings[1], torch.abs(sent_embeddings[0]-sent_embeddings[1])], 1)
        logits = self.classifier(pool_embedding)
        return logits

class PromptEncoder(torch.nn.Module):
    def __init__(self, bert_name_or_path):
        super().__init__()
        self.config = BertConfig.from_pretrained(bert_name_or_path)
        self.bert = BertForMaskedLM.from_pretrained(bert_name_or_path, config=self.config)
        for parameter in self.bert.named_parameters():
            # if parameter[0].startswith(("embeddings")):
            #     parameter[1].requires_grad=False
            logger.info(f"{parameter[0]} {parameter[1].shape} {parameter[1].requires_grad}")
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        return outputs

if __name__ == "__main__":
    # model = CrossEncoder('hfl/chinese-roberta-wwm-ext')
    # model = TowerEncoder('hfl/chinese-roberta-wwm-ext')
    # model = PromptEncoder('hfl/chinese-roberta-wwm-ext')
    
    model = BertForMaskedLM.from_pretrained("hfl/chinese-roberta-wwm-ext")
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    inputs = tokenizer(["两句话意思是否[MASK][MASK]","两句话意思的的[MASK][MASK]"], return_tensors="pt")
    logger.info("inputs:")
    logger.info(inputs)
    labels = tokenizer(["两句话意思是否相同", "两句话意思是否不同"], return_tensors="pt")["input_ids"]
    labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
    logger.info("labels:")
    logger.info(labels) 
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
    
    logits = outputs.logits
    
    for i in range(2):
        mask_token_index = (inputs.input_ids[i] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)

    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=False)
    logger.info("mask_token_index:")
    logger.info(mask_token_index)

    for i in range(mask_token_index.shape[0]):
        logger.info(mask_token_index[i])
        logger.info(logits[i])


    # for i in range(2):
    #     # mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero()
    #     mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[i].nonzero(as_tuple=True)[0]
    #     logger.info(mask_token_index) 
    #     predicted_token_id = logits[i, mask_token_index].argmax(axis=-1)
    #     tokenizer.decode(predicted_token_id)

    # mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero()
    # logger.info("mask_token_index:")
    # logger.info(mask_token_index)
    # logger.info("predicted_token:")
    # predicted_token_ids = logits[:, mask_token_index].argmax(axis=-1)
    # logger.info(predicted_token_ids)
    # for predicted_token_id in predicted_token_ids:
    #     logger.info(predicted_token_id)
    #     logger.info(tokenizer.decode(predicted_token_id))
    
    # logger.info("labels:")
    # labels = tokenizer(["两句话意思是否相同", "两句话意思是否不同"], return_tensors="pt")["input_ids"]
    # logger.info(labels)    
    # labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)
    # logger.info(labels)
    # outputs = model(**inputs, labels=labels)
    # logger.info(round(outputs.loss.item(), 2))

    # answers = labels[:, mask_token_index]
    # logger.info(answers)
