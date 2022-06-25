# coding=utf-8

import logging
import random
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import Dataset, TensorDataset, DataLoader

@dataclass(frozen=True)
class InputExample:
    texts: List[str]
    label: int

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def load_data(file):
    with open(file, "r") as fp:
        lines = fp.read().splitlines()
        lines = [line.strip().split("\t") for line in lines]
    data = []
    if len(lines[0]) == 2:
        for row in lines:
            data.append(InputExample(texts=[row[0], row[1]], label=0))
    else:
        for row in lines:
            text_a = row[1]
            text_b = row[2]
            data.append(InputExample(texts=[text_a, text_b], label=int(row[0])))
    return data

def gen_repeat_word(sentence, tokenizer, dup_rate=0.25):
    '''
    @function: 重复句子中的部分token

    @input:
    sentence: string，输入语句

    @return:
    dup_sentence: string，重复token后生成的句子
    '''
    word_tokens = tokenizer.tokenize(sentence)

    # dup_len ∈ [0, max(2, int(dup_rate ∗ N))]
    max_len = max(2, int(dup_rate * len(word_tokens)))
    # 防止随机挑选的数值大于token数量
    dup_len = min(random.choice(range(max_len+1)), len(word_tokens))

    random_indices = random.sample(range(len(word_tokens)), dup_len)
    # print(max_len, dup_len, random_indices)

    dup_word_tokens = []
    for index, word in enumerate(word_tokens):
        dup_word_tokens.append(word)
        if index in random_indices:
            dup_word_tokens.append(word)
    dup_sentence = "".join(dup_word_tokens)
    if "##" not in sentence and "##" in dup_sentence:
        dup_sentence = sentence
    return dup_sentence

class CrossDataset(Dataset):
    def __init__(self, inputExamples, tokenizer, max_len=64, dup_word=False):
        self.inputExamples = inputExamples
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dup_word = dup_word
    
    def __len__(self):
        return len(self.inputExamples)
    
    def __getitem__(self, index):
        inputExample = self.inputExamples[index]
        
        text_a, text_b = inputExample.texts[0], inputExample.texts[1]
        if self.dup_word:
            sample_ratio = random.random()
            if sample_ratio<0.10:
                text_a = gen_repeat_word(text_a, self.tokenizer, dup_rate=0.25)
            elif sample_ratio<0.20:
                text_b = gen_repeat_word(text_b, self.tokenizer, dup_rate=0.25)
            elif sample_ratio<0.30:
                text_a = gen_repeat_word(text_a, self.tokenizer, dup_rate=0.25)
                text_b = gen_repeat_word(text_b, self.tokenizer, dup_rate=0.25)

        inputs = self.tokenizer.encode_plus(
            text_a, text_b,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )
        inputs["labels"] = inputExample.label

        return {
            'input_ids': torch.tensor(inputs["input_ids"], dtype=torch.long),
            'attention_mask': torch.tensor(inputs["attention_mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs["token_type_ids"], dtype=torch.long),
            'labels': torch.tensor(inputs["labels"], dtype=torch.long)
        }

class TowerDataset(Dataset):
    def __init__(self, inputExamples, tokenizer, max_len=32):
        self.inputExamples = inputExamples
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.inputExamples)
    
    def __getitem__(self, index):
        inputExample = self.inputExamples[index]
        
        inputs1 = self.tokenizer.encode_plus(
            inputExample.texts[0],
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )

        inputs2 = self.tokenizer.encode_plus(
            inputExample.texts[1],
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )

        return {
            'input_ids1': torch.tensor(inputs1["input_ids"], dtype=torch.long),
            'input_ids2': torch.tensor(inputs2["input_ids"], dtype=torch.long),
            'attention_mask1': torch.tensor(inputs1["attention_mask"], dtype=torch.long),
            'attention_mask2': torch.tensor(inputs2["attention_mask"], dtype=torch.long),
            'token_type_ids1': torch.tensor(inputs1["token_type_ids"], dtype=torch.long),
            'token_type_ids2': torch.tensor(inputs2["token_type_ids"], dtype=torch.long),
            'labels': torch.tensor(inputExample.label, dtype=torch.long)
        }

class PromptDataset(Dataset):
    def __init__(self, inputExamples, tokenizer, max_len=64):
        self.inputExamples = inputExamples
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.template = []
        unused_id = 1
        for i in range(5):
            self.template.append(f'[unused{unused_id}]')
            unused_id += 1
        
        # self.template.append('意')
        # self.template.append('思')

        for i in range(2):
            self.template.append(f'[unused{unused_id}]')
            unused_id += 1
        
        self.template.append(self.tokenizer.mask_token)
        self.template.append(self.tokenizer.mask_token)
        self.template.append(self.tokenizer.sep_token)
        self.template_ids = self.tokenizer.convert_tokens_to_ids(self.template)
    
    def __len__(self):
        return len(self.inputExamples)
    
    def __getitem__(self, index):
        inputExample = self.inputExamples[index]

        inputs = self.tokenizer.encode_plus(
            inputExample.texts[0], inputExample.texts[1],
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding=False,
            return_token_type_ids=True
        )

        if inputExample.label==0:
            answer = "不同"
        else:
            answer = "相同"
        answer_tokens = self.tokenizer.tokenize(answer)
        answer_ids = self.tokenizer.convert_tokens_to_ids(answer_tokens)

        inputs["labels"] = []
        
        for id in inputs["input_ids"]:
            inputs["labels"].append(id)
        
        for id in self.template_ids:
            inputs["input_ids"].append(id)
            inputs["attention_mask"].append(1)
            inputs["token_type_ids"].append(0)
            if id==self.tokenizer.mask_token_id:
                answer_id = answer_ids.pop(0)
            else:
                answer_id = id
            inputs["labels"].append(answer_id)

        while len(inputs["input_ids"])<self.max_len+len(self.template_ids):
            inputs["input_ids"].append(0)
            inputs["attention_mask"].append(0)
            inputs["token_type_ids"].append(1)
            inputs["labels"].append(0)
        
        return {
            'input_ids': torch.tensor(inputs["input_ids"], dtype=torch.long),
            'attention_mask': torch.tensor(inputs["attention_mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs["token_type_ids"], dtype=torch.long),
            'labels': torch.tensor(inputs["labels"], dtype=torch.long)
        }

if __name__=="__main__":

    inputExamples = load_data("dataset/train.tsv")
    random.shuffle(inputExamples)
    total_num = len(inputExamples)
    train_num = int(0.9*total_num)

    tokenizer = BertTokenizer.from_pretrained('DMetaSoul/sbert-chinese-general-v2')
    
    MyDataset = PromptDataset
    max_len = 64
    train_dataset = MyDataset(inputExamples[0:train_num], tokenizer, max_len, dup_word=True)
    dev_dataset = MyDataset(inputExamples[train_num:], tokenizer, max_len)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=4)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=4, shuffle=False, num_workers=4)

    logger.info(f"train_num:{len(train_dataset)}")
    logger.info(f"dev_num:{len(dev_dataset)}")
    
    for i in range(5):
        logger.info(inputExamples[i])
        logger.info(train_dataset[i])
        # logger.info(train_dataset[i]['input_ids'].shape)
        # logger.info(train_dataset[i]['attention_mask'].shape)
        # logger.info(train_dataset[i]['token_type_ids'].shape)
        # logger.info(train_dataset[i]['labels'].shape)
    
    # for bs in dev_dataloader:
    #     pass
    #     print(bs)
    #     break
