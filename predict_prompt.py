# -*- coding:utf-8 -*-

import os
import argparse
import random
import logging
import json
from tqdm import tqdm

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, TensorDataset, DataLoader
import transformers
from transformers import BertTokenizerFast, BertTokenizer, BertModel, BertConfig

from utils import load_data, PromptDataset
from modeling import PromptEncoder

logger = logging.getLogger(__name__)

transformers.logging.set_verbosity_error()

class Predictor:
    def __init__(
        self, bert_name_or_path, saved_model_path, output_path, device
    ):
        """
        initiation works
        """
        self.device = device
        self.model = PromptEncoder(bert_name_or_path)
        self.model.load_state_dict(torch.load(saved_model_path))
        self.model.to(device)
        self.output_path = output_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_name_or_path)
    
    def get_answers(self, logits, input_ids):
        bs = input_ids.shape[0]
        predict_answers = []
        answers = []
        for i in range(bs):
            mask_token_index = (input_ids[i] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            predicted_token_ids = logits[i, mask_token_index].argmax(axis=-1)
            predict_answer = self.tokenizer.decode(predicted_token_ids)
            predict_answers.append(predict_answer.replace(" ", ""))
        return predict_answers
    
    def predict(self, test_dataloader):
        all_preds = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                batch = {k: v.to(self.device ) for k, v in batch.items()}
                model_outputs = self.model(input_ids=batch["input_ids"], 
                                    attention_mask=batch["attention_mask"],
                                    token_type_ids=batch["token_type_ids"],
                                    labels=None,
                                    )
                predict_answers = self.get_answers(model_outputs.logits, batch["input_ids"])
                for pred in predict_answers:
                    all_preds.append(pred)
        with open(self.output_path, "w") as fp:
            for pred in all_preds:
                fp.write(json.dumps({"label": pred}, ensure_ascii=False)+"\n")

def main():
    parser = argparse.ArgumentParser()
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    # bert_name_or_path = 'DMetaSoul/sbert-chinese-general-v2'
    bert_name_or_path = 'hfl/chinese-roberta-wwm-ext'
    save_model_name = 'roberta_prompt'
    saved_model_path = f'models/{save_model_name}.pth'
    output_path = f'predicts/{save_model_name}.json'

    dirname, basename = os.path.split(output_path) 
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    tokenizer = BertTokenizerFast.from_pretrained(bert_name_or_path)

    inputExamples = load_data("dataset/test.tsv")
    test_dataset = PromptDataset(inputExamples, tokenizer, 64)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False, num_workers=4)

    predictor = Predictor(bert_name_or_path, saved_model_path, output_path, device)
    predictor.predict(test_dataloader)

if __name__=="__main__":
    main()