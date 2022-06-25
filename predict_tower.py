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

from utils import load_data, TowerDataset
from modeling import TowerEncoder

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
        self.model = TowerEncoder(bert_name_or_path)
        self.model.load_state_dict(torch.load(saved_model_path))
        self.model.to(device)
        self.output_path = output_path
    
    def predict(self, test_dataloader):
        all_preds = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                batch = {k: v.to(self.device ) for k, v in batch.items()}
                logits = self.model(input_ids1 = batch["input_ids1"], 
                                    attention_mask1 = batch["attention_mask1"],
                                    token_type_ids1 = batch["token_type_ids1"],
                                    input_ids2 = batch["input_ids2"], 
                                    attention_mask2 = batch["attention_mask2"],
                                    token_type_ids2 = batch["token_type_ids2"],
                                    )
                preds = torch.argmax(logits, dim=1)
                for pred in preds.tolist():
                    all_preds.append(pred)
        with open(self.output_path, "w") as fp:
            for pred in all_preds:
                fp.write(json.dumps({"label": pred})+"\n")

def main():
    parser = argparse.ArgumentParser()
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    bert_name_or_path = 'DMetaSoul/sbert-chinese-general-v2'
    # bert_name_or_path = 'hfl/chinese-roberta-wwm-ext'
    save_model_name = 'tower_sbert_first_last_avg'
    saved_model_path = f'models/{save_model_name}.pth'
    output_path = f'predicts/{save_model_name}.json'

    dirname, basename = os.path.split(output_path) 
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    tokenizer = BertTokenizerFast.from_pretrained(bert_name_or_path)

    inputExamples = load_data("dataset/test.tsv")
    test_dataset = TowerDataset(inputExamples, tokenizer, 32)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False, num_workers=4)

    predictor = Predictor(bert_name_or_path, saved_model_path, output_path, device)
    predictor.predict(test_dataloader)

if __name__=="__main__":
    main()