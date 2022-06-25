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
from transformers import BertTokenizerFast, BertTokenizer, BertModel, BertConfig

from utils import load_data, TowerDataset
from modeling import TowerEncoder

logger = logging.getLogger(__name__)

transformers.logging.set_verbosity_error()

class Trainer:
    def __init__(
        self, epoch, batch_size, input_path, output_directory, bert_name_or_path, device
    ):
        """
        initiation works
        """
        self.input_path = input_path
        self.output_directory = output_directory
        self.bert_name_or_path = bert_name_or_path
        self.epoch = int(epoch)
        self.batch_size = int(batch_size)
        self.device = device
        self.model = TowerEncoder(self.bert_name_or_path)
        self.model.to(device)
        self.loss_fct = CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=1e-5)
    
    def fit(self, train_dataloader, dev_dataloader):
        for epoch in range(self.epoch):
            self.model.train()
            for i, batch in enumerate(tqdm(train_dataloader)):
                batch = {k: v.to(self.device ) for k, v in batch.items()}
                logits = self.model(input_ids1 = batch["input_ids1"], 
                                    attention_mask1 = batch["attention_mask1"],
                                    token_type_ids1 = batch["token_type_ids1"],
                                    input_ids2 = batch["input_ids2"], 
                                    attention_mask2 = batch["attention_mask2"],
                                    token_type_ids2 = batch["token_type_ids2"],
                                    )
                # logger.info(logits.view(-1, 2))
                # logger.info(batch["labels"].view(-1))
                loss = self.loss_fct(logits.view(-1, 2), batch["labels"].view(-1))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if i % 50 == 0:
                    preds = torch.argmax(logits, dim=1)
                    acc = (preds == batch["labels"]).float().mean()
                    logger.info(f"epoch:{epoch+1} step:{i+1} train loss: {loss:.4f}, acc: {acc:.4f}")
            
            self.model.eval()
            with torch.no_grad():
                dev_loss = 0
                dev_num = 0
                dev_correct_num = 0
                for i, batch in enumerate(dev_dataloader):
                    batch = {k: v.to(self.device ) for k, v in batch.items()}
                    logits = self.model(input_ids1 = batch["input_ids1"], 
                                        attention_mask1 = batch["attention_mask1"],
                                        token_type_ids1 = batch["token_type_ids1"],
                                        input_ids2 = batch["input_ids2"], 
                                        attention_mask2 = batch["attention_mask2"],
                                        token_type_ids2 = batch["token_type_ids2"],
                                        )
                    preds = torch.argmax(logits, dim=1)
                    loss = self.loss_fct(logits.view(-1, 2), batch["labels"].view(-1))
                    dev_num += batch["labels"].shape[0]
                    dev_loss += loss*batch["labels"].shape[0]
                    dev_correct_num += (preds == batch["labels"]).float().sum()
                acc = dev_correct_num/dev_num
                loss = dev_loss/dev_num
                logger.info(f"epoch:{epoch+1} dev loss: {loss:.4f}, acc: {acc:.4f}")

def main():
    parser = argparse.ArgumentParser()

    # FORMAT: python train.py -i <input_path> -o <output_directory>
    parser.add_argument("-e", "--epoch", dest="epoch", help="epoch nums")
    parser.add_argument("-b", "--batch_size", dest="batch_size", help="batch_size")
    parser.add_argument(
        "-i", "--input_path", dest="input_path", help="input file full path"
    )
    parser.add_argument(
        "-o",
        "--output_directory",
        dest="output_directory",
        help="output file full directory",
    )
    parser.add_argument(
        "-l", "--log", dest="loglevel", help="logging level", default="warning"
    )
    parser.add_argument(
        "-m", "--bert_name_or_path", dest="bert_name_or_path", help="bert_name_or_path"
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    tokenizer = BertTokenizerFast.from_pretrained(args.bert_name_or_path)
    
    # load data
    inputExamples = load_data(args.input_path)
    random.shuffle(inputExamples)
    total_num = len(inputExamples)
    train_num = int(0.9*total_num)
    batch_size = int(args.batch_size)
    
    MyDataset = TowerDataset
    max_len = 32
    train_dataset = MyDataset(inputExamples[0:train_num], tokenizer, max_len)
    dev_dataset = MyDataset(inputExamples[train_num:], tokenizer, max_len)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    logger.info(f"running device:{device}")
    logger.info(args.output_directory)

    trainer = Trainer(
        epoch=args.epoch,
        batch_size=args.batch_size,
        input_path=args.input_path,
        output_directory=args.output_directory,
        bert_name_or_path=args.bert_name_or_path,
        device=device
    )

    trainer.fit(train_dataloader, dev_dataloader)
    torch.save(trainer.model.state_dict(), args.output_directory)

    logging.info(f"train output saved to: {args.output_directory}")

if __name__ == "__main__":
    main()