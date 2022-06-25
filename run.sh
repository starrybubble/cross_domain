#!/usr/bin/bash

# -e参数指定训练轮数，使用-b参数指定batch size
# -m参数中指定所使用的bert模型路径，支持Huggingface/transformers model (如 BERT, RoBERTa, XLNet, XLM-R)
# -i指定训练文件地址 -o指定模型保存地址
TRAIN_FILE_DIR=./dataset/
PLM='hfl/chinese-roberta-wwm-ext'
# PLM='DMetaSoul/sbert-chinese-general-v2'
SAVE_MODEL_DIR=models/roberta_cls_repeat_word.pth
# if [ ! -d $SAVE_MODEL_DIR ]; then
#         mkdir -p $SAVE_MODEL_DIR
# fi
python3 train.py -e 4 -b 128 -i $TRAIN_FILE_DIR/train.tsv -o $SAVE_MODEL_DIR -m $PLM
echo 'TRAIN FINISHED'
