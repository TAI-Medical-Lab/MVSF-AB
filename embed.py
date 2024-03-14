import csv

import torch
from protein_bert_pytorch import ProteinBERT, PretrainingWrapper
import pandas as pd
import torch
import numpy as np
from tape import TAPETokenizer
from tqdm import tqdm
import os
from Bio import SeqIO

torch.cuda.set_device(3)
def get_feature(_list):
    # load model
    # model = ProteinBertModel.from_pretrained('bert-base')
    # torch.save(model, 'pretrain_bert.models')
    device = torch.device('cuda')
    model = torch.load('../cmap_final/src/models/pretrain_bert.models')
    # model = ProteinBertModel.from_pretrained('./bert-base-chinese')
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    tokenizer = TAPETokenizer(vocab='iupac')  # iupac是TAPE模型的词汇表，UniRep模型使用unirep。
    feature = []
    for seq in tqdm(_list):      # 进度条
        token_ids = torch.tensor([tokenizer.encode(seq)])
        output = model(token_ids.to(device))
        pooled_output = output[1]
        feature.append(pooled_output[0].tolist())
    _df = pd.DataFrame(np.array(feature))
    return _df



def get_feature2():
    model = ProteinBERT(
        num_tokens=21,
        num_annotation=8943,
        dim=512,
        dim_global=256,
        depth=6,
        narrow_conv_kernel=9,
        wide_conv_kernel=9,
        wide_conv_dilation=5,
        attn_heads=8,
        attn_dim_head=64
    )

    seq = torch.randint(0, 21, (2, 2048))
    mask = torch.ones(2, 2048).bool()
    annotation = torch.randint(0, 1, (2, 8943)).float()

    seq_logits, annotation_logits = model(seq, annotation, mask=mask)

def parse(f, comment="#"):
    names = []
    sequences = []

    for record in SeqIO.parse(f, "fasta"):
        names.append(record.name)
        sequences.append(str(record.seq))

    return names, sequences






if __name__ == '__main__':
    fastaPath = './datasets/seq_natural.fasta'
    outputPath = './datasets/seq_natural_embedding.csv'
    names, sequence = parse(fastaPath)
    new_sequence = []
    for seq in sequence:

        seq = seq.replace('_', '')
        seq = seq.replace('J', '')
        new_sequence.append(seq)
    rows = zip(names, new_sequence)
    with open(outputPath, 'w') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    df = get_feature(new_sequence)

    df.to_csv(outputPath, index=False)

