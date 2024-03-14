"""
Train a new model.
"""
from __future__ import annotations

import time
from sklearn.model_selection import KFold, StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import IterableDataset, dataloader
from multiprocessing.reduction import ForkingPickler
from sklearn.metrics import average_precision_score as average_precision
from tqdm import tqdm
from typing import Callable, NamedTuple, Optional
from collections import OrderedDict
import sys
import numpy as np
import argparse
import pandas as pd
import torch.optim as optim
from torch.optim import Optimizer
from src.models.mvsf import ModelAffinity
from src.utils import *
from multiprocessing.reduction import ForkingPickler
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.cuda.amp import GradScaler, autocast
from torcheval.metrics.functional import r2_score

# 解决使用multiprocessing模块时由于Tensor对象内部实现机制导致的序列化错误
default_collate_func = dataloader.default_collate
def default_collate_override(batch):
    dataloader._use_shared_memory = False
    return default_collate_func(batch)
setattr(dataloader, 'default_collate', default_collate_override)
for t in torch._storage_classes:
    if sys.version_info[0] == 2:
        if t in ForkingPickler.dispatch:
            del ForkingPickler.dispatch[t]
    else:
        if t in ForkingPickler._extra_reducers:
            del ForkingPickler._extra_reducers[t]

class TrainArguments(NamedTuple):
    cmd: str
    device: int
    train: str
    test: str
    no_augment: bool
    augment_weight: float
    weight_module1: float
    weight_module2: float
    num_epochs: int
    batch_size: int
    weight_decay: float
    lr: float
    kfolds: int
    outfile: Optional[str]
    save_prefix: Optional[str]
    checkpoint: Optional[str]
    seed: Optional[int]
    func: Callable[[TrainArguments], None]

def add_args(parser):
    data_grp = parser.add_argument_group("Data")
    contact_grp = parser.add_argument_group("Contact Module")
    train_grp = parser.add_argument_group("Training")
    misc_grp = parser.add_argument_group("Output and Device")

    # Data
    data_grp.add_argument("--train", default="datasets/pairs_sabdab.csv", help="list of training pairs")
    data_grp.add_argument("--test", default="datasets/pairs_benchmark.csv", help="list of validation/testing pairs")
    data_grp.add_argument("--seq-path", default="datasets/seq_natural.fasta")
    data_grp.add_argument("--feature-path", default="datasets/seq_natural_embedding.csv")
    data_grp.add_argument("--no-augment", default=True, help="data is automatically augmented by adding (B A) for all pairs (A B). Set this flag to not augment data",)
    data_grp.add_argument("--augment-weight", type=float, default=0.5, help="weight of augment data",)

    # Model
    contact_grp.add_argument("--weight-module1", type=float, default=1, help="weight of module1",)
    contact_grp.add_argument("--weight-module2", type=float, default=1, help="weight of module1",)

    # Training
    train_grp.add_argument("--num-epochs", type=int, default=30, help="number of epochs",)
    train_grp.add_argument("--batch-size", type=int, default=16, help="minibatch size (default: 16)",)
    train_grp.add_argument("--weight-decay", type=float, default=0.00001, help="L2 regularization /0.0001",)  # 正则化项的设置
    train_grp.add_argument("--lr", type=float, default=0.00001, help="learning rate",)
    train_grp.add_argument("--kfolds", type=int, default=10)
    train_grp.add_argument("--cross-validate", default=True, help="cross validate",)

    # Output and Device
    misc_grp.add_argument("-o", "--outfile", help="output file path (default: stdout)")
    misc_grp.add_argument("--save-prefix", help="path prefix for saving models")
    misc_grp.add_argument("-d", "--device", type=int, required=True, help="compute device to use")
    misc_grp.add_argument("--checkpoint", help="checkpoint model to start training from")
    misc_grp.add_argument("--seed", help="Set random seed", type=int)
    return parser

def predict_affinity(model, Lchain, Hchain, antigen, embedding_tensor, aaindex_feature, use_cuda):
    b = len(Hchain)
    lchain_embeddings = []
    hchain_embeddings = []
    ag_embeddings = []

    lchain_aaindex = []
    hchain_aaindex = []
    ag_aaindex = []

    for i in range(b):
        lchain_embedding = embedding_tensor[Lchain[i]]
        hchain_embedding = embedding_tensor[Hchain[i]]
        ag_embedding = embedding_tensor[antigen[i]]

        lchain_aaindex.append(aaindex_feature[Lchain[i]])
        hchain_aaindex.append(aaindex_feature[Hchain[i]])
        ag_aaindex.append(aaindex_feature[antigen[i]])

        lchain_embeddings.append(lchain_embedding)
        hchain_embeddings.append(hchain_embedding)
        ag_embeddings.append(ag_embedding)

    if use_cuda:
        lchain_embeddings = torch.stack(lchain_embeddings, 0).cuda()
        hchain_embeddings = torch.stack(hchain_embeddings, 0).cuda()
        ag_embeddings = torch.stack(ag_embeddings, 0).cuda()

        lchain_aaindex = torch.stack(lchain_aaindex, 0).cuda()
        hchain_aaindex = torch.stack(hchain_aaindex, 0).cuda()
        ag_aaindex = torch.stack(ag_aaindex, 0).cuda()



    ph = model.predict(lchain_aaindex, hchain_aaindex, ag_aaindex, lchain_embeddings, hchain_embeddings, ag_embeddings)
    return ph

def model_eval(model, test_iterator, embedding_tensors, aaindex_feature, write, weight1, weight2, use_cuda):

    p_hat = []
    true_y = []

    for lchain, hchain, antigen, y in test_iterator:

        ph = predict_affinity(model, lchain, hchain, antigen, embedding_tensors, aaindex_feature, use_cuda)
        p_hat.append(ph)
        true_y.append(y)

    y = torch.cat(true_y, 0)

    p_hat = torch.cat(p_hat, 0)
    if use_cuda:
        y.cuda()
        p_hat = torch.Tensor([x.cuda() for x in p_hat])
        p_hat.cuda()
    criterion = nn.MSELoss()
    loss = criterion(p_hat.float(), y.float())

    with torch.no_grad():
        p_hat = p_hat.float()
        y = y.float()
        max_val = 16.9138
        min_val = 5.0400
        p_hat = (p_hat * (max_val - min_val)) + min_val

        # if write:
        #     with open('pred_skempi.csv', 'a') as f:
        #         for i in range(len(y)):
        #             f.write(str(y[i].item()) + ',' + str(p_hat[i].item()) + '\n')

        rmse = torch.sqrt(torch.mean((y - p_hat) ** 2)).item()
        mae = torch.mean(torch.abs(y - p_hat)).item()
        r_2 = r2_score(y, p_hat).item()
        p = pearsonr(y, p_hat).item()

    return loss, rmse, mae, r_2, p

def train_model(args, output):
    # Create data sets
    batch_size = args.batch_size
    use_cuda = (args.device > -1) and torch.cuda.is_available()  # True
    train_fi = args.train
    train_df = pd.read_csv(train_fi)
    test_fi = args.test
    test_df = pd.read_csv(test_fi)


    # Train the model
    lr = args.lr
    # wd = args.weight_decay  # 0.0001
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    digits = int(np.floor(np.log10(num_epochs))) + 1
    save_prefix = args.save_prefix
    weight1 = args.weight_module1
    weight2 = args.weight_module2


    log(f'Using save prefix "{save_prefix}"', file=output)
    log(f"Training with SAM: lr={lr}", file=output)
    log(f"\tnum_epochs: {num_epochs}", file=output)
    log(f"\tbatch_size: {batch_size}", file=output)
    log(f"\tmodule 1 weight: {weight1}", file=output)
    log(f"\tmodule 2 weight: {weight2}", file=output)
    output.flush()


    if(args.cross_validate):
    # ===============================================cross validation=================================================
        k_folds = args.kfolds
        kfold = KFold(n_splits=k_folds, shuffle=False)
        for fold, (train_ids, test_ids) in enumerate(kfold.split(train_df)):
            print(f'******************************** FOLD {fold} ******************************')
            log(f'******************************** FOLD {fold} ******************************', file=output)
            train_df_fold = train_df.iloc[train_ids]
            test_df_fold = train_df.iloc[test_ids]
            train_df_fold = train_df_fold.reset_index(drop=True)
            test_df_fold = test_df_fold.reset_index(drop=True)

            train_df_fold.columns = ["light", "heavy", "antigen", "delta_g"]
            train_l_fold = train_df_fold["light"]
            train_h_fold = train_df_fold["heavy"]
            train_ag_fold = train_df_fold["antigen"]
            train_y_fold = torch.from_numpy(train_df_fold["delta_g"].values)
            train_y_fold = -train_y_fold

            max_val = 16.05654
            min_val = 5.0400
            train_y_fold = (train_y_fold - min_val) / (max_val - min_val)

            test_df_fold.columns = ["light", "heavy", "antigen", "delta_g"]
            test_l_fold = test_df_fold["light"]
            test_h_fold = test_df_fold["heavy"]
            test_ag_fold = test_df_fold["antigen"]
            test_y_fold = torch.from_numpy(test_df_fold["delta_g"].values)
            test_y_fold = -test_y_fold

            train_dataset_fold = PairedDataset(train_l_fold, train_h_fold, train_ag_fold, train_y_fold)
            train_iterator_fold = torch.utils.data.DataLoader(
                train_dataset_fold,
                batch_size=batch_size,
                collate_fn=collate_paired_sequences,
                shuffle=True,
                pin_memory=False,
                drop_last=False,
                # num_workers=2,
            )
            log(f"Loaded {len(train_l_fold)} training pairs", file=output)
            output.flush()

            test_dataset_fold = PairedDataset(test_l_fold, test_h_fold, test_ag_fold, test_y_fold)
            test_iterator_fold = torch.utils.data.DataLoader(
                test_dataset_fold,
                batch_size=batch_size,
                collate_fn=collate_paired_sequences,
                shuffle=False,
                pin_memory=False,
                drop_last=False,
                # num_workers=2,
            )

            all_proteins = set(train_l_fold).union(train_h_fold).union(train_ag_fold) \
                .union(test_l_fold).union(test_h_fold).union(test_ag_fold)
            fastaPath = args.seq_path
            embeddingPath = args.feature_path
            embeddings = embed_dict(fastaPath, embeddingPath)
            log("Embedded successfully...", file=output)
            aaindex_feature = seq_aaindex_dict(all_proteins, fastaPath)

            model = ModelAffinity(batch_size, use_cuda)
            model.use_cuda = use_cuda  # default is False
            if use_cuda:
                model.cuda()
            params = [p for p in model.parameters() if p.requires_grad]
            base_optimizer = optim.SGD
            optimizer = SAM(params, base_optimizer, lr=lr, weight_decay=args.weight_decay)

            batch_report_fmt = ("[{}/{}] training {:.1%}: Loss={:.6}, MSE={:.6}, MAE={:.6}")
            epoch_report_fmt = (
                "-----------------------------------Finished Epoch {}/{}: Loss={:.6}, RMSE={:.6}, MAE={:.6}, r_2={:.6}, p={:.6}")

            N = len(train_iterator_fold) * batch_size


            for epoch in range(num_epochs):
                if epoch == 10:
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
                print("lr:", optimizer.param_groups[0]['lr'])
                model.train()
                n = 0
                loss_accum = 0
                # acc_accum = 0
                mse_accum = 0
                mae_accum = 0
                optimizer.zero_grad()
                all_y = []
                all_p_hat = []
                for (lchain, hchain, antigen, y) in train_iterator_fold:

                    phat = predict_affinity(
                        model, lchain, hchain, antigen, embeddings, aaindex_feature, use_cuda=use_cuda)
                    phat = phat.float().view(-1)

                    if use_cuda:
                        y = y.cuda()
                    # y = Variable(y)
                    y = y.float()

                    criterion = nn.MSELoss()
                    b = len(y)
                    loss = criterion(phat, y)
                    loss.requires_grad_(True)
                    loss.backward()
                    # scaler.scale(loss).backward()
                    if use_cuda:
                        y = y.cpu()
                        phat = phat.cpu()
                    all_y.append(y)
                    all_p_hat.append(phat)

                    with torch.no_grad():
                        phat = phat.float()
                        y = y.float()
                        mse = torch.mean((y - phat) ** 2).item()
                        mae = torch.mean(torch.abs(y - phat)).item()
                    n += b
                    delta = b * (loss.item() - loss_accum)
                    loss_accum += delta / n
                    delta = b * (mse - mse_accum)
                    mse_accum += delta / n
                    delta = b * (mae - mae_accum)
                    mae_accum += delta / n
                    report = (n - b) // 100 < n // 100

                    optimizer.step()

                    if report:
                        tokens = [epoch + 1, num_epochs, n / N, loss_accum, mse_accum, mae_accum, ]
                        log(batch_report_fmt.format(*tokens), file=output)
                        output.flush()

                model.eval()
                with torch.no_grad():
                    if epoch+1 == 30:
                        write = True
                    else:
                        write = False
                    (inter_loss, inter_rmse, inter_mae, inter_r_2, inter_p,) = model_eval(
                        model, test_iterator_fold, embeddings, aaindex_feature, write, weight1, weight2, use_cuda=use_cuda)

                    tokens = [epoch + 1, num_epochs, inter_loss, inter_mae, inter_rmse, inter_r_2, inter_p, ]

                    # scheduler.step(inter_mse)
                    log(epoch_report_fmt.format(*tokens), file=output)
                    output.flush()

                    # Save the model (every epoch)
                    # if save_prefix is not None:
                    #     save_path = (save_prefix + "_epoch" + str(epoch + 1).zfill(digits) + ".pth")
                    #     log(f"Saving model to {save_path}", file=output)
                    #     model.cpu()
                    #     torch.save(model, save_path)
                    #     if use_cuda:
                    #         model.cuda()

                    # update learning rate
                    # scheduler.step()

                # output.flush()
            # break
    else:
        num_samples = len(train_df)
        train_df.columns = ["light", "heavy", "antigen", "delta_g"]
        train_l = train_df["light"]
        train_h = train_df["heavy"]
        train_ag = train_df["antigen"]
        train_y = torch.from_numpy(train_df["delta_g"].values)
        train_y = -train_y
        train_y = NormalizeData(train_y)

        test_df.columns = ["light", "heavy", "antigen", "delta_g"]
        test_l = test_df["light"]
        test_h = test_df["heavy"]
        test_ag = test_df["antigen"]
        test_y = torch.from_numpy(test_df["delta_g"].values)
        test_y = -test_y

        train_dataset = PairedDataset(train_l, train_h, train_ag, train_y)
        train_iterator = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collate_paired_sequences,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            # num_workers=4,
        )
        log(f"Loaded {len(train_l)} training pairs", file=output)
        output.flush()

        test_dataset = PairedDataset(test_l, test_h, test_ag, test_y)
        test_iterator = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=collate_paired_sequences,
            shuffle=False,
            pin_memory=False,
            drop_last=True,
            # num_workers=4,
        )

        log(f"Loaded {len(test_l)} test pairs", file=output)
        log("Loading embeddings...", file=output)
        output.flush()

        all_proteins = set(train_l).union(train_h).union(train_ag).union(test_l).union(test_h).union(test_ag)

        fastaPath = args.seq_path
        embeddingPath = args.feature_path
        embeddings = embed_dict(fastaPath, embeddingPath)
        log("embeded successfully...", file=output)
        aaindex_feature = seq_aaindex_dict(all_proteins, fastaPath)

        model = ModelAffinity(batch_size, use_cuda)
        if use_cuda:
            model.cuda()

        params = [p for p in model.parameters() if p.requires_grad]
        base_optimizer = optim.Adam
        optimizer = SAM(params, base_optimizer, lr=lr)
        log(f'Using save prefix "{save_prefix}"', file=output)
        log(f"Training with SAM: lr={lr}", file=output)
        log(f"\tnum_epochs: {num_epochs}", file=output)
        log(f"\tbatch_size: {batch_size}", file=output)
        log(f"\tmodule 1 weight: {weight1}", file=output)
        log(f"\tmodule 2 weight: {weight2}", file=output)
        output.flush()

        batch_report_fmt = ("[{}/{}] training {:.1%}: Loss={:.6}, MSE={:.6}, MAE={:.6}")
        epoch_report_fmt = (
            "-----------------------------------Finished Epoch {}/{}: Loss={:.6}, RMSE={:.6}, MAE={:.6}, r_2={:.6}, p={:.6}")

        N = len(train_iterator) * batch_size
        for epoch in range(num_epochs):

            model.train()
            n = 0
            loss_accum = 0
            mse_accum = 0
            mae_accum = 0
            all_y = []
            all_p_hat = []
            optimizer.zero_grad()
            for (lchain, hchain, antigen, y) in train_iterator:
                phat = predict_affinity(
                    model, lchain, hchain, antigen, embeddings, aaindex_feature, use_cuda=use_cuda)
                phat = phat.float().view(-1)

                if use_cuda:
                    y = y.cuda()
                # y = Variable(y)
                y = y.float()

                criterion = nn.MSELoss()
                b = len(y)
                loss = criterion(phat, y)
                loss.requires_grad_(True)
                loss.backward()
                # scaler.scale(loss).backward()
                if use_cuda:
                    y = y.cpu()
                    phat = phat.cpu()

                all_y.append(y)
                all_p_hat.append(phat)

                with torch.no_grad():
                    phat = phat.float()
                    y = y.float()
                    mse = torch.mean((y - phat) ** 2).item()
                    mae = torch.mean(torch.abs(y - phat)).item()
                n += b
                delta = b * (loss.item() - loss_accum)
                loss_accum += delta / n
                delta = b * (mse - mse_accum)
                mse_accum += delta / n
                delta = b * (mae - mae_accum)
                mae_accum += delta / n
                report = (n - b) // 100 < n // 100

                optimizer.step()
                if report:
                    tokens = [epoch + 1, num_epochs, n / N, loss_accum, mse_accum, mae_accum, ]
                    log(batch_report_fmt.format(*tokens), file=output)
                    output.flush()

            model.eval()
            with torch.no_grad():

                (inter_loss, inter_rmse, inter_mae, inter_r_2, inter_p,) = model_eval(
                    model, test_iterator, embeddings, aaindex_feature, weight1, weight2, use_cuda=use_cuda)
                tokens = [epoch + 1, num_epochs, inter_loss, inter_mae, inter_rmse, inter_r_2, inter_p, ]
                # scheduler.step(inter_mse)
                log(epoch_report_fmt.format(*tokens), file=output)
                output.flush()

                # Save the model (every epoch)
                # if save_prefix is not None:
                #     save_path = (save_prefix + "_epoch" + str(epoch + 1).zfill(digits) + ".pth")
                #     log(f"Saving model to {save_path}", file=output)
                #     model.cpu()
                #     torch.save(model, save_path)
                #     if use_cuda:
                #         model.cuda()

                # update learning rate
                # scheduler.step()

    # Save the model (final)
    # if save_prefix is not None:
    #     save_path = save_prefix + "_final.pth"
    #     log(f"Saving final model to {save_path}", file=output)
    #     model.cpu()
    #     torch.save(model, save_path)
    #     if use_cuda:
    #         model.cuda()

def main(args):
    output = args.outfile
    if output is None:
        output = sys.stdout
    else:
        output = open(output, "w")

    log(f'Called as: {" ".join(sys.argv)}', file=output, print_also=True)

    # Set the device
    device = args.device
    use_cuda = (device > -1) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)
        log(
            f"Using CUDA device {device} - {torch.cuda.get_device_name(device)}",
            file=output,
            print_also=True,
        )
    else:
        log("Using CPU", file=output, print_also=True)
        device = "cpu"

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    train_model(args, output)

    output.close()




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())

