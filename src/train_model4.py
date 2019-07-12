import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time, random, sys
import sklearn.metrics
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append('.')
from utils import AverageMeter, cycle, CosineLR, WaveDataset, calculate_per_class_lwlrap
from models import EnvNetv2

# set parameters
NUM_FOLD = 5
NUM_CLASS = 80
SEED = 42
NUM_EPOCH =4 #80*3
NUM_CYCLE = 2 #80
BATCH_SIZE = 64
LR = [1e-1, 1e-6]
FOLD_LIST = [1, 2, 3, 4, 5]
FOLD_LIST = [1, 2,]
CROP_LENGTH = 133300
LOAD_DIR = "../models/envnet_model4_0"
OUTPUT_DIR = "../models/envnet_model4"

cudnn.benchmark = True
starttime = time.time()


def main():
    # load table data
    df_train = pd.read_csv("../input/train_curated.csv")
    df_noisy = pd.read_csv("../input/train_noisy.csv")
    df_test = pd.read_csv("../input/sample_submission.csv")
    labels = df_test.columns[1:].tolist()
    for label in labels:
        df_train[label] = df_train['labels'].apply(lambda x: label in x)
        df_noisy[label] = df_noisy['labels'].apply(lambda x: label in x)

    df_train['path'] = "../input/train_curated/" + df_train['fname']
    df_test['path'] = "../input/test/" + df_train['fname']
    df_noisy['path'] = "../input/train_noisy/" + df_noisy['fname']

    # fold splitting
    folds = list(KFold(n_splits=NUM_FOLD, shuffle=True, random_state=SEED).split(np.arange(len(df_train))))

    # Training
    log_columns = ['epoch', 'bce', 'lwlrap', 'bce_noisy', 'lwlrap_noisy', 'val_bce', 'val_lwlrap', 'time']
    for fold, (ids_train_split, ids_valid_split) in enumerate(folds):
        if fold+1 not in FOLD_LIST: continue
        print("fold: {}".format(fold + 1))
        train_log = pd.DataFrame(columns=log_columns)

        # build model
        model = EnvNetv2(NUM_CLASS).cuda()
        # model.load_state_dict(torch.load("{}/weight_fold_{}_epoch_400.pth".format(LOAD_DIR, fold+1)))

        # prepare data loaders
        df_train_fold = df_train.iloc[ids_train_split].reset_index(drop=True)
        dataset_train = WaveDataset(df_train_fold['path'], df_train_fold[labels].values,
                                    crop=CROP_LENGTH, crop_mode='random', padding=CROP_LENGTH//2,
                                    mixup=True, scaling=1.25, gain=6,
                                    )
        train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=1, pin_memory=True,
                                  )

        df_valid = df_train.iloc[ids_valid_split].reset_index(drop=True)
        dataset_valid = WaveDataset(df_valid['path'], df_valid[labels].values, padding=CROP_LENGTH//2)
        valid_loader = DataLoader(dataset_valid, batch_size=1,
                                  shuffle=False, num_workers=1, pin_memory=True,
                                  )

        dataset_noisy = WaveDataset(df_noisy['path'], df_noisy[labels].values,
                                    crop=CROP_LENGTH, crop_mode='random', padding=CROP_LENGTH//2,
                                    mixup=True, scaling=1.25, gain=6,
                                   )
        noisy_loader = DataLoader(dataset_noisy, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=1, pin_memory=True,
                                  )
        noisy_itr = cycle(noisy_loader)

        # set optimizer and loss
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LR[0],momentum = 0.9,nesterov = True)
        scheduler = CosineLR(optimizer, step_size_min=LR[1], t0=len(train_loader) * NUM_CYCLE, tmult=1)

        # training
        for epoch in range(NUM_EPOCH):
            # train for one epoch
            bce, lwlrap, bce_noisy, lwlrap_noisy = train((train_loader, noisy_itr), model, optimizer, scheduler, epoch)

            # evaluate on validation set
            val_bce, val_lwlrap = validate(valid_loader, model)

            # print log
            endtime = time.time() - starttime
            print("Epoch: {}/{} ".format(epoch + 1, NUM_EPOCH)
                  + "CE: {:.4f} ".format(bce)
                  + "LwLRAP: {:.4f} ".format(lwlrap)
                  + "Noisy CE: {:.4f} ".format(bce_noisy)
                  + "Noisy LWLRAP: {:.4f} ".format(lwlrap_noisy)
                  + "Valid CE: {:.4f} ".format(val_bce)
                  + "Valid LWLRAP: {:.4f} ".format(val_lwlrap)
                  + "sec: {:.1f}".format(endtime)
                  )

            # save log and weights
            train_log_epoch = pd.DataFrame(
                [[epoch+1, bce, lwlrap, bce_noisy, lwlrap_noisy, val_bce, val_lwlrap, endtime]],
                columns=log_columns)
            train_log = pd.concat([train_log, train_log_epoch])
            train_log.to_csv("{}/train_log_fold{}.csv".format(OUTPUT_DIR, fold+1), index=False)
            if (epoch+1)%NUM_CYCLE==0:
                torch.save(model.state_dict(), "{}/weight_fold_{}_epoch_{}.pth".format(OUTPUT_DIR, fold+1, epoch+1))


def train(train_loaders, model, optimizer, scheduler, epoch):
    train_loader, noisy_itr = train_loaders
    bce_avr = AverageMeter()
    bce_noisy_avr = AverageMeter()
    criterion_bce = nn.BCEWithLogitsLoss().cuda()
    sigmoid = nn.Sigmoid().cuda()

    # switch to train mode
    model.train()

    # training
    preds = np.zeros([0, NUM_CLASS], np.float32)
    y_true = np.zeros([0, NUM_CLASS], np.float32)
    preds_noisy = np.zeros([0, NUM_CLASS], np.float32)
    y_true_noisy = np.zeros([0, NUM_CLASS], np.float32)
    for i, (input, target) in enumerate(train_loader):
        # get batches
        input = torch.autograd.Variable(input.cuda())
        target = torch.autograd.Variable(target.cuda())

        input_noisy, target_noisy = next(noisy_itr)
        input_noisy = torch.autograd.Variable(input_noisy.cuda())
        target_noisy = torch.autograd.Variable(target_noisy.cuda())

        # compute output
        output = model(input)
        bce = criterion_bce(sigmoid(output), target)
        output_noisy = model.noisy(input_noisy)
        bce_noisy = criterion_bce(sigmoid(output_noisy), target_noisy)
        loss = bce + bce_noisy
        pred = sigmoid(output)
        pred = pred.data.cpu().numpy()
        pred_noisy = sigmoid(output_noisy)
        pred_noisy = pred_noisy.data.cpu().numpy()

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # record log
        bce_avr.update(bce.data, input.size(0))
        bce_noisy_avr.update(bce_noisy.data, input.size(0))
        preds = np.concatenate([preds, pred])
        y_true = np.concatenate([y_true, target.data.cpu().numpy()])
        preds_noisy = np.concatenate([preds_noisy, pred_noisy])
        y_true_noisy = np.concatenate([y_true_noisy, target_noisy.data.cpu().numpy()])

    # calc metric
    per_class_lwlrap, weight_per_class = calculate_per_class_lwlrap(y_true, preds)
    lwlrap = np.sum(per_class_lwlrap * weight_per_class)
    per_class_lwlrap, weight_per_class = calculate_per_class_lwlrap(y_true_noisy, preds_noisy)
    lwlrap_noisy = np.sum(per_class_lwlrap * weight_per_class)

    return bce_avr.avg.item(), lwlrap, bce_noisy_avr.avg.item(), lwlrap_noisy


def validate(val_loader, model):
    bce_avr = AverageMeter()
    sigmoid = nn.Sigmoid().cuda()
    criterion_bce = nn.BCEWithLogitsLoss().cuda()

    # switch to eval mode
    model.eval()

    # validate
    preds = np.zeros([0, NUM_CLASS], np.float32)
    y_true = np.zeros([0, NUM_CLASS], np.float32)
    for i, (input, target) in enumerate(val_loader):
        # get batches
        input = torch.autograd.Variable(input.cuda())
        target = torch.autograd.Variable(target.cuda())

        # compute output
        with torch.no_grad():
            output = model(input)
            bce = criterion_bce(sigmoid(output), target)
            pred = sigmoid(output)
            pred = pred.data.cpu().numpy()

        # record log
        bce_avr.update(bce.data, input.size(0))
        preds = np.concatenate([preds, pred])
        y_true = np.concatenate([y_true, target.data.cpu().numpy()])

    # calc metric
    per_class_lwlrap, weight_per_class = calculate_per_class_lwlrap(y_true, preds)
    lwlrap = np.sum(per_class_lwlrap * weight_per_class)

    return bce_avr.avg.item(), lwlrap


if __name__ == '__main__':
    main()