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
from utils import AverageMeter, cycle, CosineLR, MelDataset, calculate_per_class_lwlrap
from models import ResNet

# set parameters
NUM_FOLD = 5
NUM_CLASS = 80
SEED = 42
NUM_EPOCH = 64*7
NUM_CYCLE = 64
BATCH_SIZE = 64
LR = [1e-3, 1e-6]
FOLD_LIST = [1, 2, 3, 4, 5]
CROP_LENGTH = 512
C_SEMI = 20
TEMPERATURE = 2
CROP_RATE = 0.25
LOAD_DIR = "../models/resnet_model1"
OUTPUT_DIR = "../models/resnet_model2"

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

    df_train['path'] = "../input/mel128/train/" + df_train['fname']
    df_test['path'] = "../input/mel128/test/" + df_train['fname']
    df_noisy['path'] = "../input/mel128/noisy/" + df_noisy['fname']

    # calc sampling weight
    df_train['weight'] = 1
    df_noisy['weight'] = len(df_train) / len(df_noisy)

    # generate pseudo label with sharpening
    tmp = np.load("../input/pseudo_label/preds_noisy.npy").mean(axis=(0,1))
    tmp = tmp ** TEMPERATURE
    tmp = tmp / tmp.sum(axis=1)[:, np.newaxis]
    df_noisy_pseudo = df_noisy.copy()
    df_noisy_pseudo[labels] = tmp

    # fold splitting
    folds = list(KFold(n_splits=NUM_FOLD, shuffle=True, random_state=SEED).split(np.arange(len(df_train))))
    folds_noisy = list(KFold(n_splits=NUM_FOLD, shuffle=True, random_state=SEED).split(np.arange(len(df_noisy))))

    # Training
    log_columns = ['epoch', 'bce', 'lwlrap', 'bce_noisy', 'lwlrap_noisy', 'semi_mse', 'val_bce', 'val_lwlrap', 'time']
    for fold, (ids_train_split, ids_valid_split) in enumerate(folds):
        if fold+1 not in FOLD_LIST: continue
        print("fold: {}".format(fold + 1))
        train_log = pd.DataFrame(columns=log_columns)

        # build model
        model = ResNet(NUM_CLASS).cuda()
        model.load_state_dict(torch.load("{}/weight_fold_{}_epoch_512.pth".format(LOAD_DIR, fold+1)))

        # prepare data loaders
        df_train_fold = df_train.iloc[ids_train_split].reset_index(drop=True)
        dataset_train = MelDataset(df_train_fold['path'], df_train_fold[labels].values,
                                    crop=CROP_LENGTH, crop_mode='additional', crop_rate=CROP_RATE,
                                    mixup=True, freqmask=True, gain=True,
                                    )
        train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=1, pin_memory=True,
                                  )

        df_valid = df_train.iloc[ids_valid_split].reset_index(drop=True)
        dataset_valid = MelDataset(df_valid['path'], df_valid[labels].values,)
        valid_loader = DataLoader(dataset_valid, batch_size=1,
                                  shuffle=False, num_workers=1, pin_memory=True,
                                  )

        dataset_noisy = MelDataset(df_noisy['path'], df_noisy[labels].values,
                                    crop=CROP_LENGTH, crop_mode='additional', crop_rate=CROP_RATE,
                                    mixup=True, freqmask=True, gain=True,
                                   )
        noisy_loader = DataLoader(dataset_noisy, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=1, pin_memory=True,
                                  )
        noisy_itr = cycle(noisy_loader)

        df_semi = pd.concat([df_train.iloc[ids_train_split], df_noisy_pseudo.iloc[folds_noisy[fold][0]]]).reset_index(drop=True)
        semi_sampler = torch.utils.data.sampler.WeightedRandomSampler(df_semi['weight'].values, len(df_semi))
        dataset_semi = MelDataset(df_semi['path'], df_semi[labels].values,
                                   crop=CROP_LENGTH, crop_mode='additional', crop_rate=CROP_RATE,
                                   mixup=True, freqmask=True, gain=True,
                                   )
        semi_loader = DataLoader(dataset_semi,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=1, pin_memory=True,
                                 sampler=semi_sampler,
                                 )
        semi_itr = cycle(semi_loader)

        # set optimizer and loss
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR[0])
        scheduler = CosineLR(optimizer, step_size_min=LR[1], t0=len(train_loader) * NUM_CYCLE, tmult=1)

        # training
        for epoch in range(NUM_EPOCH):
            # train for one epoch
            bce, lwlrap, bce_noisy, lwlrap_noisy, mse_semi = train((train_loader, noisy_itr, semi_itr), model, optimizer, scheduler, epoch)

            # evaluate on validation set
            val_bce, val_lwlrap = validate(valid_loader, model)

            # print log
            endtime = time.time() - starttime
            print("Epoch: {}/{} ".format(epoch + 1, NUM_EPOCH)
                  + "CE: {:.4f} ".format(bce)
                  + "LwLRAP: {:.4f} ".format(lwlrap)
                  + "Noisy CE: {:.4f} ".format(bce_noisy)
                  + "Noisy LWLRAP: {:.4f} ".format(lwlrap_noisy)
                  + "Semi MSE: {:.4f} ".format(mse_semi)
                  + "Valid CE: {:.4f} ".format(val_bce)
                  + "Valid LWLRAP: {:.4f} ".format(val_lwlrap)
                  + "sec: {:.1f}".format(endtime)
                  )

            # save log and weights
            train_log_epoch = pd.DataFrame(
                [[epoch+1, bce, lwlrap, bce_noisy, lwlrap_noisy, mse_semi, val_bce, val_lwlrap, endtime]],
                columns=log_columns)
            train_log = pd.concat([train_log, train_log_epoch])
            train_log.to_csv("{}/train_log_fold{}.csv".format(OUTPUT_DIR, fold+1), index=False)
            if (epoch+1)%NUM_CYCLE==0:
                torch.save(model.state_dict(), "{}/weight_fold_{}_epoch_{}.pth".format(OUTPUT_DIR, fold+1, epoch+1))


def train(train_loaders, model, optimizer, scheduler, epoch):
    train_loader, noisy_itr, semi_itr = train_loaders
    bce_avr = AverageMeter()
    bce_noisy_avr = AverageMeter()
    mse_semi_avr = AverageMeter()
    criterion_bce = nn.BCEWithLogitsLoss().cuda()
    criterion_mse = nn.MSELoss().cuda()
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

        input_semi, target_semi = next(semi_itr)
        input_semi = torch.autograd.Variable(input_semi.cuda())
        target_semi = torch.autograd.Variable(target_semi.cuda())

        # compute output
        output = model(input)
        bce = criterion_bce(output, target)
        output_noisy = model.noisy(input_noisy)
        bce_noisy = criterion_bce(sigmoid(output_noisy), target_noisy)
        output_semi = model(input_semi)
        mse_semi = criterion_mse(sigmoid(output_semi), target_semi)

        loss = bce + bce_noisy + C_SEMI * mse_semi
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
        mse_semi_avr.update(mse_semi.data, input.size(0))
        preds = np.concatenate([preds, pred])
        y_true = np.concatenate([y_true, target.data.cpu().numpy()])
        preds_noisy = np.concatenate([preds_noisy, pred_noisy])
        y_true_noisy = np.concatenate([y_true_noisy, target_noisy.data.cpu().numpy()])

    # calc metric
    per_class_lwlrap, weight_per_class = calculate_per_class_lwlrap(y_true, preds)
    lwlrap = np.sum(per_class_lwlrap * weight_per_class)
    per_class_lwlrap, weight_per_class = calculate_per_class_lwlrap(y_true_noisy, preds_noisy)
    lwlrap_noisy = np.sum(per_class_lwlrap * weight_per_class)

    return bce_avr.avg.item(), lwlrap, bce_noisy_avr.avg.item(), lwlrap_noisy, mse_semi_avr.avg.item()


def validate(val_loader, model):
    bce_avr = AverageMeter()
    sigmoid = torch.nn.Sigmoid().cuda()
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
            bce = criterion_bce(output, target)
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