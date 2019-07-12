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
NUM_EPOCH = 64*8
NUM_CYCLE = 64
BATCH_SIZE = 64
LR = [1e-3, 1e-6]
FOLD_LIST = [1, 2, 3, 4, 5]
SLICE_LENGTH = 512
LOAD_DIR = "../models/resnet_model1"
OUTPUT_DIR = "../input/pseudo_label"

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

    # fold splitting
    folds = list(KFold(n_splits=NUM_FOLD, shuffle=True, random_state=SEED).split(np.arange(len(df_train))))

    # build model
    model = ResNet(NUM_CLASS).cuda()

    # set generator
    dataset_noisy = MelDataset(df_noisy['path'], df_noisy[labels].values)
    noisy_loader = DataLoader(dataset_noisy, batch_size=1,
                              shuffle=False, num_workers=1, pin_memory=True,
                              )

    # predict
    preds_noisy = np.zeros([NUM_FOLD, NUM_EPOCH//NUM_CYCLE, len(df_noisy), NUM_CLASS], np.float32)
    for fold, (ids_train_split, ids_valid_split) in enumerate(folds):
        for cycle in range(NUM_EPOCH//NUM_CYCLE):
            print("fold: {} cycle: {}, sec: {:.1f}".format(fold+1, cycle+1, time.time()-starttime))
            model.load_state_dict(torch.load("{}/weight_fold_{}_epoch_{}.pth".format(
                LOAD_DIR, fold+1, NUM_CYCLE*(cycle+1))))
            preds_noisy[fold, cycle] = predict(noisy_loader, model)

        np.save("{}/preds_noisy.npy".format(OUTPUT_DIR), preds_noisy)


def predict(test_loader, model):
    sigmoid = nn.Sigmoid().cuda()

    # switch to eval mode
    model.eval()

    preds = np.zeros([0, NUM_CLASS], np.float32)
    for i, (input, target) in enumerate(test_loader):
        input = torch.autograd.Variable(input.cuda())

        # compute output
        with torch.no_grad():
            output = model(input)
            pred = sigmoid(output)
            pred = pred.data.cpu().numpy()

        # measure accuracy and record loss
        preds = np.concatenate([preds, pred])
    return preds


if __name__ == '__main__':
    main()