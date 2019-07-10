import numpy as np
import pandas as pd
import os
import time
import random
from multiprocessing import Pool
import cv2
import librosa
import gc
import shutil
from scipy.io import wavfile
import concurrent.futures
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
from math import cos, pi

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

sys.path.append('.')
from tools.utils import AverageMeter, cycle, CosineLR

# set parameters
IMG_SIZE = 256
NUM_FOLD = 5
NUM_CLASS = 5
SEED = 42
NUM_EPOCH = 64 * 2
NUM_CYCLE = 64
BATCH_SIZE = 32
BATCH_SIZE_VALID = 16
FOLD_LIST = [1,2,3,4,5]
LR_RANGE = [1e-4, 1e-6]
cudnn.benchmark = True
starttime0 = time.time()


def main():
    # table data load
    starttime = time.time()
    df_train = pd.read_csv("../input/train_curated.csv")
    df_test = pd.read_csv("../input/sample_submission.csv")
    df_noise = pd.read_csv("../input/train_noisy.csv")
    labels = df_test.columns[1:].tolist()

    df_train['path'] = "../input/mel128/train/" + df_train['fname']
    df_test['path'] = "../input/mel128/test/" + df_train['fname']
    df_noise['path'] = "../input/mel128/noise/" + df_noise['fname']

    # fold splitting
    folds = list(StratifiedKFold(n_splits=NUM_FOLD, shuffle=True, random_state=SEED).split(
        np.arange(len(df_train)), df_train['diagnosis'].values))


    # prepare transforms
    transformations = Compose([  # 224, 224, 3
        #         Resize(IMG_SIZE, IMG_SIZE, cv2.INTER_LANCZOS4),
        ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, interpolation=1, border_mode=0,
                         p=0.5),
        RandomCrop(height=IMG_SIZE * 7 // 8, width=IMG_SIZE * 7 // 8, p=1),
        Cutout(p=0.5, num_holes=4, max_h_size=IMG_SIZE // 8, max_w_size=IMG_SIZE // 8),
        HorizontalFlip(p=0.5),
        #     Normalize(),
    ])
    transformations_valid = Compose([  # 1024, 1024, 3
        #         Resize(IMG_SIZE, IMG_SIZE, cv2.INTER_LANCZOS4),
        #     Normalize(),
    ])

    # Training
    log_columns = ['epoch', 'bce', 'acc', 'qwk', 'val_bce', 'val_acc', 'val_qwk', 'time']

    for fold in range(NUM_FOLD):
        if fold + 1 not in FOLD_LIST: continue
        starttime = time.time()
        print("fold: {}".format(fold + 1))

        # build model
        model = ResNet(num_classes=NUM_CLASS).cuda()

        # train dataset
        dataset_train = XDataset(df_train.iloc[folds[fold][0]].reset_index(drop=True),
                                 img_dir='../input/aptos256/train/',
                                 transform=transformations,
                                 crop='random',
                                 mixup=True,
                                 )
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            df_train['weight'][folds[fold][0]].values, len(df_train))
        # dataloader
        train_loader = DataLoader(dataset_train,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False,
                                  num_workers=1,
                                  pin_memory=True,
                                  sampler=train_sampler,
                                  )

        # valid dataset
        dataset_valid = XDataset(df_train.iloc[folds[fold][1]].reset_index(drop=True),
                                 img_dir='../input/aptos256/train/',
                                 transform=transformations_valid,
                                 )
        valid_loader = DataLoader(dataset_valid,
                                  batch_size=BATCH_SIZE_VALID,
                                  shuffle=False,
                                  num_workers=1,
                                  pin_memory=True
                                  )

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_RANGE[0])  # Adam
        scheduler = CosineLR(optimizer, step_size_min=LR_RANGE[1], t0=len(train_loader) * NUM_CYCLE,
                             tmult=1)  # Cyclic lr

        train_log = pd.DataFrame(columns=log_columns)
        for epoch in range(NUM_EPOCH):  # epoch数 = NUM_EPOCH0
            # train 1 epoch
            bce, acc, qwk = train(train_loader, model, optimizer, scheduler, epoch)

            #  validation
            val_bce, val_acc, val_qwk = validate(valid_loader, model)

            # save log
            endtime = time.time() - starttime
            train_log_epoch = pd.DataFrame([[epoch + 1, bce, acc, qwk, val_bce, val_acc, val_qwk, endtime]],
                                           columns=log_columns)
            train_log = pd.concat([train_log, train_log_epoch])
            train_log.to_csv("train_log_fold{}.csv".format(fold + 1), index=False)

            # display log
            print("Epoch: {}/{} ".format(epoch + 1, NUM_EPOCH)
                  + "BCE: {:.3f} ".format(bce)
                  + "Acc: {:.3f} ".format(acc)
                  + "QWK: {:.3f} ".format(qwk)
                  + "Valid BCE: {:.3f} ".format(val_bce)
                  + "Valid Acc: {:.3f} ".format(val_acc)
                  + "Valid QWK: {:.3f} ".format(val_qwk)
                  + "Sec: {:.1f} ".format(time.time() - starttime)
                  )
            if (epoch + 1) % NUM_CYCLE == 0:
                torch.save(model.state_dict(), "weight_fold_{}_epoch_{}.pth".format(fold + 1, epoch + 1))
            if val_qwk == train_log['val_qwk'].max():
                torch.save(model.state_dict(), "weight_fold_{}_bestqwk.pth".format(fold + 1))

        torch.save(optimizer.state_dict(), 'optimizer_fold_{}_epoch_{}.pth'.format(fold + 1, epoch + 1))
        print("fold, min_valid_bce, min_valid_bce_epoch, max_valid_qwk, max_valid_qwk_epoch")
        print("{}, {:.6f}, {}, {:.6f}, {}".format(
            fold + 1,
            train_log['val_bce'].values.min(),
            train_log['val_bce'].values.argmin() + 1,
            train_log['val_qwk'].values.max(),
            train_log['val_qwk'].values.argmax() + 1,
        ))


def train(train_loader, model, optimizer, scheduler, epoch):
    ce_avr = AverageMeter()

    criterion_bce = nn.BCELoss().cuda()
    softmax = torch.nn.Softmax(dim=1).cuda()

    # switch to train mode
    model.train()
    starttime = time.time()
    preds = np.zeros([0, NUM_CLASS], np.float32)
    y_true = np.zeros([0, NUM_CLASS], np.float32)
    for i, (input, target) in enumerate(train_loader):
        # prepare batches
        input = torch.autograd.Variable(input.cuda(async = True))
        target = torch.autograd.Variable(target.cuda(async = True))

        # get model outputs
        output = model(input)
        ce = criterion_bce(softmax(output), target)

        loss = ce
        # compute gradient and do SGD step
        optimizer.zero_grad()  # # 勾配の初期化
        loss.backward()
        optimizer.step()
        scheduler.step()

        pred = softmax(output)
        pred = pred.data.cpu().numpy()
        ce_avr.update(ce.data, input.size(0))
        preds = np.concatenate([preds, pred])
        #         print(target.data.cpu().numpy().shape)
        y_true = np.concatenate([y_true, target.data.cpu().numpy()])
    #         if (i+1)%1==0:
    #             print("Step: {}/{} ".format(i + 1, len(train_loader))
    #               + "BCE: {:.3f} ".format(ce_avr.avg.item())
    #               + "Sec: {:.1f} ".format(time.time()-starttime)
    #               )
    acc = (y_true.argmax(axis=1) == preds.argmax(axis=1)).mean()
    qwk = metrics.cohen_kappa_score(y_true.argmax(axis=1), preds.argmax(axis=1),
                                    labels=np.arange(NUM_CLASS), weights='quadratic')
    return ce_avr.avg.item(), acc, qwk


def validate(val_loader, model):
    ce_avr = AverageMeter()

    criterion_bce = nn.BCELoss().cuda()
    softmax = torch.nn.Softmax(dim=1).cuda()

    # switch to train mode
    model.eval()

    starttime = time.time()
    preds = np.zeros([0, NUM_CLASS], np.float32)
    y_true = np.zeros([0, NUM_CLASS], np.float32)
    for i, (input, target) in enumerate(val_loader):
        # prepare batches
        input = torch.autograd.Variable(input.cuda(async = True))
        target = torch.autograd.Variable(target.cuda(async = True))

        # get model outputs
        output = model(input)
        ce = criterion_bce(softmax(output), target)

        loss = ce

        pred = softmax(output)
        pred = pred.data.cpu().numpy()
        ce_avr.update(ce.data, input.size(0))
        preds = np.concatenate([preds, pred])
        y_true = np.concatenate([y_true, target.data.cpu().numpy()])
    acc = (y_true.argmax(axis=1) == preds.argmax(axis=1)).mean()
    qwk = metrics.cohen_kappa_score(y_true.argmax(axis=1), preds.argmax(axis=1),
                                    labels=np.arange(NUM_CLASS), weights='quadratic')
    return ce_avr.avg.item(), acc, qwk


class ResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet, self).__init__()

        self.num_classes = num_classes
        self.mode = 'train'

        self.base_model = pretrainedmodels.__dict__['resnet152'](num_classes=1000, pretrained='imagenet')

        self.conv1 = self.base_model.conv1
        self.bn1 = self.base_model.bn1
        self.relu = self.base_model.relu
        self.maxpool = self.base_model.maxpool
        self.layer1 = self.base_model.layer1
        self.layer2 = self.base_model.layer2
        self.layer3 = self.base_model.layer3
        self.layer4 = self.base_model.layer4
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        #         self.last_linear = nn.Linear(self.base_model.layer4[1].conv1.in_channels, num_classes)
        self.last_linear = nn.Sequential(
            nn.Linear(self.base_model.layer4[1].conv1.in_channels, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, NUM_CLASS),
        )

    def forward(self, input):
        bs, ch, h, w = input.size()
        x0 = self.conv1(input)  # ; print('layer conv1 ',x.size()) # [8, 64, 112, 112]
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)  # ; print('layer 1 ',x.size()) # [8, 1024, 28, 28])
        x2 = self.layer2(x1)  # ; print('layer 2 ',x.size()) # [8, 1024, 28, 28])
        x3 = self.layer3(x2)  # ; print('layer 3 ',x.size()) # [8, 1024, 28, 28])
        x4 = self.layer4(x3)  # ; print('layer 4 ',x.size()) # [8, 2048, 14, 14])
        x = self.gap(x4).view(bs, -1)
        x = self.last_linear(x)

        return x


class XDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, crop='center',
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 mixup=False,
                 ):
        self.X_train = df['id_code']
        self.y_train = df['diagnosis']
        self.img_dir = img_dir
        self.transform = transform
        self.crop = crop
        self.mean = np.array(mean, np.float32)
        self.std = np.array(std, np.float32)
        self.mixup = mixup

    def do_mixup(self, img, target, alpha=0.4):
        idx2 = np.random.randint(0, len(self.X_train))
        img_path2 = "{}/{}.png".format(self.img_dir, self.X_train[idx2])
        img2 = cv2.imread(img_path2)[:, :, ::-1]
        if self.transform is not None:
            img2 = self.transform(image=img2)['image']
        target2 = np.eye(NUM_CLASS)[self.y_train[idx2]].astype(np.float32)

        rate = np.random.beta(alpha, alpha)
        img = img * rate + img2 * (1 - rate)
        target = target * rate + target2 * (1 - rate)
        target *= [1, 1, 2, 1, 4]
        target /= target.sum()
        return img, target

    def do_crop(self, img):
        h, w, _ = img.shape
        if self.crop == 'center':
            if h >= w:
                img_new = img[(h - w) // 2:(h - w) // 2 + w]
            else:
                img_new = img[:, (w - h) // 2:(w - h) // 2 + h]
        elif self.crop == 'random':
            if h > w:
                shift = np.random.randint(0, h - w)
                img_new = img[shift:shift + w]
            elif h == w:
                img_new = img
            else:
                shift = np.random.randint(0, w - h)
                img_new = img[:, shift:shift + h]
        return img_new

    def __getitem__(self, index):
        img_path = "{}/{}.png".format(self.img_dir, self.X_train[index])
        #         print(img_path)
        img = cv2.imread(img_path)[:, :, ::-1]
        #         print(img.shape)
        img = self.do_crop(img)
        #         print(img.shape)
        if self.transform is not None:
            img = self.transform(image=img)['image']
        target = np.eye(NUM_CLASS)[self.y_train[index]].astype(np.float32)
        if self.mixup and np.random.random() < 0.5:
            img, label = self.do_mixup(img, target)
        img = img.transpose([2, 0, 1]).astype(np.float32) / 255
        img = (img - self.mean[:, np.newaxis, np.newaxis]) / self.std[:, np.newaxis, np.newaxis]
        return img, target

    def __len__(self):
        return len(self.X_train)


if __name__ == '__main__':
    main()