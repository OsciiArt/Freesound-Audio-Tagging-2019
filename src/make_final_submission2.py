import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time, random, sys, os
import sklearn.metrics
from sklearn.model_selection import KFold
from multiprocessing import Pool
import gc
import shutil
from scipy.io import wavfile
import librosa
import concurrent.futures

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

sys.path.append('.')
from utils import AverageMeter, cycle, CosineLR, MelDataset, calculate_per_class_lwlrap
from models import ResNet, EnvNetv2


# set parameters
NUM_FOLD = 5
NUM_CLASS = 80
SEED = 42
BATCH_DIR = "batch"
SIZE_LIMIT = 20000000
WIDTH_LIMIT = 80000
MAX_LEN = 1400000
MAX_PAD = 32000
MAX_BATCHSIZE = 512
LEN_DF_WAV_LIMIT = 500000000
LEN_DF_MEL_LIMIT = 2000000000
NUMBATCH_PER_NUMDATA = 1 / 30
MAX_PATIENCE = 0.2
wav_dir = "../input/test/"

RES_LIST = [
    {'dir': '../models/resnet_model1',
     'epoch': [2*64,4*64,7*64,8*64],
     'pad': [8],
     },
    {'dir': '../models/resnet_model2',
     'epoch': [1*64,2*64,4*64,6*64,7*64],
     'pad': [8],
     },
    {'dir': '../models/resnet_model3',
     'epoch': [2*64,4*64,6*64],
     'pad': [8],
     },
]
ENV_LIST = [
    {'dir': '../models/envnet_model4',
     'epoch': [2*80,3*80],
     'pad': [8000],
     'acitivation': 'sigmoid',
     },
    {'dir': '../models/envnet_model5',
     'epoch': [3 * 80, 5 * 80],
     'pad': [8000],
     'acitivation': 'softmax',
     },
    {'dir': '../models/envnet_model6',
     'epoch': [1*80,2*80,4*80],
     'pad': [8000],
     'acitivation': 'softmax',
     },
]
LEN_RES_EPOCH = 0
for i in range(len(RES_LIST)):
    LEN_RES_EPOCH = max(LEN_RES_EPOCH, len(RES_LIST[i]['epoch']))
LEN_RES_PAD = 0
for i in range(len(RES_LIST)):
    LEN_RES_PAD = max(LEN_RES_PAD, len(RES_LIST[i]['pad']))
LEN_ENV_EPOCH = 0
for i in range(len(ENV_LIST)):
    LEN_ENV_EPOCH = max(LEN_ENV_EPOCH, len(ENV_LIST[i]['epoch']))
LEN_ENV_PAD = 0
for i in range(len(ENV_LIST)):
    LEN_ENV_PAD = max(LEN_ENV_EPOCH, len(ENV_LIST[i]['pad']))
starttime0 = time.time()

# cudnn speed up
cudnn.benchmark = True


def main():
    ### fix seed
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # table data load
    starttime = time.time()
    df_test = pd.read_csv("../input/sample_submission.csv")
    labels = df_test.columns[1:].tolist()
    df_test['path'] = "{}/".format(wav_dir) + df_test['fname']
    print("table data loading done. {:.1f}/{:.1f}".format(time.time() - starttime, time.time() - starttime0))

    # get data length
    starttime = time.time()
    p = Pool(2)
    len_list = p.map(get_len, df_test['path'].values)
    df_test['length'] = len_list
    print("getting data length done. {:.1f}/{:.1f}".format(time.time() - starttime, time.time() - starttime0))

    # data sort
    starttime = time.time()
    df_test_sort = df_test.copy()
    df_test_sort['index'] = np.arange(len(df_test_sort))
    df_test_sort = df_test_sort.sort_values(['length', 'index']).reset_index(drop=True)
    print("data sort done. {:.1f}/{:.1f}".format(time.time() - starttime, time.time() - starttime0))

    # batch splitting
    starttime = time.time()
    NUM_BATCH_LIMIT = 50 + int(len(df_test_sort)*NUMBATCH_PER_NUMDATA)
    print("num batch limit: {}".format(NUM_BATCH_LIMIT))
    patience_rate = 0
    patience_rate_tmp = 0
    num_batch, count = get_num_batch(df_test_sort, patience_rate)
    print("patience_rate_tmp: {:.2f}, patience_rate_tmp: {:.2f}, num_batch: {:3d}".format(
        patience_rate, patience_rate_tmp, num_batch))
    while num_batch > NUM_BATCH_LIMIT and patience_rate_tmp < MAX_PATIENCE:
        patience_rate_tmp += 0.01
        num_batch_tmp, count_tmp = get_num_batch(df_test_sort, patience_rate_tmp)
        if num_batch_tmp < num_batch:
            num_batch = num_batch_tmp
            count = count_tmp
            patience_rate = patience_rate_tmp
        print("patience_rate_tmp: {:.2f}, patience_rate_tmp: {:.2f}, num_batch_tmp: {:3d}".format(
            patience_rate, patience_rate_tmp, num_batch_tmp))
    num_batch, count = get_num_batch(df_test_sort, patience_rate)
    print("num batch: {}, rate of padding patience: {:.2f}".format(num_batch, patience_rate))
    print("batch splitting done. {:.1f}/{:.1f}".format(time.time() - starttime, time.time() - starttime0))

    # store batch id
    starttime = time.time()
    batch_list = []
    for i in range(num_batch):
        batch_list += [i] * count[i][1]
    df_test_sort['batch'] = batch_list
    print(df_test_sort[['path', 'length', 'batch']].head())
    print("save batch id done. {:.1f}/{:.1f}".format(time.time() - starttime, time.time() - starttime0))

    # split dataframe if too big
    starttime = time.time()
    df_mel_split = get_df_split(df_test_sort, LEN_DF_MEL_LIMIT)
    print("df_mel_split")
    for i in range(len(df_mel_split)):
        print("{}: num data: {}, total length: {}".format(i + 1, len(df_mel_split[i]), df_mel_split[i]['length'].sum()))
    print("dataframe splitting done. {:.1f}/{:.1f}".format(time.time() - starttime, time.time() - starttime0))

    # ### EnvNet part
    # build model
    model = EnvNetv2(NUM_CLASS).cuda()
    model.eval()

    # split df for EnvNet
    df_wav_split = get_df_split(df_test_sort, LEN_DF_WAV_LIMIT)
    print("df_wav_split")
    for i in range(len(df_wav_split)):
        print("{}: num data: {}, total length: {}".format(i + 1, len(df_wav_split[i]), df_wav_split[i]['length'].sum()))

    print("predict wav...")

    # parallel threading
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    threadA = executor.submit(get_mel_batch, df_mel_split[0])
    threadB = executor.submit(predict_wav_split, model, df_wav_split[0], ENV_LIST)
    preds_wav_split = []
    preds_wav_split.append(threadB.result())
    executor.shutdown()
    print("parallel threading done.", time.time() - starttime, time.time() - starttime0)

    # do remain EnvNet prediction
    if len(df_wav_split) > 1:
        for split in range(1, len(df_wav_split)):
            preds_wav_split.append(predict_wav_split(model, df_wav_split[split], ENV_LIST))
            print("envnet prediction split {}/{}, done. {:.1f}/{:.1f}".format(
                split + 1, len(df_wav_split), time.time() - starttime, time.time() - starttime0))
    preds_test_wav = np.concatenate(preds_wav_split, axis=4)
    print("all envnet predict done.", time.time() - starttime, time.time() - starttime0)

    # build model
    starttime = time.time()
    model = ResNet(NUM_CLASS).cuda()
    model.eval()
    print("building ResNet model done. {:.1f}/{:.1f}".format(time.time() - starttime, time.time() - starttime0))

    # predict split #1
    preds_test_mel = []
    preds_test_mel.append(predict_mel_split(model, df_mel_split[0], RES_LIST))
    shutil.rmtree(BATCH_DIR)
    print("mel prediction of split {} done. {:.1f}/{:.1f}".format(1, time.time() - starttime, time.time() - starttime0))

    # process remain split
    if len(df_mel_split) > 1:
        for split in range(1, len(df_mel_split)):
            # mel preprocessing
            starttime = time.time()
            df_test_sort_tmp = df_mel_split[split]
            get_mel_batch(df_test_sort_tmp)
            print("mel preprocessing of split {} done. {:.1f}/{:.1f}".format(
                split + 1, time.time() - starttime, time.time() - starttime0))
            preds_test_mel.append(predict_mel_split(model, df_test_sort_tmp, RES_LIST))
            shutil.rmtree(BATCH_DIR)
            print("mel prediction of split {} done. {:.1f}/{:.1f}".format(
                split + 1, time.time() - starttime, time.time() - starttime0))

    print("all prediction done. {:.1f}/{:.1f}".format(time.time() - starttime, time.time() - starttime0))

    # concat
    starttime = time.time()
    preds_test_mel = np.concatenate(preds_test_mel, axis=4)
    print("preds_test_mel.shape", preds_test_mel.shape)
    print("concat done.", time.time() - starttime, time.time() - starttime0)

    # make submission
    preds_test_avr = (
            + preds_test_mel[:, 0, :len(RES_LIST[0]['epoch'])].mean(axis=(0, 1, 2)) * 4 / 13
            + preds_test_mel[:, 1, :len(RES_LIST[1]['epoch'])].mean(axis=(0, 1, 2)) * 3 / 13
            + preds_test_mel[:, 2, :len(RES_LIST[2]['epoch'])].mean(axis=(0, 1, 2)) * 3 / 13
            + preds_test_wav[:, 0, :len(ENV_LIST[0]['epoch'])].mean(axis=(0, 1, 2)) * 1 / 13
            + preds_test_wav[:, 1, :len(ENV_LIST[1]['epoch'])].mean(axis=(0, 1, 2)) * 1 / 13
            + preds_test_wav[:, 2, :len(ENV_LIST[2]['epoch'])].mean(axis=(0, 1, 2)) * 1 / 13)
    print(preds_test_mel.shape, preds_test_wav.shape)
    print(preds_test_avr.shape)
    df_test_sort = df_test_sort.sort_values(['length', 'index']).reset_index(drop=True)
    df_test_sort[labels] = preds_test_avr
    df_test_sort = df_test_sort.sort_values('index').reset_index(drop=True)
    df_test_sort[['fname'] + labels].to_csv("../output/submission1.csv", index=None)
    print("save submission done. {:.1f}/{:.1f}".format(time.time() - starttime, time.time() - starttime0))


def check_size_limit(width, num_batch):
    if (width + MAX_PAD * 2) * num_batch > SIZE_LIMIT:
        return True
    else:
        return False


def get_num_batch(df_test_sort, patience_rate):
    i = 0
    len_now = df_test_sort['length'][i]
    patience = int(len_now * patience_rate)
    count = []
    while (i < len(df_test_sort)):
        len_now = df_test_sort['length'][i]
        patience = int(len_now * patience_rate)
        if len(count) == 0 or count[-1][0] + patience < len_now or count[-1][1] >= MAX_BATCHSIZE:
            count.append([len_now, 1])
        elif check_size_limit(len_now, count[-1][1] + 1):
            count.append([len_now, 1])
        else:
            count[-1][1] += 1
        i += 1
    return len(count), count


def predict_mel_split(model, df_split, RES_LIST):
    starttime = time.time()
    batch_idx = [df_split['batch'].min(), df_split['batch'].max() + 1]
    preds_test_mel_tmp = np.zeros([
        NUM_FOLD,
        len(RES_LIST),
        len(RES_LIST[0]['epoch']),
        len(RES_LIST[0]['pad']),
        len(df_split), NUM_CLASS], np.float32)

    dataset_valid = BatchDataset(df_split, 0)
    valid_loader = DataLoader(dataset_valid,
                              batch_size=1,
                              shuffle=False,
                              num_workers=1,
                              pin_memory=True,
                              collate_fn=my_collate
                              )
    for i in range(len(RES_LIST)):
        model_dir = RES_LIST[i]['dir']
        epoch_list = RES_LIST[i]['epoch']
        pad_list = RES_LIST[i]['pad']
        for fold in range(NUM_FOLD):
            for k, epoch in enumerate(epoch_list):
                model.load_state_dict(
                    torch.load("{}/weight_fold_{}_epoch_{}.pth".format(model_dir, fold + 1, epoch)))
                for j, pad in enumerate(pad_list):
                    print("fold: {}, dir: {}, epoch: {}, pad: {}, sec: {:.1f}".format(
                        fold + 1, model_dir, epoch, pad, time.time() - starttime))
                    dataset_valid.pad = pad
                    preds_test_mel_tmp[fold, i, k, j] = predict_resnet(model, valid_loader)
    return preds_test_mel_tmp


def get_mel_batch(df_split):
    print(1)
    df_split['path'] = "{}/".format(wav_dir) + df_split['fname']
    print(df_split[['path', 'batch']])
    os.makedirs(BATCH_DIR, exist_ok=True)
    print(2)
    p = Pool(2)  # 最大プロセス数=2
    batch_idx = [df_split['batch'].min(), df_split['batch'].max() + 1]
    for i in range(batch_idx[0], batch_idx[1]):
        df_tmp = df_split[df_split['batch'] == i].reset_index(drop=True)
        args = []
        slice = int(np.ceil((df_tmp['length'].values[-1] + 1) / 347))
        slice = np.min([slice, WIDTH_LIMIT])
        for j in range(len(df_tmp)):
            args.append([df_tmp['path'][j], slice])
        batch = p.map(preprocess_mel, args)
        batch = np.array(batch)
        np.save("{}/{}.npy".format(BATCH_DIR, i), batch)


def predict_wav_split(model, df, ENV_LIST):
    starttime = time.time()
    batch_idx = [df['batch'].min(), df['batch'].max() + 1]
    p = Pool(2)  # 最大プロセス数=2
    batch_list = []
    for i in range(batch_idx[0], batch_idx[1]):
        df_tmp = df[df['batch'] == i].reset_index(drop=True)
        args = []
        slice = df_tmp['length'].values[-1]
        for j in range(len(df_tmp)):
            args.append([df_tmp['path'][j], slice])
        batch = p.map(preprocess_wav, args)
        batch = np.array(batch)
        batch_list.append(batch)
    print("batch making done, sec: {:.1f}".format(time.time() - starttime))

    # envnet predict
    starttime = time.time()
    print("predict valid...")
    preds_test_wav = np.zeros([
        NUM_FOLD,
        len(ENV_LIST),
        len(ENV_LIST[0]['epoch']),
        len(ENV_LIST[0]['pad']),
        len(df), NUM_CLASS], np.float32)

    dataset_valid = BatchWavDataset(batch_list, 0)
    valid_loader = DataLoader(dataset_valid,
                              batch_size=1,
                              shuffle=False,
                              num_workers=1,
                              pin_memory=True,
                              collate_fn=my_collate
                              )
    for i in range(len(ENV_LIST)):
        model_dir = ENV_LIST[i]['dir']
        epoch_list = ENV_LIST[i]['epoch']
        pad_list = ENV_LIST[i]['pad']
        activation = ENV_LIST[i]['acitivation']

        for fold in range(NUM_FOLD):
            for k, epoch in enumerate(epoch_list):
                model.load_state_dict(
                    torch.load("{}/weight_fold_{}_epoch_{}.pth".format(model_dir, fold + 1, epoch),
                               map_location='cuda:0'))
                for j, pad in enumerate(pad_list):
                    print("fold: {}, dir: {}, epoch: {}, pad: {}, sec: {:.1f}".format(
                        fold + 1, model_dir, epoch, pad, time.time() - starttime))
                    dataset_valid.pad = pad
                    preds_test_wav[fold, i, k, j] = predict_envnet(model, valid_loader, activation)
    return preds_test_wav


def get_df_split(df, size_limit):
    num_batch = df['batch'].max() + 1
    sum_len = df['length'].sum()
    df_split = []
    begin = 0
    sum_tmp = 0
    print("base df shape", df.shape)
    for i in range(num_batch):
        sum_tmp += df['length'][df['batch'] == i].sum()
        if sum_tmp > size_limit:
            df_split.append(
                df[(begin <= df['batch']) & (df['batch'] < i)].reset_index(drop=True))
            sum_tmp = df['length'][df['batch'] == i].sum()
            begin = i
    df_split.append(df[(begin <= df['batch'])].reset_index(drop=True))
    return df_split


def my_collate(batch):
    return torch.Tensor(batch[0])


def get_len(path):
    _, data = wavfile.read(path)
    len_data = len(data)
    if len(data) > MAX_LEN:
        len_data = MAX_LEN
        print("File length {} is too long! This file is sliced to {}.".format(len(data), MAX_LEN))

    return len_data


def get_wav(path):
    _, snd = wavfile.read(path)
    return snd


def get_mel(wave):
    wave = wave.astype(np.float32) / 32768.0
    data = librosa.feature.melspectrogram(
        wave,
        sr=44100,
        n_mels=128,
        hop_length=347 * 1,
        n_fft=128 * 20,
        fmin=20,
        fmax=44100 // 2,
    ).astype(np.float32)
    return data


def preprocess_mel(args):
    path, slice = args
    wav = get_wav(path)
    mel = get_mel(wav)
    mel_new = np.zeros([mel.shape[0], slice], np.float32)
    if mel.shape[1] > slice:
        print("wav length: {}, mel length: {}".format(wav.shape[0], mel.shape[1]))
        print("Mel file is sliced")
        mel_new[:] = mel[:, :slice]
    else:
        mel_new[:, :mel.shape[1]] = mel
    mel_new = librosa.power_to_db(mel_new)
    mel_new = mel_new.reshape([1, mel_new.shape[0], mel_new.shape[1]])
    return mel_new


def preprocess_wav(args):
    path, slice = args
    wav = get_wav(path)
    pad = (slice - len(wav)) // 2
    wav_new = np.zeros([1, 1, slice], np.int16)

    if wav.shape[0] > slice:
        print("wav length: {}".format(wav.shape[0]))
        print("Wav file is sliced")
        wav_new[0, 0, :] = wav[:slice]
    else:
        wav_new[0, 0, pad:pad + len(wav)] = wav
    return wav_new


class BatchDataset(Dataset):
    def __init__(self, df, pad=0):
        self.len_batch = df['batch'].max() - df['batch'].min() + 1
        self.X = np.arange(self.len_batch) + df['batch'].min()
        self.pad = pad

    def __getitem__(self, index):
        batch_base = np.load("{}/{}.npy".format(BATCH_DIR, self.X[index]))
        batch_pad = np.zeros(batch_base.shape[:-1] + (batch_base.shape[-1] + self.pad * 2,), np.float32)
        batch_max = batch_base.max(axis=(1, 2, 3)) - 80
        batch_max = np.maximum(batch_max, -100)
        batch_pad[:] = batch_max[:, np.newaxis, np.newaxis, np.newaxis, ]
        if self.pad != 0:
            batch_pad[:, :, :, self.pad:-self.pad] = batch_base
        else:
            batch_pad[:] = batch_base
        batch_pad = (batch_pad - batch_pad.mean(axis=(1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis, ]) / (
                batch_pad.std(axis=(1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis, ] + 1e-7)
        return batch_pad

    def __len__(self):
        return self.len_batch


class BatchWavDataset(Dataset):
    def __init__(self, batch_list, pad=0):
        self.X = batch_list
        self.pad = pad

    def __getitem__(self, index):
        batch_base = self.X[index]
        if batch_base.shape[-1] + self.pad * 2 < 20580:
            pad = int(np.ceil((20580 - batch_base.shape[-1]) / 2))
        else:
            pad = self.pad
        if pad != 0:
            batch = np.zeros([batch_base.shape[0], 1, 1, batch_base.shape[-1] + pad * 2], np.float32)
            batch[:, :, :, pad:-pad] = batch_base.astype(np.float32) / 32768.0
        else:
            batch = batch_base.astype(np.float32) / 32768.0
        return batch

    def __len__(self):
        return len(self.X)


def predict_resnet(model, dataloader):
    sigmoid = torch.nn.Sigmoid().cuda()
    preds = []
    model.eval()
    for i, input in enumerate(dataloader):
        input = input.cuda(async=True)
        with torch.no_grad():
            pred = sigmoid(model(input)).data.cpu().numpy()
        preds.append(pred)
    preds = np.concatenate(preds)
    return preds


def predict_envnet(model, dataloader, activation='sigmoid'):
    sigmoid = torch.nn.Sigmoid().cuda()
    softmax = torch.nn.Softmax(dim=1).cuda()
    if activation == 'sigmoid':
        f_act = sigmoid
    elif activation == 'softmax':
        f_act = softmax
    preds = []
    model.eval()
    for i, input in enumerate(dataloader):
        input = input.cuda(async=True)
        with torch.no_grad():
            pred = f_act(model(input)).data.cpu().numpy()
        preds.append(pred)
    preds = np.concatenate(preds)
    return preds


if __name__ == '__main__':
    main()