import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.dataset import Dataset
from math import cos, pi
import librosa
from scipy.io import wavfile
import random

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cycle(iterable):
    """
    convert dataloader to iterator
    :param iterable:
    :return:
    """
    while True:
        for x in iterable:
            yield x


class CosineLR(_LRScheduler):
    """cosine annealing.
    """
    def __init__(self, optimizer, step_size_min=1e-5, t0=100, tmult=2, curr_epoch=-1, last_epoch=-1):
        self.step_size_min = step_size_min
        self.t0 = t0
        self.tmult = tmult
        self.epochs_since_restart = curr_epoch
        super(CosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        self.epochs_since_restart += 1

        if self.epochs_since_restart > self.t0:
            self.t0 *= self.tmult
            self.epochs_since_restart = 0

        lrs = [self.step_size_min + (
                0.5 * (base_lr - self.step_size_min) * (1 + cos(self.epochs_since_restart * pi / self.t0)))
               for base_lr in self.base_lrs]

        return lrs


class MelDataset(Dataset):
    def __init__(self, X, y, crop=-1,
                 mixup=False, freqmask=False, gain=False,
                 crop_mode='original',crop_rate=0.25
                 ):
        self.X= X
        self.y= y
        self.crop = crop
        self.mixup = mixup
        self.freqmask = freqmask
        self.gain = gain
        self.crop_mode = crop_mode
        self.crop_rate = crop_rate

    def do_additional_crop(self, img):
        len_img = img.shape[1]
        img_new = np.zeros([img.shape[0], self.crop], np.float32)
        rate = np.random.random() * (1 - self.crop_rate) + self.crop_rate
        if np.random.random() < 0.5: rate = 1

        if img.shape[1] <= self.crop:
            len_crop = int(img.shape[1] * rate)
            if img.shape[1] - len_crop == 0:
                shift_crop = 0
            else:
                shift_crop = np.random.randint(0, img.shape[1] - len_crop)
            img = img[:, shift_crop:shift_crop + len_crop]
            if self.crop - len_crop == 0:
                shift = 0
            else:
                shift = np.random.randint(0, self.crop - len_crop)
            img_new[:, shift:shift + len_crop] = img
        else:
            shift = np.random.randint(0, img.shape[1] - self.crop)
            img_new = img[:, shift:shift + self.crop]
            len_crop = int(self.crop * rate)
            if self.crop - len_crop == 0:
                shift_crop = 0
            else:
                shift_crop = np.random.randint(0, self.crop - len_crop)
            img_new[:shift_crop] = 0
            img_new[shift_crop + len_crop:] = 0
        return img_new

    def do_random_crop(self, img):
        img_new = np.zeros([img.shape[0], self.crop], np.float32)
        if img.shape[1] < self.crop:
            shift = np.random.randint(0, self.crop - img.shape[1])
            img_new[:, shift:shift + img.shape[1]] = img
        elif img.shape[1] == self.crop:
            img_new = img
        else:
            shift = np.random.randint(0, img.shape[1] - self.crop)
            img_new = img[:, shift:shift + self.crop]
        return img_new

    def do_crop(self, img):
        if self.crop_mode == 'random':
            return self.do_random_crop(img)
        elif self.crop_mode == 'additional':
            return self.do_additional_crop(img)
        elif self.crop_mode == 'original':
            return img

    def do_mixup(self, img, label, alpha=1.):
        idx = np.random.randint(0, len(self.X))
        img2 = np.load("{}.npy".format(self.X[idx][:-4]))
        img2 = self.do_crop(img2)

        label2 = self.y[idx].astype(np.float32)

        rate = np.random.beta(alpha, alpha)
        img = img * rate + img2 * (1 - rate)
        label = label * rate + label2 * (1 - rate)
        return img, label


    def do_freqmask(self, img, max=32):
        coord = np.random.randint(0, img.shape[0])
        width = np.random.randint(8, max)
        cut = np.array([coord - width, coord + width])
        cut = np.clip(cut, 0, img.shape[0])
        img[cut[0]:cut[1]] = 0
        return img

    def do_gain(self, img, max=0.1):
        rate = 1 - max + np.random.random() * max * 2
        return img * rate

    def __getitem__(self, index):
        img = np.load("{}.npy".format(self.X[index][:-4]))
        img = self.do_crop(img)
        label = self.y[index].astype(np.float32)

        if self.mixup and np.random.random() < 0.5:
            img, label = self.do_mixup(img, label)
        if self.gain and np.random.random() < 0.5:
            img = self.do_gain(img)
        if self.freqmask and np.random.random() < 0.5:
            img = self.do_freqmask(img)

        img = librosa.power_to_db(img)
        img = (img - img.mean()) / (img.std() + 1e-7)
        img = img.reshape([1, img.shape[0], img.shape[1]])

        return img, label

    def __len__(self):
        return len(self.X)


def compute_gain(sound, fs, min_db=-80.0, mode='RMSE'):
    if fs == 16000:
        n_fft = 2048
    elif fs == 44100:
        n_fft = 4096
    else:
        raise Exception('Invalid fs {}'.format(fs))
    stride = n_fft // 2

    gain = []
    for i in range(0, len(sound) - n_fft + 1, stride):
        if mode == 'RMSE':
            g = np.mean(sound[i: i + n_fft] ** 2)
        elif mode == 'A_weighting':
            spec = np.fft.rfft(np.hanning(n_fft + 1)[:-1] * sound[i: i + n_fft])
            power_spec = np.abs(spec) ** 2
            a_weighted_spec = power_spec * np.power(10, a_weight(fs, n_fft) / 10)
            g = np.sum(a_weighted_spec)
        else:
            raise Exception('Invalid mode {}'.format(mode))
        gain.append(g)

    gain = np.array(gain)
    gain = np.maximum(gain, np.power(10, min_db / 10))
    gain_db = 10 * np.log10(gain)

    return gain_db


def mix(sound1, sound2, r, fs):
    gain1 = np.max(compute_gain(sound1, fs))  # Decibel
    gain2 = np.max(compute_gain(sound2, fs))
    t = 1.0 / (1 + np.power(10, (gain1 - gain2) / 20.) * (1 - r) / r)
    sound = ((sound1 * t + sound2 * (1 - t)) / np.sqrt(t ** 2 + (1 - t) ** 2))
    sound = sound.astype(np.float32)

    return sound


class WaveDataset(Dataset):
    def __init__(self, X, y,
                 crop=-1, crop_mode='original', padding=0,
                 mixup=False, scaling=-1, gain=-1,
                 fs=44100,
                 ):
        self.X = X
        self.y = y
        self.crop = crop
        self.crop_mode = crop_mode
        self.padding = padding
        self.mixup = mixup
        self.scaling = scaling
        self.gain = gain
        self.fs = fs

    def preprocess(self, sound):
        for f in self.preprocess_funcs:
            sound = f(sound)

        return sound

    def do_padding(self, snd):
        snd_new = np.pad(snd, self.padding, 'constant')
        return snd_new

    def do_crop(self, snd):
        if self.crop_mode=='random':
            shift = np.random.randint(0, snd.shape[0] - self.crop)
            snd_new = snd[shift:shift + self.crop]
        else:
            snd_new = snd
        return snd_new

    def do_gain(self, snd):
        snd_new = snd * np.power(10, random.uniform(-self.gain, self.gain) / 20.0)
        return snd_new

    def do_scaling(self, snd, interpolate='Nearest'):
        scale = np.power(self.scaling, random.uniform(-1, 1))
        output_size = int(len(snd) * scale)
        ref = np.arange(output_size) / scale
        if interpolate == 'Linear':
            ref1 = ref.astype(np.int32)
            ref2 = np.minimum(ref1+1, len(snd)-1)
            r = ref - ref1
            snd_new = snd[ref1] * (1-r) + snd[ref2] * r
        elif interpolate == 'Nearest':
            snd_new = snd[ref.astype(np.int32)]
        else:
            raise Exception('Invalid interpolation mode {}'.format(interpolate))

        return snd_new

    def do_mixup(self, snd, label, alpha=1):
        idx2 = np.random.randint(0, len(self.X))
        _, snd2 = wavfile.read("{}".format(self.X[idx2]))
        label2 = self.y[idx2].astype(np.float32)
        if self.scaling!=-1:
            snd2 = self.do_scaling(snd2)
        snd2 = self.do_padding(snd2)
        snd2 = self.do_crop(snd2)

        rate = np.random.beta(alpha, alpha)
        snd_new = mix(snd, snd, rate, self.fs)
        label_new = label * rate + label2 * (1 - rate)
        return snd_new, label_new

    def __getitem__(self, index):
        _, snd = wavfile.read("{}".format(self.X[index]))
        label = self.y[index].astype(np.float32)
        if self.scaling!=-1:
            snd = self.do_scaling(snd)
        snd = self.do_padding(snd)
        snd = self.do_crop(snd)
        if self.mixup:
            snd, label = self.do_mixup(snd, label)
        if self.gain!=-1:
            snd = self.do_gain(snd)
        snd = snd.reshape([1, 1, -1]).astype(np.float32) / 32768.0
        return snd, label

    def __len__(self):
        return len(self.X)


def _one_sample_positive_class_precisions(scores, truth):
    """Calculate precisions for each true class for a single sample.

    Args:
      scores: np.array of (num_classes,) giving the individual classifier scores.
      truth: np.array of (num_classes,) bools indicating which classes are true.

    Returns:
      pos_class_indices: np.array of indices of the true classes for this sample.
      pos_class_precisions: np.array of precisions corresponding to each of those
        classes.
    """
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    # Only calculate precisions if there are some true classes.
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    # Retrieval list of classes for this sample.
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)
    # Which of these is a true label?
    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    # Precision of retrieval list truncated at each hit, in order of pos_labels.
    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits


# All-in-one calculation of per-class lwlrap.

def calculate_per_class_lwlrap(truth, scores):
    """Calculate label-weighted label-ranking average precision.

    Arguments:
      truth: np.array of (num_samples, num_classes) giving boolean ground-truth
        of presence of that class in that sample.
      scores: np.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.

    Returns:
      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
        class.
      weight_per_class: np.array of (num_classes,) giving the prior of each
        class within the truth labels.  Then the overall unbalanced lwlrap is
        simply np.sum(per_class_lwlrap * weight_per_class)
    """
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    # Space to store a distinct precision value for each class on each sample.
    # Only the classes that are true for each sample will be filled in.
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = (
            _one_sample_positive_class_precisions(scores[sample_num, :],
                                                  truth[sample_num, :]))
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (
            precision_at_hits)
    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    # Form average of each column, i.e. all the precisions assigned to labels in
    # a particular class.
    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    # overall_lwlrap = simple average of all the actual per-class, per-sample precisions
    #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)
    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
    #                = np.sum(per_class_lwlrap * weight_per_class)
    return per_class_lwlrap, weight_per_class