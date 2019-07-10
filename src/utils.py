from torch.optim.lr_scheduler import _LRScheduler


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
    def __init__(self, df, load_dir, slice=-1, mixup=False,
                 cutout=False, freqmask=False, gain=False,
                 slice_mode='original',
                 ):
        self.X_train = df['path']
        self.y_train = df[labels].values
        self.slice = slice
        self.mixup = mixup
        self.freqmask = freqmask
        self.gain = gain
        self.load_dir = load_dir
        self.slice_mode = slice_mode

    def do_additional_slice(self, img, min_rate=0.25):
        len_img = img.shape[1]
        img_new = np.zeros([img.shape[0], self.slice], np.float32)
        rate = np.random.random() * (1 - min_rate) + min_rate
        if np.random.random() < 0.5: rate = 1

        if img.shape[1] <= self.slice:
            len_slice = int(img.shape[1] * rate)
            if img.shape[1] - len_slice == 0:
                shift_slice = 0
            else:
                shift_slice = np.random.randint(0, img.shape[1] - len_slice)
            img = img[:, shift_slice:shift_slice + len_slice]
            if self.slice - len_slice == 0:
                shift = 0
            else:
                shift = np.random.randint(0, self.slice - len_slice)
            img_new[:, shift:shift + len_slice] = img
        else:
            shift = np.random.randint(0, img.shape[1] - self.slice)
            img_new = img[:, shift:shift + self.slice]
            len_slice = int(self.slice * rate)
            if self.slice - len_slice == 0:
                shift_slice = 0
            else:
                shift_slice = np.random.randint(0, self.slice - len_slice)
            img_new[:shift_slice] = 0
            img_new[shift_slice + len_slice:] = 0
        return img_new

    def do_random_slice(self, img):
        img_new = np.zeros([img.shape[0], self.slice], np.float32)
        if img.shape[1] < self.slice:
            shift = np.random.randint(0, self.slice - img.shape[1])
            img_new[:, shift:shift + img.shape[1]] = img
        elif img.shape[1] == self.slice:
            img_new = img
        else:
            shift = np.random.randint(0, img.shape[1] - self.slice)
            img_new = img[:, shift:shift + self.slice]
        return img_new

    def do_slice(self, img):
        if self.slice_mode == 'random':
            return self.do_random_slice(img)
        elif self.slice_mode == 'additional':
            return self.do_additional_slice(img)
        elif self.slice_mode == 'original':
            return img

    def do_mixup(self, img, label, alpha=1.):
        idx = np.random.randint(0, len(self.X_train))
        img2 = np.load("{}.npy".format(self.X_train[idx][:-4]))
        img2 = self.do_slice(img2)

        label2 = self.y_train[idx].astype(np.float32)

        rate = np.random.beta(alpha, alpha)
        img = img * rate + img2 * (1 - rate)
        label = label * rate + label2 * (1 - rate)
        return img, label

    def do_white(self, img):
        shift = np.random.randint(0, self.white_noise.shape[1] - self.slice)
        white_noise_slice = self.white_noise[:, shift:shift + self.slice] * np.random.rand()
        img += white_noise_slice
        return img

    def do_flip(self, img):
        return img[:, ::-1]

    def do_cutout_h(self, img, max=32):
        coord = np.random.randint(0, img.shape[0])
        width = np.random.randint(8, max)
        cut = np.array([coord - width, coord + width])
        cut = np.clip(cut, 0, img.shape[0])
        img[cut[0]:cut[1]] = 0
        return img

    def do_cutout_w(self, img, max=32):
        coord = np.random.randint(0, img.shape[1])
        width = np.random.randint(8, max)
        cut = np.array([coord - width, coord + width])
        cut = np.clip(cut, 0, img.shape[1])
        img[:, cut[0]:cut[1]] = 0
        return img

    def do_highpass(self, img):
        th = np.random.randint(0, img.shape[0])
        img[th:] = 0
        return img

    def cutout_bug(self, img):
        coordx = np.sort(np.random.randint(0, self.slice, 2))
        coordy = np.sort(np.random.randint(0, 128, 2))
        img[coordx[0]:coordx[1]] = 0
        return img

    def do_resize(self, img, max=0.1):
        rate = 1 - max + np.random.random() * max * 2
        img_tmp = cv2.resize(img, (int(self.slice * rate), img.shape[0],))
        if rate > 1:
            img_new = img_tmp[:, :img.shape[1]]
        else:
            img_new = np.zeros_like(img)
            img_new[:, :img_tmp.shape[1]] = img_tmp
        return img_new

    def do_gain(self, img, max=0.1):
        rate = 1 - max + np.random.random() * max * 2
        return img * rate

    def do_warp(self, img, max=64):
        left = np.random.randint(0, img.shape[1])
        right = np.min([img.shape[1], left + np.random.randint(8, max)])
        tmp = img[:, left:img.shape[1] - (right - left)]
        img[:, left:right] = 0
        img[:, right:] = tmp
        #         print(left, right, tmp.shape)
        return img

    def __getitem__(self, index):
        img = np.load("{}.npy".format(self.X_train[index][:-4]))
        img = self.do_slice(img)
        label = self.y_train[index].astype(np.float32)

        for i in range(NUM_MIX):
            if self.mixup and np.random.random() < 0.5:
                img, label = self.do_mixup(img, label)
        if self.gain and np.random.random() < 0.5:
            img = self.do_gain(img)
        if self.resize and np.random.random() < 0.5:
            img = self.do_resize(img)
        if self.white and np.random.random() < 0.5:
            img = self.do_white(img)

        if self.cutout_h and np.random.random() < 0.5:
            img = self.do_cutout_h(img)

        img = librosa.power_to_db(img)
        img = (img - img.mean()) / (img.std() + 1e-7)
        img = img.reshape([1, img.shape[0], img.shape[1]])

        return img, label

    def __len__(self):
        return len(self.X_train)