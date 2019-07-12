import numpy as np
import pandas as pd
import time
import librosa

# parameters
SAMPLE_RATE = 44100
N_MELS = 128
HOP_LENGTH = 347
N_FFT = 128*20
FMIN = 20
FMAX = SAMPLE_RATE//2

starttime = time.time()


def convert(df, input_dir, output_dir):
    for i in range(len(df)):
        if (i+1)%100==0: print("{}/{}, sec: {:.1f}".format(i+1, len(df), time.time()-starttime))
        file_path = "{}/{}".format(input_dir, df['fname'][i])
        data, _ = librosa.core.load(file_path, sr=SAMPLE_RATE, res_type="kaiser_fast")
        data = librosa.feature.melspectrogram(
            data,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            hop_length=HOP_LENGTH, # 1sec -> 128
            n_fft=N_FFT,
            fmin=FMIN,
            fmax=FMAX,
        ).astype(np.float32)
        np.save("{}/{}.npy".format(output_dir, df['fname'][i][:-4]), data)


def main():
    # load table data
    df_train = pd.read_csv("../input/train_curated.csv")
    df_noisy = pd.read_csv("../input/train_noisy.csv")
    df_test = pd.read_csv("../input/sample_submission.csv")

    # convert to logmel
    print("converting train data...")
    convert(df_train, "../input/train_curated/", "../input/mel128/train")
    print("converting noisy data...")
    convert(df_noisy, "../input/train_noisy/", "../input/mel128/noisy")
    print("converting test data...")
    convert(df_test, "../input/test/", "../input/mel128/test")


if __name__ == '__main__':
    main()