from torch.utils.data import Dataset
import os
import random
import time
import torch
import copy
import glob
import librosa
import numpy as np

'''
    [load_dataset.py]
    data load and preprocessing

    class load_datasets() {
        files_name
        datasets
    }
    
    preprocessing {
        data crop and padding
        down sampling
        mfcc
        argumentation
    }
'''

class load_datasets(Dataset):
    def __init__(self, args):
        start = time.time()
        self.files_name = []
        self.datasets = self.data_preprocessing(args)
        print('\t({:.3f} sec)'.format(time.time() - start))

    def __len__(self):
        return len(self.datasets[0])
    
    def __getitem__(self, idx):
        return self.datasets[0][idx], self.datasets[1][idx]

    def size(self, idx):
        return self.datasets[0].shape[idx]

    def data_preprocessing(self, args):
        if args.mode == 'train' or args.mode == 'key_train':
            self.sample_path = os.path.join(args.path, args.train_path)
        elif args.mode == 'test' or args.mode == 'key_test':
            self.sample_path = os.path.join(args.path, args.test_path)
        elif args.mode == 'dev':
            self.sample_path = os.path.join(args.path,'dev/')
        elif args.mode == 'evaluate':
            self.sample_path = os.path.join(args.path,'evaluate/')

        # load files and labels
        tf_folder = os.listdir(self.sample_path)
        files, labels = [], []
        for tf in tf_folder:
            if not 'true' in tf and not 'false' in tf:
                print('[Error] load_dataset.py (1)')
                exit(1)

            tmp_file = glob.glob(self.sample_path + tf + '/*')
            tmp_label = [1 if tf == 'true' else 0 for _ in range(len(tmp_file))]
            files.extend(tmp_file)
            labels.extend(tmp_label)
        self.files_name = copy.deepcopy(files)

        # load data and preprocessing
        mfcc_data, rolled_data, rolled_label = [], [], []
        for idx, file in enumerate(files):
            # data load and preprocessing(1) data crop and padding
            data, sr = librosa.load(path=file, duration=args.duration)
            if data.shape[0] / sr < args.duration:
                data = np.pad(data, (0, sr * args.duration - data.shape[0]))

            # preprocessing(2) down sampling
            data = librosa.resample(y=data, orig_sr=sr, target_sr=args.sample_rate, res_type='fft')
            
            # preprocessing(3) extract mfcc features
            mfcc = librosa.feature.mfcc(y=data, sr=args.sample_rate)
            mfcc_data.append(mfcc)

            # preprocessing(4) arguemtation
            cnt = random.randint(1, 5)
            direction = -1
            for _ in range(cnt):
                rolled_mfcc = np.roll(a=mfcc, shift=direction * random.randint(1, 8), axis=1)
                rolled_data.append(rolled_mfcc)
                rolled_label.append(labels[idx])
            
            _data = mfcc_data + rolled_data
            _label = labels + rolled_label

            ret_data = np.asarray(_data)
            ret_data = np.transpose(ret_data, (0, 2, 1))

            ret_label = np.asarray(_label)
            
        return torch.FloatTensor(ret_data), torch.FloatTensor(ret_label)