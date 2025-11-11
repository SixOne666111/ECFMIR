from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import logging

class AudioDataset(Dataset):
    def __init__(self, args, base_attrs, mode='train'):
        self.args = args
        self.mode = mode  # 'train', 'dev', 'test'
        self.logger = logging.getLogger(args.logger_name)

        audio_feats_path = os.path.join(base_attrs['data_path'], args.audio_data_path, args.audio_feats_path)
        if not os.path.exists(audio_feats_path):
            raise Exception('Error: The directory of audio features is empty.')

        # 只加载特征字典
        with open(audio_feats_path, 'rb') as f:
            self.audio_feats = pickle.load(f)

        self.data_index = base_attrs[f'{mode}_data_index']
        self.audio_max_length = base_attrs['benchmarks']['max_seq_lengths']['audio']

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        key = self.data_index[idx]
        feat = np.array(self.audio_feats[key], dtype=np.float32)  # 用 float32 减少内存
        feat = self.__padding(feat)
        return feat

    def __padding(self, feat):
        audio_length = feat.shape[0]
        audio_max_length = self.audio_max_length
        if audio_length >= audio_max_length:
            return feat[:audio_max_length, :]

        pad_len = audio_max_length - audio_length
        if self.args.padding_mode == 'zero':
            pad = np.zeros([pad_len, feat.shape[-1]], dtype=np.float32)
        else:
            mean, std = feat.mean(), feat.std()
            pad = np.random.normal(mean, std, (pad_len, feat.shape[-1])).astype(np.float32)

        if self.args.padding_loc == 'start':
            feat = np.concatenate((pad, feat), axis=0)
        else:
            feat = np.concatenate((feat, pad), axis=0)

        return feat
