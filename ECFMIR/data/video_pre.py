from torch.utils.data import Dataset
import os
import numpy as np
import pickle
import logging

class VideoDataset(Dataset):
    def __init__(self, args, base_attrs, mode='train'):
        self.args = args
        self.mode = mode  # 'train', 'dev', 'test'
        self.logger = logging.getLogger(args.logger_name)

        video_feats_path = os.path.join(base_attrs['data_path'], args.video_data_path, args.video_feats_path)
        if not os.path.exists(video_feats_path):
            raise Exception('Error: The directory of video features is empty.')

        # 仅加载 video feats 字典，不做 padding
        with open(video_feats_path, 'rb') as f:
            self.video_feats = pickle.load(f)

        self.data_index = base_attrs[f'{mode}_data_index']
        self.video_max_length = base_attrs['benchmarks']['max_seq_lengths']['video']

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        key = self.data_index[idx]
        feat = np.array(self.video_feats[key], dtype=np.float32)

        # 特殊情况处理：如 MIntRec 中 squeeze(1)
        if self.args.dataset == 'MIntRec':
            feat = feat.squeeze(1)

        feat = self.__padding(feat)
        return feat

    def __padding(self, feat):
        video_length = feat.shape[0]
        video_max_length = self.video_max_length

        if video_length >= video_max_length:
            return feat[:video_max_length, :]

        pad_len = video_max_length - video_length
        if self.args.padding_mode == 'zero':
            pad = np.zeros([pad_len, feat.shape[-1]], dtype=np.float32)
        else:  # normal
            mean, std = feat.mean(), feat.std()
            pad = np.random.normal(mean, std, (pad_len, feat.shape[-1])).astype(np.float32)

        if self.args.padding_loc == 'start':
            feat = np.concatenate((pad, feat), axis=0)
        else:
            feat = np.concatenate((feat, pad), axis=0)

        return feat

