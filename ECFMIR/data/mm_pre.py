from torch.utils.data import Dataset
import torch

__all__ = ['MMDataset']

class MMDataset(Dataset):
    def __init__(self, label_ids, text_feats, video_feats, audio_feats):
        self.label_ids = label_ids  # 原始 list
        self.text_feats = text_feats  # 应为 Dataset 或 list
        self.video_feats = video_feats  # 应为 Dataset 或 list
        self.audio_feats = audio_feats  # 应为 Dataset 或 list
        self.size = len(label_ids)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        sample = {
            'label_ids': torch.tensor(self.label_ids[index]),
            'text_feats': torch.tensor(self.text_feats[index]),
            'video_feats': torch.tensor(self.video_feats[index]),
            'audio_feats': torch.tensor(self.audio_feats[index])
        }
        return sample
