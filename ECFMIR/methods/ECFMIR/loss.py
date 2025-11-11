import torch
import torch.nn.functional as F

def topK_distance(feats, labels, k_positive=1, k_negative=1, margin=1.0):
    """
    feats: [B, D] - 某模态或模态对的特征（如 pooled_feats["text"]）
    labels: [B] - 类别标签
    k_positive: 选择的最远正样本数量
    k_negative: 选择的最近负样本数量
    margin: margin 用于损失计算
    """
    # 计算 pairwise 欧几里得距离 [B, B]
    dist = torch.cdist(feats, feats, p=2)

    # 标签相等的为正样本，否则为负样本（注意排除自身）
    labels = labels.unsqueeze(1)  # [B, 1]
    mask_pos = (labels == labels.T) & (~torch.eye(len(labels), device=labels.device, dtype=torch.bool))
    mask_neg = (labels != labels.T)

    # 选择 top-k 最远的正样本
    pos_dist = torch.where(mask_pos, dist, torch.tensor(float('-inf'), device=feats.device))
    top_k_pos_dist, _ = torch.topk(pos_dist, k=k_positive, dim=1)

    # 选择 top-k 最近的负样本
    neg_dist = torch.where(mask_neg, dist, torch.tensor(float('inf'), device=feats.device))
    top_k_neg_dist, _ = torch.topk(neg_dist, k=k_negative, dim=1, largest=False)

    # 计算 top-k distance loss
    loss = F.relu(margin + top_k_pos_dist.mean(dim=1) - top_k_neg_dist.mean(dim=1))
    return loss.mean()