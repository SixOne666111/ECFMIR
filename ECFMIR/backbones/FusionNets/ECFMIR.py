import torch
import torch.nn.functional as F
from ..SubNets.FeatureNets import BERTEncoder
from ..SubNets.transformers_encoder.transformer import TransformerEncoder
from torch import nn


__all__ = ['ECFMIR']


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            # nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.net(x)


class GeometricCalibration(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.update_proj = nn.Linear(dim, dim)
        self.gate = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, latent_variance):
        latent_variance_update_proj = latent_variance * torch.sigmoid(self.update_proj(latent_variance))
        return latent_variance + latent_variance_update_proj


class ConfidenceLens(nn.Module):
    def __init__(self, input_dim, latent_dim=32, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim

        blocks = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        ]
        for _ in range(2):
            blocks += [
                nn.Linear(hidden_dim, hidden_dim),
                ResidualBlock(hidden_dim),
                nn.GELU()
            ]
        self.feature_encoder = nn.Sequential(*blocks)

        self.variance_layer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim),
            nn.Softplus()
        )

        self.geometric_calibration = GeometricCalibration(latent_dim)

        self.focal_modulation_net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, feature_representation):
        latent_representation = self.feature_encoder(feature_representation)

        variance = self.variance_layer(latent_representation)

        variance = self.geometric_calibration(variance)
        variance = torch.clamp(variance, min=1e-6)

        focal_coefficient = 0.6 + 2 * self.focal_modulation_net(latent_representation)

        unreliability_index = torch.clamp(torch.log(1 + variance.mean(1, keepdim=True)), max=5.0)
        confidence_score = torch.exp(-focal_coefficient * unreliability_index)
        confidence_score = torch.clamp(confidence_score, min=1e-6, max=1.0)

        return confidence_score


class ECFMIR(nn.Module):

    def __init__(self, args):
        super(ECFMIR, self).__init__()

        # 文本子网络
        self.text_subnet = BERTEncoder.from_pretrained(args.text_backbone, cache_dir=args.cache_path)

        self.text_input_dim = args.text_feat_dim
        self.audio_input_dim = args.audio_feat_dim
        self.video_input_dim = args.video_feat_dim
        self.projected_dim = args.dst_feature_dims
        self.combined_dim = 9 * self.projected_dim

        # 投影层
        self.text_proj = nn.Sequential(nn.Linear(self.text_input_dim, self.projected_dim, bias=True), nn.ReLU())
        self.audio_proj = nn.Sequential(nn.Linear(self.audio_input_dim, self.projected_dim, bias=True), nn.ReLU())
        self.video_proj = nn.Sequential(nn.Linear(self.video_input_dim, self.projected_dim, bias=True), nn.ReLU())

        # 单模态置信度估计器
        self.text_CLens = ConfidenceLens(self.projected_dim)
        self.audio_CLens = ConfidenceLens(self.projected_dim)
        self.video_CLens = ConfidenceLens(self.projected_dim)

        # 两两组合置信度估计器
        self.ta_CLens = ConfidenceLens(2 * self.projected_dim)
        self.tv_CLens = ConfidenceLens(2 * self.projected_dim)
        self.av_CLens = ConfidenceLens(2 * self.projected_dim)

        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.combined_dim, 3*self.projected_dim, bias=True),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(3*self.projected_dim, args.num_labels, bias=True)
        )

    def forward(self, text_input, video_input, audio_input):
        # 编码各模态
        text_feat = self.text_subnet(text_input)                  # (B, L, D)

        # 投影
        projected_text = self.text_proj(text_feat)                # (B, L, d)
        projected_audio = self.audio_proj(audio_input)            # (B, L, d)
        projected_video = self.video_proj(video_input)            # (B, L, d)
        # 平均池化
        pooled_text = projected_text.mean(dim=1)                  # (B, d)
        pooled_audio = projected_audio.mean(dim=1)                # (B, d)
        pooled_video = projected_video.mean(dim=1)                # (B, d)
        # 两两组合特征
        ta_feat = torch.cat([pooled_text, pooled_audio], dim=-1)  # (B, 2d)
        tv_feat = torch.cat([pooled_text, pooled_video], dim=-1)  # (B, 2d)
        av_feat = torch.cat([pooled_audio, pooled_video], dim=-1) # (B, 2d)

        # 单模态置信度
        text_score = self.text_CLens(pooled_text, sample=True)
        audio_score = self.audio_CLens(pooled_audio, sample=True)
        video_score = self.video_CLens(pooled_video, sample=True)
        weighted_text = pooled_text * text_score
        weighted_audio = pooled_audio * audio_score
        weighted_video = pooled_video * video_score

        # 两两置信度
        ta_score = self.ta_CLens(ta_feat, sample=True)
        tv_score = self.tv_CLens(tv_feat, sample=True)
        av_score = self.av_CLens(av_feat, sample=True)
        weighted_ta = ta_feat * ta_score
        weighted_tv = tv_feat * tv_score
        weighted_av = av_feat * av_score

        # 融合所有信息
        fused = torch.cat([
            weighted_text, weighted_audio, weighted_video,
            weighted_ta, weighted_tv, weighted_av,
        ], dim=-1)


        logits = self.classifier(fused)

        return {
            'logits': logits,
            'fused': fused,

        }


