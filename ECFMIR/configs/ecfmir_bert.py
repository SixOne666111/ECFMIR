class Param:
    def __init__(self, args):
        self.common_param = self._get_common_parameters(args)
        self.hyper_param = self._get_hyper_parameters(args)

    def _get_common_parameters(self, args):
        common_parameters = {
            'padding_mode': 'zero',         # 序列填充方式：'zero' 或 'normal'
            'padding_loc': 'end',           # 填充位置：'start' 或 'end'
            'need_aligned': False,          # 是否对不同模态进行对齐
            'eval_monitor': 'f1',           # 验证指标：'loss' 或 'f1'、'acc' 等
            'train_batch_size': 16,         # 训练 batch 大小
            'eval_batch_size': 8,           # 验证 batch 大小
            'test_batch_size': 8,           # 测试 batch 大小
            'wait_patience': 8             # Early Stopping 的耐心步数
        }
        return common_parameters

    def _get_hyper_parameters(self, args):
        hyper_parameters = {
            # 优化器 & 学习率调度
            'num_train_epochs': 100,        # 训练轮数
            'lr': 2.5e-5,                   # 初始学习率
            'weight_decay': 0.1,           # 权重衰减 (L2 正则)
            'grad_clip': -1.0,               # 梯度裁剪阈值

            # 特征处理
            'dst_feature_dims': 1024,        # 投影后的特征维度（统一模态）

            # 损失函数权重（多任务）
            # 'lambda_trust': 1,            # 置信度对齐损失项权重
            # 'temperature': 8,  # 单模态对齐使用的温度系数，通常设为 4~8
            # 'lambda_trust_pair': 0.9,  # 两两模态 logits 与主预测之间的 KL
            # 'triplet_trust': 1,  # triplet_trust的权重



        }
        return hyper_parameters
