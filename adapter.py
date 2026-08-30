import torch
import torch.nn as nn
import torch.nn.functional as F

class DistributionFeatureAdapter(nn.Module):
    """
    将小模型的特征映射到大模型特征空间：
    student_feature -> MLP升维 -> scale缩放 + shift位移
    """
    def __init__(self, original_dim, aim_dim, hidden_dim=None):
        super(DistributionFeatureAdapter, self).__init__()

        self.expand_mlp = nn.Sequential(
            nn.Linear(original_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, aim_dim),
            nn.LeakyReLU()
        )

        # 缩放参数 gamma
        self.scale = nn.Parameter(torch.ones(aim_dim))

        # 位移参数 beta
        self.shift = nn.Parameter(torch.zeros(aim_dim))

    def forward(self, student_feature):
        """
        student_feature: [batch_size, student_dim]
        return: [batch_size, teacher_dim]
        """
        feature = self.expand_mlp(student_feature)
        feature = self.scale * feature + self.shift
        return feature