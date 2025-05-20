
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.embed_size = 128
#         self.hidden_size = 256
#         self.pre_length = configs.pred_len
#         self.feature_size = configs.enc_in
#         self.seq_length = configs.seq_len
#         self.channel_independence = configs.channel_independence
        
#         self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
#         self.trend_layer = nn.Linear(self.embed_size, self.embed_size)
#         self.season_layer = nn.Linear(self.embed_size, self.embed_size)
#         self.channel_mixer = nn.Linear(self.feature_size, self.feature_size)  # Mix channels
#         self.fc = nn.Linear(self.seq_length * self.embed_size, self.pre_length)

#     def tokenEmb(self, x):
#         x = x.permute(0, 2, 1)
#         x = x.unsqueeze(3)
#         y = self.embeddings
#         return x * y

#     def time_process(self, x, B, N, T):
#         trend = x.mean(dim=2, keepdim=True).expand_as(x)
#         season = x - trend
        
#         trend = self.trend_layer(trend)
#         season = self.season_layer(season)
#         x = trend + season
        
#         if self.channel_independence != '1':  # Mix channels when not independent
#             x_mixed = self.channel_mixer(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#             x = x + x_mixed
#         return x

#     def forward(self, x):
#         B, T, N = x.shape
#         x = self.tokenEmb(x)
#         bias = x
#         x = self.time_process(x, B, N, T)
#         x = x + bias
#         x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
#         return x
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2311.06184.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        
        self.embed_size = 128  # embed_size
        self.hidden_size = 256  # hidden_size
        self.feature_size = configs.enc_in  # channels
        self.seq_len = configs.seq_len
        self.channel_independence = configs.channel_independence
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.trend_layer = nn.Linear(self.embed_size, self.embed_size)
        self.season_layer = nn.Linear(self.embed_size, self.embed_size)
        self.channel_mixer = nn.Linear(self.feature_size, self.feature_size)  # Mix channels
        self.fc = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len)
        )

    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        y = self.embeddings
        return x * y

    def time_process(self, x, B, N, T):
        # Channel mixing logic (similar to the simplified version)
        trend = x.mean(dim=2, keepdim=True).expand_as(x)
        season = x - trend
        
        trend = self.trend_layer(trend)
        season = self.season_layer(season)
        x = trend + season
        
        # If channel independence is set to '0', mix channels
        if self.channel_independence != '1':  # Mix channels when not independent
            x_mixed = self.channel_mixer(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x = x + x_mixed
        return x

    def forecast(self, x_enc):
        # x: [Batch, Input length, Channel]
        B, T, N = x_enc.shape
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x_enc)
        bias = x
        # [B, N, T, D]
        x = self.time_process(x, B, N, T)
        x = x + bias
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            raise ValueError('Only forecast tasks implemented yet')
