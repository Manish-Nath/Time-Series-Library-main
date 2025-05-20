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

#         # Embedding Layer
#         self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

#         # Decomposition MLPs
#         self.trend_layer = nn.Linear(self.embed_size, self.embed_size)
#         self.season_layer = nn.Linear(self.embed_size, self.embed_size)

#         # Channel Mixing
#         self.channel_mixer = nn.Linear(self.feature_size, self.feature_size)

#         # Mixture of Experts (MoE)
#         self.num_experts = 4
#         self.experts = nn.ModuleList([
#             nn.Linear(self.embed_size, self.embed_size) for _ in range(self.num_experts)
#         ])
#         self.gate = nn.Linear(self.embed_size, self.num_experts)

#         # Forecast Head
#         self.fc = nn.Linear(self.seq_length * self.embed_size, self.pre_length)

#     def tokenEmb(self, x):
#         # x: [B, T, N] → [B, N, T, D]
#         x = x.permute(0, 2, 1).unsqueeze(3)
#         return x * self.embeddings

#     def moe_layer(self, x):
#         # x: [B, N, T, D]
#         B, N, T, D = x.shape
#         x_flat = x.reshape(B * N * T, D)  # [B*N*T, D]
#         gate_logits = self.gate(x_flat)  # [B*N*T, num_experts]
#         gate_probs = F.softmax(gate_logits, dim=-1)  # [B*N*T, num_experts]

#         expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)  # [B*N*T, num_experts, D]
#         gate_probs = gate_probs.unsqueeze(-1)  # [B*N*T, num_experts, 1]

#         moe_output = (expert_outputs * gate_probs).sum(dim=1)  # [B*N*T, D]
#         return moe_output.reshape(B, N, T, D)

#     def time_process(self, x, B, N, T):
#         trend = x.mean(dim=2, keepdim=True).expand_as(x)
#         season = x - trend

#         trend = self.trend_layer(trend)
#         season = self.season_layer(season)
#         x = trend + season

#         if self.channel_independence != '1':
#             x_mixed = self.channel_mixer(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
#             x = x + x_mixed
#         return x

#     def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
#         B, T, N = x_enc.shape
#         x = self.tokenEmb(x_enc)               # [B, N, T, D]
#         bias = x
#         x = self.time_process(x, B, N, T)      # decomposition + mixing
#         x = self.moe_layer(x)                  # MoE enhancement
#         x = x + bias                           # Residual
#         x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)  # [B, pre_len, N]
#         return x[:, -self.pre_length:, :]      # Only last pre_length predictions
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchdiffeq import odeint

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.embed_size = 128
        self.hidden_size = 256
        self.pre_length = configs.pred_len
        self.feature_size = configs.enc_in
        self.seq_length = configs.seq_len
        self.channel_independence = configs.channel_independence
        
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.trend_layer = nn.Linear(self.embed_size, self.embed_size)
        self.season_layer = nn.Linear(self.embed_size, self.embed_size)
        self.channel_mixer = nn.Linear(self.feature_size, self.feature_size)
        
        # Neural ODE Head
        self.ode_input_proj = nn.Linear(self.seq_length * self.embed_size, self.hidden_size)
        self.ode_func = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.ode_output_proj = nn.Linear(self.hidden_size, self.pre_length)

    def tokenEmb(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        y = self.embeddings
        return x * y

    # Custom function for ODE
    def ode_func_forward(self, t, x):
        return self.ode_func(x)

    def ode_head(self, x):
        B, N, _ = x.shape
        x = self.ode_input_proj(x)  # [B, N, hidden_size]
        t = torch.tensor([0, 1], dtype=torch.float32, device=x.device)
        # Using the custom function to match the expected signature
        x = odeint(self.ode_func_forward, x, t, method='rk4')[-1]  # [B, N, hidden_size]
        return self.ode_output_proj(x).permute(0, 2, 1)  # [B, pre_length, N]

    def time_process(self, x, B, N, T):
        trend = x.mean(dim=2, keepdim=True).expand_as(x)
        season = x - trend
        
        trend = self.trend_layer(trend)
        season = self.season_layer(season)
        x = trend + season
        
        if self.channel_independence != '1':
            x_mixed = self.channel_mixer(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x = x + x_mixed
        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, T, N = x_enc.shape
        x = self.tokenEmb(x_enc)
        bias = x
        x = self.time_process(x, B, N, T)
        x = x + bias
        x = self.ode_head(x.reshape(B, N, -1))  # Apply Neural ODE head
        return x[:, -self.pre_length:, :]  # Return last pre_length steps
