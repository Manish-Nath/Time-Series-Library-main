# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ChannelMixer(nn.Module):
#     def __init__(self, n_vars, d_model, dropout=0.1):
#         super(ChannelMixer, self).__init__()
#         self.n_vars = n_vars
#         self.d_model = d_model
        
#         # MLP for channel mixing
#         self.mixer = nn.Sequential(
#             nn.Linear(n_vars * d_model, n_vars * d_model * 2),  # Expand
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(n_vars * d_model * 2, n_vars * d_model),  # Contract
#             nn.Dropout(dropout)
#         )
#         self.norm = nn.LayerNorm(d_model)

#     def forward(self, x):
#         # x: [bs x n_vars x seq_len x d_model]
#         bs, n_vars, seq_len, d_model = x.shape
        
#         # Flatten across variables and features for mixing
#         x_flat = x.reshape(bs, seq_len, n_vars * d_model)  # [bs x seq_len x (n_vars * d_model)]
        
#         # Apply channel mixing
#         x_mixed = self.mixer(x_flat)  # [bs x seq_len x (n_vars * d_model)]
        
#         # Reshape back to original dimensions
#         x_mixed = x_mixed.reshape(bs, n_vars, seq_len, d_model)  # [bs x n_vars x seq_len x d_model]
        
#         # Residual connection and normalization
#         x = self.norm(x + x_mixed)
#         return x
    
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from layers.SelfAttention_Family import FullAttention, AttentionLayer
# # from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from layers.SelfAttention_Family import ProbSparseAttention, AttentionLayer  # Updated import
# # from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
# # import numpy as np


# # class FlattenHead(nn.Module):
# #     def __init__(self, n_vars, nf, target_window, head_dropout=0):
# #         super().__init__()
# #         self.n_vars = n_vars
# #         self.flatten = nn.Flatten(start_dim=-2)
# #         self.linear = nn.Linear(nf, target_window)
# #         self.dropout = nn.Dropout(head_dropout)

# #     def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
# #         x = self.flatten(x)
# #         x = self.linear(x)
# #         x = self.dropout(x)
# #         return x


# # class EnEmbedding(nn.Module):
# #     def __init__(self, n_vars, d_model, patch_len, dropout):
# #         super(EnEmbedding, self).__init__()
# #         # Patching
# #         self.patch_len = patch_len

# #         self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
# #         self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
# #         self.position_embedding = PositionalEmbedding(d_model)

# #         self.dropout = nn.Dropout(dropout)

# #     def forward(self, x):
# #         # do patching
# #         n_vars = x.shape[1]
# #         glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

# #         x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
# #         x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
# #         # Input encoding
# #         x = self.value_embedding(x) + self.position_embedding(x)
# #         x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
# #         x = torch.cat([x, glb], dim=2)
# #         x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
# #         return self.dropout(x), n_vars


# # class Encoder(nn.Module):
# #     def __init__(self, layers, norm_layer=None, projection=None):
# #         super(Encoder, self).__init__()
# #         self.layers = nn.ModuleList(layers)
# #         self.norm = norm_layer
# #         self.projection = projection

# #     def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
# #         for layer in self.layers:
# #             x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

# #         if self.norm is not None:
# #             x = self.norm(x)

# #         if self.projection is not None:
# #             x = self.projection(x)
# #         return x


# # class EncoderLayer(nn.Module):
# #     def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
# #                  dropout=0.1, activation="relu"):
# #         super(EncoderLayer, self).__init__()
# #         d_ff = d_ff or 4 * d_model
# #         self.self_attention = self_attention
# #         self.cross_attention = cross_attention
# #         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
# #         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
# #         self.norm1 = nn.LayerNorm(d_model)
# #         self.norm2 = nn.LayerNorm(d_model)
# #         self.norm3 = nn.LayerNorm(d_model)
# #         self.dropout = nn.Dropout(dropout)
# #         self.activation = F.relu if activation == "relu" else F.gelu

# #     def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
# #         B, L, D = cross.shape
# #         x = x + self.dropout(self.self_attention(
# #             x, x, x,
# #             attn_mask=x_mask,
# #             tau=tau, delta=None
# #         )[0])
# #         x = self.norm1(x)

# #         x_glb_ori = x[:, -1, :].unsqueeze(1)
# #         x_glb = torch.reshape(x_glb_ori, (B, -1, D))
# #         x_glb_attn = self.dropout(self.cross_attention(
# #             x_glb, cross, cross,
# #             attn_mask=cross_mask,
# #             tau=tau, delta=delta
# #         )[0])
# #         x_glb_attn = torch.reshape(x_glb_attn,
# #                                    (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
# #         x_glb = x_glb_ori + x_glb_attn
# #         x_glb = self.norm2(x_glb)

# #         y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

# #         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
# #         y = self.dropout(self.conv2(y).transpose(-1, 1))

# #         return self.norm3(x + y)


# # class Model(nn.Module):
# #     def __init__(self, configs):
# #         super(Model, self).__init__()
# #         self.task_name = configs.task_name
# #         self.features = configs.features
# #         self.seq_len = configs.seq_len
# #         self.pred_len = configs.pred_len
# #         self.use_norm = configs.use_norm
# #         self.patch_len = configs.patch_len
# #         self.patch_num = int(configs.seq_len // configs.patch_len)
# #         self.n_vars = 1 if configs.features == 'MS' else configs.enc_in
# #         # Embedding
# #         self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)

# #         self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
# #                                                    configs.dropout)

# #         # Encoder-only architecture with ProbSparseAttention
# #         self.encoder = Encoder(
# #             [
# #                 EncoderLayer(
# #                     AttentionLayer(
# #                         ProbSparseAttention(False, configs.factor, attention_dropout=configs.dropout,
# #                                             output_attention=False),  # Replaced FullAttention
# #                         configs.d_model, configs.n_heads),
# #                     AttentionLayer(
# #                         ProbSparseAttention(False, configs.factor, attention_dropout=configs.dropout,
# #                                             output_attention=False),  # Replaced FullAttention
# #                         configs.d_model, configs.n_heads),
# #                     configs.d_model,
# #                     configs.d_ff,
# #                     dropout=configs.dropout,
# #                     activation=configs.activation,
# #                 )
# #                 for l in range(configs.e_layers)
# #             ],
# #             norm_layer=torch.nn.LayerNorm(configs.d_model)
# #         )
# #         self.head_nf = configs.d_model * (self.patch_num + 1)
# #         self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
# #                                 head_dropout=configs.dropout)

# #     def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
# #         if self.use_norm:
# #             # Normalization from Non-stationary Transformer
# #             means = x_enc.mean(1, keepdim=True).detach()
# #             x_enc = x_enc - means
# #             stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
# #             x_enc /= stdev

# #         _, _, N = x_enc.shape

# #         en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))
# #         ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)

# #         enc_out = self.encoder(en_embed, ex_embed)
# #         enc_out = torch.reshape(
# #             enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
# #         # z: [bs x nvars x d_model x patch_num]
# #         enc_out = enc_out.permute(0, 1, 3, 2)

# #         dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
# #         dec_out = dec_out.permute(0, 2, 1)

# #         if self.use_norm:
# #             # De-Normalization from Non-stationary Transformer
# #             dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
# #             dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))

# #         return dec_out

# #     def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
# #         if self.use_norm:
# #             # Normalization from Non-stationary Transformer
# #             means = x_enc.mean(1, keepdim=True).detach()
# #             x_enc = x_enc - means
# #             stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
# #             x_enc /= stdev

# #         _, _, N = x_enc.shape

# #         en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
# #         ex_embed = self.ex_embedding(x_enc, x_mark_enc)

# #         enc_out = self.encoder(en_embed, ex_embed)
# #         enc_out = torch.reshape(
# #             enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
# #         # z: [bs x nvars x d_model x patch_num]
# #         enc_out = enc_out.permute(0, 1, 3, 2)

# #         dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
# #         dec_out = dec_out.permute(0, 2, 1)

# #         if self.use_norm:
# #             # De-Normalization from Non-stationary Transformer
# #             dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
# #             dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

# #         return dec_out

# #     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
# #         if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
# #             if self.features == 'M':
# #                 dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
# #                 return dec_out[:, -self.pred_len:, :]  # [B, L, D]
# #             else:
# #                 dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
# #                 return dec_out[:, -self.pred_len:, :]  # [B, L, D]
# #         else:
# #             return None
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from layers.SelfAttention_Family import FullAttention, AttentionLayer
# from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
# import numpy as np

# class FlattenHead(nn.Module):
#     def __init__(self, n_vars, nf, target_window, head_dropout=0):
#         super().__init__()
#         self.n_vars = n_vars
#         self.flatten = nn.Flatten(start_dim=-2)
#         self.linear = nn.Linear(nf, target_window)
#         self.dropout = nn.Dropout(head_dropout)

#     def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
#         x = self.flatten(x)
#         x = self.linear(x)
#         x = self.dropout(x)
#         return x

# class EnEmbedding(nn.Module):
#     def __init__(self, n_vars, d_model, patch_len, dropout):
#         super(EnEmbedding, self).__init__()
#         self.patch_len = patch_len
#         self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
#         self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
#         self.position_embedding = PositionalEmbedding(d_model)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         n_vars = x.shape[1]
#         glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))
#         x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
#         x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
#         x = self.value_embedding(x) + self.position_embedding(x)
#         x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
#         x = torch.cat([x, glb], dim=2)
#         x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
#         return self.dropout(x), n_vars

# class TemporalConvNetwork(nn.Module):
#     def __init__(self, d_model, kernel_size=3, dilations=[1, 2, 4, 8], dropout=0.1):
#         super(TemporalConvNetwork, self).__init__()
#         self.d_model = d_model
#         self.dilations = dilations
        
#         # Multi-scale TCN with dilated convolutions
#         self.convs = nn.ModuleList([
#             nn.Conv1d(d_model, d_model, kernel_size, 
#                      padding=(kernel_size-1)*dilation//2, 
#                      dilation=dilation)
#             for dilation in dilations
#         ])
#         self.norm = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = nn.ReLU()
        
#         # Projection to combine multi-scale outputs
#         self.out_proj = nn.Linear(d_model * len(dilations), d_model)

#     def forward(self, x):
#         # x: [bs*nvars, seq_len, d_model]
#         x = x.transpose(1, 2)  # [bs*nvars, d_model, seq_len]
#         residual = x
        
#         # Apply dilated convolutions at different scales
#         conv_outputs = []
#         for conv in self.convs:
#             out = conv(x)
#             out = self.activation(out)
#             conv_outputs.append(out)
        
#         # Concatenate multi-scale outputs
#         combined = torch.cat(conv_outputs, dim=1)  # [bs*nvars, d_model * len(dilations), seq_len]
#         combined = combined.transpose(1, 2)  # [bs*nvars, seq_len, d_model * len(dilations)]
        
#         # Project back to original dimension
#         out = self.out_proj(combined)  # [bs*nvars, seq_len, d_model]
#         out = self.dropout(self.norm(out))
        
#         return out + residual.transpose(1, 2)  # Residual connection

# class Encoder(nn.Module):
#     def __init__(self, layers, norm_layer=None, projection=None):
#         super(Encoder, self).__init__()
#         self.layers = nn.ModuleList(layers)
#         self.norm = norm_layer
#         self.projection = projection

#     def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
#         for layer in self.layers:
#             x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
#         if self.norm is not None:
#             x = self.norm(x)
#         if self.projection is not None:
#             x = self.projection(x)
#         return x

# class EncoderLayer(nn.Module):
#     def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
#                  dropout=0.1, activation="relu"):
#         super(EncoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.self_attention = self_attention
#         self.cross_attention = cross_attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu

#     def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
#         B, L, D = cross.shape
#         x = x + self.dropout(self.self_attention(
#             x, x, x,
#             attn_mask=x_mask,
#             tau=tau, delta=None
#         )[0])
#         x = self.norm1(x)

#         x_glb_ori = x[:, -1, :].unsqueeze(1)
#         x_glb = torch.reshape(x_glb_ori, (B, -1, D))
#         x_glb_attn = self.dropout(self.cross_attention(
#             x_glb, cross, cross,
#             attn_mask=cross_mask,
#             tau=tau, delta=delta
#         )[0])
#         x_glb_attn = torch.reshape(x_glb_attn,
#                                    (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
#         x_glb = x_glb_ori + x_glb_attn
#         x_glb = self.norm2(x_glb)

#         y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)
#         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
#         y = self.dropout(self.conv2(y).transpose(-1, 1))
#         return self.norm3(x + y)

# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.task_name = configs.task_name
#         self.features = configs.features
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.use_norm = configs.use_norm
#         self.patch_len = configs.patch_len
#         self.patch_num = int(configs.seq_len // configs.patch_len)
#         self.n_vars = 1 if configs.features == 'MS' else configs.enc_in
        
#         # Embedding
#         self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)
#         self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
#                                                    configs.dropout)
        
#         # Enhanced TCN replacing Fourier Attention
#         self.tcn = TemporalConvNetwork(configs.d_model, kernel_size=3, dilations=[1, 2, 4, 8], 
#                                      dropout=configs.dropout)

#         # Encoder with FullAttention
#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         FullAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                     output_attention=False),
#                         configs.d_model, configs.n_heads),
#                     AttentionLayer(
#                         FullAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                     output_attention=False),
#                         configs.d_model, configs.n_heads),
#                     configs.d_model,
#                     configs.d_ff,
#                     dropout=configs.dropout,
#                     activation=configs.activation,
#                 )
#                 for l in range(configs.e_layers)
#             ],
#             norm_layer=torch.nn.LayerNorm(configs.d_model)
#         )
#         self.head_nf = configs.d_model * (self.patch_num + 1)
#         self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
#                                 head_dropout=configs.dropout)

#     def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
#         if self.use_norm:
#             means = x_enc.mean(1, keepdim=True).detach()
#             x_enc = x_enc - means
#             stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
#             x_enc /= stdev

#         _, _, N = x_enc.shape
#         en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))
#         en_embed = self.tcn(en_embed)  # Apply enhanced TCN
#         ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)

#         enc_out = self.encoder(en_embed, ex_embed)
#         enc_out = torch.reshape(
#             enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
#         enc_out = enc_out.permute(0, 1, 3, 2)
#         dec_out = self.head(enc_out)
#         dec_out = dec_out.permute(0, 2, 1)

#         if self.use_norm:
#             dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
#             dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
#         return dec_out

#     def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
#         if self.use_norm:
#             means = x_enc.mean(1, keepdim=True).detach()
#             x_enc = x_enc - means
#             stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
#             x_enc /= stdev

#         _, _, N = x_enc.shape
#         en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
#         en_embed = self.tcn(en_embed)  # Apply enhanced TCN
#         ex_embed = self.ex_embedding(x_enc, x_mark_enc)

#         enc_out = self.encoder(en_embed, ex_embed)
#         enc_out = torch.reshape(
#             enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
#         enc_out = enc_out.permute(0, 1, 3, 2)
#         dec_out = self.head(enc_out)
#         dec_out = dec_out.permute(0, 2, 1)

#         if self.use_norm:
#             dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
#             dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
#         return dec_out

#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
#             if self.features == 'M':
#                 dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
#                 return dec_out[:, -self.pred_len:, :]
#             else:
#                 dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
#                 return dec_out[:, -self.pred_len:, :]
#         else:
#             return None
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import numpy as np

# Assuming these are available in your layers module
from layers.SelfAttention_Family import AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding

# Flatten Head (unchanged)
class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

# Encoder Embedding (unchanged)
class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class FourierSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, factor=5, dropout=0.1, top_k=10):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.top_k = top_k  # Number of key positions to attend to
        self.factor = factor
        self.scale = 1. / (d_model // n_heads) ** 0.5

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, H, E = queries.shape  # Already split into heads by AttentionLayer
        _, S, _, _ = keys.shape

        # Fourier analysis on keys
        keys_flat = keys.permute(0, 2, 1, 3).reshape(B * H, S, E)  # [B*H, S, E]
        freqs = fft.rfft(keys_flat, dim=1, norm='ortho')  # [B*H, S//2+1, E]
        freq_mags = torch.abs(freqs).mean(dim=-1)  # [B*H, S//2+1]
        
        # Ensure top_k does not exceed available frequency bins
        max_k = freq_mags.shape[1]  # S//2 + 1
        k = min(self.top_k, S, max_k)  # Limit k to the smallest of top_k, S, and freq bins
        _, top_indices = freq_mags.topk(k, dim=1)  # [B*H, k]

        # Create sparse mask
        sparse_mask = torch.zeros(B * H, S, device=keys.device).bool()
        for bh in range(B * H):
            sparse_mask[bh, top_indices[bh]] = True
        sparse_mask = sparse_mask.view(B, H, S)  # [B, H, S]

        # Attention computation
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * self.scale  # [B, H, L, S]
        scores = scores.masked_fill(~sparse_mask.unsqueeze(2), float('-inf'))  # Apply sparsity

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhls,bshe->blhe", attn, values)  # [B, L, H, E]

        return out.contiguous(), attn

# New Dynamic Temporal Convolution Network
class DynamicTemporalConvNetwork(nn.Module):
    def __init__(self, d_model, max_kernel_size=5, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=k, padding=(k-1)//2)
            for k in range(3, max_kernel_size + 1, 2)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.gate = nn.Linear(d_model, len(self.convs))  # Learn weights for each conv
        self.out_proj = nn.Linear(d_model * len(self.convs), d_model)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, D, L]
        residual = x
        
        # Dynamic weighting based on input
        gate_input = x.mean(dim=-1)  # [B, D]
        weights = F.softmax(self.gate(gate_input), dim=-1)  # [B, num_convs]
        
        conv_outputs = []
        for i, conv in enumerate(self.convs):
            out = conv(x) * weights[:, i].view(-1, 1, 1)  # Weight each conv
            out = self.activation(out)
            conv_outputs.append(out)
        
        combined = torch.cat(conv_outputs, dim=1)
        combined = combined.transpose(1, 2)
        out = self.out_proj(combined)
        out = self.dropout(self.norm(out))
        return out + residual.transpose(1, 2)

# Encoder (unchanged structure, updated layers)
class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = AttentionLayer(
            FourierSparseAttention(d_model, n_heads, dropout=dropout, top_k=10),
            d_model, n_heads
        )
        self.cross_attention = AttentionLayer(
            FourierSparseAttention(d_model, n_heads, dropout=dropout, top_k=10),
            d_model, n_heads
        )
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = x.shape  # Adjusted to use x.shape instead of cross.shape for clarity
        x = x + self.dropout(self.self_attention(
            x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0])
        x = self.norm1(x)

        # Extract global token and apply cross-attention
        x_glb_ori = x[:, -1:, :]  # [B, 1, D], keep as is without extra reshape
        x_glb_attn = self.cross_attention(
            x_glb_ori, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta)[0]  # [B, 1, D]
        x_glb = x_glb_ori + self.dropout(x_glb_attn)  # [B, 1, D]
        x_glb = self.norm2(x_glb)

        # Combine and apply FFN
        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)  # [B, L, D]
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)# Main Model
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name  # 'long_term_forecast' or 'short_term_forecast'
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = True
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in

        # Embedding
        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)
        self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, 
                                                 configs.freq, configs.dropout)
        
        # Dynamic TCN
        self.tcn = DynamicTemporalConvNetwork(configs.d_model, max_kernel_size=5, dropout=configs.dropout)

        # Encoder with Fourier Sparse Attention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    configs.d_model, configs.n_heads, configs.d_ff, dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                              head_dropout=configs.dropout)

    def normalize(self, x):
        """Robust normalization from iTransformer"""
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev
        return x, means, stdev

    def denormalize(self, x, means, stdev):
        """De-normalization from iTransformer"""
        if self.features == 'M':
            return x * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)) + \
                   (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        else:
            return x * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1)) + \
                   (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            x_enc, means, stdev = self.normalize(x_enc)

        _, _, N = x_enc.shape
        en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))
        en_embed = self.tcn(en_embed)
        ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            dec_out = self.denormalize(dec_out, means, stdev)
        return dec_out

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            x_enc, means, stdev = self.normalize(x_enc)

        _, _, N = x_enc.shape
        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        en_embed = self.tcn(en_embed)
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            dec_out = self.denormalize(dec_out, means, stdev)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            if self.features == 'M':
                dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

# # Example usage (for reference, not part of the model file)
# if __name__ == "__main__":
#     class Configs:
#         task_name = 'long_term_forecast'
#         features = 'M'
#         seq_len = 96
#         pred_len = 720
#         patch_len = 48
#         enc_in = 7  # Example for ETTh1 multivariate
#         d_model = 512
#         n_heads = 8
#         e_layers = 2
#         d_ff = 2048
#         dropout = 0.1
#         embed = 'timeF'
#         freq = 'h'
#         activation = 'gelu'

#     configs = Configs()
#     model = Model(configs)
#     x_enc = torch.randn(16, 96, 7)  # [batch_size, seq_len, n_vars]
#     x_mark_enc = torch.randn(16, 96, 4)  # [batch_size, seq_len, time_features]
#     x_dec = torch.randn(16, 720, 7)  # [batch_size, pred_len, n_vars]
#     x_mark_dec = torch.randn(16, 720, 4)  # [batch_size, pred_len, time_features]
    
#     output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
#     print(output.shape)  # Expected: [16, 720, 7]