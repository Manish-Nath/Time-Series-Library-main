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
# from layers.SelfAttention_Family import ProbSparseAttention, AttentionLayer
# from layers.Embed import PositionalEmbedding  # Assuming DataEmbedding_inverted is separate
# import numpy as np

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from layers.SelfAttention_Family import ProbSparseAttention, AttentionLayer
# from layers.Embed import PositionalEmbedding
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from layers.SelfAttention_Family import ProbSparseAttention, AttentionLayer
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


# class FourierAttention(nn.Module):
#     def __init__(self, d_model, n_heads, dropout=0.1):
#         super(FourierAttention, self).__init__()
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.q_proj = nn.Linear(d_model, d_model)
#         self.k_proj = nn.Linear(d_model, d_model)
#         self.v_proj = nn.Linear(d_model, d_model)
#         self.out_proj = nn.Linear(d_model, d_model)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         # x: [bs * n_vars, seq_len, d_model]
#         bs_nvars, seq_len, d_model = x.shape
        
#         # FFT to frequency domain
#         x_freq = torch.fft.rfft(x, dim=1)  # [bs * n_vars, seq_len//2 + 1, d_model]
#         freq_len = x_freq.shape[1]
        
#         # Split into real and imaginary for attention
#         x_freq_real = x_freq.real
#         x_freq_imag = x_freq.imag
        
#         # Multi-head attention in frequency domain
#         q = self.q_proj(x_freq_real).view(bs_nvars, freq_len, self.n_heads, -1).transpose(1, 2)
#         k = self.k_proj(x_freq_real).view(bs_nvars, freq_len, self.n_heads, -1).transpose(1, 2)
#         v = self.v_proj(x_freq_imag).view(bs_nvars, freq_len, self.n_heads, -1).transpose(1, 2)
        
#         scores = torch.matmul(q, k.transpose(-1, -2)) / (self.d_model // self.n_heads) ** 0.5
#         attn = F.softmax(scores, dim=-1)
#         attn = self.dropout(attn)
#         out = torch.matmul(attn, v).transpose(1, 2).reshape(bs_nvars, freq_len, d_model)
        
#         # Inverse FFT to return to time domain
#         out = torch.complex(out, torch.zeros_like(out))  # Simplified: using real part only
#         out = torch.fft.irfft(out, n=seq_len, dim=1)  # [bs * n_vars, seq_len, d_model]
        
#         return self.out_proj(out)


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
        
#         # Fourier Attention
#         self.fourier_attn = FourierAttention(configs.d_model, configs.n_heads, configs.dropout)

#         # Encoder with ProbSparseAttention
#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         ProbSparseAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                             output_attention=False),
#                         configs.d_model, configs.n_heads),
#                     AttentionLayer(
#                         ProbSparseAttention(False, configs.factor, attention_dropout=configs.dropout,
#                                             output_attention=False),
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
#         en_embed = self.fourier_attn(en_embed)  # Apply Fourier attention
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
#         en_embed = self.fourier_attn(en_embed)  # Apply Fourier attention
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


# # ProbSparseAttention (already included in your code)
# class ProbSparseAttention(nn.Module):
#     def __init__(self, mask_flag=True, factor=5, attention_dropout=0.1, output_attention=False):
#         super(ProbSparseAttention, self).__init__()
#         self.factor = factor
#         self.mask_flag = mask_flag
#         self.dropout = nn.Dropout(attention_dropout)
#         self.output_attention = output_attention

#     def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
#         B, L_Q, H, D = queries.shape
#         _, L_K, _, _ = keys.shape
#         scores = torch.einsum("blhd,bkhd->bhlk", queries, keys) / (D ** 0.5)
#         if self.mask_flag and attn_mask is not None:
#             scores.masked_fill_(attn_mask, -1e9)
#         top_k = min(L_K, int(self.factor * np.log(L_K)))
#         _, indices = scores.topk(top_k, dim=-1)
#         mask = torch.ones_like(scores).scatter_(-1, indices, 0).bool()
#         scores.masked_fill_(mask, -1e9)
#         attn = torch.softmax(scores, dim=-1)
#         attn = self.dropout(attn)
#         output = torch.einsum("bhlk,bkhd->blhd", attn, values)
#         output = output.contiguous()
#         return (output, attn) if self.output_attention else (output, None)



import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np

class Transpose(nn.Module):
    """From PatchTST for batch normalization"""
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: 
            return x.transpose(*self.dims).contiguous()
        else: 
            return x.transpose(*self.dims)

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

class TemporalConvNetwork(nn.Module):
    def __init__(self, d_model, kernel_size=3, dilations=[1, 2, 4, 8, 16], dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dilations = dilations
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size, 
                     padding=(kernel_size-1)*dilation//2, 
                     dilation=dilation)
            for dilation in dilations
        ])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.out_proj = nn.Linear(d_model * len(dilations), d_model)

    def forward(self, x):
        x = x.transpose(1, 2)
        residual = x
        conv_outputs = []
        for conv in self.convs:
            out = conv(x)
            out = self.activation(out)
            conv_outputs.append(out)
        combined = torch.cat(conv_outputs, dim=1)
        combined = combined.transpose(1, 2)
        out = self.out_proj(combined)
        out = self.dropout(self.norm(out))
        return out + residual.transpose(1, 2)

class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross=None, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        x = x + self.dropout(self.self_attention(
            x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0])
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta)[0])
        x_glb_attn = torch.reshape(x_glb_attn,
                                 (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)

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
        
        # Enhanced TCN
        self.tcn = TemporalConvNetwork(configs.d_model, kernel_size=3, dilations=[1, 2, 4, 8, 16], 
                                     dropout=configs.dropout)

        # Simplified Cross-Attention Layer
        self.cross_attention = AttentionLayer(
            FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
            configs.d_model, n_heads=4  # Reduced heads for simplicity
        )
        self.cross_norm = nn.LayerNorm(configs.d_model)

        # Encoder with BatchNorm (inspired by PatchTST)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                               output_attention=False), configs.d_model, configs.n_heads),
                    AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                               output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff, dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(configs.d_model),
                Transpose(1, 2)
            )
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
        # Embedding for main time series
        en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))  # [B*n_vars, patch_num+1, d_model]
        en_embed = self.tcn(en_embed)  # [B*n_vars, patch_num+1, d_model]
        
        # Exogenous embedding
        ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)  # [B, seq_len, d_model]

        # Simplified cross-attention: preserve patch structure
        cross_out = self.cross_attention(
            en_embed, ex_embed.repeat(n_vars, 1, 1), ex_embed.repeat(n_vars, 1, 1), 
            attn_mask=None, tau=None, delta=None
        )[0]  # [B*n_vars, patch_num+1, d_model]
        cross_out = self.cross_norm(cross_out + en_embed)  # Residual connection

        # Alternative: Concatenate TCN and exogenous embedding
        '''
        ex_embed_avg = ex_embed.mean(dim=1, keepdim=True).repeat(1, self.patch_num + 1, 1)  # [B, patch_num+1, d_model]
        ex_embed_reshaped = ex_embed_avg.repeat(n_vars, 1, 1)  # [B*n_vars, patch_num+1, d_model]
        combined = torch.cat([en_embed, ex_embed_reshaped], dim=-1)  # [B*n_vars, patch_num+1, 2*d_model]
        combined = nn.Linear(2*configs.d_model, configs.d_model)(combined)  # [B*n_vars, patch_num+1, d_model]
        cross_out = self.cross_norm(combined)
        '''

        # Encoder
        enc_out = self.encoder(cross_out, ex_embed)  # [B*n_vars, patch_num+1, d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))  # [B, n_vars, patch_num+1, d_model]
        enc_out = enc_out.permute(0, 1, 3, 2)  # [B, n_vars, d_model, patch_num+1]
        
        # Prediction head
        dec_out = self.head(enc_out)  # [B, n_vars, pred_len]
        dec_out = dec_out.permute(0, 2, 1)  # [B, pred_len, n_vars]

        if self.use_norm:
            dec_out = self.denormalize(dec_out, means, stdev)
        return dec_out

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            x_enc, means, stdev = self.normalize(x_enc)

        _, _, N = x_enc.shape
        # Embedding for main time series
        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))  # [B*n_vars, patch_num+1, d_model]
        en_embed = self.tcn(en_embed)  # [B*n_vars, patch_num+1, d_model]
        
        # Exogenous embedding
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)  # [B, seq_len, d_model]

        # Simplified cross-attention: preserve patch structure
        cross_out = self.cross_attention(
            en_embed, ex_embed.repeat(n_vars, 1, 1), ex_embed.repeat(n_vars, 1, 1), 
            attn_mask=None, tau=None, delta=None
        )[0]  # [B*n_vars, patch_num+1, d_model]
        cross_out = self.cross_norm(cross_out + en_embed)  # Residual connection

        # Alternative: Concatenate TCN and exogenous embedding
        '''
        ex_embed_avg = ex_embed.mean(dim=1, keepdim=True).repeat(1, self.patch_num + 1, 1)  # [B, patch_num+1, d_model]
        ex_embed_reshaped = ex_embed_avg.repeat(n_vars, 1, 1)  # [B*n_vars, patch_num+1, d_model]
        combined = torch.cat([en_embed, ex_embed_reshaped], dim=-1)  # [B*n_vars, patch_num+1, 2*d_model]
        combined = nn.Linear(2*configs.d_model, configs.d_model)(combined)  # [B*n_vars, patch_num+1, d_model]
        cross_out = self.cross_norm(combined)
        '''

        # Encoder
        enc_out = self.encoder(cross_out, ex_embed)  # [B*n_vars, patch_num+1, d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))  # [B, n_vars, patch_num+1, d_model]
        enc_out = enc_out.permute(0, 1, 3, 2)  # [B, n_vars, d_model, patch_num+1]
        
        # Prediction head
        dec_out = self.head(enc_out)  # [B, n_vars, pred_len]
        dec_out = dec_out.permute(0, 2, 1)  # [B, pred_len, n_vars]

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
