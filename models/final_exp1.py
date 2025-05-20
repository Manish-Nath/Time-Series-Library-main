
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
# # import numpy as np
# # import math


# # class CustomFullAttention(nn.Module):
# #     def __init__(self, mask_flag=False, n_heads=8, attention_dropout=0.1, output_attention=False):
# #         super().__init__()
# #         self.n_heads = n_heads
# #         self.mask_flag = mask_flag
# #         self.output_attention = output_attention
# #         self.dropout = nn.Dropout(attention_dropout)

# #     def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
# #         B, L, D = queries.shape
# #         _, S, _ = keys.shape
# #         H = self.n_heads
# #         scale = 1.0 / math.sqrt(D // H)

# #         queries = queries.view(B, L, H, D // H).transpose(1, 2)
# #         keys = keys.view(B, S, H, D // H).transpose(1, 2)
# #         values = values.view(B, S, H, D // H).transpose(1, 2)

# #         scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale
# #         if self.mask_flag and attn_mask is not None:
# #             scores = scores.masked_fill(attn_mask == 0, -1e9)

# #         attn = F.softmax(scores, dim=-1)
# #         attn = self.dropout(attn)
# #         out = torch.matmul(attn, values)
# #         out = out.transpose(1, 2).contiguous().view(B, L, D)

# #         if self.output_attention:
# #             return out, attn
# #         return out, None


# # class CustomAttentionLayer(nn.Module):
# #     def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
# #         super().__init__()
# #         self.inner_attention = attention
# #         self.d_model = d_model
# #         self.n_heads = n_heads
# #         self.d_keys = d_keys or (d_model // n_heads)
# #         self.d_values = d_values or (d_model // n_heads)

# #         self.query_projection = nn.Linear(d_model, self.d_keys * n_heads)
# #         self.key_projection = nn.Linear(d_model, self.d_keys * n_heads)
# #         self.value_projection = nn.Linear(d_model, self.d_values * n_heads)
# #         self.out_projection = nn.Linear(self.d_values * n_heads, d_model)
# #         self.norm = nn.LayerNorm(d_model)

# #     def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
# #         queries = self.query_projection(queries)
# #         keys = self.key_projection(keys)
# #         values = self.value_projection(values)

# #         out, attn = self.inner_attention(queries, keys, values, attn_mask, tau, delta)
# #         out = self.out_projection(out)
# #         return out, attn


# # class MultiScaleAttention(nn.Module):
# #     def __init__(self, d_model, n_heads, dropout=0.1, output_attention=False):
# #         super().__init__()
# #         self.attention_layers = nn.ModuleList([
# #             CustomAttentionLayer(
# #                 CustomFullAttention(False, n_heads, dropout, output_attention),
# #                 d_model, n_heads
# #             ) for _ in range(3)
# #         ])
# #         self.scale_weights = nn.Parameter(torch.ones(3))
# #         self.norm = nn.LayerNorm(d_model)
# #         self.d_model = d_model
# #         self.n_heads = n_heads

# #     def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
# #         assert isinstance(queries, list), "Queries must be a list of tensors"
# #         outputs = []
# #         target_len = queries[0].shape[1]
# #         for i, (q, k, v, layer) in enumerate(zip(queries, keys, values, self.attention_layers)):
# #             if q.shape[1] > target_len:
# #                 q, k, v = q[:, :target_len, :], k[:, :target_len, :], v[:, :target_len, :]
# #             elif q.shape[1] < target_len:
# #                 pad = torch.zeros(q.shape[0], target_len - q.shape[1], q.shape[2], device=q.device)
# #                 q, k, v = torch.cat([q, pad], dim=1), torch.cat([k, pad], dim=1), torch.cat([v, pad], dim=1)
# #             out, _ = layer(q, k, v, attn_mask, tau, delta)
# #             outputs.append(out)
# #         weights = torch.softmax(self.scale_weights, dim=0)
# #         combined = sum(w * out for w, out in zip(weights, outputs))
# #         return self.norm(combined), None


# # class FlattenHead(nn.Module):
# #     def __init__(self, n_vars, nf, target_window, head_dropout=0):
# #         super().__init__()
# #         self.n_vars = n_vars
# #         self.flatten = nn.Flatten(start_dim=-2)
# #         self.linear = nn.Linear(nf, target_window)
# #         self.dropout = nn.Dropout(head_dropout)

# #     def forward(self, x):
# #         x = self.flatten(x)
# #         x = self.linear(x)
# #         x = self.dropout(x)
# #         return x


# # class EnEmbedding(nn.Module):
# #     def __init__(self, n_vars, d_model, patch_lengths, dropout):
# #         super().__init__()
# #         self.patch_lengths = patch_lengths
# #         self.embeddings = nn.ModuleList([
# #             nn.ModuleDict({
# #                 'value': nn.Linear(patch_len, d_model, bias=False),
# #                 'position': PositionalEmbedding(d_model)
# #             }) for patch_len in patch_lengths
# #         ])
# #         self.glb_tokens = nn.ParameterList([
# #             nn.Parameter(torch.randn(1, n_vars, 1, d_model)) for _ in patch_lengths
# #         ])
# #         self.dropout = nn.Dropout(dropout)

# #     def forward(self, x):
# #         n_vars = x.shape[1]
# #         outputs = []
# #         for embedding, glb_token in zip(self.embeddings, self.glb_tokens):
# #             patch_len = embedding['value'].in_features
# #             glb = glb_token.repeat(x.shape[0], 1, 1, 1)
# #             x_patched = x.unfold(dimension=-1, size=patch_len, step=patch_len)
# #             x_patched = torch.reshape(x_patched, (x_patched.shape[0] * x_patched.shape[1], x_patched.shape[2], x_patched.shape[3]))
# #             x_embed = embedding['value'](x_patched) + embedding['position'](x_patched)
# #             x_embed = torch.reshape(x_embed, (-1, n_vars, x_embed.shape[-2], x_embed.shape[-1]))
# #             x_embed = torch.cat([x_embed, glb], dim=2)
# #             x_embed = torch.reshape(x_embed, (x_embed.shape[0] * x_embed.shape[1], x_embed.shape[2], x_embed.shape[3]))
# #             outputs.append(self.dropout(x_embed))
# #         return outputs, n_vars


# # class TemporalConvNetwork(nn.Module):
# #     def __init__(self, d_model, kernel_size=3, dilations=[1, 2, 4, 8], num_blocks=2, dropout=0.1):
# #         super().__init__()
# #         self.d_model = d_model
# #         self.dilations = dilations
# #         self.num_blocks = num_blocks

# #         self.blocks = nn.ModuleList([
# #             nn.ModuleList([
# #                 nn.Conv1d(d_model, d_model, kernel_size,
# #                           padding=(kernel_size - 1) * dilation // 2,
# #                           dilation=dilation)
# #                 for dilation in dilations
# #             ]) for _ in range(num_blocks)
# #         ])
# #         self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_blocks)])
# #         self.out_projs = nn.ModuleList([nn.Linear(d_model * len(dilations), d_model) for _ in range(num_blocks)])
# #         self.dropout = nn.Dropout(dropout)
# #         self.activation = nn.ReLU()

# #     def forward(self, x):
# #         x = x.transpose(1, 2)
# #         for block, norm, out_proj in zip(self.blocks, self.norms, self.out_projs):
# #             residual = x
# #             conv_outputs = [self.activation(conv(x)) for conv in block]
# #             combined = torch.cat(conv_outputs, dim=1)
# #             combined = combined.transpose(1, 2)
# #             x = out_proj(combined)
# #             x = self.dropout(norm(x)) + residual.transpose(1, 2)
# #             x = x.transpose(1, 2)
# #         return x.transpose(1, 2)


# # class Encoder(nn.Module):
# #     def __init__(self, layers, norm_layer=None, projection=None):
# #         super().__init__()
# #         self.layers = nn.ModuleList(layers)
# #         self.norm = norm_layer
# #         self.projection = projection

# #     def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
# #         for layer in self.layers:
# #             x = layer(x, cross, x_mask, cross_mask, tau, delta)
# #         if self.norm is not None:
# #             x = self.norm(x)
# #         if self.projection is not None:
# #             x = self.projection(x)
# #         return x


# # class EncoderLayer(nn.Module):
# #     def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
# #         super().__init__()
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
# #         if isinstance(x, list):
# #             x, _ = self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)
# #             x = self.norm1(x)
# #         else:
# #             x = x + self.dropout(self.self_attention(
# #                 x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0])
# #             x = self.norm1(x)

# #         x_glb_ori = x[:, -1, :].unsqueeze(1)
# #         x_glb = torch.reshape(x_glb_ori, (B, -1, D))
# #         x_glb_attn = self.dropout(self.cross_attention(
# #             x_glb, cross, cross, attn_mask=cross_mask, tau=tau, delta=None)[0])
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
# #         super().__init__()
# #         self.task_name = configs.task_name
# #         self.features = configs.features
# #         self.seq_len = configs.seq_len
# #         self.pred_len = configs.pred_len
# #         self.use_norm = True
# #         self.patch_lengths = [4, 8, 16]
# #         self.n_vars = 1 if configs.features == 'MS' else configs.enc_in
# #         self.d_model = configs.d_model

# #         # Embedding
# #         self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_lengths, configs.dropout)
# #         self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed,
# #                                                   configs.freq, configs.dropout)

# #         # Deep TCN
# #         self.tcn = TemporalConvNetwork(configs.d_model, kernel_size=3, dilations=[1, 2, 4, 8],
# #                                       num_blocks=2, dropout=configs.dropout)

# #         # Encoder with Multi-Scale Attention
# #         self_attention = MultiScaleAttention(
# #             configs.d_model, configs.n_heads, configs.dropout, output_attention=False)
# #         cross_attention = CustomAttentionLayer(
# #             CustomFullAttention(False, configs.n_heads, configs.dropout, output_attention=False),
# #             configs.d_model, configs.n_heads)
# #         self.encoder = Encoder(
# #             [
# #                 EncoderLayer(
# #                     self_attention, cross_attention, configs.d_model, configs.d_ff,
# #                     dropout=configs.dropout, activation=configs.activation
# #                 ) for _ in range(configs.e_layers)
# #             ],
# #             norm_layer=torch.nn.LayerNorm(configs.d_model)
# #         )

# #         self.head_nf = configs.d_model * (self.seq_len // min(self.patch_lengths) + 1)
# #         self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
# #                                 head_dropout=configs.dropout)

# #     def normalize(self, x):
# #         means = x.mean(1, keepdim=True).detach()
# #         x = x - means
# #         stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
# #         x /= stdev
# #         return x, means, stdev

# #     def denormalize(self, x, means, stdev):
# #         if self.features == 'M':
# #             return x * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)) + \
# #                    (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
# #         else:
# #             return x * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1)) + \
# #                    (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))

# #     def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
# #         if self.use_norm:
# #             x_enc, means, stdev = self.normalize(x_enc)

# #         _, _, N = x_enc.shape
# #         en_embeds, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))
# #         en_embeds = [self.tcn(embed) for embed in en_embeds]
# #         ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)

# #         enc_out = self.encoder(en_embeds, ex_embed)
# #         enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
# #         enc_out = enc_out.permute(0, 1, 3, 2)
# #         dec_out = self.head(enc_out)
# #         dec_out = dec_out.permute(0, 2, 1)

# #         if self.use_norm:
# #             dec_out = self.denormalize(dec_out, means, stdev)
# #         return dec_out

# #     def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
# #         if self.use_norm:
# #             x_enc, means, stdev = self.normalize(x_enc)

# #         _, _, N = x_enc.shape
# #         en_embeds, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
# #         en_embeds = [self.tcn(embed) for embed in en_embeds]
# #         ex_embed = self.ex_embedding(x_enc, x_mark_enc)

# #         enc_out = self.encoder(en_embeds, ex_embed)
# #         enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
# #         enc_out = enc_out.permute(0, 1, 3, 2)
# #         dec_out = self.head(enc_out)
# #         dec_out = dec_out.permute(0, 2, 1)

# #         if self.use_norm:
# #             dec_out = self.denormalize(dec_out, means, stdev)
# #         return dec_out

# #     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
# #         device = x_enc.device
# #         self.to(device)
# #         if self.task_name in ['long_term_forecast', 'short_term_forecast']:
# #             if self.features == 'M':
# #                 dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
# #             else:
# #                 dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
# #             return dec_out[:, -self.pred_len:, :]
# #         return None

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
# import numpy as np
# import math


# class CustomFullAttention(nn.Module):
#     def __init__(self, mask_flag=False, n_heads=8, attention_dropout=0.1, output_attention=False):
#         super().__init__()
#         self.n_heads = n_heads
#         self.mask_flag = mask_flag
#         self.output_attention = output_attention
#         self.dropout = nn.Dropout(attention_dropout)

#     def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
#         B, L, D = queries.shape
#         _, S, _ = keys.shape
#         H = self.n_heads
#         scale = 1.0 / math.sqrt(D // H)

#         queries = queries.view(B, L, H, D // H).transpose(1, 2)
#         keys = keys.view(B, S, H, D // H).transpose(1, 2)
#         values = values.view(B, S, H, D // H).transpose(1, 2)

#         scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale
#         if self.mask_flag and attn_mask is not None:
#             scores = scores.masked_fill(attn_mask == 0, -1e9)

#         attn = F.softmax(scores, dim=-1)
#         attn = self.dropout(attn)
#         out = torch.matmul(attn, values)
#         out = out.transpose(1, 2).contiguous().view(B, L, D)

#         if self.output_attention:
#             return out, attn
#         return out, None


# class CustomAttentionLayer(nn.Module):
#     def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
#         super().__init__()
#         self.inner_attention = attention
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.d_keys = d_keys or (d_model // n_heads)
#         self.d_values = d_values or (d_model // n_heads)

#         self.query_projection = nn.Linear(d_model, self.d_keys * n_heads)
#         self.key_projection = nn.Linear(d_model, self.d_keys * n_heads)
#         self.value_projection = nn.Linear(d_model, self.d_values * n_heads)
#         self.out_projection = nn.Linear(self.d_values * n_heads, d_model)
#         self.norm = nn.LayerNorm(d_modelquirky)
#         super().__init__()
#         self.inner_attention = attention
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.d_keys = d_keys or (d_model // n_heads)
#         self.d_values = d_values or (d_model // n_heads)

#         self.query_projection = nn.Linear(d_model, self.d_keys * n_heads)
#         self.key_projection = nn.Linear(d_model, self.d_keys * n_heads)
#         self.value_projection = nn.Linear(d_model, self.d_values * n_heads)
#         self.out_projection = nn.Linear(self.d_values * n_heads, d_model)
#         self.norm = nn.LayerNorm(d_model)

#     def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
#         queries = self.query_projection(queries)
#         keys = self.key_projection(keys)
#         values = self.value_projection(values)

#         out, attn = self.inner_attention(queries, keys, values, attn_mask, tau, delta)
#         out = self.out_projection(out)
#         return out, attn


# class MultiScaleAttention(nn.Module):
#     def __init__(self, d_model, n_heads, dropout=0.1, output_attention=False):
#         super().__init__()
#         self.attention_layers = nn.ModuleList([
#             CustomAttentionLayer(
#                 CustomFullAttention(False, n_heads, dropout, output_attention),
#                 d_model, n_heads
#             ) for _ in range(3)
#         ])
#         self.scale_weights = nn.Parameter(torch.ones(3))
#         self.norm = nn.LayerNorm(d_model)
#         self.d_model = d_model
#         self.n_heads = n_heads

#     def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
#         assert isinstance(queries, list), "Queries must be a list of tensors"
#         outputs = []
#         target_len = queries[0].shape[1]
#         for i, (q, k, v, layer) in enumerate(zip(queries, keys, values, self.attention_layers)):
#             if q.shape[1] > target_len:
#                 q, k, v = q[:, :target_len, :], k[:, :target_len, :], v[:, :target_len, :]
#             elif q.shape[1] < target_len:
#                 pad = torch.zeros(q.shape[0], target_len - q.shape[1], q.shape[2], device=q.device)
#                 q, k, v = torch.cat([q, pad], dim=1), torch.cat([k, pad], dim=1), torch.cat([v, pad], dim=1)
#             out, _ = layer(q, k, v, attn_mask, tau, delta)
#             outputs.append(out)
#         weights = torch.softmax(self.scale_weights, dim=0)
#         combined = sum(w * out for w, out in zip(weights, outputs))
#         return self.norm(combined), None


# class FlattenHead(nn.Module):
#     def __init__(self, n_vars, nf, target_window, head_dropout=0):
#         super().__init__()
#         self.n_vars = n_vars
#         self.flatten = nn.Flatten(start_dim=-2)
#         self.linear = nn.Linear(nf, target_window)
#         self.dropout = nn.Dropout(head_dropout)

#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.linear(x)
#         x = self.dropout(x)
#         return x


# class EnEmbedding(nn.Module):
#     def __init__(self, n_vars, d_model, patch_lengths, dropout):
#         super().__init__()
#         self.patch_lengths = patch_lengths
#         self.embeddings = nn.ModuleList([
#             nn.ModuleDict({
#                 'value': nn.Linear(patch_len, d_model, bias=False),
#                 'position': PositionalEmbedding(d_model)
#             }) for patch_len in patch_lengths
#         ])
#         self.glb_tokens = nn.ParameterList([
#             nn.Parameter(torch.randn(1, n_vars, 1, d_model)) for _ in patch_lengths
#         ])
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         n_vars = x.shape[1]
#         outputs = []
#         for embedding, glb_token in zip(self.embeddings, self.glb_tokens):
#             patch_len = embedding['value'].in_features
#             glb = glb_token.repeat(x.shape[0], 1, 1, 1)
#             x_patched = x.unfold(dimension=-1, size=patch_len, step=patch_len)
#             x_patched = torch.reshape(x_patched, (x_patched.shape[0] * x_patched.shape[1], x_patched.shape[2], x_patched.shape[3]))
#             x_embed = embedding['value'](x_patched) + embedding['position'](x_patched)
#             x_embed = torch.reshape(x_embed, (-1, n_vars, x_embed.shape[-2], x_embed.shape[-1]))
#             x_embed = torch.cat([x_embed, glb], dim=2)
#             x_embed = torch.reshape(x_embed, (x_embed.shape[0] * x_embed.shape[1], x_embed.shape[2], x_embed.shape[3]))
#             outputs.append(self.dropout(x_embed))
#         return outputs, n_vars


# class MultiScaleConv(nn.Module):
#     def __init__(self, d_model, kernel_sizes=[3, 5, 7], dropout=0.1):
#         super().__init__()
#         self.convs = nn.ModuleList([
#             nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2)
#             for kernel_size in kernel_sizes
#         ])
#         self.norm = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = nn.ReLU()

#     def forward(self, x_list):
#         outputs = []
#         for x in x_list:
#             x = x.transpose(1, 2)  # [bs*n_vars, d_model, seq]
#             conv_outs = [self.activation(conv(x)) for conv in self.convs]
#             combined = sum(conv_out for conv_out in conv_outs) / len(conv_outs)
#             combined = combined.transpose(1, 2)  # [bs*n_vars, seq, d_model]
#             combined = self.norm(combined)
#             outputs.append(self.dropout(combined))
#         return outputs


# class Encoder(nn.Module):
#     def __init__(self, layers, norm_layer=None, projection=None):
#         super().__init__()
#         self.layers = nn.ModuleList(layers)
#         self.norm = norm_layer
#         self.projection = projection

#     def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
#         for layer in self.layers:
#             x = layer(x, cross, x_mask, cross_mask, tau, delta)
#         if self.norm is not None:
#             x = self.norm(x)
#         if self.projection is not None:
#             x = self.projection(x)
#         return x


# class EncoderLayer(nn.Module):
#     def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
#         super().__init__()
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
#         self.patch_lengths = [4, 8, 16]  # Match EnEmbedding
#         self.seq_len = 96  # Match seq_len

#     def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
#         B, L, D = cross.shape
#         if isinstance(x, list):
#             x, _ = self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)
#             x = self.norm1(x)
#         else:
#             x = x + self.dropout(self.self_attention(
#                 x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0])
#             x = self.norm1(x)

#         x_glb_ori = x[:, -1, :].unsqueeze(1)
#         x_glb = torch.reshape(x_glb_ori, (B, -1, D))
#         x_glb_attn = self.dropout(self.cross_attention(
#             x_glb, cross, cross, attn_mask=cross_mask, tau=tau, delta=None)[0])
#         x_glb_attn = torch.reshape(x_glb_attn,
#                                    (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
#         x_glb = x_glb_ori + x_glb_attn
#         x_glb = self.norm2(x_glb)

#         y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)
#         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
#         y = self.dropout(self.conv2(y).transpose(-1, 1))
#         x = self.norm3(x + y)

#         # Convert tensor to list for next layer
#         outputs = []
#         for patch_len in self.patch_lengths:
#             patch_num = self.seq_len // patch_len + 1
#             if x.shape[1] > patch_num:
#                 out = x[:, :patch_num, :]
#             else:
#                 pad = torch.zeros(x.shape[0], patch_num - x.shape[1], x.shape[2], device=x.device)
#                 out = torch.cat([x, pad], dim=1)
#             outputs.append(out)
#         return outputs


# class Model(nn.Module):
#     def __init__(self, configs):
#         super().__init__()
#         self.task_name = configs.task_name
#         self.features = configs.features
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.use_norm = True
#         self.patch_lengths = [4, 8, 16]
#         self.n_vars = 1 if configs.features == 'MS' else configs.enc_in
#         self.d_model = configs.d_model

#         # Embedding
#         self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_lengths, configs.dropout)
#         self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed,
#                                                   configs.freq, configs.dropout)

#         # Multi-Scale Convolution
#         self.multi_conv = MultiScaleConv(configs.d_model, kernel_sizes=[3, 5, 7], dropout=configs.dropout)

#         # Encoder with Multi-Scale Attention
#         self_attention = MultiScaleAttention(
#             configs.d_model, configs.n_heads, configs.dropout, output_attention=False)
#         cross_attention = CustomAttentionLayer(
#             CustomFullAttention(False, configs.n_heads, configs.dropout, output_attention=False),
#             configs.d_model, configs.n_heads)
#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     self_attention, cross_attention, configs.d_model, configs.d_ff,
#                     dropout=configs.dropout, activation=configs.activation
#                 ) for _ in range(configs.e_layers)
#             ],
#             norm_layer=torch.nn.LayerNorm(configs.d_model)
#         )

#         self.head_nf = configs.d_model * (self.seq_len // min(self.patch_lengths) + 1)
#         self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
#                                 head_dropout=configs.dropout)

#     def normalize(self, x):
#         means = x.mean(1, keepdim=True).detach()
#         x = x - means
#         stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
#         x /= stdev
#         return x, means, stdev

#     def denormalize(self, x, means, stdev):
#         if self.features == 'M':
#             return x * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)) + \
#                    (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
#         else:
#             return x * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1)) + \
#                    (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))

#     def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
#         if self.use_norm:
#             x_enc, means, stdev = self.normalize(x_enc)

#         _, _, N = x_enc.shape
#         en_embeds, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))
#         en_embeds = self.multi_conv(en_embeds)
#         ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)

#         enc_out = self.encoder(en_embeds, ex_embed)
#         enc_out = enc_out[0]  # Take first scale for head
#         enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
#         enc_out = enc_out.permute(0, 1, 3, 2)
#         dec_out = self.head(enc_out)
#         dec_out = dec_out.permute(0, 2, 1)

#         if self.use_norm:
#             dec_out = self.denormalize(dec_out, means, stdev)
#         return dec_out

#     def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
#         if self.use_norm:
#             x_enc, means, stdev = self.normalize(x_enc)

#         _, _, N = x_enc.shape
#         en_embeds, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
#         en_embeds = self.multi_conv(en_embeds)
#         ex_embed = self.ex_embedding(x_enc, x_mark_enc)

#         enc_out = self.encoder(en_embeds, ex_embed)
#         enc_out = enc_out[0]  # Take first scale for head
#         enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
#         enc_out = enc_out.permute(0, 1, 3, 2)
#         dec_out = self.head(enc_out)
#         dec_out = dec_out.permute(0, 2, 1)

#         if self.use_norm:
#             dec_out = self.denormalize(dec_out, means, stdev)
#         return dec_out

#     def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
#         device = x_enc.device
#         self.to(device)
#         if self.task_name in ['long_term_forecast', 'short_term_forecast']:
#             if self.features == 'M':
#                 dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
#             else:
#                 dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
#             return dec_out[:, -self.pred_len:, :]
#         return None


import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np

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
    def __init__(self, d_model, kernel_size=3, dilations=[1, 2, 4, 8], dropout=0.1):
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

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
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
        # No separate cross_attention; reuse self_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape  # cross: [batch_size, n_vars, d_model]
        # Step 1: Self-Attention on endogenous tokens
        x = x + self.dropout(self.self_attention(
            x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0])
        x = self.norm1(x)

        # Step 2: Self-Attention combining global token and exogenous embeddings
        x_glb_ori = x[:, -1, :].unsqueeze(1)  # [batch_size * n_vars, 1, d_model]
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))  # [batch_size, n_vars, d_model]
        # Concatenate global token and exogenous embeddings
        combined_input = torch.cat([x_glb, cross], dim=1)  # [batch_size, 2*n_vars, d_model]
        x_glb_attn = self.dropout(self.self_attention(
            x_glb, combined_input, combined_input, attn_mask=cross_mask, tau=tau, delta=delta)[0])
        # Extract output corresponding to the global token
        x_glb_attn = x_glb_attn[:, :x_glb.shape[1], :]  # [batch_size, n_vars, d_model]
        x_glb_attn = torch.reshape(x_glb_attn, 
                                  (x_glb_attn.shape[0] * x_glb_attn.shape[1], 1, x_glb_attn.shape[2]))  # [batch_size * n_vars, 1, d_model]
        x_glb = x_glb_ori + x_glb_attn  # Residual connection
        x_glb = self.norm2(x_glb)

        # Step 3: Feed-Forward
        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)  # [batch_size * n_vars, patch_num + 1, d_model]
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
        self.tcn = TemporalConvNetwork(configs.d_model, kernel_size=3, dilations=[1, 2, 4, 8], 
                                     dropout=configs.dropout)

        # Encoder with FullAttention
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