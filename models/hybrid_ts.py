import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import PatchEmbedding
from layers.Conv_Blocks import Inception_Block_V1

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=2, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, bias=True): 
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias) 
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.fc3 = nn.Linear(input_dim, output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.ln = LayerNorm(output_dim, bias=bias)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + self.fc3(x)
        out = self.ln(out)
        return out


class TimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps):
        half_dim = self.dim // 2
        emb = torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        emb = torch.exp(-emb * (math.log(self.max_period) / half_dim))
        emb = timesteps.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class Model(nn.Module):
    def __init__(self, configs, bias=True, feature_encode_dim=2, use_time_emb=True):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.hidden_dim = configs.d_model
        self.res_hidden = configs.d_model
        self.encoder_num = configs.e_layers
        self.decoder_num = configs.d_layers
        self.freq = configs.freq
        self.feature_encode_dim = feature_encode_dim
        self.decode_dim = configs.c_out
        self.temporalDecoderHidden = configs.d_ff
        self.dropout = configs.dropout
        self.use_time_emb = use_time_emb
        self.patch_len = configs.patch_len if hasattr(configs, 'patch_len') else 16
        self.top_k = configs.top_k if hasattr(configs, 'top_k') else 5
        self.num_kernels = configs.num_kernels if hasattr(configs, 'num_kernels') else 6

        # Frequency map for time features
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        self.feature_dim = freq_map[self.freq]

        # DLinear Decomposition
        self.decomp = series_decomp(configs.moving_avg)

        # PatchTST Patching
        self.patch_embedding = PatchEmbedding(
            self.hidden_dim, self.patch_len, self.patch_len, 0, self.dropout)

        # TimesNet FFT Convolution
        self.times_conv = nn.Sequential(
            Inception_Block_V1(self.hidden_dim, self.res_hidden, num_kernels=self.num_kernels),
            nn.GELU(),
            Inception_Block_V1(self.res_hidden, self.hidden_dim, num_kernels=self.num_kernels)
        )

        # Feature Encoder with Time Embeddings
        input_feature_dim = self.feature_dim + (feature_encode_dim if use_time_emb else 0)
        self.time_embedding = TimeEmbedding(feature_encode_dim) if use_time_emb else None
        self.feature_encoder = ResBlock(input_feature_dim, self.res_hidden, self.feature_encode_dim, self.dropout, bias)

        # Pre-Attention and Encoders
        self.pre_attention = MultiHeadSelfAttention(self.feature_encode_dim, num_heads=2, dropout=self.dropout)
        flatten_dim = self.seq_len + (self.seq_len + self.pred_len) * self.feature_encode_dim
        self.encoders = nn.Sequential(
            ResBlock(flatten_dim, self.res_hidden, self.hidden_dim, self.dropout, bias),
            *[ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, self.dropout, bias) 
              for _ in range(self.encoder_num - 1)]
        )

        # Decoder
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.decoders = nn.Sequential(
                *[ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, self.dropout, bias) 
                  for _ in range(self.decoder_num - 1)],
                ResBlock(self.hidden_dim, self.res_hidden, self.decode_dim * self.pred_len, self.dropout, bias)
            )
            self.temporalDecoder = ResBlock(self.decode_dim + self.feature_encode_dim, 
                                            self.temporalDecoderHidden, 1, self.dropout, bias)
            self.residual_proj = nn.Linear(self.seq_len, self.pred_len, bias=bias)

    def fft_period(self, x, k=5):
        xf = torch.fft.rfft(x, dim=1)
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, k)
        top_list = top_list.detach().cpu().numpy()
        period = x.shape[1] // top_list
        return period, abs(xf).mean(-1)[:, top_list]

    def process_seasonal(self, x):
        x_patched, n_vars = self.patch_embedding(x.permute(0, 2, 1))  # [B * nvars, patch_num, d_model]
        period_list, period_weight = self.fft_period(x, self.top_k)
        res = []
        for i in range(self.top_k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], length - (self.seq_len + self.pred_len), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                out = x
            out = out.reshape(out.shape[0], length // period, period, out.shape[2]).permute(0, 3, 1, 2).contiguous()
            out = self.times_conv(out).permute(0, 2, 3, 1).reshape(out.shape[0], -1, out.shape[3])
            res.append(out[:, :self.seq_len + self.pred_len, :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1).unsqueeze(1).unsqueeze(1).repeat(1, self.seq_len + self.pred_len, self.decode_dim, 1)
        seasonal_out = torch.sum(res * period_weight, -1)
        return seasonal_out

    def forecast(self, x_enc, x_mark_enc, x_dec, batch_y_mark):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Decomposition
        seasonal_init, trend_init = self.decomp(x_enc)
        seasonal_out = self.process_seasonal(seasonal_init)

        # Feature Encoding with Time Embeddings
        if self.use_time_emb:
            time_emb = self.time_embedding(torch.arange(self.seq_len + self.pred_len, device=x_enc.device))
            batch_y_mark = torch.cat([batch_y_mark, time_emb.expand(x_enc.shape[0], -1, -1)], dim=-1)
        feature = self.feature_encoder(batch_y_mark)
        feature = self.pre_attention(feature)

        # Combine trend and seasonal with feature
        hidden = torch.cat([trend_init, seasonal_out, feature.reshape(feature.shape[0], -1)], dim=-1)
        hidden = self.encoders(hidden)
        decoded = self.decoders(hidden).reshape(hidden.shape[0], self.pred_len, self.decode_dim)
        dec_out = self.temporalDecoder(torch.cat([feature[:, self.seq_len:], decoded], dim=-1)).squeeze(-1)
        dec_out = dec_out + self.residual_proj(x_enc)

        # Denormalization
        dec_out = dec_out * (stdev[:, 0].unsqueeze(1).repeat(1, self.pred_len))
        dec_out = dec_out + (means[:, 0].unsqueeze(1).repeat(1, self.pred_len))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, batch_y_mark, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            if batch_y_mark is None:
                batch_y_mark = torch.zeros((x_enc.shape[0], self.seq_len + self.pred_len, self.feature_dim), 
                                           device=x_enc.device).detach()
            else:
                batch_y_mark = torch.cat([x_mark_enc, batch_y_mark[:, -self.pred_len:, :]], dim=1)
            dec_out = torch.stack([self.forecast(x_enc[:, :, feature], x_mark_enc, x_dec, batch_y_mark) 
                                   for feature in range(x_enc.shape[-1])], dim=-1)
            return dec_out
        return None