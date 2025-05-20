
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
    #0.375/data/home/manishnath/Time-Series-Library-main/models/new_sota.py
# python run.py --task_name long_term_forecast --is_training 1 --model_id train --model combined  
# --data ETTh1 --pred_len 192 --e_layers 2 --factor 3 --d_model 128 --batch_size 16 mse:0.4198872447013855, mae:0.4266609251499176,rmse:0.6479870676994324
# long_term_forecast_ETTh1_96_336_combined_ETTh1_ftM_sl96_ll48_pl336_dm128_batch_size4_nh8_el2_dl1_df2028_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.4620048403739929, mae:0.4469195306301117, rmse:0.6797093749046326
# long_term_forecast_ETTh1_96_720_combined_ETTh1_ftM_sl96_ll48_pl720_dm128_batch_size8_nh8_el1_dl1_df2028_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.4621013402938843, mae:0.46428483724594116, rmse:0.6797803640365601


# long_term_forecast_ETTh2_96_96_combined_ETTh2_ftM_sl96_ll48_pl96_dm256_batch_size16_nh8_el1_dl1_df1024_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.28389832377433777, mae:0.3381497859954834, rmse:0.5328211188316345
# long_term_forecast_ETTh2_96_192_combined_ETTh2_ftM_sl96_ll48_pl192_dm256_batch_size8_nh8_el2_dl1_df1024_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.36285826563835144, mae:0.38598909974098206, rmse:0.60237717628479
# long_term_forecast_ETTh2_96_336_combined_ETTh2_ftM_sl96_ll48_pl336_dm512_batch_size16_nh8_el1_dl1_df2028_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.41016286611557007, mae:0.42381131649017334, rmse:0.6404395699501038
# long_term_forecast_ETTh2_96_720_combined_ETTh2_ftM_sl96_ll48_pl720_dm512_batch_size16_nh8_el1_dl1_df1024_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.40084460377693176, mae:0.4289475977420807, rmse:0.6331229209899902


# long_term_forecast_ETTm1_96_96_combined_ETTm1_ftM_sl96_ll48_pl96_dm128_batch_size4_nh8_el1_dl1_df2028_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.31475958228111267, mae:0.3552192747592926, rmse:0.5610343813896179
# long_term_forecast_ETTm1_96_192_combined_ETTm1_ftM_sl96_ll48_pl192_dm128_batch_size4_nh8_el1_dl1_df1024_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.3559955358505249, mae:0.38144451379776, rmse:0.5966536402702332
# long_term_forecast_ETTm1_96_336_combined_ETTm1_ftM_sl96_ll48_pl336_dm512_batch_size4_nh8_el1_dl1_df1024_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.38764065504074097, mae:0.4020441174507141, rmse:0.622607946395874
# long_term_forecast_ETTm1_96_720_combined_ETTm1_ftM_sl96_ll48_pl720_dm128_batch_size32_nh8_el2_dl1_df4048_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.44945070147514343, mae:0.4419509172439575, rmse:0.6704108715057373



# long_term_forecast_ETTm2_96_96_combined_ETTm2_ftM_sl96_ll48_pl96_dm256_batch_size8_nh8_el1_dl1_df256_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.1698801964521408, mae:0.2541079521179199, rmse:0.41216525435447693
# long_term_forecast_ETTm2_96_192_combined_ETTm2_ftM_sl96_ll48_pl192_dm128_batch_size16_nh8_el1_dl1_df512_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.23577995598316193, mae:0.2976422607898712, rmse:0.4855717718601227
# long_term_forecast_ETTm2_96_336_combined_ETTm2_ftM_sl96_ll48_pl336_dm512_batch_size4_nh8_el1_dl1_df4048_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.29451581835746765, mae:0.3342248499393463, rmse:0.5426931381225586
# long_term_forecast_ETTm2_96_720_combined_ETTm2_ftM_sl96_ll48_pl720_dm128_batch_size32_nh8_el1_dl1_df256_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.390125572681427, mae:0.3940282464027405, rmse:0.624600350856781




# long_term_forecast_weather_96_96_combined_custom_ftM_sl96_ll48_pl96_dm256_batch_size8_nh8_el2_dl1_df4048_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.1565317064523697, mae:0.2017393261194229, rmse:0.39564087986946106

# long_term_forecast_train_combined_custom_ftM_sl96_ll48_pl96_dm512_batch_size8_nh8_el3_dl1_df1024_expand2_dc4_fc5_ebtimeF_dtTrue_Exp_0  
# mse:0.15491650998592377, mae:0.20131489634513855, rmse:0.393594354391098