export CUDA_VISIBLE_DEVICES=1
# model_name=TimeXer_P
model_name=final_exp1

# long_term_forecast_ETTm2_96_96_combined_ETTm2_ftM_sl96_ll48_pl96_dm256_batch_size8_nh8_el1_dl1_df256_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.1698801964521408, mae:0.2541079521179199, rmse:0.41216525435447693
# long_term_forecast_ETTm2_96_192_combined_ETTm2_ftM_sl96_ll48_pl192_dm128_batch_size16_nh8_el1_dl1_df512_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.23577995598316193, mae:0.2976422607898712, rmse:0.4855717718601227
# long_term_forecast_ETTm2_96_336_combined_ETTm2_ftM_sl96_ll48_pl336_dm512_batch_size4_nh8_el1_dl1_df4048_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.29451581835746765, mae:0.3342248499393463, rmse:0.5426931381225586
# long_term_forecast_ETTm2_96_720_combined_ETTm2_ftM_sl96_ll48_pl720_dm128_batch_size32_nh8_el1_dl1_df256_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.390125572681427, mae:0.3940282464027405, rmse:0.624600350856781
python -u run.py \
 --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1\
  --n_heads 8\
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads 8\
  --d_model 256 \
  --batch_size 8 \
  --des 'exp' \
  --itr 1\
  --d_ff 256
  

python -u run.py \
 --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_192 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 1\
  --n_heads 8\
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads 8\
  --d_model 128 \
  --batch_size 16 \
  --des 'exp' \
  --itr 1\
  --d_ff 512
  

python -u run.py \
 --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_336 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 1\
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads 8\
  --d_model 512 \
  --batch_size 4 \
  --des 'exp' \
  --itr 1\
  --d_ff 4048



python -u run.py \
 --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_720 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2\
  --n_heads 8\
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads 8\
  --d_model 128 \
  --batch_size 32 \
  --des 'exp' \
  --itr 1\
  --d_ff 256

