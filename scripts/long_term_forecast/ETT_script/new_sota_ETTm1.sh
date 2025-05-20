export CUDA_VISIBLE_DEVICES=0
# model_name=new_sota
model_name=final_exp1

# long_term_forecast_ETTm1_96_96_combined_ETTm1_ftM_sl96_ll48_pl96_dm128_batch_size4_nh8_el1_dl1_df2028_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.31475958228111267, mae:0.3552192747592926, rmse:0.5610343813896179
python -u run.py \
 --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96 \
  --model $model_name \
  --data ETTm1 \
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
  --d_model 128 \
  --batch_size 4 \
  --des 'exp' \
  --itr 1\
  --d_ff 2028
  

# long_term_forecast_ETTm1_96_192_combined_ETTm1_ftM_sl96_ll48_pl192_dm128_batch_size4_nh8_el1_dl1_df1024_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.3559955358505249, mae:0.38144451379776, rmse:0.5966536402702332
python -u run.py \
 --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_192 \
  --model $model_name \
  --data ETTm1 \
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
  --batch_size 4 \
  --des 'exp' \
  --itr 1\
  --d_ff 1024
  
# long_term_forecast_ETTm1_96_336_combined_ETTm1_ftM_sl96_ll48_pl336_dm512_batch_size4_nh8_el1_dl1_df1024_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.38764065504074097, mae:0.4020441174507141, rmse:0.622607946395874
python -u run.py \
 --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_336 \
  --model $model_name \
  --data ETTm1 \
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
  --d_ff 1024


# long_term_forecast_ETTm1_96_720_combined_ETTm1_ftM_sl96_ll48_pl720_dm128_batch_size32_nh8_el2_dl1_df4048_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.44945070147514343, mae:0.4419509172439575, rmse:0.6704108715057373



python -u run.py \
 --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_720 \
  --model $model_name \
  --data ETTm1 \
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
  --d_ff 4048

