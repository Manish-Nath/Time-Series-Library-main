export CUDA_VISIBLE_DEVICES=0
# model_name=new_sota

model_name=final_exp1






python -u run.py \
 --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data ETTh2 \
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
  --batch_size 16 \
  --des 'exp' \
  --itr 1\
  --d_ff 1024
  

# long_term_forecast_ETTh2_96_96_combined_ETTh2_ftM_sl96_ll48_pl96_dm256_batch_size16_nh8_el1_dl1_df1024_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.28389832377433777, mae:0.3381497859954834, rmse:0.5328211188316345
python -u run.py \
 --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2\
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
  --d_ff 1024
  
# long_term_forecast_ETTh2_96_192_combined_ETTh2_ftM_sl96_ll48_pl192_dm256_batch_size8_nh8_el2_dl1_df1024_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.36285826563835144, mae:0.38598909974098206, rmse:0.60237717628479


python -u run.py \
 --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_336 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2\
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads 8\
  --d_model 512 \
  --batch_size 16 \
  --des 'exp' \
  --itr 1\
  --d_ff 2028


# long_term_forecast_ETTh2_96_336_combined_ETTh2_ftM_sl96_ll48_pl336_dm512_batch_size16_nh8_el1_dl1_df2028_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.41016286611557007, mae:0.42381131649017334, rmse:0.6404395699501038
# long_term_forecast_ETTh2_96_720_combined_ETTh2_ftM_sl96_ll48_pl720_dm512_batch_size16_nh8_el1_dl1_df1024_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.40084460377693176, mae:0.4289475977420807, rmse:0.6331229209899902


python -u run.py \
 --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_720 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 1\
  --n_heads 8\
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --n_heads 8\
  --d_model 512 \
  --batch_size 16 \
  --des 'exp' \
  --itr 1\
  --d_ff 1024