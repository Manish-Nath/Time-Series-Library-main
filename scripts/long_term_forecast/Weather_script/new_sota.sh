export CUDA_VISIBLE_DEVICES=1

model_name=new_sota

# long_term_forecast_train_combined_custom_ftM_sl96_ll48_pl96_dm512_batch_size8_nh8_el3_dl1_df1024_expand2_dc4_fc5_ebtimeF_dtTrue_Exp_0  
# mse:0.15491650998592377, mae:0.20131489634513855, rmse:0.393594354391098

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 1024 \
  --batch_size 8 \
  --itr 1 \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 256 \
  --batch_size 4 \
  --itr 1


# long_term_forecast_weather_96_192_combined_custom_ftM_sl96_ll48_pl192_dm128_batch_size4_nh8_el2_dl1_df256_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.20463970303535461, mae:0.2485852688550949, rmse:0.452371209859848
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 4048\
  --batch_size 8 \
  --itr 1
# long_term_forecast_weather_96_336_combined_custom_weather.csv_ftM_sl96_ll48_pl336_dm128_batch_size8_nh8_el2_dl1_df4048_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.2610996961593628, mae:0.2899330258369446, rmse:0.510979175567627
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 4048\
  --batch_size 8 \
  --itr 1

# long_term_forecast_weather_96_720_combined_custom_weather.csv_ftM_sl96_ll48_pl720_dm128_batch_size8_nh8_el2_dl1_df4048_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.34033825993537903, mae:0.3424927890300751, rmse:0.583385169506073