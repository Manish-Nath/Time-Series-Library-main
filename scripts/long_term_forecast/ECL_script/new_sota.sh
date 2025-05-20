
export CUDA_VISIBLE_DEVICES=0

model_name=new_sota

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 4048 \
  --use_multi_gpu\
  --batch_size 4 \
  --itr 1
# long_term_forecast_electricity_96_192_new_sota_custom_electricity.csv_ftM_sl96_ll48_pl192_dm512_batch_size4_nh8_el2_dl1_df1024_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.15640878677368164, mae:0.253147155046463, rmse:0.395485520362854
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id ECL_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 3 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --batch_size 4 \
  --itr 1

# long_term_forecast_electricity_96_336_new_sota_custom_electricity.csv_ftM_sl96_ll48_pl336_dm512_batch_size4_nh8_el3_dl1_df4048_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.17408451437950134, mae:0.2717980742454529, rmse:0.4172343611717224
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id ECL_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 4 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --batch_size 4 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --model_id ECL_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 3 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model 512\
  --d_ff 2048\
  --des 'Exp' \
  --batch_size 4 \
  --itr 1


# long_term_forecast_ECL_96_720_new_sota_custom_electricity.csv_ftM_sl96_ll48_pl720_dm512_batch_size4_nh8_el3_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0  
# mse:0.20261536538600922, mae:0.3007258474826813, rmse:0.4501281678676605

# long_term_forecast_electricity_96_96_new_sota_custom_electricity.csv_ftM_sl96_ll48_pl96_dm512_batch_size4_nh8_el2_dl1_df4048_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.13773958384990692, mae:0.2371286302804947, rmse:0.37113285064697266
