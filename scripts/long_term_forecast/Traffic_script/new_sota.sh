export CUDA_VISIBLE_DEVICES=0

model_name=new_tcn_

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 512 \
  --d_ff 4048 \
  --des 'Exp' \
  --batch_size 4 \
  --learning_rate 0.001 \
  --itr 1

# long_term_forecast_traffic_96_96_combined_custom_traffic.csv_ftM_sl96_ll48_pl96_dm512_batch_size4_nh8_el1_dl1_df4048_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.4387943744659424, mae:0.279102087020874, rmse:0.6624155640602112

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 512 \
  --d_ff 2028 \
  --des 'Exp' \
  --batch_size 4 \
  --learning_rate 0.001 \
  --itr 1

# long_term_forecast_traffic_96_192_combined_custom_traffic.csv_ftM_sl96_ll48_pl192_dm512_batch_size4_nh8_el1_dl1_df2028_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.45929524302482605, mae:0.286650687456131, rmse:0.6777132749557495
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 512 \
  --d_ff 2028 \
  --des 'Exp' \
  --batch_size 4 \
  --learning_rate 0.001 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --model_id traffic_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 512 \
  --d_ff 4048 \
  --des 'Exp' \
  --batch_size 4 \
  --learning_rate 0.001 \
  --itr 1

# long_term_forecast_traffic_96_336_combined_custom_traffic.csv_ftM_sl96_ll48_pl336_dm512_batch_size4_nh8_el2_dl1_df2028_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.48038211464881897, mae:0.29366663098335266, rmse:0.6930960416793823
# long_term_forecast_traffic_96_720_combined_custom_traffic.csv_ftM_sl96_ll48_pl720_dm512_batch_size4_nh8_el2_dl1_df4048_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.5187297463417053, mae:0.3085053563117981, rmse:0.7202289700508118
