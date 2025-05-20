export CUDA_VISIBLE_DEVICES=0
model_name=new_sota
# model_name=final_exp1
# model_name=new_tcn_






python -u run.py \
 --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
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
  --batch_size 16 \
  --des 'exp' \
  --itr 1\
  --d_ff 2028


python -u run.py \
 --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
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
  --d_model 128 \
  --batch_size 16 \
  --des 'exp' \
  --itr 1\
  --d_ff 2048
  
python -u run.py \
 --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
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
  --d_model 128 \
  --batch_size 4 \
  --des 'exp' \
  --itr 1\
  --d_ff 2028




python -u run.py \
 --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data ETTh1 \
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
  --d_model 128 \
  --batch_size 8 \
  --des 'exp' \
  --itr 1\
  --d_ff 2028  # python run.py --task_name long_term_forecast --is_training 1 --model_id train --model combined  
# --data ETTh1 --pred_len 192 --e_layers 2 --factor 3 --d_model 128 --batch_size 16 mse:0.4198872447013855, mae:0.4266609251499176,rmse:0.6479870676994324
# long_term_forecast_ETTh1_96_336_combined_ETTh1_ftM_sl96_ll48_pl336_dm128_batch_size4_nh8_el2_dl1_df2028_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.4620048403739929, mae:0.4469195306301117, rmse:0.6797093749046326
# long_term_forecast_ETTh1_96_720_combined_ETTh1_ftM_sl96_ll48_pl720_dm128_batch_size8_nh8_el1_dl1_df2028_expand2_dc4_fc3_ebtimeF_dtTrue_exp_0  
# mse:0.4621013402938843, mae:0.46428483724594116, rmse:0.6797803640365601
