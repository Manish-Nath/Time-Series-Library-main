# export CUDA_VISIBLE_DEVICES=0
# model_name=new_sota

# # Define hyperparameter combinations
# d_models=(128 256 512)
# batch_sizes=(4 8 16 32)
# e_layers=(3 4 5)
# d_ffs=(256 512 1024 2028 4048)  # Added d_ff variations

# pred_len=96  # Fixed prediction length

# # Loop through all combinations
# for d_model in "${d_models[@]}"; do
#     for batch_size in "${batch_sizes[@]}"; do
#         for e_layer in "${e_layers[@]}"; do
#             for d_ff in "${d_ffs[@]}"; do
#                 echo "Running: d_model=$d_model, batch_size=$batch_size, e_layers=$e_layer, d_ff=$d_ff"
#                 python -u run.py \
#                   --task_name long_term_forecast \
#                   --is_training 1 \
#                   --root_path ./dataset/ \
#                   --data_path electricity.csv \
#                   --model_id "electricity_96_${pred_len}" \
#                   --model $model_name \
#                   --data custom \
#                   --features M \
#                   --seq_len 96 \
#                   --label_len 48 \
#                   --pred_len $pred_len \
#                   --e_layers $e_layer \
#                   --d_ff $d_ff \
#                   --factor 3 \
#                   --enc_in 321 \
#                   --dec_in 321 \
#                   --c_out 321 \
#                   --d_model $d_model \
#                   --batch_size $batch_size \
#                   --des 'exp' \
#                   --itr 1 | tee -a results.txt
#             done
#         done
#     done
# done



# export CUDA_VISIBLE_DEVICES=1
# model_name=new_sota

# # Define hyperparameter combinations
# d_models=(512)
# batch_sizes=(8 16 32)
# e_layers=(3 4)
# d_ffs=(256 512 1024 2028 4048)  # Added d_ff variations
# pred_len=336  # Fixed prediction length

# # Loop through all combinations
# for d_model in "${d_models[@]}"; do
#     for batch_size in "${batch_sizes[@]}"; do
#         for e_layer in "${e_layers[@]}"; do
#             for d_ff in "${d_ffs[@]}"; do
#                 echo "Running: d_model=$d_model, batch_size=$batch_size, e_layers=$e_layer, d_ff=$d_ff"
#                 taskset -c 0-47 python -u run.py \
#                   --task_name long_term_forecast \
#                   --is_training 1 \
#                   --root_path ./dataset/ \
#                   --data_path electricity.csv \
#                   --model_id "electricity_96_${pred_len}" \
#                   --model $model_name \
#                   --data custom \
#                   --features M \
#                   --seq_len 96 \
#                   --label_len 48 \
#                   --pred_len $pred_len \
#                   --e_layers $e_layer \
#                   --d_ff $d_ff \
#                   --factor 3 \
#                   --enc_in 321 \
#                   --dec_in 321 \
#                   --c_out 321 \
#                   --d_model $d_model \
#                   --batch_size $batch_size \
#                   --des 'exp' \
#                   --itr 1 | tee -a results.txt
#             done
#         done
#     done
# done


export CUDA_VISIBLE_DEVICES=1
model_name=new_sota

# 
# Define hyperparameter combinations
d_models=(256 512)
batch_sizes=(4 8 16 32)
e_layers=(1 2)
d_ffs=(256 512 1024 2028 4048)  # Added d_ff variations

pred_len=192  # Fixed prediction length

# Loop through all combinations
for d_model in "${d_models[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        for e_layer in "${e_layers[@]}"; do
            for d_ff in "${d_ffs[@]}"; do
                echo "Running: d_model=$d_model, batch_size=$batch_size, e_layers=$e_layer, d_ff=$d_ff"
                taskset -c 0-47 python -u run.py \
                  --task_name long_term_forecast \
                  --is_training 1 \
                  --root_path ./dataset/ \
                  --data_path electricity.csv \
                  --model_id "electricity_96_${pred_len}" \
                  --model $model_name \
                  --data custom \
                  --features M \
                  --seq_len 96 \
                  --label_len 48 \
                  --pred_len $pred_len \
                  --e_layers $e_layer \
                  --d_ff $d_ff \
                  --factor 3 \
                  --enc_in 321 \
                  --dec_in 321 \
                  --c_out 321 \
                  --d_model $d_model \
                  --batch_size $batch_size \
                  --des 'exp' \
                  --itr 1 | tee -a results.txt
            done
        done
    done
done



# export CUDA_VISIBLE_DEVICES=1
# model_name=new_sota

# # Define hyperparameter combinations
# d_models=(128 256 512)
# batch_sizes=(4 8 16 32)
# e_layers=(1 2 3 4)
# d_ffs=(256 512 1024 2028 4048)  # Added d_ff variations

# pred_len=720  # Fixed prediction length

# # Loop through all combinations
# for d_model in "${d_models[@]}"; do
#     for batch_size in "${batch_sizes[@]}"; do
#         for e_layer in "${e_layers[@]}"; do
#             for d_ff in "${d_ffs[@]}"; do
#                 echo "Running: d_model=$d_model, batch_size=$batch_size, e_layers=$e_layer, d_ff=$d_ff"
#                 taskset -c 0-47 python -u run.py \
#                   --task_name long_term_forecast \
#                   --is_training 1 \
#                   --root_path ./dataset/ \
#                   --data_path electricity.csv \
#                   --model_id "electricity_96_${pred_len}" \
#                   --model $model_name \
#                   --data custom \
#                   --features M \
#                   --seq_len 96 \
#                   --label_len 48 \
#                   --pred_len $pred_len \
#                   --e_layers $e_layer \
#                   --d_ff $d_ff \
#                   --factor 3 \
#                   --enc_in 321 \
#                   --dec_in 321 \
#                   --c_out 321 \
#                   --d_model $d_model \
#                   --batch_size $batch_size \
#                   --des 'exp' \
#                   --itr 1 | tee -a results.txt
#             done
#         done
#     done
# done


# long_term_forecast_ECL_96_720_new_sota_custom_electricity.csv_ftM_sl96_ll48_pl720_dm512_batch_size4_nh8_el3_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0  
# mse:0.20261536538600922, mae:0.3007258474826813, rmse:0.4501281678676605

