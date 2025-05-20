# # # export CUDA_VISIBLE_DEVICES=0
# # # model_name=combined







# # # python -u run.py \
# # #  --task_name long_term_forecast \
# # #   --is_training 1 \
# # #   --root_path ./dataset/ \
# # #   --data_path ETTh1.csv \
# # #   --model_id ETTh1_96_96 \
# # #   --model $model_name \
# # #   --data ETTh1 \
# # #   --features M \
# # #   --seq_len 96 \
# # #   --label_len 48 \
# # #   --pred_len 96 \
# # #   --e_layers 1\
# # #   --n_heads 8\
# # #   --factor 3 \
# # #   --enc_in 7 \
# # #   --dec_in 7 \
# # #   --c_out 7 \
# # #   --n_heads 8\
# # #   --d_model 128 \
# # #   --batch_size 16 \
# # #   --des 'exp' \
# # #   --itr 1\
# # #   --d_ff 2028
  


# # # python -u run.py \
# # #  --task_name long_term_forecast \
# # #   --is_training 1 \
# # #   --root_path ./dataset/ \
# # #   --data_path ETTh1.csv \
# # #   --model_id ETTh1_96_192 \
# # #   --model $model_name \
# # #   --data ETTh1 \
# # #   --features M \
# # #   --seq_len 96 \
# # #   --label_len 48 \
# # #   --pred_len 192 \
# # #   --e_layers 2\
# # #   --n_heads 8\
# # #   --factor 3 \
# # #   --enc_in 7 \
# # #   --dec_in 7 \
# # #   --c_out 7 \
# # #   --n_heads 8\
# # #   --d_model 128 \
# # #   --batch_size 16 \
# # #   --des 'exp' \
# # #   --itr 1\
# # #   --d_ff 2048
  
# # # python -u run.py \
# # #  --task_name long_term_forecast \
# # #   --is_training 1 \
# # #   --root_path ./dataset/ \
# # #   --data_path ETTh1.csv \
# # #   --model_id ETTh1_96_336 \
# # #   --model $model_name \
# # #   --data ETTh1 \
# # #   --features M \
# # #   --seq_len 96 \
# # #   --label_len 48 \
# # #   --pred_len 336 \
# # #   --e_layers 2\
# # #   --factor 3 \
# # #   --enc_in 7 \
# # #   --dec_in 7 \
# # #   --c_out 7 \
# # #   --n_heads 8\
# # #   --d_model 128 \
# # #   --batch_size 4 \
# # #   --des 'exp' \
# # #   --itr 1\
# # #   --d_ff 2028




# # # python -u run.py \
# # #  --task_name long_term_forecast \
# # #   --is_training 1 \
# # #   --root_path ./dataset/ \
# # #   --data_path ETTh1.csv \
# # #   --model_id ETTh1_96_720 \
# # #   --model $model_name \
# # #   --data ETTh1 \
# # #   --features M \
# # #   --seq_len 96 \
# # #   --label_len 48 \
# # #   --pred_len 720 \
# # #   --e_layers 1\
# # #   --n_heads 8\
# # #   --factor 3 \
# # #   --enc_in 7 \
# # #   --dec_in 7 \
# # #   --c_out 7 \
# # #   --n_heads 8\
# # #   --d_model 128 \
# # #   --batch_size 8 \
# # #   --des 'exp' \
# # #   --itr 1\
# # #   --d_ff 2028

# # # export CUDA_VISIBLE_DEVICES=0
# # # model_name=combined

# # # # Define hyperparameter combinations
# # # d_models=(128 256 512)
# # # batch_sizes=(4 8 16 32)
# # # e_layers=(1 2)
# # # d_ffs=(256 512 1024 2028 4048)  # Added d_ff variations

# # # pred_len=96  # Fixed prediction length

# # # # Loop through all combinations
# # # for d_model in "${d_models[@]}"; do
# # #     for batch_size in "${batch_sizes[@]}"; do
# # #         for e_layer in "${e_layers[@]}"; do
# # #             for d_ff in "${d_ffs[@]}"; do
# # #                 echo "Running: d_model=$d_model, batch_size=$batch_size, e_layers=$e_layer, d_ff=$d_ff"
# # #                 python -u run.py \
# # #                   --task_name long_term_forecast \
# # #                   --is_training 1 \
# # #                   --root_path ./dataset/ \
# # #                   --data_path ETTm1.csv \
# # #                   --model_id "ETTm1_96_${pred_len}" \
# # #                   --model $model_name \
# # #                   --data ETTm1 \
# # #                   --features M \
# # #                   --seq_len 96 \
# # #                   --label_len 48 \
# # #                   --pred_len $pred_len \
# # #                   --e_layers $e_layer \
# # #                   --d_ff $d_ff \
# # #                   --factor 3 \
# # #                   --enc_in 7 \
# # #                   --dec_in 7 \
# # #                   --c_out 7 \
# # #                   --d_model $d_model \
# # #                   --batch_size $batch_size \
# # #                   --des 'exp' \
# # #                   --itr 1 | tee -a results.txt
# # #             done
# # #         done
# # #     done
# # # done



# # export CUDA_VISIBLE_DEVICES=0
# # model_name=combined

# # # Define hyperparameter combinations
# # d_models=(512)
# # batch_sizes=(16 32)
# # e_layers=(1 2)
# # d_ffs=(256 512 1024 2028 4048)  # Added d_ff variations

# # pred_len=192  # Fixed prediction length

# # # Loop through all combinations
# # for d_model in "${d_models[@]}"; do
# #     for batch_size in "${batch_sizes[@]}"; do
# #         for e_layer in "${e_layers[@]}"; do
# #             for d_ff in "${d_ffs[@]}"; do
# #                 echo "Running: d_model=$d_model, batch_size=$batch_size, e_layers=$e_layer, d_ff=$d_ff"
# #                 python -u run.py \
# #                   --task_name long_term_forecast \
# #                   --is_training 1 \
# #                   --root_path ./dataset/ \
# #                   --data_path ETTm1.csv \
# #                   --model_id "ETTm1_96_${pred_len}" \
# #                   --model $model_name \
# #                   --data ETTm1 \
# #                   --features M \
# #                   --seq_len 96 \
# #                   --label_len 48 \
# #                   --pred_len $pred_len \
# #                   --e_layers $e_layer \
# #                   --d_ff $d_ff \
# #                   --factor 3 \
# #                   --enc_in 7 \
# #                   --dec_in 7 \
# #                   --c_out 7 \
# #                   --d_model $d_model \
# #                   --batch_size $batch_size \
# #                   --des 'exp' \
# #                   --itr 1 | tee -a results.txt
# #             done
# #         done
# #     done
# # done

# # export CUDA_VISIBLE_DEVICES=0
# # model_name=combined

# # # Define hyperparameter combinations
# # d_models=(128 256 512)
# # batch_sizes=(4 8 16 32)
# # e_layers=(1 2)
# # d_ffs=(256 512 1024 2028 4048)  # Added d_ff variations

# # pred_len=336  # Fixed prediction length

# # # Loop through all combinations
# # for d_model in "${d_models[@]}"; do
# #     for batch_size in "${batch_sizes[@]}"; do
# #         for e_layer in "${e_layers[@]}"; do
# #             for d_ff in "${d_ffs[@]}"; do
# #                 echo "Running: d_model=$d_model, batch_size=$batch_size, e_layers=$e_layer, d_ff=$d_ff"
# #                 python -u run.py \
# #                   --task_name long_term_forecast \
# #                   --is_training 1 \
# #                   --root_path ./dataset/ \
# #                   --data_path ETTm1.csv \
# #                   --model_id "ETTm1_96_${pred_len}" \
# #                   --model $model_name \
# #                   --data ETTm1 \
# #                   --features M \
# #                   --seq_len 96 \
# #                   --label_len 48 \
# #                   --pred_len $pred_len \
# #                   --e_layers $e_layer \
# #                   --d_ff $d_ff \
# #                   --factor 3 \
# #                   --enc_in 7 \
# #                   --dec_in 7 \
# #                   --c_out 7 \
# #                   --d_model $d_model \
# #                   --batch_size $batch_size \
# #                   --des 'exp' \
# #                   --itr 1 | tee -a results.txt
# #             done
# #         done
# #     done
# # done



# # export CUDA_VISIBLE_DEVICES=0
# # model_name=combined

# # # Define hyperparameter combinations
# # d_models=(128 256 512)
# # batch_sizes=(4 8 16 32)
# # e_layers=(1 2)
# # d_ffs=(256 512 1024 2028 4048)  # Added d_ff variations

# # pred_len=720  # Fixed prediction length

# # # Loop through all combinations
# # for d_model in "${d_models[@]}"; do
# #     for batch_size in "${batch_sizes[@]}"; do
# #         for e_layer in "${e_layers[@]}"; do
# #             for d_ff in "${d_ffs[@]}"; do
# #                 echo "Running: d_model=$d_model, batch_size=$batch_size, e_layers=$e_layer, d_ff=$d_ff"
# #                 python -u run.py \
# #                   --task_name long_term_forecast \
# #                   --is_training 1 \
# #                   --root_path ./dataset/ \
# #                   --data_path ETTm1.csv \
# #                   --model_id "ETTm1_96_${pred_len}" \
# #                   --model $model_name \
# #                   --data ETTm1 \
# #                   --features M \
# #                   --seq_len 96 \
# #                   --label_len 48 \
# #                   --pred_len $pred_len \
# #                   --e_layers $e_layer \
# #                   --d_ff $d_ff \
# #                   --factor 3 \
# #                   --enc_in 7 \
# #                   --dec_in 7 \
# #                   --c_out 7 \
# #                   --d_model $d_model \
# #                   --batch_size $batch_size \
# #                   --des 'exp' \
# #                   --itr 1 | tee -a results.txt
# #             done
# #         done
# #     done
# # done


# # export CUDA_VISIBLE_DEVICES=1
# # model_name=combined

# # # Define hyperparameter combinations
# # d_models=(256 512)
# # batch_sizes=(4 8 16 32)
# # e_layers=(1 2)
# # d_ffs=(256 512 1024 2028 4048)  # Added d_ff variations

# # pred_len=96  # Fixed prediction length

# # # Loop through all combinations
# # for d_model in "${d_models[@]}"; do
# #     for batch_size in "${batch_sizes[@]}"; do
# #         for e_layer in "${e_layers[@]}"; do
# #             for d_ff in "${d_ffs[@]}"; do
# #                 echo "Running: d_model=$d_model, batch_size=$batch_size, e_layers=$e_layer, d_ff=$d_ff"
# #                 python -u run.py \
# #                   --task_name long_term_forecast \
# #                   --is_training 1 \
# #                   --root_path ./dataset/ \
# #                   --data_path traffic.csv \
# #                   --model_id "traffic_96_${pred_len}" \
# #                   --model $model_name \
# #                   --data custom \
# #                   --features M \
# #                   --seq_len 96 \
# #                   --label_len 48 \
# #                   --pred_len $pred_len \
# #                   --e_layers $e_layer \
# #                   --d_ff $d_ff \
# #                   --factor 3 \
# #                   --enc_in 862 \
# #                   --dec_in 862 \
# #                   --c_out 862 \
# #                   --d_model $d_model \
# #                   --batch_size $batch_size \
# #                   --des 'exp' \
# #                   --itr 1 | tee -a results.txt
# #             done
# #         done
# #     done
# # done



# # export CUDA_VISIBLE_DEVICES=1
# # model_name=combined

# # # Define hyperparameter combinations
# # d_models=(128 256 512)
# # batch_sizes=(4 8 16 32)
# # e_layers=(1 2)
# # d_ffs=(256 512 1024 2028 4048)  # Added d_ff variations

# # pred_len=192  # Fixed prediction length

# # # Loop through all combinations
# # for d_model in "${d_models[@]}"; do
# #     for batch_size in "${batch_sizes[@]}"; do
# #         for e_layer in "${e_layers[@]}"; do
# #             for d_ff in "${d_ffs[@]}"; do
# #                 echo "Running: d_model=$d_model, batch_size=$batch_size, e_layers=$e_layer, d_ff=$d_ff"
# #                 python -u run.py \
# #                   --task_name long_term_forecast \
# #                   --is_training 1 \
# #                   --root_path ./dataset/ \
# #                   --data_path traffic.csv \
# #                   --model_id "traffic_96_${pred_len}" \
# #                   --model $model_name \
# #                   --data custom \
# #                   --features M \
# #                   --seq_len 96 \
# #                   --label_len 48 \
# #                   --pred_len $pred_len \
# #                   --e_layers $e_layer \
# #                   --d_ff $d_ff \
# #                   --factor 3 \
# #                   --enc_in 862 \
# #                   --dec_in 862 \
# #                   --c_out 862 \
# #                   --d_model $d_model \
# #                   --batch_size $batch_size \
# #                   --des 'exp' \
# #                   --itr 1 | tee -a results.txt
# #             done
# #         done
# #     done
# # done

# export CUDA_VISIBLE_DEVICES=1
# model_name=combined

# # Define hyperparameter combinations
# d_models=(512)
# batch_sizes=(16)
# e_layers=(4)
# d_ffs=(2028 4048)  # Added d_ff variations

# pred_len=336  # Fixed prediction length

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
#                   --data_path traffic.csv \
#                   --model_id "traffic_96_${pred_len}" \
#                   --model $model_name \
#                   --data custom \
#                   --features M \
#                   --seq_len 96 \
#                   --label_len 48 \
#                   --pred_len $pred_len \
#                   --e_layers $e_layer \
#                   --d_ff $d_ff \
#                   --factor 3 \
#                   --enc_in 862 \
#                   --dec_in 862 \
#                   --c_out 862 \
#                   --d_model $d_model \
#                   --batch_size $batch_size \
#                   --des 'exp' \
#                   --itr 1 | tee -a results.txt
#             done
#         done
#     done
# done


# export CUDA_VISIBLE_DEVICES=1
# model_name=combined

# # Define hyperparameter combinations
# d_models=(512)
# batch_sizes=(32)
# e_layers=(2 4)
# d_ffs=(256 512 1024 2028 4048)  # Added d_ff variations

# pred_len=336  # Fixed prediction length

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
#                   --data_path traffic.csv \
#                   --model_id "traffic_96_${pred_len}" \
#                   --model $model_name \
#                   --data custom \
#                   --features M \
#                   --seq_len 96 \
#                   --label_len 48 \
#                   --pred_len $pred_len \
#                   --e_layers $e_layer \
#                   --d_ff $d_ff \
#                   --factor 3 \
#                   --enc_in 862 \
#                   --dec_in 862 \
#                   --c_out 862 \
#                   --d_model $d_model \
#                   --batch_size $batch_size \
#                   --des 'exp' \
#                   --itr 1 | tee -a results.txt
#             done
#         done
#     done
# done



export CUDA_VISIBLE_DEVICES=1
model_name=combined

# Define hyperparameter combinations
d_models=(512)
batch_sizes=(4)
e_layers=(5)
d_ffs=(2028)  # Added d_ff variations

pred_len=720  # Fixed prediction length

# Loop through all combinations
for d_model in "${d_models[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        for e_layer in "${e_layers[@]}"; do
            for d_ff in "${d_ffs[@]}"; do
                echo "Running: d_model=$d_model, batch_size=$batch_size, e_layers=$e_layer, d_ff=$d_ff"
                python -u run.py \
                  --task_name long_term_forecast \
                  --is_training 1 \
                  --root_path ./dataset/ \
                  --data_path traffic.csv \
                  --model_id "traffic_96_${pred_len}" \
                  --model $model_name \
                  --data custom \
                  --features M \
                  --seq_len 96 \
                  --label_len 48 \
                  --pred_len $pred_len \
                  --e_layers $e_layer \
                  --d_ff $d_ff \
                  --factor 3 \
                  --enc_in 862 \
                  --dec_in 862 \
                  --c_out 862 \
                  --d_model $d_model \
                  --batch_size $batch_size \
                  --des 'exp' \
                  --itr 1 | tee -a results.txt
            done
        done
    done
done
