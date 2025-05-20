#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
model_name=new_sota
des='Exp'
patch_len=24

# Define hyperparameter ranges
e_layers_list=(1 2 3 4)
d_model_list=(256 512 1024)
batch_size_list=(4 8 16 32)
d_ff_list=(128 512 1024)

# Define dataset files
datasets=("PJM.csv" "BE.csv" "FR.csv" "DE.csv")

for data_path in "${datasets[@]}"; do
  model_id_prefix="${data_path%%.*}_168_24"
  echo "===== Starting experiments for $data_path ====="
  
  for e_layers in "${e_layers_list[@]}"; do
    for d_model in "${d_model_list[@]}"; do
      for batch_size in "${batch_size_list[@]}"; do
        for d_ff in "${d_ff_list[@]}"; do
          model_id="${model_id_prefix}_el${e_layers}_dm${d_model}_bs${batch_size}_ff${d_ff}"
          echo "Running $model_id"

          # Run with CPU affinity 0-47
          taskset -c 0-47 python3 -u run.py \
            --is_training 1 \
            --task_name long_term_forecast \
            --root_path ./dataset/EPF/ \
            --data_path $data_path \
            --model_id $model_id \
            --model $model_name \
            --data custom \
            --features MS \
            --seq_len 168 \
            --pred_len 24 \
            --e_layers $e_layers \
            --enc_in 3 \
            --dec_in 3 \
            --c_out 1 \
            --des $des \
            --patch_len $patch_len \
            --d_model $d_model \
            --d_ff $d_ff \
            --batch_size $batch_size \
            --itr 1
        done
      done
    done
  done
done
