export CUDA_VISIBLE_DEVICES=0
model_name=combined

# Define hyperparameter combinations
d_models=(128 256 512)
batch_sizes=(4 8 16)
e_layers=(1 2)
d_ffs=(1024 2028 512)  # Added d_ff variations

pred_len=96  # Fixed prediction length

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
                  --data_path ETTh1.csv \
                  --model_id "ETTh1_96_${pred_len}" \
                  --model $model_name \
                  --data ETTh1 \
                  --features M \
                  --seq_len 96 \
                  --label_len 48 \
                  --pred_len $pred_len \
                  --e_layers $e_layer \
                  --d_ff $d_ff \
                  --factor 3 \
                  --enc_in 7 \
                  --dec_in 7 \
                  --c_out 7 \
                  --d_model $d_model \
                  --batch_size $batch_size \
                  --des 'exp' \
                  --itr 1 | tee -a results.txt
            done
        done
    done
done
