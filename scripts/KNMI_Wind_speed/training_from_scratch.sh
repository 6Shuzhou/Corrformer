#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --is_training 1 \
  --root_path /root/Codes/code/data/knmi_wind_speed/ \
  --data_path None \
  --model_id knmi_wind_speed_48_24_1ETCN_1DTCN \
  --model Corrformer \
  --data Knmi_Wind_speed \
  --features M \
  --seq_len 48 \
  --label_len 24 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 1 \
  --factor_temporal 1 \
  --factor_spatial 1 \
  --enc_tcn_layers 1 \
  --dec_tcn_layers 1 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --node_num 33 \
  --node_list 3,11 \
  --des 'Exp_training_from_scratch' \
  --itr 1 \
  --d_model 768 \
  --batch_size 12 \
  --n_heads 16 \
  --train_epochs 10\
  --patience 3 \
  --transfer_type none \
  --learning_rate 0.00001
