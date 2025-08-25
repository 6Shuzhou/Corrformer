#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# 简单迁移学习：直接加载预训练权重并全参数微调
python -u run.py \
  --is_training 1 \
  --root_path /root/Codes/code/data/knmi_humidity/ \
  --data_path None \
  --model_id knmi_humidity_48_24_1ETCN_1DTCN \
  --model Corrformer \
  --data Knmi_Humidity \
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
  --node_num 32 \
  --node_list 4,8 \
  --des 'Exp_simple_transfer' \
  --itr 1 \
  --d_model 768 \
  --batch_size 8 \
  --n_heads 16 \
  --train_epochs 5 \
  --patience 1 \
  --transfer_type simple \
  --pretrained_model '/root/Codes/code/data/pretrained_global_temp_model' \
  --learning_rate 0.0001
