#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# export CUDA_VISIBL  --batch_size 12 \
#   --n_heads 16 \
#   --train_epochs 20 \
#   --patience 5 \
#   --transfer_type multilevel \
#   --pretrained_model '/root/Codes/code/data/pretrained_global_wind_model' \
#   --freeze_embedding \
#   --freeze_encoder \
#   --progressive_schedule '2:decoder,5:encoder,10:embedding' \
#   --learning_rate 0.00005 \
#   --embedding_lr 0.000005 \
#   --encoder_lr 0.00001 \
#   --decoder_lr 0.00005# 多层级迁移学习：渐进式解冻 + 分层学习率
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
  --des 'Exp_multilevel_transfer' \
  --itr 1 \
  --d_model 768 \
  --batch_size 8 \
  --n_heads 16 \
  --train_epochs 15 \
  --patience 6 \
  --transfer_type multilevel \
  --pretrained_model '/root/Codes/code/data/pretrained_global_wind_model' \
  --freeze_embedding \
  --freeze_encoder \
  --progressive_schedule '3:decoder,7:encoder,12:embedding' \
  --learning_rate 0.0001 \
  --embedding_lr 0.00001 \
  --encoder_lr 0.00005 \
  --decoder_lr 0.0001
