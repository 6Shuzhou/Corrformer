#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# 更激进的多层级迁移学习：更早解冻 + 更平衡的学习率
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
  --des 'Exp_multilevel_aggressive' \
  --itr 1 \
  --d_model 768 \
  --batch_size 12 \
  --n_heads 16 \
  --train_epochs 15 \
  --patience 3 \
  --transfer_type multilevel \
  --pretrained_model '/root/Codes/code/data/pretrained_global_wind_model' \
  --freeze_embedding \
  --freeze_encoder \
  --progressive_schedule '1:decoder,3:encoder,6:embedding' \
  --learning_rate 0.00001 \
  --embedding_lr 0.000005 \
  --encoder_lr 0.000008 \
  --decoder_lr 0.00001
