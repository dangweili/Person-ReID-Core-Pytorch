#!/usr/bin/env python

python ./script/experiment/train_hacnn_cls.py \
    --sys_device_ids="(1,)" \
    --dataset="rap2" \
    --split=trainval \
    --test_split=test \
    --batch_size=32 \
    --resize="(160,64)" \
    --workers=2 \
    --optimize_adam=True \
    --sgd_momentum=0.9 \
    --staircase_decay_at_epochs="(150, 225)" \
    --base_lr=0.0005 \
    --total_epochs=150 \
    --exp_decay_at_epoch=1 \
    --epochs_per_val=5 \
    --epochs_per_save=50 \
    --eval_type="['sq']" \
    --run=1 \
    --resume=False \
    --ckpt_file="" \
    --test_only=False \
    --model_weight_file=""
