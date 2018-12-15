#!/usr/bin/env python

python ./script/experiment/train_apr_cls.py \
    --sys_device_ids="(1,)" \
    --dataset="rap2" \
    --split=trainval \
    --test_split=test \
    --batch_size=32 \
    --resize="(224,224)" \
    --workers=4 \
    --optimize_adam=False \
    --sgd_momentum=0.9 \
    --staircase_decay_at_epochs="(50, 100)" \
    --base_lr=0.001 \
    --total_epochs=125 \
    --exp_decay_at_epoch=1 \
    --epochs_per_val=5 \
    --epochs_per_save=25 \
    --loss_att_weight=1.0 \
    --eval_type="['sq']" \
    --run=14 \
    --resume=False \
    --ckpt_file="" \
    --test_only=False \
    --model_weight_file=""
