#!/bin/bash
#train the hand segmentation task
#utilize TensorFlow Object Detection API("train.py")

python train.py --logtostderr --train_dir=/home/zhc/projects/netease/FreashBird/segmentation/handsegmentation/training/training_log/ --pipeline_config_path=/home/zhc/projects/netease/FreashBird/segmentation/handsegmentation/training/ssd_mobilenet_v1_hand.config
