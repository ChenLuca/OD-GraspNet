#!/bin/bash

network_name='odr_1_im'
dataset='jacquard'
datset_path='../../datasets/Jacquard_Dataset/all/'
python3 train_network.py --network ${network_name} --dataset ${dataset} --dataset-path ${datset_path} --description ${network_name}_${dataset} --use-dropout 0

network_name='odr_2_im'
dataset='jacquard'
datset_path='../../datasets/Jacquard_Dataset/all/'
python3 train_network.py --network ${network_name} --dataset ${dataset} --dataset-path ${datset_path} --description ${network_name}_${dataset} --use-dropout 0

network_name='odr_3_im'
dataset='jacquard'
datset_path='../../datasets/Jacquard_Dataset/all/'
python3 train_network.py --network ${network_name} --dataset ${dataset} --dataset-path ${datset_path} --description ${network_name}_${dataset} --use-dropout 0

network_name='odr_4_im'
dataset='jacquard'
datset_path='../../datasets/Jacquard_Dataset/all/'
python3 train_network.py --network ${network_name} --dataset ${dataset} --dataset-path ${datset_path} --description ${network_name}_${dataset} --use-dropout 0
