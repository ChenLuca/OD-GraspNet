#!/bin/bash

network_name='od_1_csp'
dataset='jacquard'
datset_path='../../datasets/Jacquard_Dataset/all/'
python3 train_network.py --network ${network_name} --dataset ${dataset} --dataset-path ${datset_path} --description ${network_name}_${dataset} --use-dropout 0

network_name='od_2_csp'
dataset='jacquard'
datset_path='../../datasets/Jacquard_Dataset/all/'
python3 train_network.py --network ${network_name} --dataset ${dataset} --dataset-path ${datset_path} --description ${network_name}_${dataset} --use-dropout 0

network_name='od_3_csp'
dataset='jacquard'
datset_path='../../datasets/Jacquard_Dataset/all/'
python3 train_network.py --network ${network_name} --dataset ${dataset} --dataset-path ${datset_path} --description ${network_name}_${dataset} --use-dropout 0

network_name='od_4_csp'
dataset='jacquard'
datset_path='../../datasets/Jacquard_Dataset/all/'
python3 train_network.py --network ${network_name} --dataset ${dataset} --dataset-path ${datset_path} --description ${network_name}_${dataset} --use-dropout 0
