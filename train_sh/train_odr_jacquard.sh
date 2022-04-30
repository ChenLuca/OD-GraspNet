#!/bin/bash

network_name='odr_1'
dataset='jacquard'
datset_path='../../datasets/Jacquard_Dataset/all/'
python3 train_network.py --network ${network_name} --dataset ${dataset} --dataset-path ${datset_path} --description ${network_name}_${dataset} --use-dropout 0

network_name='odr_2'
dataset='jacquard'
datset_path='../../datasets/Jacquard_Dataset/all/'
python3 train_network.py --network ${network_name} --dataset ${dataset} --dataset-path ${datset_path} --description ${network_name}_${dataset} --use-dropout 0

network_name='odr_3'
dataset='jacquard'
datset_path='../../datasets/Jacquard_Dataset/all/'
python3 train_network.py --network ${network_name} --dataset ${dataset} --dataset-path ${datset_path} --description ${network_name}_${dataset} --use-dropout 0

network_name='odr_4'
dataset='jacquard'
datset_path='../../datasets/Jacquard_Dataset/all/'
python3 train_network.py --network ${network_name} --dataset ${dataset} --dataset-path ${datset_path} --description ${network_name}_${dataset} --use-dropout 0
