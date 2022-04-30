
#!/bin/bash

python3 evaluate.py --network logs/od_1_jacquard/epoch_49_iou_0.89 --dataset jacquard --dataset-path ../../datasets/Jacquard_Dataset/all/ --iou-eval

python3 evaluate.py --network logs/od_1_im_jacquard/epoch_41_iou_0.89 --dataset jacquard --dataset-path ../../datasets/Jacquard_Dataset/all/ --iou-eval

python3 evaluate.py --network logs/od_1_csp_jacquard/epoch_45_iou_0.87 --dataset jacquard --dataset-path ../../datasets/Jacquard_Dataset/all/ --iou-eval

python3 evaluate.py --network logs/od_1_im_csp_jacquard/epoch_49_iou_0.88 --dataset jacquard --dataset-path ../../datasets/Jacquard_Dataset/all/ --iou-eval



python3 evaluate.py --network logs/odr_1_jacquard/epoch_49_iou_0.88 --dataset jacquard --dataset-path ../../datasets/Jacquard_Dataset/all/ --iou-eval

python3 evaluate.py --network logs/odr_1_im_jacquard/epoch_47_iou_0.89 --dataset jacquard --dataset-path ../../datasets/Jacquard_Dataset/all/ --iou-eval

python3 evaluate.py --network logs/odr_1_csp_jacquard/epoch_49_iou_0.88 --dataset jacquard --dataset-path ../../datasets/Jacquard_Dataset/all/ --iou-eval

python3 evaluate.py --network logs/odr_1_im_csp_jacquard/epoch_49_iou_0.89 --dataset jacquard --dataset-path ../../datasets/Jacquard_Dataset/all/ --iou-eval