7.21 cubs botton_up的实验
1.cub的lrx1,多seed 五个实验
CUDA_VISIBLE_DEVICES=0 python train.py --config-file /home/workspace/chaohao/ggk/TOAST/visual_classification/configs/cub.yaml MODEL.TRANSFER_TYPE "lora" MODEL.MODEL_ROOT /home/workspace/chaohao/ggk/TOAST/visual_classification/pretrained/vit_b_bottom_up.pth DATA.DATAPATH /data1/chaohao/ggk/CUB_200_2011 OUTPUT_DIR /home/workspace/chaohao/ggk/TOAST/visual_classification/output/CUB/bottom_up_lrx1 DATA.BATCH_SIZE "32" SOLVER.BASE_LR "0.01" MODEL.TYPE "vit_bottom_up" SEED "8" 
2.lrx5，多seed 五个实验
CUDA_VISIBLE_DEVICES=3 python train.py --config-file /home/workspace/chaohao/ggk/TOAST/visual_classification/configs/cub.yaml MODEL.TRANSFER_TYPE "lora" MODEL.MODEL_ROOT /home/workspace/chaohao/ggk/TOAST/visual_classification/pretrained/vit_b_bottom_up.pth DATA.DATAPATH /data1/chaohao/ggk/CUB_200_2011 OUTPUT_DIR /home/workspace/chaohao/ggk/TOAST/visual_classification/output/CUB/bottom_up_lrx5 DATA.BATCH_SIZE "32" SOLVER.BASE_LR "0.05" MODEL.TYPE "vit_bottom_up" SEED "42" 

3.lrx10 5个

CUDA_VISIBLE_DEVICES=5 python train.py --config-file /home/workspace/chaohao/ggk/TOAST/visual_classification/configs/cub.yaml MODEL.TRANSFER_TYPE "lora" MODEL.MODEL_ROOT /home/workspace/chaohao/ggk/TOAST/visual_classification/pretrained/vit_b_bottom_up.pth DATA.DATAPATH /data1/chaohao/ggk/CUB_200_2011 OUTPUT_DIR /home/workspace/chaohao/ggk/TOAST/visual_classification/output/CUB/bottom_up_lrx10 DATA.BATCH_SIZE "32" SOLVER.BASE_LR "0.1" MODEL.TYPE "vit_bottom_up" SEED "42"

4.qk,qv,kv,qkv,mlp
CUDA_VISIBLE_DEVICES=6 python train.py --config-file /home/workspace/chaohao/ggk/TOAST/visual_classification/configs/cub.yaml MODEL.TRANSFER_TYPE "lora" MODEL.MODEL_ROOT /home/workspace/chaohao/ggk/TOAST/visual_classification/pretrained/vit_b_bottom_up.pth DATA.DATAPATH /data1/chaohao/ggk/CUB_200_2011 OUTPUT_DIR /home/workspace/chaohao/ggk/TOAST/visual_classification/output/CUB/qkvmlp DATA.BATCH_SIZE "32" SOLVER.BASE_LR "0.01" MODEL.TYPE "vit_bottom_up" SEED "0"


5.o的seed0 lrx1
CUDA_VISIBLE_DEVICES=4 python train.py --config-file /home/workspace/chaohao/ggk/TOAST/visual_classification/configs/cub.yaml MODEL.TRANSFER_TYPE "lora" MODEL.MODEL_ROOT /home/workspace/chaohao/ggk/TOAST/visual_classification/pretrained/vit_b_bottom_up.pth DATA.DATAPATH /data1/chaohao/ggk/CUB_200_2011 OUTPUT_DIR /data1/chaohao/ggk/output/CUB/qkvomlp DATA.BATCH_SIZE "32" SOLVER.BASE_LR "0.01" MODEL.TYPE "vit_bottom_up" SEED "0"

6.q k v o mlp lrx1
CUDA_VISIBLE_DEVICES=0 python train.py --config-file /home/workspace/chaohao/ggk/TOAST/visual_classification/configs/cub.yaml MODEL.TRANSFER_TYPE "lora" MODEL.MODEL_ROOT /home/workspace/chaohao/ggk/TOAST/visual_classification/pretrained/vit_b_bottom_up.pth DATA.DATAPATH /data1/chaohao/ggk/CUB_200_2011 OUTPUT_DIR /data1/chaohao/ggk/output/CUB/qkvomlp DATA.BATCH_SIZE "32" SOLVER.BASE_LR "0.01" MODEL.TYPE "vit_bottom_up" SEED "0"

7.lrx20
conda activate prompt
CUDA_VISIBLE_DEVICES=5 python train.py --config-file /home/workspace/chaohao/ggk/TOAST/visual_classification/configs/cub.yaml MODEL.TRANSFER_TYPE "lora" MODEL.MODEL_ROOT /home/workspace/chaohao/ggk/TOAST/visual_classification/pretrained/vit_b_bottom_up.pth DATA.DATAPATH /data1/chaohao/ggk/CUB_200_2011 OUTPUT_DIR /data1/chaohao/ggk/output/CUB/lrx20 DATA.BATCH_SIZE "32" SOLVER.BASE_LR "0.2" MODEL.TYPE "vit_bottom_up" SEED "42"