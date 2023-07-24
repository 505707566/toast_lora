import torch

# 加载模型
checkpoint = torch.load('/home/workspace/chaohao/ggk/TOAST/visual_classification/output/lrx2/StanfordCars/vit_fb_ppt_small_patch16_224/lr0.02_wd0.0001/run8/val_StanfordCars_logits_87.pth')

# 提取模型的状态字典
# model_state_dict = checkpoint['joint_logits']
model_state_dict = checkpoint['targets']
# print(checkpoint.keys())
# 打印状态字典
for key in model_state_dict:
    print(key)