import os
import re

def get_epoch_from_filename(filename):
    # 从文件名中提取epoch数
    match = re.search('epoch(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return None

def main():
    target_path = "/home/workspace/chaohao/ggk/TOAST/visual_classification/output/lrx5/StanfordCars/vit_fb_ppt_small_patch16_224/lr0.05_wd0.0001/run2"  # 将这里替换为你的目标路径

    model_files = []
    logits_files = []
    for filename in os.listdir(target_path):
        if filename.endswith(".pth"):
            if 'ckpt' in filename:
                model_files.append(filename)
            elif 'logits' in filename:
                logits_files.append(filename)

    # 对文件名列表按epoch数排序
    model_files.sort(key=get_epoch_from_filename)
    logits_files.sort(key=get_epoch_from_filename)

    # 删除除最后一个以外的所有.pth文件
    for filename in model_files[:-1]:
        os.remove(os.path.join(target_path, filename))
    for filename in logits_files[:-1]:
        os.remove(os.path.join(target_path, filename))

if __name__ == "__main__":
    main()
