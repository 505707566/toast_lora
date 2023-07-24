import torch
import numpy as np
from sklearn.metrics import accuracy_score
import os
from scipy.special import softmax

# 获取所有的.pth文件
folder_path = "/home/workspace/chaohao/ggk/TOAST/visual_classification/output/ensemble_cub/qkvmlp"
pth_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pth')]

# 加载所有.pth文件
logits_list = []
targets_list = []

for filename in pth_files:
    data = torch.load(filename)
    logits_list.append(data['joint_logits'])
    targets_list.append(data['targets'])

# 对每个logits进行softmax
softmax_logits_list = [softmax(logits, axis=1) for logits in logits_list]

# 将所有softmax后的logits加总
total_softmax_logits = np.sum(softmax_logits_list, axis=0)

# 计算平均值
average_softmax_logits = total_softmax_logits / len(softmax_logits_list)

# 将softmax后的logits转换为类别预测
predictions_softmax = np.argmax(average_softmax_logits, axis=1)

# 加载真实的类别标签
# 假设所有.pth文件的targets都是一样的，所以我们只需要取第一个文件的targets即可
true_labels = targets_list[0]

# 计算分类准确率
accuracy_softmax = accuracy_score(true_labels, predictions_softmax)
print('Classification accuracy after softmax: ', accuracy_softmax)
