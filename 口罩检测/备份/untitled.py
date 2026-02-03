# 1.加载数据并进行数据处理
import warnings
warnings.filterwarnings('ignore')
import cv2
from PIL import Image
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torchvision

# 数据集路径
data_path = "./datasets/5f680a696ec9b83bb0037081-momodel/data/image"

def processing_data(data_path, height=224, width=224, batch_size=32, test_split=0.1):
    transforms = T.Compose([
        T.Resize((height, width)),
        T.RandomHorizontalFlip(0.1),
        T.RandomVerticalFlip(0.1),
        T.ToTensor(),
        T.Normalize([0], [1]),
    ])

    dataset = ImageFolder(data_path, transform=transforms)
    train_size = int((1-test_split)*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    valid_data_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True)
    return train_data_loader, valid_data_loader

train_data_loader, valid_data_loader = processing_data(data_path=data_path, height=160, width=160, batch_size=64)

# 2.如果有预训练模型，则加载预训练模型；如果没有则不需要加载
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# 加载 ResNet50 的预训练模型
model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# 3.创建模型和训练模型，训练模型时尽量将模型保存在 results 文件夹
epochs = 30
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10)
criterion = nn.CrossEntropyLoss()

best_loss = 1e9
best_model_weights = copy.deepcopy(model.state_dict())
loss_list = []

print('开始训练...')
for epoch in range(epochs):
    model.train()
    for batch_idx, (x, y) in tqdm(enumerate(train_data_loader, 1)):
        x = x.to(device)
        y = y.to(device)
        pred_y = model(x)
        loss = criterion(pred_y, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss < best_loss:
            best_model_weights = copy.deepcopy(model.state_dict())
            best_loss = loss
            
        loss_list.append(loss)

    print('step:' + str(epoch + 1) + '/' + str(epochs) + ' || Total Loss: %.4f' % (loss))

# 保存最佳模型
torch.save(best_model_weights, './results/temp.pth', _use_new_zipfile_serialization=False)
print('Finish Training.')

# 4.评估模型，将自己认为最佳模型保存在 result 文件夹，其余模型备份在项目中其它文件夹，方便您加快测试通过。
loss_val = [index.item() for index in loss_list]
plt.plot(loss_val, label="loss")
plt.legend()
plt.show()