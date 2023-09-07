import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from torchvision.datasets import YourDataset  # あなたのデータセットに置き換えてください
from torchvision.models import resnet18  # または他のモデルを選択してください
from PIL import Image

import glob
import numpy as np
import cv2
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset

from sklearn.metrics import confusion_matrix
import random

import matplotlib.pyplot as plt

# 乱数固定
seed=42
# Python random
random.seed(seed)
# Numpy
np.random.seed(seed)
# Pytorch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True

# ハイパーパラメータ
batch_size = 4
learning_rate = 0.001
epochs = 100

# データの前処理
transform = transforms.Compose([
    transforms.Resize((255, 255)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 画像の正規化
])

from torch.utils.data import Dataset
class YakinikuDataset(Dataset):
    def __init__(self, transform=None):
        ng_path_list = glob.glob("yaketenai/*")
        ok_path_list = glob.glob("yaketeru/*")
        self.path_list = ng_path_list + ok_path_list
        ng_labels = np.zeros(len(ng_path_list))
        ok_labels = np.ones(len(ok_path_list))
        self.labels = np.concatenate((ng_labels, ok_labels))
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.path_list[idx]).convert("RGB")  # 画像をRGB形式で読み込み

        if self.transform:
            image = self.transform(image)
        return image, int(self.labels[idx])


# ニューラルネットワークモデルの定義（ResNet-18を例として使用）
model = resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 2)  # 2クラス分類なので出力ユニット数は2
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 損失関数とオプティマイザの定義
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

yakiniku_dataset = YakinikuDataset(transform=transform)
all_cm=np.array([[0,0],[0,0]])
kf = KFold(n_splits=5, shuffle=True)
for _fold, (train_index, valid_index) in enumerate(kf.split(range(len(yakiniku_dataset)))):
    print(F"valid [{_fold+1}/5]")
    train_dataset = Subset(yakiniku_dataset, train_index)
    valid_dataset = Subset(yakiniku_dataset, valid_index)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(valid_dataset, 1, shuffle=False)
    
    train_loss_list = []
    valid_loss_list = []

    # 学習ループ
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 検証ループ
        valid_loss = 0.0
        model.eval()
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels)
            valid_loss += loss.item()

        train_loss_list.append(train_loss / len(train_loader))
        valid_loss_list.append(valid_loss / len(test_loader))
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss / len(train_loader)}, Valid Loss: {valid_loss / len(test_loader)}")

    # テストループ
    model.eval()
    correct = 0
    total = 0
    pred_array = np.array([])
    label_array = np.array([])
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pred = predicted.to('cpu').detach().numpy().copy()
            pred_array = np.concatenate((pred_array, pred))
            labels = labels.to('cpu').detach().numpy().copy()
            label_array = np.concatenate((label_array, labels))
            
    cm=confusion_matrix(label_array,pred_array)
    all_cm+=cm
    torch.save(model.state_dict(), 'cnn_model.pth')
tn, fp, fn, tp = all_cm.flatten()

print("accuracy:",(tp+tn)/(tp+tn+fp+fn))
print("precision:",tp/(tp+fp))
print("recall:",tp/(tp+fn))
print("F1-measure:",(2*tp)/(2*tp+fp+fn))
print("混同行列\n",all_cm)

plt.plot(train_loss_list)
plt.plot(valid_loss_list)
plt.show()
