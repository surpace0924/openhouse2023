import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from PIL import Image

import glob
import numpy as np
import cv2

# データの前処理
transform = transforms.Compose([
    transforms.Resize((255, 255)),  # 画像のサイズを[255, 255]にリサイズ
    transforms.ToTensor(),  # 画像をテンソルに変換
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 画像の正規化
])

# ニューラルネットワークモデルの定義（ResNet-18を例として使用）
model = resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 2)  # 2クラス分類なので出力ユニット数は2
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 保存された重みを読み込む
model.load_state_dict(torch.load('cnn_model.pth'))


def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


#カメラの設定　デバイスIDは0
cap = cv2.VideoCapture(0)

#繰り返しのためのwhile文
while True:
    #カメラからの画像取得
    _, img = cap.read()
    img = cv2.resize(img, (255, 255))

    image = cv2pil(img)
    image = transform(image).contiguous().view(1, 3, 255, 255)
    image = image.to(device)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    if int(predicted[0]) == 0:
        result = "NG"
    else:
        result = "OK"

    cv2.putText(img, result, (0, 50), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 5, cv2.LINE_AA)
    #カメラの画像の出力
    cv2.imshow('camera' , img)

    #繰り返し分から抜けるためのif文
    key =cv2.waitKey(10)
    if key == 27:
        break

#メモリを解放して終了するためのコマンド
cap.release()
cv2.destroyAllWindows()
