# -*- coding: utf-8 -*- 
import cv2
import pickle
from sklearn.svm import SVC
import numpy as np

def get_feature_vector(imgs):
    std_list, arg_list = [], []
    for img in imgs:
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Red=cv2.calcHist([im_rgb],[0],None,[256],[0,256])
        for i in range(50):
            Red[i]=0
        std_list.append(np.std(Red))
        arg_list.append(np.argmax(Red))
    X = np.array([arg_list, std_list])
    X = X.T
    return X


def model(img):
    with open("model.pickle",mode="rb") as f:
        clf=pickle.load(f)
    
    imgs = np.array([img])
    print(imgs.shape)
    X = get_feature_vector(imgs)
    print(X)

    y_pred = clf.predict(X)
    print(y_pred)   
    if y_pred[0]==0:
        return "NG"
    else:
        return "OK"

#カメラの設定　デバイスIDは0
cap = cv2.VideoCapture(0)

#繰り返しのためのwhile文
while True:
    #カメラからの画像取得
    _, img = cap.read()
    img = cv2.resize(img, (255, 255))

    result = model(img)
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
