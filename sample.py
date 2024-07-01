import os 
import numpy as np
import pandas as pd
# fakeimages = os.listdir("data\\DeepFake")
# realimages = os.listdir("data\\Real")

# fakeLabel = ["FAKE"]*len(fakeimages)
# realLabel = ["REAL"]*len(realimages)

# # print(len(set(fakeimages))==len(fakeimages))
# # print(len(set(realimages))==len(realimages))
# # print(set(fakeimages).intersection(set(realimages)))

# combinedImages = fakeimages + realimages
# # print(len(combinedImages)== len(fakeimages)+len(realimages))
# combinedLabels = fakeLabel+realLabel
# # print(len(combinedLabels)== len(fakeLabel)+len(realLabel))

# combinedData = np.array([np.array(combinedImages),np.array(combinedLabels)]).T
# # print(combinedData)
# df = pd.DataFrame(combinedData, columns=['image', 'label'])
# print(df.head())
# df = df.sample(frac=1).reset_index(drop=True)
# print(df.head())
# df.to_csv("data.csv", index=False)

import cv2
import matplotlib.pyplot as plt
data = pd.read_csv("data.csv")
print(len(data)//5)

# for i in os.listdir("images"):
#     image = cv2.imread("images\\"+i)
#     print(image)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     print(image)
#     # plt.imshow(image)
#     plt.show()
#     break

def custom_train_test_split(df, test_ratio = 0.2):
    randomLst = list(range(len(df)))
    count_train = len(df) - int(len(df)*test_ratio)
    xtrain = []
    ytrain = []
    xtest = []
    ytest = []
    idx = 0
    while idx<count_train:
        i = np.random.choice(randomLst)
        randomLst.pop(randomLst.index(i))
        image = cv2.imread("images\\"+df.iloc[i]['image'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        xtrain.append(image)
        ytrain.append(df.iloc[i]['label'])
        idx+=1
    print(len(randomLst))
    for i in randomLst:
        image = cv2.imread("images\\"+df.iloc[i]['image'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        xtest.append(image)
        ytest.append(df.iloc[i]['label'])
    return xtrain, ytrain, xtest, ytest
Xtr, Ytr, Xte, Yte =  custom_train_test_split(data)

# print(Ytr)
# print()
# print(Yte)