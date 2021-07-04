# %%
import paddle
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import json


# %%

img_name_list = os.listdir('data_old')
data=[]
# 48,60

for img_name in img_name_list:
    img=plt.imread('data_old/'+img_name).transpose([2,0,1])[0]
    data.append({
        'text':img_name[0],
        'img_array':cv2.resize(img,[48,60]).tolist()
    })

# %%
with open('data.json','w') as f:
    json.dump(data,f)