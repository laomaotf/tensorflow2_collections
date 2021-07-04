# %%
from cv2 import data
import matplotlib.pyplot as plt
import cv2
import json
import numpy as np

# %%
text_img=plt.imread('data/国标一级汉字（3755个，按拼音排序）.jpg')
text_img=text_img.transpose([2,0,1])[0]

# %%
y=[sum(x) for x in text_img]
c=[i for i in range(len(y)) if y[i]>1210000]
cc=[(c[i],c[i+1]) for i in range(len(c)-1)]
cc_x=[x for x in cc if (x[1]-x[0])>50]

y=[sum(x) for x in text_img.transpose([1,0])]
c=[i for i in range(len(y)) if y[i]>1705000]
cc=[(c[i],c[i+1]) for i in range(len(c)-1)]
cc_y=[x for x in cc if (x[1]-x[0])>50]

# %%
def get_img_by_id(i):
    # rows and columns
    row=int((i)/56)
    column=i % 56
    c_row=cc_x[row]
    c_column=cc_y[column]
    img=text_img[c_row[0]:c_row[1]+1]
    img=img.transpose([1,0])
    img=img[c_column[0]:c_column[1]+1]
    img=img.transpose([1,0])
    
    return img

# %%

text_list=[]
with open("data/国标一级汉字（3755个，按拼音排序）.txt",'r') as f:
    text_list=list(f.read())

# %%

data=[]
for i in range(3755):
    data.append({
        'text':text_list[i],
        'id':i,
        'img_array':cv2.resize(get_img_by_id(i),[64,64]).tolist()
    })

# %%

with open('data.json','w') as f:
    json.dump(data,f)

# %%

with open('data.json','r') as f:
    data=json.load(f)