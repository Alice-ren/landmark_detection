import PIL as img
import cv2
from matplotlib import pyplot as plt
import csv
import time
import pandas as pd
import sys
import os
from pandas.io.parsers import read_csv
import numpy as np


def draw_landmark():

    img=cv2.imread("/home/rentingting/project/mtcnn/MTCNN_Caffe/examples/LKSrc/landmark_detection/scripts/imgs/1.jpg")    
    df = pd.read_csv('/home/rentingting/project/mtcnn/MTCNN_Caffe/examples/LKSrc/landmark_detection/scripts/fkp_output.csv', sep=',')
            
    for indexs in df.index:
       x = df.loc[indexs].values[0]
       y = df.loc[indexs].values[1]
       x = int(x)
       y = int(y)
       pos = (x,y)
       cv2.circle(img, pos, 3, color=(0, 0, 255))
       print pos
    cv2.imwrite("result0.jpg", img)
       


if __name__ == '__main__':
   draw_landmark()
 








































 #print df.iloc[0,[0]]
    #print df.iloc[0,[1]]
    #print df.iloc[0,[0,1]]
    #cols = df.columns[:-1] 
    # print cols
    #for indexs in df.index:  
    #  for  i in range(len(df.loc[indexs].values)):  
    #    if(indexs == 0): 
    #        print('******************************') 
    #        print(indexs,i)  
    #        print(df.loc[indexs].values[i])   
    #        print('******************************') 
    #        x = df.loc[indexs].values[i]
    #        y = df.loc[indexs].values[i+1]
    #    
    #        pos = (x,y) 
    #        cv2.circle(im, pos, 3, color=(0, 255, 255))


