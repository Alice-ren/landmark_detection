import csv 
import time
import pandas as pd
import numpy as np
from PIL import Image
import sys
import csv
import os
from pandas.io.parsers import read_csv
#
#img_num = 0
##with open("/home/rentingting/data/kaggle/test.csv","r") as csvfile_img:
#with open("/home/rentingting/project/mtcnn/MTCNN_Caffe/examples/LKSrc/landmark_detection/scripts/fkp_output.csv","r") as csvfile_img:
#    reader_img = csv.reader(csvfile_img)
#    img_somerow = pd.read_csv('/home/rentingting/project/mtcnn/MTCNN_Caffe/examples/LKSrc/landmark_detection/scripts/fkp_output.csv',nrows = 10)
#    print img_somerow
#    img_id = pd.read_csv('/home/rentingting/data/kaggle/test.csv',usecols=[0])
#    img_data = pd.read_csv('/home/rentingting/data/kaggle/test.csv',usecols=[1])
#    print img_data
#    print img_id
# #   for rows in img_data:
#   # for i,rows in enumerate(img_data):
##        if i == img_num:
##            arry = rows
##            img_arry = np.array(arry)
##            print(img_arry)
##        img_arry = img_arry.reshape([48,48])
##        img = Image.fromarray(img_arry)
##        img = img.resize([384,384],Img.ANTIALIAS)
##        img.save("/home/rentingting/project/mtcnn/MTCNN_Caffe/examples/LKSrc/landmark_detection/scripts/image.jpg")
##        print("img saved ...")
##    img_num = img_num +  1    
#    print(img_num)
#
##
  
#convert.py version

TRAIN_CSV = '/home/rentingting/data/kaggle/training.csv'
def csv_getimage():
        dataframe = read_csv(os.path.expanduser(TRAIN_CSV))
        dataframe['Image'] = dataframe['Image'].apply(lambda img: np.fromstring(img,sep = ' '))
        dataframe = dataframe.dropna()
        dataframe[['1','1']]  #choose w row z col of the dataframe
        data = np.vstack(dataframe['Image'].values)
        
        label =dataframe[dataframe.columns[:-1]].values
        label = (label-48) / 48
        data,label = shuffle(data,label,random_state = 0)

        return data,label

if __name__ == '__main__':
        data,label = csv_to_hd5()
        data = data.reshape(-1,1,96,96)
        data_train = data[:-100,:,:,:]
        data_val = data[-100:,:,:,:]

        label=label.reshape(-1,1,1,30)
        label_train=label[:-100,:,:,:]
        label_val=label[-100:,:,:,:]

        fhandle =h5py.File('train.hd5','w')
        fhandle.create_dataset('data',data=data_train,compression='gzip',compression_opts=4)
        fhandle.create_dataset('label',data=label_train,compression='gzip',compression_opts=4)
        fhandle.close()

        fhandle =h5py.File('val.hd5','w')
        fhandle.create_dataset('data',data=data_train,compression='gzip',compression_opts=4)
        fhandle.create_dataset('label',data=label_train,compression='gzip',compression_opts=4)
        fhandle.close()

def station_test_data(csvfile):
    data = pd.read_csv(csvfile)
    data[data==9999]=np.nan
    dataset = data
    
    return dataset
def draw_csv(csvfile,varname,outdir="."): 
    dataset = station_test_data(csvfile)
    keys =["TEM","RHU","PRE_1h","WIN_S_Max","GST"] 
    lon = dataset['Lon'].values 
    lat = dataset['Lat'].values 
    limit = [120,136,40,54]  
    llon ,llat,vvalue=[],[],[]
    for varname in keys:
        print 'var=',varname
        value = dataset[varname].values
        for ii in range(len(value)):
            if np.float(value[ii]) > -500 and np.float(value[ii])<500:
                vvalue.append(value[ii])
                llon.append(lon[ii])
                llat.append(lat[ii])  

        x,y,z = np.asarray(llon),np.asarray(llat),np.asarray(vvalue) 



def csv_match(id_list,key,input_file,output_file):
    with open(input_file, 'rb') as f:
        reader = csv.DictReader(f)
        img_id = pd.read_csv('/home/rentingting/data/kaggle/test.csv',usecols=[0])
        landmark = pd.read_csv('/home/rentingting/project/mtcnn/MTCNN_Caffe/examples/LKSrc/landmark_detection/scripts/fkp_output.csv',usecols[1])
        print landmark
        print img_id
        print img_data
        
        for row in reader:
          
        #   print row  
         # print(row['first_name'], row['last_name'])
        #rows = [row for row in reader if row[key] in set(id_list)]
          
#    header = rows[0].keys()
 #   print header
    with open(output_file, 'w') as f:
        f.write(','.join(header))
        f.write('\n')
        for data in rows:
            f.write(",".join(data[h] for h in header))
            f.write('\n')

def csv_match2(refer_list, key1, key2, input_file, output_file):
    with open(input_file, 'rb') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if (row[key1] in set(refer_list)) or (row[key2] in set(refer_list))]

    header = rows[0].keys()
    with open(output_file, 'w') as f:
        f.write(','.join(header))
        f.write('\n')
        for data in rows:
            f.write(",".join(data[h] for h in header))
            f.write('\n')

TEST_CSV = ''
def draw_landmark(): 
    os.path.expanduser()  
    
if __name__ == '__main__':
   
   lst=['23']
   csv_match(lst,'ImageID','/home/rentingting/data/kaggle/test.csv','/home/rentingting/project/mtcnn/MTCNN_Caffe/examples/LKSrc/landmark_detection/scripts/fkp_output.csv')
#   csv_match2(lst,'ImageID','Image','/home/rentingting/data/kaggle/test.csv','/home/rentingting/project/mtcnn/MTCNN_Caffe/examples/LKSrc/landmark_detection/scripts/fkp_output.csv')
   
