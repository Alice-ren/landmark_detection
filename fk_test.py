import numpy as np
import pandas as pd
import caffe

MODEL_FILE ='./fk_deploy.prototxt'
PRETRAINED ='./caffe_landmark_train_iter_5000.caffemodel'

dataframe = pd.read_csv('/home/rentingting/data/kaggle/test.csv',header = 0)
dataframe['Image'] = dataframe['Image'].apply(lambda im:np.fromstring(im,sep=' '))
data = np.vstack(dataframe['Image'].values)
data = data.reshape([-1,96,96])
data = data.astype(np.float32)

data = data/255.
data = data.reshape(-1,1,96,96)

net = caffe.Net(MODEL_FILE,PRETRAINED,caffe.TEST)
caffe.set_mode_gpu()

total_images = data.shape[0]
print 'total images to be predicted:',total_images

dataL = np.zeros([total_images,1,1,1],np.float32)

net.set_input_arrays(data.astype(np.float32),dataL.astype(np.float32))
pred = net.forward()
predicted = net.blobs['ip2'].data*96

print 'predicted',predicted

print 'predicted shape:',predicted.shape
print 'saving to csv..'

np.savetxt('fkp_output.csv',predicted,delimiter =",")

