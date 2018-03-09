import os
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import h5py

TRAIN_CSV = '/home/rentingting/data/kaggle/training.csv'
def csv_to_hd5():
	dataframe = read_csv(os.path.expanduser(TRAIN_CSV))
	dataframe['Image'] = dataframe['Image'].apply(lambda img: np.fromstring(img,sep = ' '))
	dataframe = dataframe.dropna()
	data = np.vstack(dataframe['Image'].values) / 255.
	
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


