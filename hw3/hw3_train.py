

import pandas as pd
import numpy as np
import sys
from keras.models import Sequential,load_model
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Activation, Conv2D, Dropout, AveragePooling2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras import regularizers

def read_data():
	raw = pd.read_csv(sys.argv[1])

	label = raw.iloc[:,0]
	data = raw.iloc[:,1]
	
	
	data = data.str.split(expand = True).astype('float32').values
	

	return data.reshape(-1,48,48,1),np_utils.to_categorical(label, 7)

def val(data,label):
	num = label.shape[0]
	index = np.random.permutation(num)
	x = data[:int(num*0.9)]
	y = label[:int(num*0.9)]
	val_x = data[int(num*0.9):]
	val_y = label[int(num*0.9):]
	return x,y,val_x,val_y





def training(x,y,val_x,val_y,num=0):
	
	model = Sequential()
	datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.1 , shear_range=0.1, horizontal_flip=True)
	model.add(Conv2D(80, kernel_size=(5, 5), input_shape=x.shape[1:4], padding='valid'))
	model.add(LeakyReLU(1./30))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.25))

	

	model.add(Conv2D(160, kernel_size=(3, 3), padding='same'))
	model.add(LeakyReLU(1./30))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.3))

	model.add(Conv2D(500, kernel_size=(5, 5), padding='same'))
	model.add(LeakyReLU(1./30))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.35))

	model.add(Conv2D(500, kernel_size=(3, 3), padding='same'))
	model.add(LeakyReLU(1./30))
	model.add(BatchNormalization())
	model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
	model.add(Dropout(0.4))

	model.add(Flatten())


	model.add(Dense(500, activation='relu'))
	model.add(LeakyReLU(1./30))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))


	model.add(Dense(500, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	model.add(Dense(7, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


	lr_reducer = ReduceLROnPlateau(factor=0.9, cooldown=0, patience=3, min_lr=10**(-5))
	earlyStopping = EarlyStopping(monitor='val_acc', patience=15, verbose=0, mode='auto')
	checkpointer = ModelCheckpoint(filepath='model'+str(num)+'_{epoch:05d}_{val_acc:.5f}.h5', save_best_only=True,period=1,monitor='val_acc')

	#lr_reducer = ReduceLROnPlateau(factor=0.1, cooldown=0, patience=5, min_lr=0.5e-6)
	#earlyStopping = EarlyStopping(monitor='val_acc', patience=8, verbose=0, mode='auto')
	#checkpointer = ModelCheckpoint(filepath='model_{epoch:05d}_{val_acc:05f}.h5', save_best_only=True,period=1,monitor='val_acc')

	print('Model fitting!')

	history = model.fit_generator(datagen.flow(x, y, batch_size=200), steps_per_epoch=x.shape[0]//200+1, epochs=8000, validation_data=(val_x, val_y), max_queue_size=100, callbacks=[lr_reducer, earlyStopping, checkpointer])
	loss, accuracy = model.evaluate(val_x, val_y)
	print('test loss: ',loss)
	print('test accuracy: ',accuracy)

	#model.save('model'+str(num)+'.h5')


def test():
	raw = pd.read_csv("test.csv")
	data = raw.iloc[:,1]
	data = data.str.split(expand = True).astype('float32').values
	data = data.reshape(-1,48,48,1)

	model = load_model('model.h5')

	y = model.predict(data)
	ans = np.argmax(y,axis=1)

	with open('result.csv','w') as output:
		output.write('id,label\n')
		for i in range(y.shape[0]):
			output.write(str(i)+','+str(ans[i])+'\n')


if __name__ == '__main__':
	data, label = read_data()
	train_x,train_y,val_x,val_y = val(data,label)
	#print(train_x.shape)
	for i in range(10):
		training(train_x,train_y,val_x,val_y,i)
	#test()
