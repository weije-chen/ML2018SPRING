import numpy as np
import pandas as pd
import os
from keras import regularizers
from keras.models import Model
from keras.layers import Input,Dense, Dropout,BatchNormalization,Dot,Flatten,Add,Multiply
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau,CSVLogger
from keras.regularizers import l2
from keras.models import load_model

#def rmse(y_true, y_pred):
#	return np.sqrt(np.sum(np.square(y_pred-y_true))/(y_true.shape[0]))

user_path = 'data/users.csv'
movie_path = 'data/movies.csv'
train_path = 'data/train.csv'
test_path = 'data/test.csv'

data = pd.read_csv(train_path,engine='python')
ratio = int(data.shape[0]*0.1)

user = np.array(data['UserID'][:],dtype=np.int32)
movie = np.array(data['MovieID'][:],dtype=np.int32)
rate = np.array(data['Rating'][:],dtype=np.int32)
#print(data)

X_val = []
Y_val = []
Z_val = []
X = []
Y = []
Z = []

for i in range(user.shape[0]):
	if i%10==1.0:
		X_val.append(user[i])
		Y_val.append(movie[i])
		Z_val.append(rate[i])
	else:
		X.append(user[i])
		Y.append(movie[i])
		Z.append(rate[i])
X_val = np.array(X_val)
Y_val = np.array(Y_val)
Z_val = np.array(Z_val)
X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)

print(X.shape)
print(Y.shape)
print(Z.shape)

print(X_val.shape)
print(Y_val.shape)
print(Z_val.shape)




def build_model():
	userID = Input(shape=[1])
	movieID = Input(shape=[1])
	emb_user = Embedding(6040,64, embeddings_initializer='random_normal',embeddings_regularizer=l2(0.00001))(userID)
	emb_movie = Embedding(3952,64,embeddings_initializer='random_normal',embeddings_regularizer=l2(0.00001))(movieID)
	bias_user = Embedding(6040,1, embeddings_initializer = 'zeros',embeddings_regularizer=l2(0.00001))(userID)
	bias_movie = Embedding(3952,1,embeddings_initializer = 'zeros',embeddings_regularizer=l2(0.00001))(movieID)

	emb_user = Flatten()(emb_user)
	emb_movie = Flatten()(emb_movie)
	bias_user = Flatten()(bias_user)
	bias_movie = Flatten()(bias_movie)

	#emb_movie = Dropout(0.5)(emb_movie)
	#emb_user = Dropout(0.5)(emb_user)


	rating = Dot(axes=1,normalize=True)([emb_user,emb_movie])
	#rating_sqr = Dot(axes=1,normalize=True)([np.square(emb_user),np.square(emb_movie)])

	rating = Add()([rating,bias_movie,bias_user])
	adam = Adam()
	model =  Model(inputs=[userID,movieID],outputs=rating)
	model.compile( loss= 'mse', optimizer='adam')
	print(model.summary())

	return model

def train(model):
	earlystopping = EarlyStopping(monitor='loss', patience = 5, verbose=0, mode='min')
	#lr_reducer = ReduceLROnPlateau(factor=0.1, cooldown=0, patience=2, min_lr=0.5e-6)
	checkpoint = ModelCheckpoint(filepath='./model.h5', 
								 verbose=1,
								 save_best_only=True,
								 save_weights_only=False,
								 monitor='loss',
								 mode='min' )
	csv_logger = CSVLogger('./model.csv')

	ratio = int(data.shape[0]*0.1)

	history = model.fit([X,Y], Z,
						validation_data=([X_val,Y_val], Z_val),
						epochs=1000, 
						verbose=1,
						batch_size=256,
						callbacks=[checkpoint,csv_logger,earlystopping] )


def test():
	print(X_val[:50])
	print(Y_val[:50])
	print(Z_val[:50])
	model = load_model('model.h5')
	#pred = model.predict([X[:1000],Y[:1000]])

	#score = np.sqrt(((pred - Z[:1000])**2).mean())

	#print('training data is ',score)
	pred = model.predict([X_val,Y_val])

	score = np.sqrt(((pred - Z_val)**2).mean())
	print('\n-----------------')
	print('rmse is %f',score)
	print('-----------------\n')




if __name__ == '__main__':
	model = build_model()
	train(model)
	test()

