import numpy as np
import pandas as pd
import os,sys
from keras import regularizers
from keras.models import Model
from keras.layers import Input,Dense, Dropout,BatchNormalization,Dot,Flatten,Add
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from keras.regularizers import l2
from keras.models import load_model
#def rmse(y_true, y_pred):
#	return np.sqrt(np.sum(np.square(y_pred-y_true))/(y_true.shape[0]))

movie_path = 'data/movies.csv'
test_path = sys.argv[1]

data = pd.read_csv(test_path,engine='python')

userID = Input(shape=[1])
movieID = Input(shape=[1])

emb_user = Embedding(6040,64, embeddings_initializer='random_normal',embeddings_regularizer=l2(0.00001))(userID)
emb_movie = Embedding(3952,64,embeddings_initializer='random_normal',embeddings_regularizer=l2(0.00001))(movieID)
#bias_user = Embedding(6040,1, embeddings_initializer='random_normal',embeddings_regularizer=l2(0.00001))(userID)
#bias_movie = Embedding(3952,1,embeddings_initializer='random_normal',embeddings_regularizer=l2(0.00001))(movieID)

emb_user = Flatten()(emb_user)
emb_movie = Flatten()(emb_movie)
#bias_user = Flatten()(bias_user)
#bias_movie = Flatten()(bias_movie)

rating = Dot(axes=1,normalize=True)([emb_user,emb_movie])
#rating = Add()([rating])
adam = Adam()
model =  Model(inputs=[userID,movieID],outputs=rating)
model.compile( loss= 'mse', optimizer='adam')
print(model.summary())

user = np.array(data['UserID'][:],dtype=np.int32)
movie = np.array(data['MovieID'][:],dtype=np.int32)

model = load_model('model.h5')
print("predict data")
predict = model.predict([user,movie])
print("write data")
with open(sys.argv[2],'w') as o:
	o.write('TestDataID,Rating\n')
	for i in range(predict.shape[0]):
		o.write(str(i+1)+','+str(predict[i][0])+'\n')
print("ok")

