import sys, argparse, os
import keras
import _pickle as pk
import numpy as np
import pandas as pd

from keras import regularizers
from keras.models import Model
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional,BatchNormalization,Masking
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from gensim.models import word2vec
import keras.backend.tensorflow_backend as K
import re
from keras.layers.advanced_activations import LeakyReLU


parser = argparse.ArgumentParser(description='Sentiment classification')
parser.add_argument('action', choices=['train','test','semi'])

# training argument
parser.add_argument('--batch_size', default=512, type=float)
parser.add_argument('--nb_epoch', default=500, type=int)
parser.add_argument('--val_ratio', default=0.1, type=float)
parser.add_argument('--vocab_size', default=20000, type=int)
parser.add_argument('--max_length', default=40,type=int)

# model parameter
parser.add_argument('--loss_function', default='binary_crossentropy')
parser.add_argument('--cell', default='GRU', choices=['LSTM','GRU'])
parser.add_argument('-emb_dim', '--embedding_dim', default=128, type=int)
parser.add_argument('-hid_siz', '--hidden_size', default=512, type=int)
parser.add_argument('--dropout_rate', default=0.3, type=float)
parser.add_argument('-lr','--learning_rate', default=0.001,type=float)
parser.add_argument('--threshold', default=0.1,type=float)

# output path for your prediction
parser.add_argument('--result_path', default='result.csv',)

# put model in the same directory
parser.add_argument('--load_model', default = None)
parser.add_argument('--save_dir', default = 'model/')
parser.add_argument('--train_path', default = '/data/training_label.txt')
parser.add_argument('--semi_path', default = '/data/training_nolabel.txt')
parser.add_argument('--test_path',default = '/data/testing_data.txt')
parser.add_argument('--mode',default = None)
args = parser.parse_args()

train_path = 'data/training_label.txt'
test_path = 'data/testing_data.txt'
semi_path = 'data/training_nolabel.txt'


# build model
def simpleRNN(args,size):
	inputs = Input(shape=(size))

	# Embedding layer
	masking = Masking(input_shape=size)(inputs)

	# RNN 
	return_sequence = True
	dropout_rate = 0.3
	RNN_output = Bidirectional(GRU(100, activation='tanh', dropout=dropout_rate))(masking)




	

	RNN_output = BatchNormalization()(RNN_output)
	# DNN layer
	
	outputs = Dense(80,kernel_regularizer=regularizers.l2(0.1))(RNN_output)
	outputs = LeakyReLU(1./20)(outputs)
	#outputs = Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.1))(outputs)
	outputs = BatchNormalization()(outputs)
	outputs = Dropout(dropout_rate)(outputs)
	outputs = Dense(1, activation='sigmoid')(outputs)
		
	model =  Model(inputs=inputs,outputs=outputs)

	# optimizer
	adam = Adam()
	print ('compile model...')

	# compile model
	model.compile( loss='binary_crossentropy', optimizer=adam, metrics=[ 'accuracy',])
	
	return model

def split_data(X,Y,ratio):
	cut = int(ratio*X.shape[0])
	return (X[cut:],Y[cut:]),(X[:cut],Y[:cut])


def main():
	# limit gpu memory usage
	global data
	global data_no
	



	if args.action == 'train':
		data = pd.read_csv(args.train_path, sep="\+\+\+\$\+\+\+", engine='python', header=None, names=['label', 'text'])
		data['text'] = data['text'].apply(lambda x: re.sub('[^a-zA-Z0-9\s\?\!]','',x.lower()))
		#data['text'] = data['text'].apply(lambda x: (x.lower()))
		y_train = data['label'].values

	elif args.action == 'test':
		data = pd.read_csv(args.test_path, sep="\n", skiprows=1, engine='python', names=['text'])
		data = data['text'].str.split(',', 1 , expand=True)
		data['text'] = data[1].apply(lambda x: re.sub('[^a-zA-Z0-9\s\?\!]','',x.lower()))
		#data['text'] = data[1].apply(lambda x:(x.lower()))
	word_dim = 120
	word2vec_model = word2vec.Word2Vec.load('emb')


	i = 0
	tmp = []
	for row in data.text.str.split():
		try:
			tmp.append(np.pad(word2vec_model[row],((args.max_length-len(row),0), (0,0)), mode='constant'))
		except:
			string = []
			for ele in row:
				if ele in word2vec_model:
					string.append(ele)
			if len(string) == 0 :
				tmp.append(np.zeros([args.max_length,word_dim]).astype('float32'))
			else:
				tmp.append(np.pad(word2vec_model[string],((args.max_length-len(string),0), (0,0)), mode='constant'))
		i += 1
		print("\rtraining data : " + repr(i), end="", flush=True)

	data = np.array(tmp)
#####read data#####
	print(data.shape)


	print ('Loading data...')    

	# prepare tokenizer
	print ('get Tokenizer...')



# initial model
	print ('initial model...')
	model = simpleRNN(args,data.shape[1:])    
	print (model.summary())

	if args.action == 'test':	
		model.load_weights('model.h5')



# training
	if args.action == 'train':
		(X,Y),(X_val,Y_val) = split_data(data,y_train, args.val_ratio)
		earlystopping = EarlyStopping(monitor='val_acc', patience = 5, verbose=0, mode='auto')
		lr_reducer = ReduceLROnPlateau(factor=0.1, cooldown=0, patience=2, min_lr=0.5e-6)
		checkpoint = ModelCheckpoint(filepath='model.h5', 
									 verbose=1,
									 save_best_only=True,
									 save_weights_only=True,
									 monitor='val_acc',
									 mode='max' )
		history = model.fit(X, Y, 
							validation_data=(X_val, Y_val),
							epochs=args.nb_epoch, 
							batch_size=args.batch_size,
							verbose=2,
							callbacks=[checkpoint,earlystopping] )

# testing
	elif args.action == 'test' :
		#raise Exception ('Implement your testing function')
		print("Predict test data\n")
		predict = model.predict(data,verbose=2)
		with open(args.result_path,'w') as output:
			predict[predict>=0.5] = 1
			predict[predict<0.5] = 0
			output.write('id,label\n')
			for i in range(predict.shape[0]):
				output.write(str(i)+','+str(int(predict[i][0]))+'\n')




 # semi-supervised training


if __name__ == '__main__':
		main()

   

