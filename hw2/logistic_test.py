import numpy as np
import pandas as pd
import sys
import math


def loaddata():
	train_x = pd.read_csv(sys.argv[1]).as_matrix().astype('int')
	test_x = pd.read_csv(sys.argv[3]).as_matrix().astype('int')
	train_y = pd.read_csv(sys.argv[2],header = None).as_matrix().astype('int')

	temp = np.concatenate((train_x,test_x), axis=0)
	a = [0,10,78,79]
	#a.extend([i for i in range(27,43)])
	#a.extend([i for i in range(50,65)])


	

	max_x = np.max(temp,axis=0)
	min_x = np.min(temp,axis=0)
	temp = (temp-min_x)/(max_x-min_x)

	temp = choose_parameter(temp,a)

	temp = np.concatenate((np.ones((temp.shape[0],1)),temp), axis=1)

	return temp[:train_x.shape[0],:],temp[train_x.shape[0]:,:],train_y

	

	
def choose_parameter(X,index):

	X = np.concatenate((X,X[:,index], X[:,index]**2,X[:,index]**3,X[:,index]**4,np.log(X[:,index] + 1e-10)),axis=1)
	return X

def val(x,y,ratio):
	order = np.random.permutation(x.shape[0])
	train_x = x[order[:int(ratio*x.shape[0])]]
	train_y = y[order[:int(ratio*x.shape[0])]]
	val_x = x[order[int(ratio*x.shape[0]):]]
	val_y = y[order[int(ratio*x.shape[0]):]]
	return train_x,train_y,val_x,val_y

def acc_count(y_pred,y):
	
	y_pred[y_pred>=0.5] = 1
	y_pred[y_pred<0.5] = 0
	return np.mean(1-np.abs(y-y_pred))

def sigmoid(x):
	res = 1 / (1.0 + np.exp(-x))
	return np.clip(res, 1e-8, 1-(1e-8))

def logistic(x,y,epoch,lr,mom,lamdba):
	print(x.shape)

	#w = np.ones((x.shape[1],1))
	#G = np.ones((x.shape[1],1))
	error = 0
	
	w = np.random.randn(x.shape[1],1).reshape(x.shape[1],1) / x.shape[1] / x.shape[0]
	G = np.random.randn(x.shape[1],1).reshape(x.shape[1],1) / x.shape[1] / x.shape[0]
	
	acc_record = []
	last_step = 0
	for i in range(1,epoch+1):
		
		
			
		y_pred = sigmoid(x.dot(w)) 
		diff = y_pred - y
		cost = -np.mean(y*np.log(y_pred+1e-20) + (1-y)*np.log(1-y_pred+1e-20))  
		grad = x.T.dot(diff) + mom*last_step + lamdba*w
		G += grad**2
		w -= lr*grad / np.sqrt(G)
		last_step = lr*grad / np.sqrt(G)
		acc = acc_count(y_pred,y)
			
		if i % 200 == 0:
			print('epoch : %d | cost : %f | acc : %f' %(i,cost,acc))
			if math.isnan(acc):
				print("Training again")
				error = 1
				break
			acc_record.append(acc)
		
	np.save('model.npy',w)
	return acc,error

def test_predict(X):
	w = np.load('model.npy')
	y = sigmoid(X.dot(w))
	y[y>=0.5] = 1
	y[y< 0.5] = 0    
	
	finalString = "id,label\n"
	with open(sys.argv[4], "w") as f:
		for i in range(len(y)) :
			finalString = finalString + str(i+1) + "," + str(int(y[i][0])) + "\n"
		f.write(finalString)
	
	








if __name__ == '__main__':

	train_x, test_x,train_y = loaddata()
	
	#train_x, train_y, val_x, val_y = val(train_x,train_y,0.9)

	#while 1:
	#	acc,error = logistic(train_x,train_y,epoch=8000,lr = 0.5,mom=0 ,lamdba = 1)
	#	if error == 0:
	#		break

	#acc_valid = acc_count(sigmoid(val_x.dot(np.load('model.npy'))),val_y)

	#print('train_acc : %f | valid_acc : %f' %(acc,acc_valid))

	test_predict(test_x)




	