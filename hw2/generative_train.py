import numpy as np
import pandas as pd
import sys
import math


def loaddata():
	train_x = pd.read_csv("train_X").as_matrix().astype('int')
	test_x = pd.read_csv("test_X").as_matrix().astype('int')
	train_y = pd.read_csv("train_Y",header = None).as_matrix().astype('int')

	temp = np.concatenate((train_x,test_x), axis=0)
	a = [0,10,78,79]
	#a.extend([i for i in range(27,43)])
	#a.extend([i for i in range(50,65)])


	

	max_x = np.max(temp,axis=0)
	min_x = np.min(temp,axis=0)
	temp = (temp-min_x)/(max_x-min_x)

	#temp = choose_parameter(temp,a)

	#temp = np.concatenate((np.ones((temp.shape[0],1)),temp), axis=1)

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

def acc_count(val_x,mean1,mean2,sigma,cnt1,cnt2,val_y):
	sigma_inverse = np.linalg.pinv(sigma)
	w = np.dot((mean1-mean2),sigma_inverse)
	x = val_x.T
	b = -0.5*np.dot(np.dot([mean1],sigma_inverse),mean1) +  0.5*np.dot(np.dot([mean2],sigma_inverse),mean2)
	a = np.dot(w,x)+b
	y = sigmoid(a) 
	y[y>=0.5] = 1
	y[y<0.5] = 0
	print("The val data acc: %f" %np.mean(1-np.abs(y-val_y)))

def sigmoid(x):
	res = 1 / (1.0 + np.exp(-x))
	return np.clip(res, 1e-8, 1-(1e-8))

def generative(x,y):
	mul1 = np.zeros((x.shape[1],))
	mul2 = np.zeros((x.shape[1],))
	cnt1 = 0
	cnt2 = 0

	for i in range(x.shape[0]):
		if y[i]==1:
			mul1 = mul1 + x[i]
			cnt1 +=1
		else:
			mul2 = mul2 + x[i]
			cnt2 += 1

	mul1 /= cnt1
	mul2 /= cnt2
	

	sigma1 = np.zeros((x.shape[1],x.shape[1]))
	sigma2 = np.zeros((x.shape[1],x.shape[1]))


	for i in range(x.shape[0]):
		if y[i] == 1:
			sigma1 += np.dot(np.transpose([x[i]-mul1]),[x[i]-mul1])
		else:
			sigma2 += np.dot(np.transpose([x[i]-mul2]),[x[i]-mul2])
	sigma1 /= cnt1
	sigma2 /= cnt2
	
	shared_sigma = (float(cnt1)/x.shape[0])*sigma1 + (float(cnt2)/x.shape[0])*sigma2

	return cnt1, cnt2,mul1, mul2, shared_sigma

def test_predict(x,mean1,mean2,sigma,cnt1,cnt2):
	sigma_inverse = np.linalg.pinv(sigma)
	w = np.dot((mean1-mean2),sigma_inverse)
	x = x.T
	b = -0.5*np.dot(np.dot([mean1],sigma_inverse),mean1) +  0.5*np.dot(np.dot([mean2],sigma_inverse),mean2)
	a = np.dot(w,x)+b

	y = sigmoid(a)
	y[y>=0.5] = 1
	y[y< 0.5] = 0    
	
	

	finalString = "id,label\n"
	with open("result.csv", "w") as f:
		for i in range(len(y)) :
			finalString = finalString + str(i+1) + "," + str(int(y[i])) + "\n"
		f.write(finalString)
	
	








if __name__ == '__main__':

	train_x, test_x,train_y = loaddata()
	
	train_x, train_y, val_x, val_y = val(train_x,train_y,0.9)

	
	cnt1,cnt2,mean1, mean2, sigma = generative(train_x,train_y)
	print(mean1.shape,mean2.shape,sigma.shape)
	acc_count(val_x,mean1,mean2,sigma,cnt1,cnt2,val_y)

	#print('train_acc : %f | valid_acc : %f' %(acc,acc_valid))

	test_predict(test_x,mean1,mean2,sigma,cnt1,cnt2)




	