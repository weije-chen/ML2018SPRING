import sys
import csv 
import math
import random
import numpy as np

x = []
y = []

test_x = []
test_y = []

lr = 1                       # learning rate
iter = 120000                   # iteration

def loadtrain():
	global x 
	global y
	#一個維度儲存一種污染物的資訊
	data = []
	for i in range(18):
		data.append([])


	n_row = 0
	text = open('train.csv', 'r', encoding='big5') 
	row = csv.reader(text , delimiter=",")
	for r in row:
		# 第0列為header沒有資訊
		if n_row != 0:
			# 每一列只有第3-27格有值(1天內24小時的數值)
			for i in range(3,27):
				if r[i] != "NR":
					data[(n_row-1)%18].append(float(r[i]))
				else:
					data[(n_row-1)%18].append(float(0))
		n_row = n_row+1
	text.close()

	data_after = []
	for i in range(18):
		data_after.append([])

	for i in range(len(data[0])):
		if data[9][i] == 0.0 or data[8][i] == 0:
			pass
		else:
			for j in range(18):
				data_after[j].append(data[j][i])

	data_after = np.array(data_after)
	mean = np.mean(data_after,axis=1)
	std = np.std(data_after,axis=1)

	for i in range(12):
		temp = []
		for j in range(18):
			temp.append([])

		for j in range(480):
			if data[9][i*480+j] <= 0.0: # or data[8][i*480+j] <= 0.0 or data[7][i*480+j] <= 0.0 or data[6][i*480+j] <= 0.0 or data[5][i*480+j] <= 0.0:
				temp = []
				for a in range(18):
					temp.append([])

			else:
				if len(temp[0]) == 9:
					x.append([])
					
					x[-1].extend(temp[4])
					x[-1].extend(temp[5])
					x[-1].extend(temp[6])
					x[-1].extend(temp[7])
					x[-1].extend(temp[8])			# decide parameter
					x[-1].extend(temp[9])
					#for k in range(9):
					#	for m in range(5,10):
					#		x[-1].append(temp[m][k]**2)
							#x[-1].append(temp[m][k]**3)
						
						
					y.append(data[9][i*480+j])
					#y.append((data[9][i*480+j]-mean[9])/std[9])
					

					for s in range(18):
						del temp[s][0]
						temp[s].append(data[s][i*480+j])
						#temp[s].append((data[s][i*480+j]-mean[s])/std[s])
				else:
					for s in range(18):
						temp[s].append(data[s][i*480+j])
						#temp[s].append((data[s][i*480+j]-mean[s])/std[s])
						
				
	x = np.array(x)
	y = np.array(y)
	x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
	#print(x[0])

	for i in range(200):
		x = np.delete(x, 1000,0)
		y = np.delete(y,1000,0)

	for i in range(500):
		x = np.delete(x, -1,0)
		y = np.delete(y,-1,0)

#def cut():
#	global test_x,test_y,x,y
#	
#	start = int(sys.argv[2])*500
#	
#	for i in range(260):
#		test_x.append(x[start])
#		test_y.append(y[start])
#		x = np.delete(x,start,0)
#		y = np.delete(y,start,0)
#
#	test_x = np.array(test_x)
#	test_y = np.array(test_y)
#
#	for i in range(9):
#		x = np.delete(x,start,0)
#		y = np.delete(y,start,0)


def train():
	w = np.zeros(len(x[0]))         # initial weight vector
	x_t = x.transpose()
	s_gra = np.zeros(len(x[0]))

	for i in range(iter):
		hypo = np.dot(x,w)
		loss = hypo - y
		cost = np.sum((loss)**2) / len(x)
		cost_a  = math.sqrt(cost)
		gra = np.dot(x_t,loss)
		s_gra += gra**2
		ada = np.sqrt(s_gra)
		w = w - lr * gra/ada
		print ('iteration: %d | Cost: %f  ' % ( i,cost_a))	

	np.save('model.npy',w)
	#print(w)



def test():
	w = np.load('model.npy')

	test_x = []
	data = []
	n_row = 0
	text = open(sys.argv[1] ,"r")
	row = csv.reader(text , delimiter= ",")

	for i in range(18):
		data.append([])

	for r in row:
		if n_row%18 >= 4 and  n_row%18 <= 9 :
			for i in range(2,11):
				if float(r[i]) > 0:
					data[n_row%18].append(float(r[i]))
				else:
					if i == 2:
						data[n_row%18].append(float(r[3]))
					elif i == 10:
						data[n_row%18].append(float(r[9]))
					else:
						data[n_row%18].append((float(r[i-1])+float(r[i+1]))/2.0)
		else:
			for i in range(2,11):
				if r[i] !="NR":
					data[n_row%18].append(float(r[i]))
				else:
					data[n_row%18].append(0)

		if  n_row%18 == 17:
			test_x.append([])
				
			test_x[-1].extend(data[4])
			test_x[-1].extend(data[5])
			test_x[-1].extend(data[6])
			test_x[-1].extend(data[7])
			test_x[-1].extend(data[8])			# decide parameter
			test_x[-1].extend(data[9])
		#	for i in range(9):
		#		for j in range(4,10):
		#			test_x[-1].append(data[j][i]**2)
		#			test_x[-1].append(data[j][i]**3)

			data = []
			for i in range(18):
				data.append([])
			
		n_row = n_row+1

	text.close()

	


	

	test_x = np.array(test_x)
	# test_x = np.concatenate((test_x,test_x**2), axis=1)
	# 增加平方項
	
	test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)
	# 增加bias項  


	ans = []
	for i in range(len(test_x)):
		ans.append(["id_"+str(i)])
		a = np.dot(w,test_x[i])
		if a < 0:
			a = 0
		ans[i].append(a)



	filename = sys.argv[2]
	text = open(filename, "w+")
	s = csv.writer(text,delimiter=',',lineterminator='\n')
	s.writerow(["id","value"])
	for i in range(len(ans)):
		s.writerow(ans[i]) 
	text.close()



#def test_val():
#	w = np.load('model.npy')
#	mean = np.load('mean.npy')
#	std = np.load('std.npy')
#	ans = []
#	for i in range(len(test_x)):
#		a = np.dot(w,test_x[i])
#		ans.append(a)
#
#	ans = np.array(ans)
#	loss = (ans - test_y) #*std[9]
#	cost = math.sqrt(np.sum(loss**2)/len(ans))
#
#
#	ans = []
#	for i in range(len(test_x)):
#		ans.append(["id_"+str(i)])
#		a = np.dot(w,test_x[i])
#		if a < 0:
#			a = 0
#		ans[i].append(a)
#		ans[i].append(test_y[i])
#
#	filename = sys.argv[2]
#	text = open(filename, "w+")
#	s = csv.writer(text,delimiter=',',lineterminator='\n')
#	s.writerow(["id","value"])
#	for i in range(len(ans)):
#		s.writerow(ans[i]) 
#	print ('Test Cost: %f  ' % (cost))

	


if __name__ == '__main__':
	test()

		
