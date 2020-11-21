import csv
import numpy as np
import math 
import random
from copy import copy, deepcopy

# data_folder = "linear_regression/data/"
data_folder = "data/"

airline_delay_causes = []
columns = []
target_variable = "DEP_DELAY"
target_col = []

with open(data_folder+'2017.csv', newline='\n') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	line_count = 0
	for row in csv_reader:
		if line_count == 0:
			columns = row.copy()
			line_count += 1
		else:
			airline_delay_causes.append(row)
			line_count += 1

target = 11
# features = [1, 2, 9, 10, 12, 17, 19, 25, 26, 27, 28, 29]
# features = [9, 10, 12, 25, 27, 29]
features = [9, 25, 29]
useful_data_X = []
useful_data_Y = []
for row in airline_delay_causes:
	tmp = []
	for col in features:
		if row[col] == "":
			break
		else:
			tmp.append(float(row[col]))
	if len(tmp) == len(features):
		useful_data_X.append(tmp)
		useful_data_Y.append(float(row[target]))

data_train_features = np.array(useful_data_X)
data_train_target = np.array(useful_data_Y)

def norm(V):
	ans = 0
	for i in V:
		ans += i**2
	return math.sqrt(ans)

def LeastSquareMethod(A, y):
	AT = np.transpose(A)
	AInv = np.linalg.inv(np.matmul(AT, A))
	return np.matmul(AInv, np.matmul(AT, y))

def LeastSquareMethod_GD(A, y):
	col_A = len(A[0])
	W = np.array([1. for row in range(col_A)])
	step = 1e-8
	h = 1e-8
	L = 0
	Lnew = MSE(W, A, y)
	num_steps = 200
	while abs(L - Lnew) > 1e-6 and num_steps > 0:
		L = Lnew
		W -= step * grad(MSE, h, W, A, y)
		Lnew = MSE(W, A, y)
		num_steps -= 1
	return np.array(W)

def grad(fun, h, W, X, Y):
	col_X = len(X[0])
	ans = [0 for i in range(col_X)]
	Wn = W.copy()
	for i in range(col_X):
		Wn[i] += h
		ans[i] = (MSE(Wn, X, Y) - MSE(W, X, Y))/h
		Wn[i] -= h
	return np.array(ans)

def SGD(A, y):
	col_A = len(A[0])
	W = np.array([1. for row in range(col_A)])
	step = 1e-8
	h = 1e-8
	L = 0
	Lnew = MSE(W, A, y)
	num_steps = 50
	while abs(L - Lnew) > 1e-5 and num_steps > 0:
		for j in range(len(A)):
			L = Lnew
			W -= step * grad(MSE, h, W, [A[j]], [y[j]])
			Lnew = MSE(W, A, y)
		num_steps -= 1
	return np.array(W)

def AdaGrad(A, y):
	col_A = len(A[0])
	W = np.array([1. for row in range(col_A)])
	step = 1e-3
	h = 1e-3
	eps = 1
	L = MSE(W, A, y)
	Gnii = [0 for i in range(col_A)]
	temp = grad(MSE, h, W, A, y)
	for i in range(col_A):
		Gnii[i] = (temp[i])**2
		W[i] -= step*temp[i]/math.sqrt(Gnii[i] + eps)
	Lnew = MSE(W, A, y)
	num_steps = 100
	while abs(L - Lnew) > 1e-2 or num_steps > 0:
		L = Lnew
		temp = grad(MSE, h, W, A, y)
		for i in range(col_A):
			Gnii[i] += (temp[i])**2
			W[i] -= step*temp[i]/math.sqrt(Gnii[i] + eps)
		Lnew = MSE(W, A, y)
		num_steps -= 1
	return np.array(W)

def RMSProp(A, y, gamma=0.5):
	col_A = len(A[0])
	W = np.array([1. for row in range(col_A)])
	step = 1e-3
	h = 1e-3
	eps = 1e-2
	L = MSE(W, A, y)
	Eg = 0
	tmp = 0
	tmpnew = grad(MSE, h, W, A, y)
	Eg = gamma*(tmp**2) + (1-gamma)*(tmpnew**2)
	W -= step * tmpnew / np.sqrt(Eg + eps)
	Lnew = MSE(W, A, y)
	num_steps = 100
	while abs(L - Lnew) > 1e-2 or num_steps > 0:
		L = Lnew
		tmp = tmpnew
		tmpnew = grad(MSE, h, W, A, y)
		Eg = gamma*(tmp**2) + (1-gamma)*(tmpnew**2)
		W -= step*tmpnew/np.sqrt(Eg + eps)
		Lnew = MSE(W, A, y)
		num_steps -= 1
	return np.array(W)

def Adam(A, y, gamma=0.9, beta=[0.95, 0.95]):
	col_A = len(A[0])
	W = np.array([1. for row in range(col_A)])
	step = 1e-3
	h = 1e-3
	eps = 1e-2
	L = 0
	Lnew = MSE(W, A, y)
	tmp = grad(MSE, h, W, A, y)
	mt = beta[0]*0 + (1-beta[0])*tmp
	_mt = mt/(1-beta[0])
	vt = beta[1]*0 + (1-beta[1])*(tmp**2)
	_vt = vt/(1-beta[1])
	W = W - step*_mt/np.sqrt(_vt + eps)
	L = Lnew
	Lnew = MSE(W, A, y)
	num_steps = 100
	while abs(L - Lnew) > 1e-3 or num_steps > 0:
		L = Lnew
		tmp = grad(MSE, h, W, A, y)
		mt = beta[0]*mt + (1-beta[0])*tmp
		_mt = mt/(1-beta[0])
		vt = vt
		vt = beta[1]*vt + (1-beta[1])*(tmp**2)
		_vt = vt/(1-beta[1])
		W = W - step*_mt/np.sqrt(_vt + eps)
		Lnew = MSE(W, A, y)
		num_steps -= 1
	return np.array(W)


def PolynomRegression(W, X):
	row_X = len(X)
	col_X = len(X[0])
	ans = np.array([0 for i in range(row_X)])
	for i in range(col_X):
		for j in range(row_X):
			ans[j] += W[i] * X[j][i]
	ans += np.array([random.random() for i in range(row_X)])
	return ans

def R2(W, X, Y):
	if len(X[0]) != len(W) or len(X) != len(Y):
		raise Exception("Len of vectors should be same!")
	ym = sum(Y)/len(Y)
	SSreg = 0
	for i in range(len(X)):
		SSreg += (np.dot(W, X[i]) - ym)**2
	SSreg /= len(Y)
	SStot = 0
	for i in range(len(Y)):
		SStot += (Y[i] - ym)**2
	SStot /= len(Y)
	return 1 - (SSreg / SStot)

def MSE(W, X, Y):
	if len(X[0]) != len(W) or len(X) != len(Y):
		raise Exception("Len of vectors should be same!")
	ans = 0
	for i in range(len(X)):
		ans += (np.dot(W, X[i]) - Y[i])**2
	return ans/len(X)

def MSE_l1(W, X, Y, alpha):
	if len(X[0]) != len(W) or len(X) != len(Y):
		raise Exception("Len of vectors should be same!")
	ans = 0
	for i in range(len(X)):
		ans += (np.dot(W, X[i]) - Y[i])**2
	return (ans + alpha*norm(W))/len(W)

def MSE_l2(W, X, Y, alpha):
	if len(X[0]) != len(W) or len(X) != len(Y):
		raise Exception("Len of vectors should be same!")
	ans = 0
	for i in range(len(X)):
		ans += (np.dot(W, X[i]) - Y[i])**2
	return (ans + alpha*(norm(W)**2))/len(W)


W = LeastSquareMethod(data_train_features[:1000], data_train_target[:1000])
print("\nW (LSM) = ", W)
print("LSM MSE (train) = ", MSE(W, data_train_features[:1000], data_train_target[:1000]))
print("LSM R2  (train) = ", R2(W, data_train_features[:1000], data_train_target[:1000]))
print("LSM MSE (test) = ", MSE(W, data_train_features[1000:2000], data_train_target[1000:2000]))
print("LSM R2  (test) = ", R2(W, data_train_features[1000:2000], data_train_target[1000:2000]))

W = LeastSquareMethod_GD(data_train_features[:1000], data_train_target[:1000])
print("\nW (LSM GD) = ", W)
print("LSM GD MSE (train) = ", MSE(W, data_train_features[:1000], data_train_target[:1000]))
print("LSM GD R2  (train) = ", R2(W, data_train_features[:1000], data_train_target[:1000]))
print("LSM GD MSE (test) = ", MSE(W, data_train_features[1000:2000], data_train_target[1000:2000]))
print("LSM GD R2  (test) = ", R2(W, data_train_features[1000:2000], data_train_target[1000:2000]))

W = SGD(data_train_features[:1000], data_train_target[:1000])
print("\nW (SGD) = ", W)
print("SGD MSE (train) = ", MSE(W, data_train_features[:1000], data_train_target[:1000]))
print("SGD R2  (train) = ", R2(W, data_train_features[:1000], data_train_target[:1000]))
print("SGD MSE (test) = ", MSE(W, data_train_features[1000:2000], data_train_target[1000:2000]))
print("SGD R2  (test) = ", R2(W, data_train_features[1000:2000], data_train_target[1000:2000]))
 
W = AdaGrad(data_train_features[:1000], data_train_target[:1000])
print("\nW (AdaGrad) = ", W)
print("AdaGrad MSE (train) = ", MSE(W, data_train_features[:1000], data_train_target[:1000]))
print("AdaGrad R2  (train) = ", R2(W, data_train_features[:1000], data_train_target[:1000]))
print("AdaGrad MSE (test) = ", MSE(W, data_train_features[1000:2000], data_train_target[1000:2000]))
print("AdaGrad R2  (test) = ", R2(W, data_train_features[1000:2000], data_train_target[1000:2000]))

W = RMSProp(data_train_features[:1000], data_train_target[:1000])
print("\nW (RMSProp) = ", W)
print("RMSProp MSE (train) = ", MSE(W, data_train_features[:1000], data_train_target[:1000]))
print("RMSProp R2  (train) = ", R2(W, data_train_features[:1000], data_train_target[:1000]))
print("RMSProp MSE (test) = ", MSE(W, data_train_features[1000:2000], data_train_target[1000:2000]))
print("RMSProp R2  (test) = ", R2(W, data_train_features[1000:2000], data_train_target[1000:2000]))

W = Adam(data_train_features[:1000], data_train_target[:1000])
print("\nW (Adam) = ", W)
print("Adam MSE (train) = ", MSE(W, data_train_features[:1000], data_train_target[:1000]))
print("Adam R2  (train) = ", R2(W, data_train_features[:1000], data_train_target[:1000]))
print("Adam MSE (test) = ", MSE(W, data_train_features[1000:2000], data_train_target[1000:2000]))
print("Adam R2  (test) = ", R2(W, data_train_features[1000:2000], data_train_target[1000:2000]))
