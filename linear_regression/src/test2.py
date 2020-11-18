import csv
import numpy as np
import math 
import random

data_folder = "linear_regression/data/"

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

def grad(method, h, W, X, Y, alpha):
	col_X = len(X[0])
	ans = [0 for i in range(col_X)]
	Wn = W.copy()
	for i in range(col_X):
		Wn[i] += h
		ans[i] = (method(Wn, X, Y, alpha) - method(W, X, Y, alpha))/h
		Wn[i] -= h
	return np.array(ans)

def SGD(method, A, y, alpha=None, beta=None, gamma=None):
	col_A = len(A[0])
	W = np.array([1. for row in range(col_A)])
	step = 1e-4
	h = 1e-4
	L = 0
	Lnew = method(W, A, y, alpha)
	while abs(L - Lnew) > 1e-3:
		for j in range(len(A)) :
			L = Lnew
			W -= step * grad(method, h, W, [A[j]], [y[j]], alpha)
			Lnew = MSE(W, A, y)
	return np.array(W)

def AdaGrad(method, A, y, alpha=None, beta=None, gamma=None):
	col_A = len(A[0])
	W = np.array([1. for row in range(col_A)])
	step = 1e-2
	h = 1e-2
	eps = 1
	L = method(W, A, y, alpha)
	Gnii = [0 for i in range(col_A)]
	temp = grad(method, h, W, A, y, alpha)
	for i in range(col_A):
		Gnii[i] = (temp[i])**2
		W[i] -= step*temp[i]/math.sqrt(Gnii[i] + eps)
	Lnew = MSE(W, A, y)
	while abs(L - Lnew) > 1:
		L = Lnew
		temp = grad(method, h, W, A, y, alpha)
		for i in range(col_A):
			Gnii[i] += (temp[i])**2
			W[i] -= step*temp[i]/math.sqrt(Gnii[i] + eps)
		Lnew = method(W, A, y, alpha)
	return np.array(W)

def RMSProp(method, A, y, alpha=None, beta=None, gamma=0.9):
	if gamma is None:
		gamma=0.9
	col_A = len(A[0])
	W = np.array([1. for row in range(col_A)])
	step = 1e-3
	h = 1e-3
	eps = 1e-2
	L = method(W, A, y, alpha)
	Eg = 0
	tmp = 0
	tmpnew = grad(method, h, W, A, y, alpha)
	Eg = gamma*(tmp**2) + (1-gamma)*(tmpnew**2)
	W -= step * tmpnew / np.sqrt(Eg + eps)
	Lnew = method(W, A, y, alpha)
	while abs(L - Lnew) > 1e-1:
		L = Lnew
		tmp = tmpnew
		tmpnew = grad(method, h, W, A, y, alpha)
		Eg = gamma*(tmp**2) + (1-gamma)*(tmpnew**2)
		W -= step*tmpnew/np.sqrt(Eg + eps)
		Lnew = method(W, A, y, alpha)
	return np.array(W)

def Adam(method, A, y, alpha=None, beta=[0.95, 0.95], gamma=0.9):
	if beta is None:
		beta=[0.95, 0.95]
	if gamma is None:
		gamma=0.9
	col_A = len(A[0])
	W = np.array([1. for row in range(col_A)])
	step = 1e-3
	h = 1e-3
	eps = 1e-2
	L = 0
	Lnew = method(W, A, y, alpha)
	tmp = grad(method, h, W, A, y, alpha)
	mt = beta[0]*0 + (1-beta[0])*tmp
	_mt = mt/(1-beta[0])
	vt = beta[1]*0 + (1-beta[1])*(tmp**2)
	_vt = vt/(1-beta[1])
	W = W - step*_mt/np.sqrt(_vt + eps)
	L = Lnew
	Lnew = method(W, A, y, alpha)
	while abs(L - Lnew) > 1e-2:
		L = Lnew
		tmp = grad(method, h, W, A, y, alpha)
		mt = beta[0]*mt + (1-beta[0])*tmp
		_mt = mt/(1-beta[0])
		vt = vt
		vt = beta[1]*vt + (1-beta[1])*(tmp**2)
		_vt = vt/(1-beta[1])
		W = W - step*_mt/np.sqrt(_vt + eps)
		Lnew = method(W, A, y, alpha)
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

def MSE(W, X, Y, alpha=None):
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

def CrossValidation(methodGD, methodReg, X, Y, alpha=None, beta=None, gamma=None, folds=5):
	Wf = []
	onefold = int(1000/folds)
	for i in range(folds):
		Wf.append(methodGD(methodReg, X[i*onefold: (i+1)*onefold], Y[i*onefold: (i+1)*onefold], alpha, beta, gamma))
	W = sum(Wf)/len(Wf)
	W = SGD(MSE_l1, data_train_features[:1000], data_train_target[:1000], alpha)
	print("\nW " + str(methodGD) + " (" + str(methodReg.__name__) + ") = ", W)
	print(str(methodGD.__name__) + " MSE (train) = ", methodReg(W, data_train_features[:1000], data_train_target[:1000], alpha))
	print(str(methodGD.__name__) + " R2  (train) = ", R2(W, data_train_features[:1000], data_train_target[:1000]))
	print(str(methodGD.__name__) + " MSE (test) = ", methodReg(W, data_train_features[1000:2000], data_train_target[1000:2000], alpha))
	print(str(methodGD.__name__) + " R2  (test) = ", R2(W, data_train_features[1000:2000], data_train_target[1000:2000]))


W = LeastSquareMethod(data_train_features[:1000], data_train_target[:1000])
print("\nW (LSM) = ", W)
print("LSM MSE (train) = ", MSE(W, data_train_features[:1000], data_train_target[:1000]))
print("LSM R2 (train) = ", R2(W, data_train_features[:1000], data_train_target[:1000]))
print("LSM MSE (test) = ", MSE(W, data_train_features[1000:2000], data_train_target[1000:2000]))
print("LSM R2 (test) = ", R2(W, data_train_features[1000:2000], data_train_target[1000:2000]))

alpha = 0.8

# CrossValidation(SGD, MSE_l1, data_train_features, data_train_target, alpha)
# CrossValidation(SGD, MSE_l2, data_train_features, data_train_target, alpha)

# CrossValidation(AdaGrad, MSE_l1, data_train_features, data_train_target, alpha)
# CrossValidation(AdaGrad, MSE_l2, data_train_features, data_train_target, alpha)

CrossValidation(RMSProp, MSE_l1, data_train_features, data_train_target, alpha)
CrossValidation(RMSProp, MSE_l2, data_train_features, data_train_target, alpha)

CrossValidation(Adam, MSE_l1, data_train_features, data_train_target, alpha)
CrossValidation(Adam, MSE_l2, data_train_features, data_train_target, alpha)