import csv
import numpy as np
import math 

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

print(columns)

# if len(empty_cols) > 0:
# 	for col in empty_cols:
# 		del columns[col]
# 	del columns[target_index]
# 	for row in airline_delay_causes:
# 		for col in empty_cols:
# 			del row[col]
# 		target_col.append(row[target_index])
# 		del row[target_index]
# for i in range(len(columns)):
# 	print(str(i) + " " + columns[i]+ " " + airline_delay_causes[1][i])

target_index = 11
columns_for_use = [1, 2, 9, 10, 12, 17, 19, 25, 26, 27, 28, 29]
useful_data_X = []
useful_data_Y = []
for row in airline_delay_causes:
	tmp = []
	for col in columns_for_use:
		if row[col] == "":
			break
		else:
			tmp.append(float(row[col]))
	if len(tmp) == len(columns_for_use):
		useful_data_X.append(tmp)
		useful_data_Y.append(row[target_index])

def norm(V):
	ans = 0
	for i in V:
		ans += i**2
	return math.sqrt(ans)

def sumvc(X, a):
	ans = X.copy()
	for i in range(len(X)):
		ans[i] += a
	return ans

def sumvv(X, Y):
	if len(X) != len(Y):
		raise Exception("Len of vectors should be same!")
	ans = X.copy()
	for i in range(len(Y)):
		ans[i] += Y[i]
	return ans

def subvc(X, a):
	ans = X.copy()
	for i in range(len(X)):
		ans[i] -= a
	return ans

def subvv(X, Y):
	if len(X) != len(Y):
		raise Exception("Len of vectors should be same!")
	ans = X.copy()
	for i in range(len(Y)):
		ans[i] -= Y[i]
	return ans

def multvc(X, c):
	ans = X.copy()
	for i in range(len(X)):
		ans[i] *= c
	return ans

def multvv(X, Y):
	if len(X) != len(Y):
		raise Exception("Len of vectors should be same!")
	ans = X.copy()
	for i in range(len(Y)):
		ans[i] *= Y[i]
	return ans

def summc(A, c):
	if len(A) == 0:
		return []
	rows_A = len(A)
	cols_A = len(A[0])
	ans = [[0 for row in range(cols_A)] for col in range(rows_A)]
	for i in range(rows_A):
		for j in range(cols_A):
			ans[i][j] = A[i][j] + c
	return ans

def summm(A, B):
	if len(A) == 0 or len(B) == 0:
		return []
	if len(A) != len(B) or len(A[0]) != len(B[0]):
		raise Exception("Dim of matrix should be same!")
	rows_A = len(A)
	cols_B = len(B[0])
	ans = [[0 for row in range(cols_B)] for col in range(rows_A)]
	for i in range(rows_A):
		for j in range(cols_B):
			ans[i][j] = A[i][j] + B[i][j]
	return ans

def submc(A, c):
	if len(A) == 0:
		return []
	rows_A = len(A)
	cols_A = len(A[0])
	ans = [[0 for row in range(cols_A)] for col in range(rows_A)]
	for i in range(rows_A):
		for j in range(cols_A):
			ans[i][j] = A[i][j] - c
	return ans

def submm(A, B):
	if len(A) == 0 or len(B) == 0:
		return []
	if len(A) != len(B) or len(A[0]) != len(B[0]):
		raise Exception("Dim of matrix should be same!")
	rows_A = len(A)
	cols_B = len(B[0])
	ans = [[0 for row in range(cols_B)] for col in range(rows_A)]
	for i in range(rows_A):
		for j in range(cols_B):
			ans[i][j] = A[i][j] - B[i][j]
	return ans

def multmc(A, c):
	if len(A) == 0:
		return []
	rows_A = len(A)
	cols_A = len(A[0])
	ans = [[0 for col in range(cols_A)] for row in range(rows_A)]
	for i in range(rows_A):
		for j in range(cols_A):
			ans[i][j] = A[i][j] * c
	return ans

def multmv(A, V):
	if len(A) == 0:
		return []
	rows_A = len(A)
	cols_A = len(A[0])
	rows_V = len(V)
	if cols_A != rows_V:
		raise Exception("Cannot multiply the matrix and vector. Incorrect dimensions.")
	ans = [0 for row in range(rows_A)]
	for i in range(rows_A):
		for j in range(cols_A):
			ans[i] += A[i][j] * V[j]
	return ans

def multmm(A, B):
	if len(A) == 0 or len(B) == 0:
		return []
	rows_A = len(A)
	cols_A = len(A[0])
	rows_B = len(B)
	cols_B = len(B[0])
	if cols_A != rows_B:
		raise Exception("Cannot multiply the two matrices. Incorrect dimensions.")
	ans = [[0 for row in range(cols_B)] for col in range(rows_A)]
	for i in range(rows_A):
		for j in range(cols_B):
			for k in range(cols_A):
				ans[i][j] += A[i][k] * B[k][j]
	return ans

def transpm(A):
	if len(A) == 0:
		return []
	rows_A = len(A)
	cols_A = len(A[0])
	ans = [[0 for row in range(rows_A)] for col in range(cols_A)]
	for i in range(rows_A):
		for j in range(cols_A):
			ans[j][i] = A[i][j]
	return ans

def minorm(m,i,j):
	return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

def detm(m):
	#base case for 2x2 matrix
	if len(m) == 2:
		return m[0][0]*m[1][1]-m[0][1]*m[1][0]

	determinant = 0
	for c in range(len(m)):
		determinant += ((-1)**c)*m[0][c]*detm(minorm(m,0,c))
	return determinant

def invm(m):
	determinant = detm(m)
	if determinant < 1e-6:
		raise Exception("Determinant of a matrix is zero!")
	#special case for 2x2 matrix:
	if len(m) == 2:
		return [[m[1][1]/determinant, -1*m[0][1]/determinant],
				[-1*m[1][0]/determinant, m[0][0]/determinant]]

	cofactors = []
	for r in range(len(m)):
		cofactorRow = []
		for c in range(len(m)):
			minor = minorm(m,r,c)
			cofactorRow.append(((-1)**(r+c)) * detm(minor))
		cofactors.append(cofactorRow)
	cofactors = transpm(cofactors)
	for r in range(len(cofactors)):
		for c in range(len(cofactors)):
			cofactors[r][c] = cofactors[r][c]/determinant
	return cofactors

def LeastSquareMethod(A, y):
	AT = transpm(A)
	AInv = invm(multmm(AT, A))
	return multmv(AInv, multmv(AT, y))

def SGD(A, y):
	W = [1. for row in range(len(y))]
	step = 1e-4
	h = 1e-4
	L = 0
	Lnew = MSE(W, A, y)
	while abs(L - Lnew)>1e-3:
		for i in range(len(y)):
			L = Lnew
			Wn = W.copy()
			Wn[i] += h
			tmp = MSE(Wn, A, y)
			W[i] -= step*tmp
			Lnew = MSE(W, A, y)
	return W

def AdaGrad(A, y):
	W = [1. for row in range(len(y))]
	step = 1e-4
	h = 1e-4
	L = 0
	Lnew = MSE(W, A, y)
	Gnii = [0 for i in range(len(y))]
	for i in range(len(y)):
		Wn = W.copy()
		Wn[i] += h
		Gnii[i] = MSE(Wn, A, y)**2
	while abs(L - Lnew) > 1e-3:
		L = Lnew
		for i in range(len(y)):
			Wn = W.copy()
			Wn[i] += h
			tmp = MSE(Wn, A, y)
			Gnii[i] += tmp**2
			W[i] -= step*tmp/math.sqrt(Gnii[i] + h)
		Lnew = MSE(W, A, y)
	return W

def RMSProp(A, y, gamma=0.9):
	W = [1. for row in range(len(y))]
	step = 1e-4
	h = 1e-4
	L = 0
	Lnew = MSE(W, A, y)
	Eg = 0
	Wn = sumvc(W, h)
	tmp = 0
	tmpnew = MSE(Wn, A, y)
	Eg = gamma*(tmp**2) + (1-gamma)*(tmp**2)
	W = subvc(W, step*tmp/math.sqrt(Eg + h))
	L = Lnew
	Lnew = MSE(W, A, y)
	while abs(L - Lnew)>1e-3:
		L = Lnew
		tmp = tmpnew
		Wn = sumvc(W, h)
		tmpnew = MSE(Wn, A, y)
		Eg = gamma*(tmp**2) + (1-gamma)*(tmpnew**2)
		W = subvc(W, step*tmpnew/math.sqrt(Eg + h))
		Lnew = MSE(W, A, y)
	return W

def Adam(A, y, gamma=0.9, beta=[0.9, 0.9]):
	W = [1. for row in range(len(y))]
	step = 1e-4
	h = 1e-4
	L = 0
	Lnew = MSE(W, A, y)
	Wn = sumvc(W, h)
	tmp = MSE(Wn, A, y)
	mt = beta[0]*0 + (1-beta[0])*tmp
	_mt = mt/(1-beta[0])
	vt = beta[1]*0 + (1-beta[1])*(tmp**2)
	_vt = vt/(1-beta[1])
	W = subvc(W, step*_mt/math.sqrt(_vt + h))
	L = Lnew
	Lnew = MSE(W, A, y)
	while abs(L - Lnew)>1e-3:
		L = Lnew
		Wn = sumvc(W, h)
		tmp = MSE(Wn, A, y)
		mt = beta[0]*mt + (1-beta[0])*tmp
		_mt = mt/(1-beta[0])
		vt = vt
		vt = beta[1]*vt + (1-beta[1])*(tmp**2)
		_vt = vt/(1-beta[1])
		W = subvc(W, step*_mt/math.sqrt(_vt + h))
		Lnew = MSE(W, A, y)
	return W

def quadv(V):
	return multvv(V, V)

def PolynomRegression(W, X):
	col_X = len(X)
	ans = multvc(X.copy(), W[0])
	for i in range(1, col_X):
		for j in range(col_X):
			ans[i] += multvc(X[j]**i, W[i])
	return ans

# minimize
def MSE(W, X, Y):
	if len(X) != len(W) or len(X) != len(Y) or len(Y) != len(W):
		return Exception("Len of vectors should be same!")
	ans = 0
	for i in range(len(X)):
		ans += (W[i]*X[i] - Y[i])**2
	return sum(ans)/len(W)

# minimize
def MSE_l1(W, X, Y, alpha):
	if len(X) != len(W) or len(X) != len(Y) or len(Y) != len(W):
		return Exception("Len of vectors should be same!")
	ans = 0
	for i in range(len(X)):
		ans += (W[i]*X[i] - Y[i])**2
	return (sum(ans) + alpha*norm(W))/len(W)

# minimize
def MSE_l2(W, X, Y, alpha):
	if len(X) != len(W) or len(X) != len(Y) or len(Y) != len(W):
		return Exception("Len of vectors should be same!")
		ans = 0
	for i in range(len(X)):
		ans += (W[i]*X[i] - Y[i])**2
	return (sum(ans) + alpha*(norm(W)**2))/len(W)

test_data = [7., 1., 1903., 14., 0., 1., 67., 1., 78., 0., 0.]

try:
	W = LeastSquareMethod(useful_data_X[:10], useful_data_Y[:10])
	print(W)
	print(PolynomRegression(W, test_data))
except Exception as e:
	print(e)

