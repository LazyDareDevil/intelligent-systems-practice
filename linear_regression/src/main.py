import csv
import numpy as np
import math 
from pathlib import Path


data_folder = Path("linear_regression/data/")

airline_delay_causes = []
columns = []
target_variable = "DEP_DELAY"
target_col = []

with open(data_folder/'2017.csv', newline='\n') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	line_count = 0
	for row in csv_reader:
		if line_count == 0:
			columns = row.copy()
			line_count += 1
		else:
			airline_delay_causes.append(row)
			line_count += 1

empty_cols = []
target_index = 0
for col in columns:
	if len(col) == 0:
		empty_cols.append(columns.index(col))
	if col == target_variable:
		target_index = columns.index(col)

if len(empty_cols) > 0:
	for col in empty_cols:
		del columns[col]
	del columns[target_index]
	for row in airline_delay_causes:
		for col in empty_cols:
			del row[col]
		target_col.append(row[target_index])
		del row[target_index]

airline_delay_causes = np.array(airline_delay_causes)
columns = np.array(columns)
target_col = np.array(target_col)

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
		return Exception("Len of vectors should be same!")
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
		return Exception("Len of vectors should be same!")
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
		return Exception("Len of vectors should be same!")
	ans = X.copy()
	for i in range(len(Y)):
		ans[i] *= Y[i]
	return ans

def summc(A, c):
	if len(A) == 0:
		return []
	rows_A = len(A)
	cols_A = len(A[0])
	ans = A.copy()
	for i in range(rows_A):
		for j in range(cols_A):
			ans[i][j] += c
	return ans

def summm(A, B):
	if len(A) == 0 or len(B) == 0:
		return []
	if len(A) != len(B) or len(A[0]) != len(B[0]):
		return Exception("Dim of matrix should be same!")
	rows_A = len(A)
	cols_B = len(B[0])
	ans = A.copy()
	for i in range(rows_A):
		for j in range(cols_B):
			ans[i][j] += B[i][j]
	return ans

def submc(A, c):
	if len(A) == 0:
		return []
	rows_A = len(A)
	cols_A = len(A[0])
	ans = A.copy()
	for i in range(rows_A):
		for j in range(cols_A):
			ans[i][j] -= c
	return ans

def submm(A, B):
	if len(A) == 0 or len(B) == 0:
		return []
	if len(A) != len(B) or len(A[0]) != len(B[0]):
		return Exception("Dim of matrix should be same!")
	rows_A = len(A)
	cols_B = len(B[0])
	ans = A.copy()
	for i in range(rows_A):
		for j in range(cols_B):
			ans[i][j] -= B[i][j]
	return ans

def multmc(A, c):
	if len(A) == 0:
		return []
	rows_A = len(A)
	cols_A = len(A[0])
	ans = A.copy()
	for i in range(cols_A):
		for j in range(rows_A):
			ans[i][j] *= c
	return ans

def multmv(A, V):
	if len(A) == 0:
		return []
	rows_A = len(A)
	cols_A = len(A[0])
	row_V = len(V)
	if cols_A != row_V:
		return Exception("Cannot multiply the matrix and vector. Incorrect dimensions.")
	ans = A.copy()
	for i in range(rows_A):
		for j in range(cols_A):
			ans[i][j] *= V[j]
	return ans

def multmm(A, B):
	if len(A) == 0 or len(B) == 0:
		return []
	rows_A = len(A)
	cols_A = len(A[0])
	rows_B = len(B)
	cols_B = len(B[0])
	if cols_A != rows_B:
		return Exception("Cannot multiply the two matrices. Incorrect dimensions.")
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
	ans = [[0 for row in range(cols_A)] for col in range(rows_A)]
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
	return multmm(invm(multmm(transpm(A), A)), multmv(transpm(A), y))

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
	return (subvv(multvv(W, X), Y)**2)/len(W)

# minimize
def MSE_l1(W, X, Y, alpha):
	if len(X) != len(W) or len(X) != len(Y) or len(Y) != len(W):
		return Exception("Len of vectors should be same!")
	return (subvv(multvv(W, X), Y)**2)/len(W) + alpha*norm(W)

# minimize
def MSE_l2(W, X, Y, alpha):
	if len(X) != len(W) or len(X) != len(Y) or len(Y) != len(W):
		return Exception("Len of vectors should be same!")
	return (subvv(multvv(W, X), Y)**2)/len(W) + alpha*(norm(W)**2)

