import csv
import numpy as np


airline_delay_causes = []
columns = []
target_variable = "DEP_DELAY"
target_col = []

with open('2017.csv', newline='\n') as csv_file:
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

def MSE(W, X, Y):
	if len(X) != len(W) or len(X) != len(Y) or len(Y) != len(W):
		return Exception("Len of vectors should be same!")
	return (np.sum(W*X - Y)**2)/len(W)

