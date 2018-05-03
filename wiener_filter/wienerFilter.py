import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import scipy
from scipy import signal


def loadSignal(fileName):
	# load any given signal
	return np.loadtxt(fileName)

def signalFilteration(signal, c, sigma2v, filterOrder):
	# list of Ryy's
	Ryy  = []

	# list of rows for Ryy Matrix
	rows = []

	# full correlation between the signal and itself to get Ryy's
	out = np.correlate(signal, signal, "full")

	# build the list of Ryy[0], Ryy[1], ... Ryy[N-1]
	[(Ryy.append(out[(len(out)/2) + i]), rows.append([])) for i in range(filterOrder)]

	# build rows of the matrix
	[rows[i].append(Ryy[abs(i-j)]) for j in range(filterOrder) for i in range(filterOrder)]
			
	# put the initial element of the last matrix
	mat  = [Ryy[0] - sigma2v]

	# put the other elements based on the filter order
	[mat.append(Ryy[i]) for i in range(1, filterOrder)]
		
	# build the final matrices
	Ryy_matrix = np.array(rows)
	matrix = (np.array(mat)).reshape(filterOrder, 1)
	c_matrix = c * (np.eye(filterOrder))

	# inverse the desired ones
	Ryy_matrix = inv(Ryy_matrix)
	c_matrix = inv(c_matrix)

	# apply matrix multiplication
	h_matrix = np.matmul(Ryy_matrix, np.matmul(c_matrix, matrix))

	# reshape the h matrix into a row vector
	h_matrix = h_matrix.reshape(1, filterOrder)[0]

	# return the filtered signal
	return np.convolve(signal, h_matrix)[:len(distorted_ECG)]
	

if __name__=='__main__':
	# read model Parameters c and sigmav^2 from user
	c = -3
	var = 0.02

	# read filter order from user
	filter_order = int(raw_input('Enter Filter Order: '))	# 3

	# load distorted signal
	distorted_ECG = loadSignal('distorted_ECG.txt')

	# load the original signal 
	Original_ECG = loadSignal('Original_ECG.txt')

	# make them zero mean
	Original_ECG -= np.mean(Original_ECG)
	distorted_ECG -= np.mean(distorted_ECG)
	
	n = np.array(range(len(distorted_ECG)))

	# apply filter
	filteredSignal = signalFilteration(distorted_ECG, c, var, filter_order)
	filteredSignal = np.convolve(filteredSignal,[0.25, 0.25, 0.25, 0.25],'same')

	# show the result
	plt.figure('Distorted Vs. Filtered')
	plt.xlim(0,len(n))
	plt.plot(n, filteredSignal, 'b', label='Filtered Signal')
	plt.plot(n, distorted_ECG, 'r', label='Distored Signal')
	plt.savefig('output_filtered_distorted.png')
	plt.legend()

	# plot it with filtered one
	plt.figure("Original vs Filtered")
	plt.xlim(0,len(n))
	plt.plot(n, filteredSignal, 'b', label='Filtered Signal')
	plt.plot(n, Original_ECG, 'g', label='Original Signal')
	plt.savefig('outut_filtered_original.png')
	plt.legend()


	# get the mean square error
	meanSqrErr = np.mean(np.sqrt((Original_ECG - filteredSignal)**2))

	print(meanSqrErr)

	plt.show()