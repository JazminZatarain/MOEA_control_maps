import numpy as np
#different vector sizes are averaged across all seeds for each parameter
# the resulting average matrix shows x-bins of number of function evaluations, y-first parameter in the samples file 

def average_groups(a, N): # N is number of groups and a is input array
	n = len(a)
	m = n//N
	w = np.full(N,m)
	w[:n-m*N] += 1
	sums = np.add.reduceat(a, np.r_[0,w.cumsum()[:-1]])
	return np.true_divide(sums,w)

def average_metrics(input_dir,algorithm,problem,sort_id,n_parameters,n_seeds,metric_index,n_bins):
	average_vector=np.array([])
	for idx in range(len(sort_id)):
		binned_metric=np.array([])
		for seed in range(1,n_seeds+1):

			data=np.loadtxt(input_dir+algorithm+'_'+problem+'_S'+str(seed)+'_P'+str(sort_id[idx])+'.metrics', skiprows=0)
			metric=data[:,metric_index] 
			binned=(average_groups(metric, n_bins))
			binned_metric=np.append(binned_metric,binned)


		reshaped=np.reshape(binned_metric,(n_seeds,n_bins))
		average=np.mean(reshaped, axis=0) 
		average_vector=np.append(average_vector,average)

	average_matrix=np.reshape(average_vector,(n_parameters,n_bins))

	return average_matrix
















