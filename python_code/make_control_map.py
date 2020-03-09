import numpy as np
from average_metrics import average_metrics
import os
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from matplotlib.ticker import MaxNLocator

param_dir='../parameters/'
input_dir='../data_metrics/'
plot_dir='../plots/'

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

problem='LakeProblem'
algorithm='Borg'
metric_label='Hypervolume'
name=metric_label+'_'+ algorithm+'_'+problem
n_bins=90 #metric vectors into equal bins to average across seeds
n_parameters=20
n_seeds=5
metric_index=0 #hypervolume; 1 -generational distance; 2- igd, 3- spacing; 4- epsilon indicator; 5- maximum pareto front error

params=np.loadtxt(param_dir+algorithm+'_Samples.txt', usecols=0) # in this case population size is column 0
sorted_params=np.sort(params) #
sort_id=np.argsort(params)+1 # 

average_matrix=average_metrics(input_dir,algorithm,problem,sort_id,n_parameters,n_seeds,metric_index,n_bins)

### for normalized values #####
#max_metric=np.max(average_matrix)
#min_metric=np.min(average_matrix)
#normalized_matrix=(average_matrix-min_metric)/(max_metric-min_metric) # optional normalization
###############################

filtered_arr=gaussian_filter(average_matrix, sigma=5) # smooths the appearance of the control map while preserving the patterns of the control map
nfe=np.linspace(1,15,num=n_bins)

fig, ax= plt.subplots()
cmap=plt.get_cmap('RdBu')
im = ax.contourf(nfe, sorted_params, filtered_arr, cmap=cmap) 

font_size=30
left=0.16; bottom=0.15; right=0.95; top=0.94; wspace=0.2; hspace=0.2

cbar=fig.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=font_size)

plt.title(name, fontsize=font_size)
plt.ylabel('Population size',fontsize=font_size)
plt.xlabel(r'Number of function evaluations $x 10^4$', fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)

plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
fig.set_size_inches(10,11)
plt.savefig(plot_dir+name+'.pdf', transparent=True)
plt.savefig(plot_dir+name+'.png', transparent=True)
plt.show()


