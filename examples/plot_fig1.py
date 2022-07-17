from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("Falsebenchmark.csv")
data['network_count']=data['network_count'].astype(str)
openmp_sep_data = data[(data['multithreading_type'] == 'openmp') & (data['is_merged'] == False)]
singlethread_sep_data = data[(data['multithreading_type'] == 'none') & (data['is_merged'] == False)]
cuda_sep_data = data[(data['multithreading_type'] == 'GPU') & (data['is_merged'] == False)]

openmp_merged_data = data[(data['multithreading_type'] == 'openmp') & (data['is_merged'] == True)]
singlethread_merged_data = data[(data['multithreading_type'] == 'none') & (data['is_merged'] == True)]
cuda_merged_data = data[(data['multithreading_type'] == 'GPU') & (data['is_merged'] == True)]
openmp40_merged_data = data[(data['multithreading_type'] == 'openmp40') & (data['is_merged'] == True)]
# figure_data = pd.DataFrame.from_dict({
#   'OpenMP seperate': openmp_sep_data,
#   'Single Thread seperate': singlethread_sep_data,
#   'CUDA seperate': cuda_sep_data,
# })
print(openmp_sep_data)
plt.plot(openmp_sep_data['network_count'], openmp_sep_data['last_run_time'], label="OpenMP seperate", marker='o')
plt.plot(singlethread_sep_data['network_count'], singlethread_sep_data['last_run_time'], label="Single thread seperate", marker='o')
plt.plot(cuda_sep_data['network_count'], cuda_sep_data['last_run_time'], label="CUDA seperate", marker='o')

plt.plot(openmp_merged_data['network_count'], openmp_merged_data['last_run_time'], label="OpenMP merged", marker='o')
plt.plot(singlethread_merged_data['network_count'], singlethread_merged_data['last_run_time'], label="Single thread merged", marker='o')
plt.plot(cuda_merged_data['network_count'], cuda_merged_data['last_run_time'], label="CUDA merged", marker='o')
plt.plot(openmp40_merged_data['network_count'], openmp40_merged_data['last_run_time'], label="OPENMP 40 merged", marker='o')
plt.yscale('log')
plt.xticks([str(i) for i in 2 ** np.arange(9)])
#plt.plot(singlethread_sep_data)
plt.legend()
plt.savefig('fig1.png')