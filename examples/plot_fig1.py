from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("Falsebenchmark_new.csv")
data = data.groupby(['device_name','network_count','duration','is_merged','multithreading_type'], as_index=False).mean()
data.to_csv('data.csv')
sep_data = data[(data['is_merged'] == False)]
len_sep_data=len(sep_data['network_count'])
sep_data = pd.DataFrame(np.repeat(sep_data.values, 9, axis=0), columns=sep_data.columns)
sep_data.sort_values(['multithreading_type'], inplace=True)
sep_data.to_csv('sep_data.csv')

sep_data[
    ['network_count','last_run_time', 'binary_run_time']
  ] = sep_data[
    ['network_count','last_run_time', 'binary_run_time']
  ].mul([i for i in 2 ** np.arange(9)]*len_sep_data, axis=0)
sep_data['network_count']=sep_data['network_count'].astype(str)
data['network_count']=data['network_count'].astype(str)

openmp_sep_data = sep_data[(sep_data['multithreading_type'] == 'openmp') ]
singlethread_sep_data = sep_data[(sep_data['multithreading_type'] == 'none')]
cuda_sep_data = sep_data[(sep_data['multithreading_type'] == 'GPU')]

openmp_merged_data = data[(data['multithreading_type'] == 'openmp') & (data['is_merged'] == True)]
singlethread_merged_data = data[(data['multithreading_type'] == 'none') & (data['is_merged'] == True)]
cuda_merged_data = data[(data['multithreading_type'] == 'GPU') & (data['is_merged'] == True)]
# openmp40_merged_data = data[(data['multithreading_type'] == 'openmp40') & (data['is_merged'] == True)]

print(openmp_sep_data)
plt.plot(openmp_sep_data['network_count'], openmp_sep_data['last_run_time'], label="OpenMP seperate", marker='o', color='r', linestyle='dashed')
plt.plot(singlethread_sep_data['network_count'], singlethread_sep_data['last_run_time'], label="Single thread seperate", marker='o', color='b', linestyle='dashed')
plt.plot(cuda_sep_data['network_count'], cuda_sep_data['last_run_time'], label="CUDA seperate", marker='o', color='g', linestyle='dashed')

plt.plot(openmp_merged_data['network_count'], openmp_merged_data['last_run_time'], label="OpenMP merged", marker='o', color='r')
plt.plot(singlethread_merged_data['network_count'], singlethread_merged_data['last_run_time'], label="Single thread merged", marker='o', color='b')
plt.plot(cuda_merged_data['network_count'], cuda_merged_data['last_run_time'], label="CUDA merged", marker='o', color='g')
# plt.plot(openmp40_merged_data['network_count'], openmp40_merged_data['last_run_time'], label="OPENMP 40 merged", marker='o')
plt.yscale('log')
plt.xticks([str(i) for i in 2 ** np.arange(9)])
plt.ylabel('seconds')
plt.xlabel('Number of networks')

#plt.plot(singlethread_sep_data)
plt.legend()
plt.savefig('fig1_new.png')