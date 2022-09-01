from os import sep
from turtle import color
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

for has_monitors in [True, False]:
  data = pd.read_csv("Truebenchmark_new.csv")
  monitor_string = "monitors" if has_monitors else "no-monitors"
  data = data[(data['has_monitors'] == has_monitors)]
  data = data.groupby(['device_name','network_count','duration','has_monitors','is_merged','multithreading_type'], as_index=False).mean()
  data.to_csv('GROUPEDTruebenchmark.csv')
  data.sort_values(['network_count'], inplace=True)
  sep_data = data[(data['is_merged'] == False)]
  len_sep_data=len(sep_data['network_count'])
  sep_data = pd.DataFrame(np.repeat(sep_data.values, 9, axis=0), columns=sep_data.columns)
  sep_data.sort_values(['multithreading_type'], inplace=True)
  sep_data[
      ['total_run_time','network_count','last_run_time', 'binary_run_time',"neurongroup_stateupdater","neurongroup_thresholder",'neurongroup_resetter',"synapses_pre","synapses_pre_push_spikes","spikemonitor","statemonitor","sum_ratemonitors"]
    ] = sep_data[
      ['total_run_time','network_count','last_run_time', 'binary_run_time',"neurongroup_stateupdater","neurongroup_thresholder",'neurongroup_resetter',"synapses_pre","synapses_pre_push_spikes","spikemonitor","statemonitor","sum_ratemonitors"]
    ].mul([i for i in 2 ** np.arange(9)]*len_sep_data, axis=0)

  sep_data.to_csv('sep_data.csv')
  sep_data['network_count']=sep_data['network_count'].astype(str)
  data['network_count']=data['network_count'].astype(str)


  openmp_sep_data = sep_data[(sep_data['multithreading_type'] == 'openmp')]
  singlethread_sep_data = sep_data[(sep_data['multithreading_type'] == 'none')]
  cuda_sep_data = sep_data[(sep_data['multithreading_type'] == 'GPU')]
  singlethread_sep_data.to_csv('singlethread_sep_data.csv')
  openmp_sep_data.to_csv('openmp_sep_data.csv')

  openmp_merged_data = data[(data['multithreading_type'] == 'openmp') & (data['is_merged'] == True)]
  singlethread_merged_data = data[(data['multithreading_type'] == 'none') & (data['is_merged'] == True)]
  cuda_merged_data = data[(data['multithreading_type'] == 'GPU') & (data['is_merged'] == True)]

  datapoints = ['total_run_time','last_run_time',"neurongroup_stateupdater","neurongroup_thresholder",'neurongroup_resetter',"synapses_pre","synapses_pre_push_spikes"]
  if has_monitors:
    datapoints.append("spikemonitor")
    datapoints.append("statemonitor")
    datapoints.append("sum_ratemonitors")
  for profiling_datapoint in datapoints:
    plt.plot(openmp_sep_data['network_count'], openmp_sep_data[profiling_datapoint], label="OpenMP seperate", marker='o', color='r', linestyle='dashed')
    plt.plot(singlethread_sep_data['network_count'], singlethread_sep_data[profiling_datapoint], label="Single thread seperate", marker='o', color='b', linestyle='dashed')
    plt.plot(cuda_sep_data['network_count'], cuda_sep_data[profiling_datapoint], label="CUDA seperate", marker='o', color='g', linestyle='dashed')

    plt.plot(openmp_merged_data['network_count'], openmp_merged_data[profiling_datapoint], label="OpenMP merged", marker='o', color='r')
    plt.plot(singlethread_merged_data['network_count'], singlethread_merged_data[profiling_datapoint], label="Single thread merged", marker='o', color='b')
    plt.plot(cuda_merged_data['network_count'], cuda_merged_data[profiling_datapoint], label="CUDA merged", marker='o', color='g')
    plt.yscale('log')
    plt.ylabel('seconds')
    plt.xlabel('Number of networks')
    plt.xticks([str(i) for i in 2 ** np.arange(9)])
    #plt.plot(singlethread_sep_data)
    plt.legend()
    plt.title(monitor_string + '-'+ profiling_datapoint)
    plt.savefig('fig2-000-new-' + monitor_string + '-'+ profiling_datapoint +'.png')
    plt.clf()