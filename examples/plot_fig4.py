from os import sep
from turtle import color
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'
data = pd.read_csv("Truebenchmark.csv")

data = data[(data['has_monitors'] == False)]
data = data.groupby(['device_name','network_count','duration','is_recompile','has_monitors','is_merged','multithreading_type'], as_index=False).mean()
data.sort_values(['network_count'], inplace=True)
sep_data = data[(data['is_merged'] == False)]
sep_data_recompile = sep_data[(sep_data['is_recompile'] == True)]
# TODO: fresh compile + 127*recompile
sep_data.to_csv('sep_data.csv')
sep_data_recompile[
    ['network_count','compilation_time','last_run_time', 'binary_run_time']
  ] = sep_data_recompile[
    ['network_count','compilation_time','last_run_time', 'binary_run_time']
  ].mul(127, axis=0)
sep_data_recompile.to_csv('sep_data_recompile_after.csv')
sep_data_compile = sep_data[(sep_data['is_recompile'] == False)]

openmp_sep_data = sep_data[(sep_data['is_recompile'] == False) & (sep_data['multithreading_type'] == 'openmp')]
openmp_recompile_sep_data = sep_data_recompile[(sep_data_recompile['multithreading_type'] == 'openmp')]

sep_data_total = sep_data_compile.append(sep_data_recompile)
sep_data_total = sep_data_total.groupby(['device_name','duration','has_monitors','is_merged','multithreading_type'], as_index=False).sum()
sep_data_total.to_csv('sep_data_total.csv')


openmp_sep_data = sep_data_total[(sep_data_total['multithreading_type'] == 'openmp')]
singlethread_sep_data = sep_data_total[(sep_data_total['multithreading_type'] == 'none')]
cuda_sep_data = sep_data_total[(sep_data_total['multithreading_type'] == 'GPU')]
cuda_sep_data.to_csv('cuda_sep_data.csv')


data = data[(data['network_count'] == 128)]


openmp_merged_data = data[(data['multithreading_type'] == 'openmp') & (data['is_merged'] == True)]
singlethread_merged_data = data[(data['multithreading_type'] == 'none') & (data['is_merged'] == True)]
cuda_merged_data = data[(data['multithreading_type'] == 'GPU') & (data['is_merged'] == True)]


# plt.yscale('log')
# plt.ylabel('seconds')
#plt.plot(singlethread_sep_data)
# Shrink current axis by 20%
# plt.tight_layout()
# plt.legend(['Compile','Setup and Finalisation','Last run time'],loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig('fig4-openmp.png', bbox_inches='tight')
# plt.clf()

plt.bar("Single sep", singlethread_sep_data['compilation_time'], label="Compile", bottom=0)
plt.bar("Single sep", singlethread_sep_data['binary_run_time']-singlethread_sep_data['last_run_time'], label="Setup and Finalisation", bottom=singlethread_sep_data['compilation_time'])
plt.bar("Single sep", singlethread_sep_data['last_run_time'],label="Last run time", bottom=singlethread_sep_data['compilation_time']+singlethread_sep_data['binary_run_time']-singlethread_sep_data['last_run_time'])

plt.gca().set_prop_cycle(None)
plt.bar("Single mer", singlethread_merged_data['compilation_time'], label="Compile", bottom=0)
plt.bar("Single mer", singlethread_merged_data['binary_run_time']-singlethread_merged_data['last_run_time'], label="Setup and Finalisation", bottom=singlethread_merged_data['compilation_time'])
plt.bar("Single mer", singlethread_merged_data['last_run_time'],label="Last run time", bottom=singlethread_merged_data['compilation_time']+singlethread_merged_data['binary_run_time']-singlethread_merged_data['last_run_time'])

plt.gca().set_prop_cycle(None)
plt.bar("OpenMP sep", openmp_sep_data['compilation_time'], label="Compile", bottom=0)
plt.bar("OpenMP sep", openmp_sep_data['binary_run_time']-openmp_sep_data['last_run_time'], label="Setup and Finalisation", bottom=openmp_sep_data['compilation_time'])
plt.bar("OpenMP sep", openmp_sep_data['last_run_time'],label="Last run time", bottom=openmp_sep_data['compilation_time']+openmp_sep_data['binary_run_time']-openmp_sep_data['last_run_time'])

plt.gca().set_prop_cycle(None)
plt.bar("OpenMP mer", openmp_merged_data['compilation_time'], label="Compile", bottom=0)
plt.bar("OpenMP mer", openmp_merged_data['binary_run_time']-openmp_merged_data['last_run_time'], label="Setup and Finalisation", bottom=openmp_merged_data['compilation_time'])
plt.bar("OpenMP mer", openmp_merged_data['last_run_time'],label="Last run time", bottom=openmp_merged_data['compilation_time']+openmp_merged_data['binary_run_time']-openmp_merged_data['last_run_time'])

# plt.yscale('log')
# plt.ylabel('seconds')
# plt.tight_layout()
# plt.legend(['Compile','Setup and Finalisation','Last run time'],loc='center left', bbox_to_anchor=(1, 0.5))
# plt.savefig('fig4-single-thread.png', bbox_inches='tight')
# plt.clf()
plt.gca().set_prop_cycle(None)

plt.bar("CUDA sep", cuda_sep_data['compilation_time'], label="Compile", bottom=0)
plt.bar("CUDA sep", cuda_sep_data['binary_run_time']-cuda_sep_data['last_run_time'], label="Setup and Finalisation", bottom=cuda_sep_data['compilation_time'])
plt.bar("CUDA sep", cuda_sep_data['last_run_time'],label="Last run time", bottom=cuda_sep_data['compilation_time']+cuda_sep_data['binary_run_time']-cuda_sep_data['last_run_time'])

plt.gca().set_prop_cycle(None)
plt.bar("CUDA mer", cuda_merged_data['compilation_time'], label="Compile", bottom=0)
plt.bar("CUDA mer", cuda_merged_data['binary_run_time']-cuda_merged_data['last_run_time'], label="Setup and Finalisation", bottom=cuda_merged_data['compilation_time'])
plt.bar("CUDA mer", cuda_merged_data['last_run_time'],label="Last run time", bottom=cuda_merged_data['compilation_time']+cuda_merged_data['binary_run_time']-cuda_merged_data['last_run_time'])


# plt.yscale('log')
plt.ylabel('seconds')
plt.xticks(['OpenMP sep',"OpenMP mer","Single sep","Single mer","CUDA sep","CUDA mer"])

# plt.tight_layout()
plt.legend(['Compile','Setup and Finalisation','Last run time'],loc='center left', bbox_to_anchor=(1, 0.5))
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.savefig('fig4-cuda.png', bbox_inches='tight')
plt.clf()