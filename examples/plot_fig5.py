from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
x = np.linspace(0,20,2)
openmp_y = 18.364056 + 58.809367*x
cuda_y = 152.981124 + 42.1145*x
plt.plot(x, openmp_y, label="OpenMP merged", marker='o', color='r')
plt.plot(x, cuda_y, label="CUDA merged", marker='o', color='g')
# plt.plot(openmp40_merged_data['network_count'], openmp40_merged_data['last_run_time'], label="OPENMP 40 merged", marker='o')
# plt.yscale('log')
# plt.xticks([str(i) for i in 2 ** np.arange(9)])
plt.ylabel('Compile time + simulation time [seconds]')
plt.xlabel('Biological real time simulated [seconds]')

#plt.plot(singlethread_sep_data)
plt.legend()
plt.savefig('fig5.png')