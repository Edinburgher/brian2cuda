#devicename = 'cuda_standalone'
devicename = 'cpp_standalone'

# number of neurons
N = 5000

# number of networks to simulate
nNetworks = 2

# duration
duration = .1

# whether to profile run
profiling = True

# folder to store plots and profiling information
resultsfolder = 'results'

# folder for the code
codefolder = 'code'

# monitors (neede for plot generation)
monitors = True

# single precision
single_precision = False

# multi threading
multi_threading = False

# run multiple PRMs
run_PRMs = True

# connect Synmapses with conditional connect call
use_conditional_connect = False

###############################################################################
## CONFIGURATION

params = {'devicename': devicename,
          'resultsfolder': resultsfolder,
          'codefolder': codefolder,
          'N': N,
          'M': nNetworks,
          'profiling': profiling,
          'monitors': monitors,
          'PRMs': run_PRMs,
          'single_precision': single_precision,
          'multi_threading': multi_threading,
          'duration': duration,
          'use_conditional_connect': use_conditional_connect}

from utils import set_prefs, update_from_command_line

# update params from command line
choices={'devicename': ['cuda_standalone', 'cpp_standalone', 'genn']}
update_from_command_line(params, choices=choices)

# do the imports after parsing command line arguments (quicker --help)
import os
import matplotlib
matplotlib.use('Agg')

from matplotlib.pyplot import figure, subplot, plot, xticks, ylabel, xlim, ylim, xlabel, clf
from numpy import zeros, arange, ones
import numpy as np

import time
start = time.time()

from brian2 import *
if params['devicename'] == 'cuda_standalone':
    import brian2cuda

# set brian2 prefs from params dict
name = set_prefs(params, prefs)

codefolder = os.path.join(params['codefolder'], name)
print('running example {}'.format(name))
print('compiling model in {}'.format(codefolder))

###############################################################################
## SIMULATION

set_device(params['devicename'], directory=codefolder, compile=True, run=True,
           debug=False)

Vr = 10*mV
theta = 20*mV
tau = 20*ms
delta = 2*ms
taurefr = 2*ms
duration = duration*second
C = 1000
sparseness = float(C)/params['N']
J = .1*mV

# default values from brian2 example (say 'reference' regime)
sigmaext = 1*mV
muext = 25*mV

eqs = """
dV/dt = (-V+muext + sigmaext * sqrt(tau) * xi)/tau : volt
"""

network = Network()

group = NeuronGroup(params['N'] * params['M'], eqs, threshold='V>theta',
                    reset='V=Vr', refractory=taurefr)
group.V = Vr

network.add(group)

conn = Synapses(group, group, on_pre='V += -J', delay=delta)
print("STARTING connecting")
subgroups = []
for m in range(0, params['M']):
    lower = m * params['N']
    upper = (m+1) * params['N']
    subgroup = group[lower:upper]
    if params['use_conditional_connect']:
        conn.connect(condition='lower <= i and i < upper and lower <= j and j < upper ', p=sparseness)
        network.add(conn)
    subgroups.append(subgroup)

# NEW better way
param_N = params['N']

if not params['use_conditional_connect']:
    conn.connect(j="k for k in sample(param_N*(i//param_N), param_N*(i//param_N+1),1, p=sparseness)")
    network.add(conn)

if params['monitors']:
    spikemon = SpikeMonitor(group)
    network.add(spikemon)
    if params['PRMs']:
        PRMs = []
        # PopulationRateMonitor results can not be "untangled" after the simulation
        for subgroup in subgroups:
            PRM = PopulationRateMonitor(subgroup)
            network.add(PRM)
            PRMs.append(PRM)


network.run(duration, report='text', profile=params['profiling'])
#
# ###############################################################################
# ## RESULTS COLLECTION

if not os.path.exists(params['resultsfolder']):
    os.mkdir(params['resultsfolder']) # for plots and profiling txt file
if params['profiling']:
    print(profiling_summary())
    profilingpath = os.path.join(params['resultsfolder'], '{}.txt'.format(name))
    with open(profilingpath, 'w') as profiling_file:
        profiling_file.write(str(profiling_summary()))
        print('profiling information saved in {}'.format(profilingpath))

if params['monitors']:
    for m in range(0, params['M']):
        lower = m * params['N']
        upper = (m+1) * params['N']
        subgroup = group[lower:upper]

        subplot(211)
        points = np.c_[spikemon.t/ms, spikemon.i]
        curPoints = points[np.where((lower <= points[:,1]) & (points[:,1] < upper))].T
        # NOT WORKING.
        # Not even deepcopy itself works, so I dont see generating multiple monitor objects after the fact happening.
        # curM = deepcopy(M)

        # This also fails on M. i is readonly., I think the monitor arrays can only be modified in the generated code
        # curM.set_states()

        plot(curPoints[0], curPoints[1], '.')
        xlim(0, duration/ms)

        subplot(212)
        plot(PRMs[m].t/ms, PRMs[m].smooth_rate(window='flat', width=0.5*ms)/Hz)
        xlim(0, duration/ms)
        #show()

        plotpath = os.path.join(params['resultsfolder'], '{}.png'.format(name + "_Network_" + str(m+1)))
        savefig(plotpath)
        print('plot saved in {}'.format(plotpath))
        clf()

    print('the generated model in {} needs to removed manually if wanted'.format(codefolder))

print('It took', time.time()-start, 'seconds.')
