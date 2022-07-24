#devicename = 'cuda_standalone'
devicename = 'cpp_standalone'

# number of neurons
N = 5000

# number of networks to simulate
nNetworks = 2

# duration
duration = 1

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
openmp = False

# run multiple PRMs
run_PRMs = True

# connect Synmapses with conditional connect call
use_conditional_connect = False

# benchmark folder
benchmarkfolder = '.'

## the preferences below only apply for cuda_standalone

# number of post blocks (None is default)
num_blocks = None

# atomic operations
atomics = True

# push synapse bundles
bundle_mode = True

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
          'openmp': openmp,
          'duration': duration,
          'use_conditional_connect': use_conditional_connect,
          'partitions': num_blocks,
          'atomics': atomics,
          'bundle_mode': bundle_mode}

from brian2.groups import neurongroup
from utils import set_prefs, update_from_command_line

update_from_command_line(params)

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


if params['devicename'] == 'cpp_standalone' and params['openmp']:
    params['cpp_threads'] = 20
    multi_threading_type = "openmp"
elif params['devicename'] == 'cuda_standalone':
    multi_threading_type = "GPU"
else:
    multi_threading_type = "none"

# set brian2 prefs from params dict
name = set_prefs(params, prefs)

codefolder = os.path.join(params['codefolder'], name)
print('running example {}'.format(name))
print('compiling model in {}'.format(codefolder))

###############################################################################
## SIMULATION

#set_device(params['devicename'], directory=codefolder, compile=True, run=True,
        #   debug=False)


###############################################################################
## SIMULATION

import brian2tools.baseexport
set_device('exporter')
from multiply_network import multiplyNetwork





Vr = 10*mV
theta = 20*mV
tau = 20*ms
delta = 2*ms
taurefr = 2*ms
duration = params['duration']*second
C = 1000
sparseness = float(C)/params['N']
J = .1*mV

# default values from brian2 example (say 'reference' regime)
sigmaext = 1*mV
muext = 25*mV

eqs = """
dV/dt = (-V+muext + sigmaext * sqrt(tau) * xi)/tau : volt
"""

#
Vr = 10 * mV
theta = 20 * mV
tau = 20 * ms
delta = 2 * ms
taurefr = 2 * ms
duration = params['duration'] * second
C = 1000
sparseness = float(C) / params['N']
J = .1 * mV

# default values from brian2 example (say 'reference' regime)
sigmaext = 1 * mV
muext = 25 * mV

eqs = """
dV/dt = (-V+muext + sigmaext * sqrt(tau) * xi)/tau : volt
"""
# set_device(params['devicename'], directory=codefolder, build_on_run=False, debug=False)


network_single = Network()

group = NeuronGroup(params['N'], eqs, threshold='V>theta',
                    reset='V=Vr', refractory=taurefr)
group.V = Vr
network_single.add(group)

conn = Synapses(group, group, on_pre='V += -J', delay=delta)
conn.connect(p=sparseness)
network_single.add(conn)

if params['monitors']:
    statemon = StateMonitor(group, 'V', record=[0])
    network_single.add(statemon)
    spikemon = SpikeMonitor(group)
    network_single.add(spikemon)
    if params['PRMs']:
        PRM = PopulationRateMonitor(group)
        network_single.add(PRM)

network_single.run(1*second)
network_multi = multiplyNetwork(device.runs, nNetworks, params)


network_multi.run(duration)
# network_single.run(duration, report='text', profile=params['profiling'])

#
