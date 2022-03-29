
from matplotlib.pyplot import figure, subplot, plot, xticks, ylabel, xlim, ylim, xlabel, clf
from numpy import zeros, arange, ones
import time
start = time.time()
#devicename = 'cuda_standalone'
devicename = 'cpp_standalone'

# number of neurons
N = 5000

# number of networks to simulate
nNetworks = 2

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
          'single_precision': single_precision,
          'num_blocks': num_blocks,
          'atomics': atomics,
          'bundle_mode': bundle_mode}

from utils import set_prefs, update_from_command_line

# update params from command line
choices={'devicename': ['cuda_standalone', 'cpp_standalone', 'genn']}
update_from_command_line(params, choices=choices)

# do the imports after parsing command line arguments (quicker --help)
import os
import matplotlib
matplotlib.use('Agg')

from brian2 import *
if params['devicename'] == 'cuda_standalone':
    import brian2cuda

# set brian2 prefs from params dict
name = set_prefs(params, prefs)

codefolder = os.path.join(params['codefolder'], name)
print('runing example {}'.format(name))
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
duration = .1*second
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
    conn.connect(condition='lower <= i and i < upper and lower <= j and j < upper ', p=sparseness)
    subgroups.append(subgroup)
network.add(conn)


# i_index = []
# for m in range(0,M):
#   for k in range(m*N,(m+1)*N):
#     for i in range(0,N):
#       i_index.__iadd__([k])
# #print(i_index)
#
# print("FINISHED i_index")
# j_index = []
# for m in range(0,M):
#   for k in range(m*N,(m+1)*N):
#     for j in range(0,N):
#       j_index.__iadd__([j+(m*N)])
# #print(j_index)
#
# print("FINISHED j_index")
#
# NOT WORKING: only one iterator allowed
#conn.connect(j='j+(m*N) for m in range(0,M) for k in range(m*N,(m+1)*N) for j in range(0,N)', skip_if_invalid=True)

# if params['monitors']:
#     M = SpikeMonitor(group)
#     LFP = PopulationRateMonitor(group)

if params['monitors']:
    Ms = []
    LFPs = []
    for subgroup in subgroups:
        monitor = SpikeMonitor(subgroup)
        LFP = PopulationRateMonitor(subgroup)
        network.add(monitor)
        network.add(LFP)
        Ms.append(monitor)
        LFPs.append(LFP)


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

    for index, subgroup in enumerate(subgroups):
        subplot(211)
        plot(Ms[index].t/ms, Ms[index].i, '.')
        xlim(0, duration/ms)

        subplot(212)
        plot(LFPs[index].t/ms, LFPs[index].smooth_rate(window='flat', width=0.5*ms)/Hz)
        xlim(0, duration/ms)
        #show()

        plotpath = os.path.join(params['resultsfolder'], '{}.png'.format(name + "_Network_" + str(index+1)))
        savefig(plotpath)
        print('plot saved in {}'.format(plotpath))
        clf()

    print('the generated model in {} needs to removed manually if wanted'.format(codefolder))

print('It took', time.time()-start, 'seconds.')
