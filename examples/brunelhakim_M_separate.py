from multiprocessing import Process

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
multi_threading = True

# run multiple PRMs
run_PRMs = True

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
          'duration': duration}

from utils import set_prefs, update_from_command_line

# update params from command line
choices={'devicename': ['cuda_standalone', 'cpp_standalone', 'genn']}
update_from_command_line(params, choices=choices)

# do the imports after parsing command line arguments (quicker --help)
import os
import matplotlib
matplotlib.use('Agg')

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


def run_sim(m):
    set_device(params['devicename'], directory=None, run=True, debug=False)
    network = Network()

    group = NeuronGroup(params['N'], eqs, threshold='V>theta',
                        reset='V=Vr', refractory=taurefr)
    group.V = Vr
    network.add(group)

    conn = Synapses(group, group, on_pre='V += -J', delay=delta)
    conn.connect(p=sparseness)
    network.add(conn)

    if params['monitors']:
        spikemon = SpikeMonitor(group)
        network.add(spikemon)
        if params['PRMs']:
            PRM = PopulationRateMonitor(group)
            network.add(PRM)

    network.run(duration, report='text', profile=params['profiling'])
    #
    # ###############################################################################
    # ## RESULTS COLLECTION

    if not os.path.exists(params['resultsfolder']):
        os.mkdir(params['resultsfolder'])  # for plots and profiling txt file
    if params['profiling']:
        print(profiling_summary())
        profilingpath = os.path.join(params['resultsfolder'], '{}_{}.txt'.format(name, str(m + 1)))
        with open(profilingpath, 'w') as profiling_file:
            profiling_file.write(str(profiling_summary()) + '\n_last_run_time = ' + str(device._last_run_time))
            print('profiling information saved in {}'.format(profilingpath))

    if params['monitors']:
        subplot(211)
        plot(spikemon.t / ms, spikemon.i, '.')
        xlim(0, duration / ms)

        if params['PRMs']:
            subplot(212)
            plot(PRM.t / ms, PRM.smooth_rate(window='flat', width=0.5 * ms) / Hz)
            xlim(0, duration / ms)
            # show()

        plotpath = os.path.join(params['resultsfolder'], '{}.png'.format(name + "_Network_" + str(m + 1)))
        savefig(plotpath)
        print('plot saved in {}'.format(plotpath))
        clf()
    device.reinit()
    device.activate()

    print('the generated model in {} needs to removed manually if wanted'.format(codefolder))

if params['multi_threading']:
    ps = []
    for m in range(0, params['M']):
        p=Process(target=run_sim, args=(m,))
        p.start()
        ps.append(p)

    for p in ps:
        p.join()
else:
    for m in range(0, params['M']):
        run_sim(m)

print('It took', time.time()-start, 'seconds.')
