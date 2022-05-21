import ctypes
from multiprocessing import Process, Value

from examples.write_results_csv import append_total_run_time, write_results_csv


def write_profiling_data(device, network, m):
    # ###############################################################################
    # ## RESULTS COLLECTION

    if not os.path.exists(params['resultsfolder']):
        os.mkdir(params['resultsfolder'])  # for plots and profiling txt file
    if True:
        multi_threading_type = 'multiprocess' if params['multi_processing'] else 'None'
        profiling_dict = dict(network.profiling_info)
        sum_ratemonitors = sum([v for (k, v) in profiling_dict.items() if 'ratemonitor' in k])
        write_results_csv(
            benchmarkfolder, network_count=params['M'],
            device_name=params['devicename'], duration=params['duration'], has_PRMs=params['PRMs'], is_merged=False,
            multithreading_type=multi_threading_type, uses_conditional_connect='N/A',
            last_run_time=device._last_run_time, compilation_time=device.timers['compile']['all'],
            binary_run_time=device.timers['run_binary'],
            neurongroup_stateupdater=profiling_dict['neurongroup_stateupdater_codeobject'],
            neurongroup_thresholder=profiling_dict['neurongroup_thresholder_codeobject'],
            neurongroup_resetter=profiling_dict['neurongroup_resetter_codeobject'],
            synapses_pre=profiling_dict['synapses_pre_codeobject'],
            synapses_pre_push_spikes=profiling_dict['synapses_pre_push_spikes'],
            spikemonitor=profiling_dict['spikemonitor_codeobject'],
            # statemonitor=profiling_dict['statemonitor_codeobject'],
            sum_ratemonitors=sum_ratemonitors,
        )
    profilingpath = os.path.join(params['resultsfolder'], '{}_{}.txt'.format(name, str(m + 1)))
    with open(profilingpath, 'w') as profiling_file:
        profiling_file.write(
            str(profiling_summary()) +
            '\n_last_run_time = ' + str(device._last_run_time) +
            '\ncompilation time = ' + str(device.timers['compile']['all']) +
            '\nbinary run time = ' + str(device.timers['run_binary'])
        )
        print('profiling information saved in {}'.format(profilingpath))


def print_results(spikemon, PRM, m):

    #
    # ###############################################################################
    # ## RESULTS COLLECTION

    if not os.path.exists(params['resultsfolder']):
        os.mkdir(params['resultsfolder'])  # for plots and profiling txt file
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



devicename = 'cuda_standalone'

# number of neurons
N = 5000

# number of networks to simulate
nNetworks = 3

# duration
duration = .1

# folder to store plots and profiling information
resultsfolder = 'results'

# folder for the code
codefolder = 'code'

# monitors (neede for plot generation)
monitors = True

# single precision
single_precision = False

# multi threading
multi_processing = False

# benchmark folder
benchmarkfolder = '.'

# run multiple PRMs
run_PRMs = True

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
          'monitors': monitors,
          'PRMs': run_PRMs,
          'single_precision': single_precision,
          'multi_processing': multi_processing,
          'duration': duration,
          'partitions': num_blocks,
          'atomics': atomics,
          'bundle_mode': bundle_mode}

from utils import set_prefs, update_from_command_line

# update params from command line
choices={'devicename': ['cuda_standalone', 'cpp_standalone', 'genn']}
update_from_command_line(params)

# do the imports after parsing command line arguments (quicker --help)
import os
import matplotlib
matplotlib.use('Agg')

import time
start = time.time()

from brian2 import *
if params['devicename'] == 'cuda_standalone':
    import brian2cuda

#params['cpp_threads'] = 16
# set brian2 prefs from params dict
name = set_prefs(params, prefs)

codefolder = os.path.join(params['codefolder'], name)
print('running example {}'.format(name))
print('compiling model in {}'.format(codefolder))

###############################################################################
## SIMULATION

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

times = {'last_run': 0.0, 'compile': 0.0, 'run_binary': 0.0}
if params['multi_processing']:
    counter = Value(ctypes.py_object, times)
else:
    set_device(params['devicename'], directory=codefolder, build_on_run=False, debug=False)

networks = {}
PRMs = {}
spikemons = {}
statemons = {}

def run_sim(m):
    if params['multi_processing']:
        set_device(params['devicename'], directory=codefolder+str(m), run=True, debug=False)
    else:
        set_device(params['devicename'], directory=codefolder, run=True, debug=False)
    networks[m] = Network()

    group = NeuronGroup(params['N'], eqs, threshold='V>theta',
                        reset='V=Vr', refractory=taurefr)
    group.V = Vr
    networks[m].add(group)

    conn = Synapses(group, group, on_pre='V += -J', delay=delta)
    conn.connect(p=sparseness)
    networks[m].add(conn)

    if params['monitors']:
        statemons[m] = StateMonitor(group, 'V', record=True)
        networks[m].add(statemons[m])
        spikemons[m] = SpikeMonitor(group)
        networks[m].add(spikemons[m])
        if params['PRMs']:
            PRMs[m] = PopulationRateMonitor(group)
            networks[m].add(PRMs[m])
    networks[m].run(duration, report='text', profile=True)
    if True or params['multi_processing']:
        if m == 0:
            write_profiling_data(device, networks[m], m)
        # print_results(spikemons[m], PRMs[m], m)
    device.reinit()
    device.activate()

    print('the generated model in {} needs to removed manually if wanted'.format(codefolder))

if params['multi_processing']:
    ps = []
    for m in range(0, params['M']):
        p=Process(target=run_sim, args=(m,))
        p.start()
        ps.append(p)

    for p in ps:
        p.join()

    times = counter.value
else:
    for m in range(0, params['M']):
        run_sim(m)
    #device.build(directory=codefolder, compile=True, run=True, debug=False, clean=True)
    #write_profiling_data(device, networks[m], m-1)
    # for m in range(0, params['M']):
    #     print_results(spikemons[m], PRMs[m], m)

    print('_last_run_time = ', device._last_run_time)
    print('compilation time = ', device.timers['compile']['all'])
    print('Binary run time: ', device.timers['run_binary'])
print('Total time: ', time.time()-start, 'seconds.')
append_total_run_time(benchmarkfolder,time.time()-start)
