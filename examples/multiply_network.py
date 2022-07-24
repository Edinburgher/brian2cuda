import pprint
from brian2 import *
import os
from utils import set_prefs
def multiplyNetwork(runs, M, params):
    name = set_prefs(params, prefs)

    codefolder = os.path.join(params['codefolder'], name)
    set_device('cpp_standalone', directory=codefolder, compile=True, run=True, debug=False)


    # pprint.pprint(runs)
    export = runs[0]
    pprint.pprint(export)
    network = Network()
    neurongroup_export = export['components']['neurongroup'][0]
    globals()['param_N'] = neurongroup_export['N']
    globals()['param_M'] = M
    pprint.pprint(neurongroup_export['identifiers'])
    for key in (neurongroup_export['identifiers']):
        globals()[key] = neurongroup_export['identifiers'][key]

    if(neurongroup_export['equations']['V']['type'] == "differential equation"):
        eqs = """
        dV/dt = """ + neurongroup_export['equations']['V']['expr'] + """ : volt
        """

    print(eqs)
    neurongroup_parameters = neurongroup_export['events']['spike']


    group = NeuronGroup(param_N * param_M, eqs, threshold=neurongroup_parameters['threshold']['code'],
                         reset=neurongroup_parameters['reset']['code'], refractory=neurongroup_parameters['refractory'],)
    for initializer in export['initializers_connectors']:
        if initializer['source'] == 'neurongroup' and initializer['type'] == 'initializer':
            pprint.pprint(initializer)
            group.__setattr__(initializer['variable'], initializer['value'])

    network.add(group)

    synapses_export = export['components']['synapses'][0]
    synapses_parameters = synapses_export['pathways'][0]

    for key in (synapses_export['identifiers']):
        globals()[key] = synapses_export['identifiers'][key]
    conn = Synapses(group, group, on_pre=synapses_parameters['code'], delay=synapses_parameters['delay'])
    print("STARTING connecting")
    for initializer in export['initializers_connectors']:
        if initializer['source'] == 'neurongroup' and initializer['type'] == 'connect':
            pprint.pprint(initializer)
            globals()['probability'] = initializer['probability']
    conn.connect(j="k for k in sample(i%param_M, param_M*param_N, param_M, p=probability)")
    network.add(conn)


    subgroups = []
    for m in range(0, param_M):
        lower = m * param_N
        upper = (m+1) * param_N
        subgroup = group[lower:upper]
        subgroups.append(subgroup)

    for subgroup in subgroups:
        PRM = PopulationRateMonitor(subgroup)
        network.add(PRM)
        #PRMs.append(PRM)

    spikemon = SpikeMonitor(group)
    network.add(spikemon)
    # TODO: generalise record
    statemon = StateMonitor(group, export['components']['statemonitor'][0]['variables'][0], record=[m*param_N for m in range(0, param_M)])
    network.add(statemon)

    return network
