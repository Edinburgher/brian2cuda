# ###############################################################################
# ## RESULTS COLLECTION
import os

def write_results_csv(folder, device_name, network_count, duration, has_PRMs, is_merged, multithreading_type, uses_conditional_connect,
                      last_run_time, compilation_time, binary_run_time,
                      neurongroup_stateupdater, neurongroup_thresholder, neurongroup_resetter,
                      synapses_pre, synapses_pre_push_spikes,
                      spikemonitor, sum_ratemonitors):
    file = os.path.join(folder, "benchmark.csv")
    is_new = os.path.exists(file)
    with open(file, "a", newline='') as f:
        if not is_new:
            # Create header
            f.write("device_name,network_count,duration,has_PRMs,is_merged,multithreading_type,uses_conditional_connect,")
            f.write("last_run_time,compilation_time,binary_run_time,")
            f.write("neurongroup_stateupdater,neurongroup_thresholder,neurongroup_resetter,")
            f.write("synapses_pre,synapses_pre_push_spikes,")
            f.write("spikemonitor,sum_ratemonitors,")
            f.write("total_run_time")
            f.write("\n")
        f.write(f'{device_name},{network_count},{duration},{has_PRMs},{is_merged},{multithreading_type},{uses_conditional_connect},')
        f.write(f'{last_run_time},{compilation_time},{binary_run_time},')
        f.write(f'{neurongroup_stateupdater},{neurongroup_thresholder}, {neurongroup_resetter},')
        f.write(f'{synapses_pre},{synapses_pre_push_spikes},')
        f.write(f'{spikemonitor},{sum_ratemonitors},')

def append_total_run_time(folder, total_run_time):
    file = os.path.join(folder, "benchmark.csv")
    with open(file, "a", newline='') as f:
        f.write(f'{total_run_time}')
        f.write("\n")