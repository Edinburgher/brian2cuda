# example script to run multiple examples, the code and results will be in:
base='benchmark_multiple_networks'
codefolder=$base/code
resultsfolder=$base/results
logfile=$base/log.txt
benchmark_results=$base/benchmark_results.txt

# Exit at first failing example run
#set -e -o pipefail

mkdir -p $base
for m in 5 10 50; do
    for duration in .1 1 10; do
        for prm_flag in "" "no-"; do
            args="--resultsfolder $resultsfolder \
                --codefolder $codefolder \
                --M $m \
                --profiling \
                --monitors \
                --duration $duration \
                --"$prm_flag"PRMs"

            echo "------------------------------------------------------------------------------------" >> $benchmark_results
            echo "RESULTS FOR $m networks, $duration seconds with $prm_flag PopulationRateMonitors" >> $benchmark_results
            echo "------------------------------------------------------------------------------------" >> $benchmark_results

            start_time=$(($(date +%s%N)/1000000))
            cmd="python brunelhakim_M_joined.py --devicename cuda_standalone $args"
            echo $cmd
            $cmd 2>&1 | tee -a $logfile
            end_time=$(($(date +%s%N)/1000000))
            elapsed=$(($end_time-$start_time))
            echo "brunelhakim_M_joined (cuda, connect with j syntax): $elapsed" >> $benchmark_results

            start_time=$(($(date +%s%N)/1000000))
            cmd="python brunelhakim_M_joined.py --devicename cuda_standalone --use-conditional-connect $args"
            echo $cmd
            $cmd 2>&1 | tee -a $logfile
            end_time=$(($(date +%s%N)/1000000))
            elapsed=$(($end_time-$start_time))
            echo "brunelhakim_M_joined (cuda, conditional_connect): $elapsed" >> $benchmark_results

            start_time=$(($(date +%s%N)/1000000))
            cmd="python brunelhakim_M_joined.py --devicename cpp_standalone $args"
            echo $cmd
            $cmd 2>&1 | tee -a $logfile
            end_time=$(($(date +%s%N)/1000000))
            elapsed=$(($end_time-$start_time))
            echo "brunelhakim_M_joined (cpp, connect with j syntax): $elapsed" >> $benchmark_results

            start_time=$(($(date +%s%N)/1000000))
            cmd="python brunelhakim_M_joined.py --devicename cpp_standalone --use-conditional-connect $args"
            echo $cmd
            $cmd 2>&1 | tee -a $logfile
            end_time=$(($(date +%s%N)/1000000))
            elapsed=$(($end_time-$start_time))
            echo "brunelhakim_M_joined (cpp, conditional_connect): $elapsed" >> $benchmark_results

            start_time=$(($(date +%s%N)/1000000))
            cmd="python brunelhakim_M_separate.py $args"
            echo $cmd
            $cmd 2>&1 | tee -a $logfile
            end_time=$(($(date +%s%N)/1000000))
            elapsed=$(($end_time-$start_time))
            echo "brunelhakim_M_separate (cpp, single thread): $elapsed" >> $benchmark_results

            start_time=$(($(date +%s%N)/1000000))
            cmd="python brunelhakim_M_separate.py --multi-threading $args"
            echo $cmd
            $cmd 2>&1 | tee -a $logfile
            end_time=$(($(date +%s%N)/1000000))
            elapsed=$(($end_time-$start_time))
            echo "brunelhakim_M_separate (cpp, multi_threading): $elapsed" >> $benchmark_results
        done
    done
done
