# example script to run multiple examples, the code and results will be in:
base='benchmark_multiple_networks'
codefolder=$base/code
resultsfolder=$base/results
logfile=$base/log.txt

# Exit at first failing example run
#set -e -o pipefail

mkdir -p $base
for m in 2 4 8 16 32 64 128 256; do
	for duration in .1; do
		for monitors_flag in "no-" "" ; do
			args="--resultsfolder $resultsfolder \
				--codefolder $codefolder \
				--M $m \
				--duration $duration \
				--"$monitors_flag"monitors \
				--PRMs"

#			cmd="python brunelhakim_M_separate.py --devicename cuda_standalone $args"
#			echo $cmd
#			$cmd 2>&1 | tee -a $logfile

#			cmd="python brunelhakim_M_separate.py $args"
#			echo $cmd
#			$cmd 2>&1 | tee -a $logfile

#			cmd="python brunelhakim_M_joined.py --devicename cpp_standalone $args"
#			echo $cmd
#			$cmd 2>&1 | tee -a $logfile

			cmd="python brunelhakim_M_joined.py --devicename cuda_standalone $args"
			echo $cmd
			$cmd 2>&1 | tee -a $logfile
		done
	done
done