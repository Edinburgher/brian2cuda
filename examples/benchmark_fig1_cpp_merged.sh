base='benchmark_fig1'
codefolder=$base/code
resultsfolder=$base/results
logfile=$base/log.txt

mkdir -p $base
for openmp in "no-" ""; do
	for m in 1 2 4 8 16 32 64 128 256; do
		args="--resultsfolder $resultsfolder \
			--codefolder $codefolder \
			--M $m \
			--duration 0.1\
			--${openmp}openmp"

		cmd="python brunelhakim_M_joined.py --devicename cpp_standalone --no-profiling  \
																				--no-monitors $args"
		echo $cmd
		$cmd 2>&1 | tee -a $logfile

	#			cmd="python brunelhakim_M_joined.py --devicename cpp_standalone $args"
	#			echo $cmd
	#			$cmd 2>&1 | tee -a $logfile
	done
done