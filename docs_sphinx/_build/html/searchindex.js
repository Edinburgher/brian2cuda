Search.setIndex({docnames:["examples/brunelhakim","examples/cobahh","examples/compartmental.bipolar_cell_cpp","examples/compartmental.bipolar_cell_cuda","examples/compartmental.bipolar_with_inputs2_cpp","examples/compartmental.bipolar_with_inputs2_cuda","examples/compartmental.bipolar_with_inputs_cpp","examples/compartmental.bipolar_with_inputs_cuda","examples/compartmental.hh_with_spikes_cpp","examples/compartmental.hh_with_spikes_cuda","examples/compartmental.hodgkin_huxley_1952_cpp","examples/compartmental.hodgkin_huxley_1952_cuda","examples/compartmental.lfp_cpp","examples/compartmental.lfp_cuda","examples/compartmental.plots.hh_with_spikes_checkresults","examples/compartmental.rall_cpp","examples/compartmental.rall_cuda","examples/compartmental.spike_initiation_cpp","examples/compartmental.spike_initiation_cuda","examples/cuba","examples/cuba_cuda","examples/index","examples/mushroombody","examples/stdp","examples/utils","index","introduction/index","introduction/preferences","reference/brian2cuda","reference/brian2cuda.codeobject.CUDAStandaloneAtomicsCodeObject","reference/brian2cuda.codeobject.CUDAStandaloneCodeObject","reference/brian2cuda.cuda_generator.CUDAAtomicsCodeGenerator","reference/brian2cuda.cuda_generator.CUDACodeGenerator","reference/brian2cuda.cuda_generator.ParallelisationError","reference/brian2cuda.cuda_prefs.compute_capability_validator","reference/brian2cuda.device.CUDAStandaloneDevice","reference/brian2cuda.device.CUDAWriter","reference/brian2cuda.device.check_codeobj_for_rng","reference/brian2cuda.device.cuda_standalone_device","reference/brian2cuda.utils","reference/brian2cuda.utils.gputools.get_available_gpus","reference/brian2cuda.utils.gputools.get_best_gpu","reference/brian2cuda.utils.gputools.get_compute_capability","reference/brian2cuda.utils.gputools.get_cuda_installation","reference/brian2cuda.utils.gputools.get_cuda_path","reference/brian2cuda.utils.gputools.get_cuda_runtime_version","reference/brian2cuda.utils.gputools.get_gpu_selection","reference/brian2cuda.utils.gputools.get_nvcc_path","reference/brian2cuda.utils.gputools.reset_cuda_installation","reference/brian2cuda.utils.gputools.reset_gpu_selection","reference/brian2cuda.utils.gputools.restore_cuda_installation","reference/brian2cuda.utils.gputools.restore_gpu_selection","reference/brian2cuda.utils.gputools.select_gpu","reference/brian2cuda.utils.logger.suppress_brian2_logs","reference/brian2cuda.utils.stringtools.append_f","reference/brian2cuda.utils.stringtools.replace_floating_point_literals"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.cpp":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,"sphinx.ext.intersphinx":1,"sphinx.ext.todo":1,"sphinx.ext.viewcode":1,sphinx:54},filenames:["examples/brunelhakim.rst","examples/cobahh.rst","examples/compartmental.bipolar_cell_cpp.rst","examples/compartmental.bipolar_cell_cuda.rst","examples/compartmental.bipolar_with_inputs2_cpp.rst","examples/compartmental.bipolar_with_inputs2_cuda.rst","examples/compartmental.bipolar_with_inputs_cpp.rst","examples/compartmental.bipolar_with_inputs_cuda.rst","examples/compartmental.hh_with_spikes_cpp.rst","examples/compartmental.hh_with_spikes_cuda.rst","examples/compartmental.hodgkin_huxley_1952_cpp.rst","examples/compartmental.hodgkin_huxley_1952_cuda.rst","examples/compartmental.lfp_cpp.rst","examples/compartmental.lfp_cuda.rst","examples/compartmental.plots.hh_with_spikes_checkresults.rst","examples/compartmental.rall_cpp.rst","examples/compartmental.rall_cuda.rst","examples/compartmental.spike_initiation_cpp.rst","examples/compartmental.spike_initiation_cuda.rst","examples/cuba.rst","examples/cuba_cuda.rst","examples/index.rst","examples/mushroombody.rst","examples/stdp.rst","examples/utils.rst","index.rst","introduction/index.rst","introduction/preferences.rst","reference/brian2cuda.rst","reference/brian2cuda.codeobject.CUDAStandaloneAtomicsCodeObject.rst","reference/brian2cuda.codeobject.CUDAStandaloneCodeObject.rst","reference/brian2cuda.cuda_generator.CUDAAtomicsCodeGenerator.rst","reference/brian2cuda.cuda_generator.CUDACodeGenerator.rst","reference/brian2cuda.cuda_generator.ParallelisationError.rst","reference/brian2cuda.cuda_prefs.compute_capability_validator.rst","reference/brian2cuda.device.CUDAStandaloneDevice.rst","reference/brian2cuda.device.CUDAWriter.rst","reference/brian2cuda.device.check_codeobj_for_rng.rst","reference/brian2cuda.device.cuda_standalone_device.rst","reference/brian2cuda.utils.rst","reference/brian2cuda.utils.gputools.get_available_gpus.rst","reference/brian2cuda.utils.gputools.get_best_gpu.rst","reference/brian2cuda.utils.gputools.get_compute_capability.rst","reference/brian2cuda.utils.gputools.get_cuda_installation.rst","reference/brian2cuda.utils.gputools.get_cuda_path.rst","reference/brian2cuda.utils.gputools.get_cuda_runtime_version.rst","reference/brian2cuda.utils.gputools.get_gpu_selection.rst","reference/brian2cuda.utils.gputools.get_nvcc_path.rst","reference/brian2cuda.utils.gputools.reset_cuda_installation.rst","reference/brian2cuda.utils.gputools.reset_gpu_selection.rst","reference/brian2cuda.utils.gputools.restore_cuda_installation.rst","reference/brian2cuda.utils.gputools.restore_gpu_selection.rst","reference/brian2cuda.utils.gputools.select_gpu.rst","reference/brian2cuda.utils.logger.suppress_brian2_logs.rst","reference/brian2cuda.utils.stringtools.append_f.rst","reference/brian2cuda.utils.stringtools.replace_floating_point_literals.rst"],objects:{"brian2cuda.codeobject":{CUDAStandaloneAtomicsCodeObject:[29,1,1,""],CUDAStandaloneCodeObject:[30,1,1,""]},"brian2cuda.codeobject.CUDAStandaloneCodeObject":{__call__:[30,2,1,""],run:[30,2,1,""]},"brian2cuda.cuda_generator":{CUDAAtomicsCodeGenerator:[31,1,1,""],CUDACodeGenerator:[32,1,1,""],ParallelisationError:[33,1,1,""]},"brian2cuda.cuda_generator.CUDACodeGenerator":{atomics_parallelisation:[32,2,1,""],conditional_write:[32,2,1,""],denormals_to_zero_code:[32,2,1,""],determine_keywords:[32,2,1,""],flush_denormals:[32,3,1,""],get_array_name:[32,4,1,""],parallelise_code:[32,2,1,""],restrict:[32,3,1,""],translate_expression:[32,2,1,""],translate_one_statement_sequence:[32,2,1,""],translate_statement:[32,2,1,""],translate_to_declarations:[32,2,1,""],translate_to_read_arrays:[32,2,1,""],translate_to_statements:[32,2,1,""],translate_to_write_arrays:[32,2,1,""],universal_support_code:[32,3,1,""]},"brian2cuda.cuda_prefs":{compute_capability_validator:[34,5,1,""]},"brian2cuda.device":{CUDAStandaloneDevice:[35,1,1,""],CUDAWriter:[36,1,1,""],check_codeobj_for_rng:[37,5,1,""],cuda_standalone_device:[38,6,1,""]},"brian2cuda.device.CUDAStandaloneDevice":{build:[35,2,1,""],check_openmp_compatible:[35,2,1,""],code_object:[35,2,1,""],code_object_class:[35,2,1,""],copy_source_files:[35,2,1,""],generate_codeobj_source:[35,2,1,""],generate_main_source:[35,2,1,""],generate_makefile:[35,2,1,""],generate_network_source:[35,2,1,""],generate_objects_source:[35,2,1,""],generate_rand_source:[35,2,1,""],generate_run_source:[35,2,1,""],generate_synapses_classes_source:[35,2,1,""],get_array_name:[35,2,1,""],get_array_read_write:[35,2,1,""],network_run:[35,2,1,""]},"brian2cuda.device.CUDAWriter":{write:[36,2,1,""]},"brian2cuda.utils":{gputools:[39,0,0,"-"],logger:[39,0,0,"-"],stringtools:[39,0,0,"-"]},"brian2cuda.utils.gputools":{get_available_gpus:[40,5,1,""],get_best_gpu:[41,5,1,""],get_compute_capability:[42,5,1,""],get_cuda_installation:[43,5,1,""],get_cuda_path:[44,5,1,""],get_cuda_runtime_version:[45,5,1,""],get_gpu_selection:[46,5,1,""],get_nvcc_path:[47,5,1,""],reset_cuda_installation:[48,5,1,""],reset_gpu_selection:[49,5,1,""],restore_cuda_installation:[50,5,1,""],restore_gpu_selection:[51,5,1,""],select_gpu:[52,5,1,""]},"brian2cuda.utils.logger":{suppress_brian2_logs:[53,5,1,""]},"brian2cuda.utils.stringtools":{append_f:[54,5,1,""],replace_floating_point_literals:[55,5,1,""]},brian2cuda:{__init__:[28,0,0,"-"],binomial:[28,0,0,"-"],codeobject:[28,0,0,"-"],cuda_generator:[28,0,0,"-"],cuda_prefs:[28,0,0,"-"],device:[28,0,0,"-"],timedarray:[28,0,0,"-"],utils:[39,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","staticmethod","Python static method"],"5":["py","function","Python function"],"6":["py","data","Python data"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:staticmethod","5":"py:function","6":"py:data"},terms:{"14f":55,"2ms":0,"32bit":27,"4ms":0,"5e6":55,"5e6f":55,"64bit":27,"case":[0,23,27,35],"class":28,"const":[0,37],"default":[0,1,19,22,23,24,27,32,35,37],"export":[28,39],"float":[0,22,23,27,42,45,52,55],"function":[24,27,28,29,30,32,35,39],"import":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,24,29,30,31,32,33,34,35,36,37,38,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55],"int":[1,19,22,23,24,37,52],"new":[43,49],"return":[24,30,32,35,37,40,42,43,45,46,47,52,54,55],"static":32,"true":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,22,23,27,32,35,37],"try":[24,27],"var":[32,35],And:27,E_e:22,For:[27,32,35],IDs:27,INTO:[8,9],The:[1,8,9,10,11,12,13,27,29,30,32,35,38,54],There:[4,5,6,7],Use:24,Used:54,__call__:30,__file__:[2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,24],__init__:[],__launch_bounds__:27,__main__:24,_ptr:[32,35],_result:[8,9],_standalon:24,_synapses_:23,abbott:23,about:39,abov:0,abstract_cod:35,acces:24,access:27,access_data:[32,35],accord:[15,16],account:44,action:24,activ:[0,22,23],actual:22,adapt:23,add:[1,23,24],add_argu:24,add_mutually_exclusive_group:24,added:[23,32],addit:35,additional_header_fil:35,additional_source_fil:35,adhara:24,after:[0,1,19,22,23],after_group:[12,13],again:48,agg:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,22,23],align:19,all:[1,22,27,32,53],alloc:27,allow:[28,35],almost:0,along:[8,9],alpha_h:[1,22],alpha_m:[1,22],alpha_n:[1,22],alphah:[8,9,10,11,12,13],alpham:[8,9,10,11,12,13],alphan:[8,9,10,11,12,13],also:37,alvarez:[1,19],amp:[2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,22],ani:[27,29,30,35],antenn:22,api:27,apost:[22,23],appear:37,append:[22,54,55],appli:[0,8,9,10,11,12,13,27],applic:[27,28],approx:23,apr:[22,23],arang:[1,22],arange_arrai:35,arcco:27,arch:24,arcsin:27,arctan:27,area:1,area_pr:[12,13],arg:[24,31,32],argpars:24,argument:[0,1,19,22,23,24,27,35],argumentpars:24,arrai:[22,35],assert:[23,24],assum:[23,55],asynchron:[4,5,6,7],atom:[0,1,19,22,23,24,27,28,29],atomics_parallelis:32,attempt:35,attribut:[32,37],attributeerror:24,automat:[27,35],avail:[32,39,40,41],averag:22,avg:[1,23],avoid:[1,27,35],axi:[10,11,22],axial:[15,16],axon:[2,3,8,9,10,11,12,13,17,18],ball:[17,18],base:[0,29,30,31,32,33,35,36,52],basenam:[2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,24],beeman:[1,19],been:[12,13],befor:[24,35],behaviour:0,below:[0,2,3,8,9,10,11,12,13,17,18],benchmark:[1,19],best:41,beta_h:[1,22],beta_m:[1,22],beta_n:[1,22],betah:[8,9,10,11,12,13],betam:[8,9,10,11,12,13],betan:[8,9,10,11,12,13],bifurc:0,bimod:23,binari:47,binomi:[22,37],binomial_match:37,binomialfunct:28,bipolar_cell_cpp:21,bipolar_cell_cuda:21,bipolar_with_inputs2_cpp:21,bipolar_with_inputs2_cuda:21,bipolar_with_inputs_cpp:21,bipolar_with_inputs_cuda:21,black:[8,9],blob:22,block:[0,1,19,22,23,27],bodi:22,bool:[24,35,37],boustani:[1,19],bower:[1,19],branch:[15,16],brett:[1,19],brian2:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,22,23,24,30,32,35,36,53],brian2cuda:[0,1,3,5,7,9,11,13,16,18,19,20,22,23,24,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55],brian2genn:[1,19],brian2genn_benchmark:22,brian:[19,22,24],brianglobalprefer:24,brunel:0,brunelhakim:21,bug:17,build:[2,3,8,9,10,11,12,13,17,18,35],build_on_run:[2,3,8,9,10,11,12,13,17,18,35],built:35,bundl:[0,1,19,22,23,27],bundle_mod:[0,1,19,22,23,24],calc_occup:27,calcul:[8,9,12,13],call:[2,3,8,9,10,11,12,13,17,18,24,30,35],can:[23,24,27],capabl:[24,34,41,42,52],carneval:[1,19],ceil:27,center:[12,13],chang:[0,2,3,8,9,10,11,12,13,17,18],channel:[2,3,4,5,6,7,15,16,17,18],check:[34,37],check_binomi:37,check_openmp_compat:35,choic:[1,22,23,24],choices_copi:24,choos:[1,23,27,41],clean:35,clip:[22,23],clock:[1,6,7,19],close:27,closest:27,cmd_arg:24,cobahh:21,code:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,22,23,24,27,28,29,30,32,35,37,55],code_object:[30,35],code_object_class:35,codefold:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,22,23,24],codegen:[24,32],codegener:[30,32],codeobj:37,codeobj_class:35,codeobject:[29,30,35,37],collect:[0,23],com:22,come:53,command:[0,1,19,22,23,24],compil:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,22,23,24,32,35],compili:1,comput:[0,1,2,3,8,9,10,11,12,13,17,18,19,24,34,41,42,52],compute_cap:52,concaten:22,condit:[27,28],conditional_writ:32,conditional_write_var:32,conduct:1,configur:[0,1,19,22,23],conn:0,connect:[0,1,4,5,6,7,12,13,19,20,22,23,27],connection_prob:23,constant:[0,1,12,13,22],content:36,convert:27,copi:[24,55],copy_source_fil:35,core:[20,24],correspond:[27,30],cos:27,cosh:[15,16,27],count:37,coupl:1,cpp_compil:35,cpp_compiler_flag:35,cpp_re:14,cpp_standalon:[0,1,2,4,6,8,10,12,15,17,19,22,23,30,35,36],cppcodegener:32,cppstandalonecodeobject:30,cppstandalonedevic:35,cppwriter:36,creat:[23,24],creation:27,cu_fil:37,cuba:[20,21],cuba_cuda:21,cuba_cuda_rasterplot:20,cuba_cuda_sp:20,cuda:[24,27,28,29,30,32,35,37,38,43,44,45,48,50],cuda_backend:[44,52],cuda_gener:[31,32,33],cuda_instal:50,cuda_path:44,cuda_pref:34,cuda_r:14,cuda_standalon:[0,1,3,5,7,9,11,13,16,18,19,20,22,23,24,27,35,44,52],cuda_visible_devic:24,cudaatomicscodegeneratorc_data_typ:28,cudacodegener:[28,31],cudastandaloneatomicscodeobject:28,cudastandalonecodeobject:[28,29,35],cudastandalonedevic:[32,38],cumsum:[10,11],curand:27,curand_rng_pseudo_default:27,current:[0,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,41],cylind:[2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18],dapost:[22,23],dapr:[22,23],data:35,davison:[1,19],debug:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,22,23,35],dec:1,declar:35,deduc:24,def:24,default_devic:24,default_float_dtyp:[20,24],default_functions_integral_convert:27,defaultclock:[8,9,12,13,15,16,17,18],defin:[27,29,30,32,44],definit:[29,30,32,35],delai:[0,23,27],delet:24,delta:0,dendrit:[2,3,4,5,6,7],denormals_to_zero_cod:32,depend:[23,24],describ:[1,19],descript:24,dest:24,destexh:[1,19],detail:[27,30,32,35,36],detect:[44,48],determine_keyword:32,dev:35,dev_no_to_cc:24,devic:[0,1,2,3,5,7,8,9,10,11,12,13,16,17,18,19,22,23,24,27,30,35,36,37,38,44,52],devicenam:[0,1,19,22,23,24],dg_ekc_ekc:22,dg_ikc_ekc:22,dg_pn_ikc:22,dge:[1,19,20,23],dgi:[1,19,20],dgs:[4,5],diagram:0,diamet:[2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18],dict:[0,1,19,22,23,24,30],dictionari:[23,24,30,32,43,46,50,51],diesmann:[1,19],differ:[8,9,10,11,12,13,28],direct_cal:35,directli:35,directori:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,22,23,35],disabl:[0,23],disable_assert:35,distanc:[8,9,15,16],distinguish:35,distrib:23,distribut:[0,8,9,10,11,12,13],djurfeldt:[1,19],document:27,doe:28,doesn:54,don:27,doubl:[27,55],draw:23,driven:[1,6,7,19,22,23],dtype:24,due:35,dummi:22,durat:[0,35],dure:30,dynam:0,dynamicarrayvari:35,e_i:22,e_k:22,e_leak:22,e_na:22,each:[23,27],effect:[1,23,27,28],efficaci:1,either:[22,35],ekc:22,ekc_ekc:22,ekc_spik:22,electrod:[12,13],elif:[1,23,24],elnath:24,els:[0,1,22,23,24,37,54],eltanin:24,ena:[1,8,9,10,11,12,13,17,18],enabl:0,end:[8,9,10,11,12,13,15,16,54],endpoint:[2,3,6,7],ensur:23,enumer:22,environ:24,environment:44,eqs:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20],eqs_ekc:22,eqs_ikc:22,eqs_neuron:23,equat:[1,8,9,10,11,12,13,22],equival:23,ermentrout:[1,19],err:24,error:[34,35],etc:32,euler:[4,5,6,7],even:28,event:[22,23,27],exact:[6,7,19],exampl:[25,32,55],except:[24,33],excitatori:[19,20,22],execut:29,exist:[0,1,8,9,10,11,12,13,15,16,17,18,19,22,23,35],exp:[1,8,9,10,11,12,13,17,18,22,27],experi:24,exponenti:[1,19],exponential_eul:[1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,22],expr:32,express:32,extend:[22,24],extens:39,extra:27,extra_compile_args_nvcc:24,extra_threshold_kernel:27,extracellular:[12,13],fake:[2,3,4,5,6,7],fallback_pref:35,fals:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,22,23,24,27,32,35,37],far:[12,13],fast:0,faster:[24,27],feature_pars:24,field:[12,13],fig:0,file:[0,1,8,9,14,19,22,23,32,35],filenam:36,fire:[0,19],first:22,fix:[1,27],flag:[0,24],flaot:55,flat:0,float32:[20,24],float64:27,floor:27,flush_denorm:32,folder:[0,1,8,9,19,22,23],follow:[0,1,19,32],forc:0,format:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,22,23,24],formula:[15,16],from:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,22,23,24,27,29,30,31,32,33,34,35,36,37,38,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55],further:0,g_ekc_ekc:22,g_ikc_ekc:22,g_k:22,g_kd:1,g_leak:22,g_max:22,g_na:[1,22],g_pn_ikc:22,g_raw:22,g_scale:22,gener:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,22,23,27,32,37],generate_codeobj_sourc:35,generate_main_sourc:35,generate_makefil:35,generate_network_sourc:35,generate_objects_sourc:35,generate_rand_sourc:35,generate_run_sourc:35,generate_synapses_classes_sourc:35,genn:[1,19],geometri:[2,3,4,5,6,7],get:[39,41,42],get_array_nam:[32,35],get_array_read_writ:35,getattr:24,gethostnam:24,github:22,give:24,given:[32,35],global:[0,27,32,35,50,51],gmax:23,gna0:[8,9,10,11,12,13],gna:[8,9,10,11,12,13,17,18],goodman:[1,19],got:24,gpu:[39,40,41,42,46,49,51,52],gpu_devic:24,gpu_id:[42,52],gpu_select:51,gputool:[28,40,41,42,43,44,45,46,47,48,49,50,51,52],grid:19,group:[0,54],gs_post:[6,7],gsyn:23,hakim:0,harri:[1,19],has:[12,13,27],hashdefine_cod:32,have:[0,22,23,24,27],header:35,help:[0,1,19,22,23,24],here:[23,32,35],heterog_delai:0,heterogen:[0,23,27],hh_with_spikes_checkresult:21,hh_with_spikes_cpp:21,hh_with_spikes_cpp_result:14,hh_with_spikes_cuda:21,hh_with_spikes_cuda_result:14,higher:27,highest:[41,52],hine:[1,19],hinf:[17,18],hist:23,hodgkin:[8,9,10,11,12,13],hodgkin_huxley_1952_cpp:21,hodgkin_huxley_1952_cuda:21,hold:23,homogen:[0,23],host:35,hostnam:24,how:27,html:[2,3,8,9,10,11,12,13,17,18],http:[2,3,8,9,10,11,12,13,17,18,22],huxlei:[8,9,10,11,12,13],i_syn:22,ic_pr:[12,13],ident:0,idx:22,ignor:[24,35],ikc:22,ikc_ekc:22,ikc_spik:22,im_pr:[12,13],implement:[1,19,27,28],inact:22,inactiv:[17,18],includ:35,independ:35,index:[19,20,23,25],indic:32,individu:[0,27],inffect:23,inform:[0,1,19,22,23,39],inhibitori:[0,19,20,22],initi:1,inject:[2,3,8,9,10,11,12,13,15,16],input:[4,5,6,7,22,23,35],inspect:0,instal:[43,44,48,50],integr:[0,19,27],intercept:[8,9],interfac:28,intern:35,interpol:1,introduct:25,invalid:23,irregularli:0,isinst:24,item:[23,24],ith:27,its:27,itself:35,jinja2:32,join:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,22,23],journal:[1,19],k_poisson:23,kei:[23,24,30,32],kernel:[27,35],kernel_paramet:35,kernel_tim:[1,19],keyerror:24,keyword:35,kwd:[31,32,35],languag:32,lansner:[1,19],larger:34,launch_bound:27,lead:0,left:[2,3,15,16],len:[1,22],length:[2,3,4,5,6,7,8,9,10,11,15,16,17,18],less:0,level:35,lfp:[0,12,13],lfp_cpp:21,lfp_cuda:21,librari:32,line:[0,1,19,22,23,24,32],linregress:[8,9],linspac:[2,3,6,7],list:[22,35,40],liter:55,littl:24,load:14,lobe:22,local:44,log10:27,log:[27,39,53],logger:[28,53],loop:[29,30,32],low:0,lower:27,lowest:41,macro:[29,30,32],made:32,main:[15,16,17,18,29,30,32,35],main_includ:35,major:[10,11],make:[22,23,24],manual:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,22,23,35],master:22,match:54,matchobject:54,matplotlib:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,22,23],mbody_exampl:22,mean:[23,35],mechan:0,member:[28,39],membran:[15,16],memori:[27,35],merop:24,meter:[2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18],method:[1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,22,30,32,35,36],miller:23,minf:[17,18],minim:34,minor:[10,11],mkdir:[0,1,8,9,10,11,12,13,15,16,17,18,19,22,23],mlfp:[12,13],mode:[1,23,35],model:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,22,23],modif:0,modifi:[23,24],modul:[24,25],mon:[17,18,23],mon_l:[2,3,4,5,6,7],mon_r:[2,3,4,5,6,7],mon_soma:[2,3,4,5,6,7],monitor:[0,1,2,3,4,5,6,7,17,18,19,22,23],more:[27,35],morpho:[2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18],morpholog:[2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18],msiemen:[1,8,9,10,11,12,13],mso:[2,3,4,5,6,7],much:24,muext:0,muller:[1,19],multipl:[2,3,8,9,10,11,12,13,17,18,23],multiprocessor:27,multitempl:[29,30],mushroom:22,mushroombodi:21,must:23,n_al:22,n_lb:22,n_lif:23,n_mb:22,n_pattern:22,n_poisson:23,n_repeat:22,name:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,22,23,24,29,30,32,35,40],name_suffix:24,namespac:[30,35],narrow:0,narrow_delaydistr:0,natschlaeg:1,natschlag:19,nb_thread:35,need:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,22,23,24,27,48,49],neither:27,net:35,network:[0,1,19,27,35],network_run:35,neural:0,neuron:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,22,23,27],neurongroup:[0,1,4,5,6,7,12,13,19,20,22,23],neurosci:[1,19],new_indic:22,next:49,nicola:0,ninf:[17,18],nkckc:22,no_post_refer:27,no_pre_refer:27,nois:23,none:[0,1,19,22,23,24,27,32,35,37],nonzero:1,normal:23,note:27,npz:[8,9,14],num_block:[0,1,19,22,23,24,27],num_spik:22,num_thread:27,number:[0,1,12,13,19,22,23,24,27,37],numpi:[14,24],nvcc:47,nvidia:[40,41],object:[24,27,28,29,30,35],occup:27,off:[12,13],ohm:[2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18],on_post:[22,23],on_pr:[0,1,4,5,6,7,19,20,22,23],one:[2,3,8,9,10,11,12,13],onli:[0,1,22,27],open:[0,1,19,22,23],openmp:20,oper:[0,1,19,22,23,27,28,29],optim:27,optimis:[1,27],option:[23,24,27,35,37],order:27,ordereddict:23,origin:0,oscil:0,oscillatori:0,other:1,otherwis:[12,13],out:23,output:[17,35],output_vari:30,overload:27,override_conditional_writ:35,overwrit:24,overwritten:35,owner:[29,30,35],p_perturb:22,p_valu:[8,9],page:25,paper:[1,19],parallel:[27,28,29,35],parallel_block:[24,27],parallelis:27,parallelise_cod:32,param:[0,1,19,22,23,24],paramet:[0,1,8,9,10,11,12,13,19,22,23,24,27,32,35,37,54,55],paramt:23,pars:[0,1,19,22,23],parse_arg:24,parser:24,particular:0,pass:27,passiv:[2,3,4,5,6,7,15,16],path:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,22,23,24,44,47],pattern:22,pecevski:[1,19],per:[1,15,16,27],pick:52,place:[12,13],plastic:23,plot:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,22,23],plotfold:[8,9,10,11,12,13,15,16,17,18],plotpath:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,22,23],plu:[15,16],pn_ikc:22,pn_spike:22,png:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,22,23],point:[2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,27,55],pointer:35,poissongroup:23,pop:24,popratemon:1,popul:22,populationratemonitor:[0,1],posit:[8,9,10,11],possibl:[8,9,10,11,12,13,24,28],post:[0,1,19,22,23],post_effect:23,postneuron:23,postsynapt:[23,27],potenti:[1,12,13],power:0,precis:[0,1,19,22,23,27,55],pref:[0,1,19,20,22,23,24,27,44,52],prefer:[0,24,25,28,34,44,52],prefix:[32,35],presynapt:27,prevent:17,princip:22,print:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,22,23,24],probabl:[1,23],problem:35,process:23,produc:0,profil:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,19,22,23,35],profiling_fil:[0,1,19,22,23],profiling_summari:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,19,22,23],profilingpath:[0,1,19,22,23],program:35,project:35,project_dir:36,propag:[0,27],provid:[32,52],pseudo:[2,3,4,5,6,7],pseudocoupl:1,push:[0,1,19,22,23,27],push_synapse_bundl:[24,27],py_random:22,queue:27,quicker:[0,1,19,22,23],r_valu:[8,9],race:[27,28],rais:[24,34],rall:[15,16],rall_cpp:21,rall_cuda:21,rand:[0,1,19,20,22,23,37],rand_cal:37,randint:22,randn:[1,22,37],randn_cal:37,random:[1,19,22,23,27],random_number_generator_ord:27,random_number_generator_typ:27,rang:[8,9,10,11,12,13,22],rate:[0,22,23],read:32,readthedoc:[2,3,8,9,10,11,12,13,17,18],recogn:24,record:[1,2,3,4,5,6,7,8,9,10,11,12,13,17,18,23],red:[8,9],reduc:0,refer:[0,25],refractori:[0,1,8,9,19,20,22],regex:39,regim:0,regist:27,regular:[4,5,6,7],relat:28,relev:27,remark:0,remov:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,22,23,24],repeat:22,replac:[2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,22,24,55],replace_floating_point_liter:54,report:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,22,23,35,40,41],report_period:35,represent:27,requir:[2,3,8,9,10,11,12,13,17,18,24],reset:[0,4,5,6,7,19,20,23,27,48,49],resist:[12,13,15,16],resiz:35,respons:24,restrict:[1,23,32],result:[0,1,8,9,19,22,23,27],resultsfold:[0,1,19,22,23,24],return_valu:30,revers:1,review:[1,19],risha:24,rochel:[1,19],rudolph:[1,19],run:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,22,23,24,30,35],run_arg:35,run_includ:35,rune:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,22,23],runtim:[22,35,45],s_mon:[19,20,23],sabik:24,safe:27,sai:0,same:[8,9,10,11,12,13,27],sampl:0,save:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,22,23],savefig:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,22,23],savez:[8,9],scalar:32,scalar_cod:[32,35],scale:[22,23],scenario:1,scipi:[8,9,12,13],script:19,search:[25,54],second:[0,1,4,5,8,9,19,20,22,23,35],see:[2,3,8,9,10,11,12,13,17,18,27,32],select:[0,1,19,22,23,27,46,49,51,52],separ:32,serial:28,set:[0,1,19,22,23,24,27,32,35,50,51],set_default:24,set_devic:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,22,23],set_pref:[0,1,19,22,23,24],sever:35,shape:14,shortest:[29,30,31,32,33,34,35,36,37,38,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55],should:[29,30,32,35],show:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,22,23,35],shown:35,shuffl:22,siemen:[1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,22],sigma:[12,13],sigmaext:0,signatur:[],similar:[0,23],simplifi:[17,18],simul:[0,1,19,22,23,35,38,52],sin:27,sinc:[23,27,32],singl:[0,1,19,22,23,27,32,55],single_precis:[0,1,19,22,23,24],sinh:[15,16,27],size:[22,27],slope:[8,9],sm_35:24,sm_:24,sm_multipli:27,small:27,smi:[40,41],smooth_rat:[0,1],snippet:32,socket:24,soma:[2,3,4,5,6,7,17,18],song:23,sort:[24,40],sorted_vari:22,sourc:[22,29,30,31,32,33,34,35,36,37,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55],space:[15,16],spars:0,spatialneuron:[2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18],specif:25,specifi:[23,35],spike:[1,8,9,19,23,27],spike_indic:22,spike_initiation_cpp:21,spike_initiation_cuda:21,spike_tim:22,spikegeneratorgroup:22,spikemon:[1,22],spikemonitor:[0,1,8,9,19,20,22,23],sqrt:[0,15,16,23,27],stabl:[2,3,8,9,10,11,12,13,17,18],standalon:[0,1,19,22,23,27,28,29,30,35,38],standard:28,stat:[8,9,12,13],statement:32,statemonitor:[1,2,3,4,5,6,7,8,9,10,11,12,13,17,18,23],static_array_spec:35,std_err:[8,9],stdout:35,stdp:21,step:[12,13],stick:[17,18],stimul:[4,5,6,7,12,13],store:[0,1,19,22,23,37],store_fals:24,store_tru:24,str:[0,1,19,22,23,24,27,35,54,55],strategi:[1,19],stream:27,strength:1,string:[24,27,32,54,55],stringtool:[28,54,55],subplot:[0,1,2,3,4,5,6,7,8,9,12,13,22,23],subthreshold:19,successfulli:35,suffix:24,suit:35,sum:[6,7,12,13],summed_updat:[12,13],support:[29,30,32,34],support_cod:[29,30,32],suppress:53,suppress_brian2_log:39,syn:1,syn_launch_bound:27,synaps:[0,1,4,5,6,7,12,13,19,20,22,23,27,35],synapses_push:27,synapt:[1,4,5,6,7,19,20,27,28],synchron:0,syntax:55,system:39,take:44,taken:22,tan:27,tanh:27,target:32,tau:[0,4,5,6,7],tau_ekc_ekc:22,tau_ikc_ekc:22,tau_lhi_ikc:22,tau_pn_ikc:22,tau_pn_lhi:22,tau_post:22,tau_pr:22,taue:[1,19,20,23],taui:[1,19,20],taum:[19,20,23],taupost:23,taupr:23,taurefr:0,team:22,tell:37,templat:32,template_kwd:35,template_nam:[29,30,35],template_sourc:[29,30],temporari:35,test:35,text:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,22,23],than:34,thei:35,them:[23,37],theori:[15,16],theta:0,thi:[1,2,3,8,9,10,11,12,13,17,18,19,22,23,24,27,32,34,35,37,41,44,48,49],those:[22,27],thread:20,threshold:[0,1,4,5,6,7,8,9,19,20,22,23,27],tight:[10,11],tight_layout:23,time:[1,2,3,4,5,6,7,8,9,10,11,12,13,17,18,19,20,23,49],timedarrai:32,titl:20,to_replac:22,todo:[24,35],tool:[1,19,39],total:27,trace:1,training_s:22,training_vari:22,translat:32,translate_express:32,translate_one_statement_sequ:32,translate_stat:32,translate_to_declar:32,translate_to_read_arrai:32,translate_to_stat:32,translate_to_write_arrai:32,transmembran:[8,9,10,11,12,13],traub_mil:22,tupl:52,two:[2,3,4,5,6,7,15,16,22,28,29,30,32],txt:[0,1,19,22,23],type:[8,9,10,11,12,13,24,27,52,54],typeerror:24,typic:[8,9,10,11,12,13],ufarad:1,umetr:1,uncoupl:1,undefin:24,underli:35,uniform:0,uniqu:[22,32,35],unit:[15,16],universal_support_cod:32,unknown:24,unless:[19,20],unnecassari:27,until:27,updat:[0,1,12,13,19,22,23],update_from_command_lin:[0,1,19,22,23,24],usag:[0,27],use:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,22,23,27,28],use_atom:[24,27],used:[22,24,27,32,35,38],used_vari:32,user:[2,3,8,9,10,11,12,13,17,18,32,44,52],uses:[27,28,29],usr:44,util:[0,1,19,21,22,23,28,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55],v_m:[12,13],v_post:[12,13],valu:[0,8,9,10,11,12,13,24,27,32],variabl:[27,29,30,35,43,44,46],variable_indic:[29,30,35],varianc:23,variant:22,variou:[12,13],vector_cod:[32,35],vectorisation_idx:37,veloc:[8,9],veri:[0,27],version:45,vibert:[1,19],vincent:0,volt:[0,1,12,13,19,20,22,23],voltag:[19,20],w_ekc_ekc:22,w_lhi_ikc:22,want:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,22,23],warn:24,weakli:0,weather:[23,27],weight:[12,13,19,20,22,23],wether:37,when:[12,13,22,27,28,32,35,48],where:0,whether:[0,1,19,22,23,35],which:[23,27,28,29,32],white:23,whole:[8,9,10,11,12,13],widen:0,wider:0,width:[0,1],window:0,with_output:35,without:24,would:55,write:[0,1,19,22,23,32,35,36],writer:35,x_post:[12,13],x_pre:[12,13],xlabel:[2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,23],xlim:0,y_post:[12,13],y_pre:[12,13],yet:[12,13],ylabel:[2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,23],you:27,your:27,ytick:[10,11],z_post:[12,13],z_pre:[12,13],zero:1,zirp:[1,19]},titles:["Example: brunelhakim","Example: cobahh","Example: bipolar_cell_cpp","Example: bipolar_cell_cuda","Example: bipolar_with_inputs2_cpp","Example: bipolar_with_inputs2_cuda","Example: bipolar_with_inputs_cpp","Example: bipolar_with_inputs_cuda","Example: hh_with_spikes_cpp","Example: hh_with_spikes_cuda","Example: hodgkin_huxley_1952_cpp","Example: hodgkin_huxley_1952_cuda","Example: lfp_cpp","Example: lfp_cuda","Example: hh_with_spikes_checkresults","Example: rall_cpp","Example: rall_cuda","Example: spike_initiation_cpp","Example: spike_initiation_cuda","Example: cuba","Example: cuba_cuda","Examples","Example: mushroombody","Example: stdp","Example: utils","Welcome to brian2cuda\u2019s documentation!","Introduction","Brian2cuda specific preferences","brian2cuda package","CUDAStandaloneAtomicsCodeObject class","CUDAStandaloneCodeObject class","CUDAAtomicsCodeGenerator class","CUDACodeGenerator class","ParallelisationError class","compute_capability_validator function","CUDAStandaloneDevice class","CUDAWriter class","check_codeobj_for_rng function","cuda_standalone_device object","utils package","get_available_gpus function","get_best_gpu function","get_compute_capability function","get_cuda_installation function","get_cuda_path function","get_cuda_runtime_version function","get_gpu_selection function","get_nvcc_path function","reset_cuda_installation function","reset_gpu_selection function","restore_cuda_installation function","restore_gpu_selection function","select_gpu function","suppress_brian2_logs function","append_f function","replace_floating_point_literals function"],titleterms:{"class":[29,30,31,32,33,35,36],"function":[34,37,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55],append_f:54,binomi:28,bipolar_cell_cpp:2,bipolar_cell_cuda:3,bipolar_with_inputs2_cpp:4,bipolar_with_inputs2_cuda:5,bipolar_with_inputs_cpp:6,bipolar_with_inputs_cuda:7,brian2cuda:[25,27,28],brunelhakim:0,check_codeobj_for_rng:37,cobahh:1,codeobject:28,compartment:21,compute_capability_valid:34,cuba:19,cuba_cuda:20,cuda_gener:28,cuda_pref:28,cuda_standalone_devic:38,cudaatomicscodegener:31,cudacodegener:32,cudastandaloneatomicscodeobject:29,cudastandalonecodeobject:30,cudastandalonedevic:35,cudawrit:36,devic:28,document:25,exampl:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],get_available_gpu:40,get_best_gpu:41,get_compute_cap:42,get_cuda_instal:43,get_cuda_path:44,get_cuda_runtime_vers:45,get_gpu_select:46,get_nvcc_path:47,gputool:39,hh_with_spikes_checkresult:14,hh_with_spikes_cpp:8,hh_with_spikes_cuda:9,hodgkin_huxley_1952_cpp:10,hodgkin_huxley_1952_cuda:11,indic:25,introduct:26,lfp_cpp:12,lfp_cuda:13,list:27,logger:39,modul:[28,39],mushroombodi:22,object:38,packag:[28,39],parallelisationerror:33,plot:21,prefer:27,rall_cpp:15,rall_cuda:16,replace_floating_point_liter:55,reset_cuda_instal:48,reset_gpu_select:49,restore_cuda_instal:50,restore_gpu_select:51,select_gpu:52,specif:27,spike_initiation_cpp:17,spike_initiation_cuda:18,stdp:23,stringtool:39,subpackag:28,suppress_brian2_log:53,tabl:25,timedarrai:28,util:[24,39],welcom:25}})