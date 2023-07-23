#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   RC_generater.py
@Time    :   2023/03/06 21:15:59
@Author  :   Li Yuan 
@Version :   1.0
@Contact :   ly1837372873@163.com
@Desc    :   定义水库层神经元的相关参数，并保存连接状态
'''
# here put the import lib

import os
import sys
from brian2 import *
import brian2
import numpy as np
import pickle as pickle
from struct import unpack


print('RC_generater')

start_scope()
#set_device('cpp_standalone', directory = './RC_simulation/output')
## setup for C++ compilation preferences
#codegen.cpp_compiler = 'gcc' #Compiler to use (uses default if empty);Should be gcc or msvc.
#prefs.devices.cpp_standalone.extra_make_args_unix = ['-j']
#prefs.devices.cpp_standalone.openmp_threads = 1
##-----------------------------------------------------------------------------------------
## basic simulation setup
##-----------------------------------------------------------------------------------------

# 时间步设置
brian2.defaultclock.dt = 0.1 * brian2.ms # 0.1
brian2.core.default_float_dtype = float32 ### can be chaged to float32 in brian 2.2
brian2.core.default_integer_dtype = int32

# clear_cache('cython')

k = sys.argv[18]

indices_path = './spikes_input/3second_9direction/'+k+'/train_index.npy'
times_path =  './spikes_input/3second_9direction/'+k+'/train_time.npy'
spike_record_path = './spike_record/3second_9direction/'
simulation_archive_path = './simulation_archive/3second_9direction/'

num_neuron = int(sys.argv[1])
work_time = int(sys.argv[2])
rest_time = int(sys.argv[3])
num_data = int(sys.argv[4])

weight_ie = str(sys.argv[5])
p_inpute = float(sys.argv[6])
p_ee = float(sys.argv[7])
w_ee = str(sys.argv[8])
p_ei = float(sys.argv[9])
w_ei = str(sys.argv[10])
p_ie = float(sys.argv[11])
w_ie = str(sys.argv[12])
p_ii = float(sys.argv[13])
w_ii = str(sys.argv[14])

input_channel = int(sys.argv[15])
ex_rate = float(sys.argv[16])
in_rate = float(sys.argv[17])


seed(1)

# start_scope()
set_device('cpp_standalone')

indices = np.load(indices_path)
times = np.load(times_path)

input_indices = indices
input_times = times

Gs = SpikeGeneratorGroup(input_channel, input_indices, input_times*brian2.ms)

# standard neuron parameters
v_rest_excite = -65.0 * mV
v_rest_inhibite = -60.0 * mV
v_reset_excite = -65.0 * mV
v_reset_inhibite = -45.0 * mV
# v_thresh_excite = -52.0 * mV
v_thresh_excite = -48.53601013154767 * mV
v_thresh_inhibite = -33.3596552351866 * mV  # -40
t_refrac_excite = 5 * brian2.ms
t_refrac_inhibite = 2 * ms

# spike conditon
v_thresh_excite_str = '(v > v_thresh_excite) and (timer > t_refrac_excite)'
v_thresh_inhibite_str = '(v > v_thresh_inhibite) and (timer > t_refrac_inhibite)'


# excitatory neuron membrane dynamics
neuron_eqs_excite = '''
        dv/dt = ((v_rest_excite -v) + (I_excite + I_inhibite) ) / (91.13229645760313 * ms)  : volt (unless refractory)
        I_excite = ge * (-v)                           : volt
        I_inhibite = gi * (-100.0 * mV - v)            : volt
        dge/dt = -ge / (1.0 * ms)                           : 1
        dgi/dt = -gi / (2.0 * ms)                           : 1
        '''
neuron_eqs_inhibite = '''
        dv/dt = ((v_rest_inhibite -v) + (I_excite + I_inhibite) ) / (8.058421025659644 * ms)  : volt (unless refractory)
        I_excite = ge * (-v)                           : volt 
        I_inhibite = gi * (-85.0 * mV - v)             : volt
        dge/dt = -ge / (1.0 * ms)                           : 1
        dgi/dt = -gi / (2.0 * ms)                           : 1
        '''
# time
neuron_eqs_excite += '\n  dtimer/dt = 1  : second'
neuron_eqs_inhibite += '\n  dtimer/dt = 1  : second'

# reset dynamics for excitatory and inhibitory neuron
reset_excite = 'v = v_reset_excite; timer = 0 * ms; ge = 0; gi = 0;'
reset_inhibite = 'v = v_reset_inhibite; timer = 0 * ms; ge = 0; gi = 0'

# synapse conductance model
model_synapse_base = 'w : 1'
spike_pre_excite = 'ge_post += w'
spike_pre_inhibite = 'gi_post += w'

# 按照80%和20%的比例设置兴奋性神经元和抑制性神经元
G1 = NeuronGroup(int(int(num_neuron)*ex_rate), neuron_eqs_excite,
                 threshold=v_thresh_excite_str, refractory=t_refrac_excite, reset=reset_excite)
G2 = NeuronGroup(int(int(num_neuron)*in_rate), neuron_eqs_inhibite,
                 threshold=v_thresh_inhibite_str, refractory=t_refrac_inhibite, reset=reset_inhibite)

# ge 和 gi为神经元自定义参数
G1.ge = 0
G2.gi = 0

on_pre = spike_pre_excite
on_post = ''
S5 = Synapses(Gs, G1, model=model_synapse_base, on_pre=on_pre, on_post=on_post)
seed(1)
S5.connect(p=p_inpute)
# np.random.seed(1)
S5.w = 'rand()*'+weight_ie


on_pre = spike_pre_excite
on_post = ''
S1 = Synapses(G1, G1, model=model_synapse_base, on_pre=on_pre, on_post=on_post)
seed(1)
S1.connect(p=p_ee)
S1.w = 'rand()*'+w_ee


on_pre = spike_pre_excite
on_post = ''
S2 = Synapses(G1, G2, model=model_synapse_base, on_pre=on_pre, on_post=on_post)
seed(1)
S2.connect(p=p_ei)
S2.w = 'rand()*'+w_ei


on_pre = spike_pre_inhibite
on_post = ''
S3 = Synapses(G2, G1, model=model_synapse_base, on_pre=on_pre, on_post=on_post)
seed(1)
S3.connect(p=p_ie)
S3.w = 'rand()*'+w_ie

on_pre = spike_pre_inhibite
on_post = ''
S4 = Synapses(G2, G2, model=model_synapse_base, on_pre=on_pre, on_post=on_post)
seed(1)
S4.connect(p=p_ii)
S4.w = 'rand()*'+w_ii

M1 = SpikeMonitor(G1)
M2 = SpikeMonitor(G2)

run((work_time+rest_time)*num_data*brian2.ms,
    report='text', report_period=10*second)

index_array1 = M1.i
time_array1 = M1.t
index_array2 = M2.i
time_array2 = M2.t

# 保存神经元发放脉冲的时间和索引
np.save(spike_record_path +'index_array1_train', index_array1)
np.save(spike_record_path +'time_array1_train', time_array1)
np.save(spike_record_path +'index_array2_train', index_array2)
np.save(spike_record_path +'time_array2_train', time_array2)

# 保存连接状态和连接权重
np.save(simulation_archive_path+'IN_RCe_i.npy', S5.i)
np.save(simulation_archive_path+'IN_RCe_j.npy', S5.j)
np.save(simulation_archive_path+'RCe_RCe_i.npy', S1.i)
np.save(simulation_archive_path+'RCe_RCe_j.npy', S1.j)
np.save(simulation_archive_path+'RCe_RCi_i.npy', S2.i)
np.save(simulation_archive_path+'RCe_RCi_j.npy', S2.j)
np.save(simulation_archive_path+'RCi_RCe_i.npy', S3.i)
np.save(simulation_archive_path+'RCi_RCe_j.npy', S3.j)
np.save(simulation_archive_path+'RCi_RCi_i.npy', S4.i)
np.save(simulation_archive_path+'RCi_RCi_j.npy', S4.j)

np.save(simulation_archive_path+'s5w.npy', S5.w)
np.save(simulation_archive_path+'s1w.npy', S1.w)
np.save(simulation_archive_path+'s2w.npy', S2.w)
np.save(simulation_archive_path+'s3w.npy', S3.w)
np.save(simulation_archive_path+'s4w.npy', S4.w)

print('done')
