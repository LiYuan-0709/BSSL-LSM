#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   RC_test.py
@Time    :   2023/03/06 21:25:54
@Author  :   Li Yuan 
@Version :   1.0
@Contact :   ly1837372873@163.com
@Desc    :   将训练的连接权重加载并做测试
'''

# here put the import lib

import os
import sys
from brian2 import *
import numpy as np
import pickle as pickle
from struct import unpack
import brian2


print('RC_test')
start_scope()

seed(1)

k = sys.argv[8]
indices_path = './spikes_input/3second_9direction/'+k+'/index_test.npy'
times_path = './spikes_input/3second_9direction/'+k+'/time_test.npy'
simulation_archive_path = './simulation_archive/3second_9direction/'
spike_record_path = './spike_record/3second_9direction/'

num_neuron = int(sys.argv[1])
work_time = int(sys.argv[2])
rest_time = int(sys.argv[3])
num_data = int(sys.argv[4])
input_channel = int(sys.argv[5])
ex_rate = float(sys.argv[6])
in_rate = float(sys.argv[7])

#set_device('cpp_standalone', directory = './RC_simulation/output')
## setup for C++ compilation preferences
#codegen.cpp_compiler = 'gcc' #Compiler to use (uses default if empty);Should be gcc or msvc.
#prefs.devices.cpp_standalone.extra_make_args_unix = ['-j']
#prefs.devices.cpp_standalone.openmp_threads = 1
##-----------------------------------------------------------------------------------------
## basic simulation setup
##-----------------------------------------------------------------------------------------
brian2.defaultclock.dt = 0.1 * brian2.ms # 0.1
brian2.core.default_float_dtype = float32 ### can be chaged to float32 in brian 2.2
brian2.core.default_integer_dtype = int32

set_device('cpp_standalone')
codegen.cpp_compiler = 'gcc'
prefs.devices.cpp_standalone.extra_make_args_unix = ['-j']

indices = np.load(indices_path)
times = np.load(times_path)
input_indices = indices
input_times = times

# standard neuron parameters
output_vector = np.array([[0 for i in range(num_neuron)]
                         for j in range(num_data)])
v_rest_excite = -65.0 * mV
v_rest_inhibite = -60.0 * mV
v_reset_excite = -65.0 * mV
v_reset_inhibite = -45.0 * mV
# v_thresh_excite = -52.0 * mV
v_thresh_excite = -48.53601013154767 * mV
v_thresh_inhibite = -33.3596552351866 * mV  # -40
t_refrac_excite = 5 * ms
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

Gs = SpikeGeneratorGroup(input_channel, input_indices, input_times*ms)

G1 = NeuronGroup(int(ex_rate*int(num_neuron)), neuron_eqs_excite,
                 threshold=v_thresh_excite_str, refractory=t_refrac_excite, reset=reset_excite)
G2 = NeuronGroup(int(in_rate*int(num_neuron)), neuron_eqs_inhibite,
                 threshold=v_thresh_inhibite_str, refractory=t_refrac_inhibite, reset=reset_inhibite)
G1.ge = 0
G2.gi = 0

s5_i = np.load(simulation_archive_path+'IN_RCe_i.npy')
s5_j = np.load(simulation_archive_path+'IN_RCe_j.npy')
s1_i = np.load(simulation_archive_path+'RCe_RCe_i.npy')
s1_j = np.load(simulation_archive_path+'RCe_RCe_j.npy')
s2_i = np.load(simulation_archive_path+'RCe_RCi_i.npy')
s2_j = np.load(simulation_archive_path+'RCe_RCi_j.npy')
s3_i = np.load(simulation_archive_path+'RCi_RCe_i.npy')
s3_j = np.load(simulation_archive_path+'RCi_RCe_j.npy')
s4_i = np.load(simulation_archive_path+'RCi_RCi_i.npy')
s4_j = np.load(simulation_archive_path+'RCi_RCi_j.npy')

on_pre = spike_pre_excite
on_post = ''
S5 = Synapses(Gs, G1, model=model_synapse_base, on_pre=on_pre, on_post=on_post)
S5.connect(i=s5_i, j=s5_j)
S5.w = np.load(simulation_archive_path+'s5w.npy')


on_pre = spike_pre_excite
on_post = ''
S1 = Synapses(G1, G1, model=model_synapse_base, on_pre=on_pre, on_post=on_post)
S1.connect(i=s1_i, j=s1_j)
S1.w = np.load(simulation_archive_path+'s1w.npy')


on_pre = spike_pre_excite
on_post = ''
S2 = Synapses(G1, G2, model=model_synapse_base, on_pre=on_pre, on_post=on_post)
S2.connect(i=s2_i, j=s2_j)
S2.w = np.load(simulation_archive_path+'s2w.npy')


on_pre = spike_pre_inhibite
on_post = ''
S3 = Synapses(G2, G1, model=model_synapse_base, on_pre=on_pre, on_post=on_post)
S3.connect(i=s3_i, j=s3_j)
S3.w = np.load(simulation_archive_path+'s3w.npy')

on_pre = spike_pre_inhibite
on_post = ''
S4 = Synapses(G2, G2, model=model_synapse_base, on_pre=on_pre, on_post=on_post)
S4.connect(i=s4_i, j=s4_j)
S4.w = np.load(simulation_archive_path+'s4w.npy')

M1 = SpikeMonitor(G1)
M2 = SpikeMonitor(G2)

run((work_time+rest_time)*num_data*ms, report='text', report_period=10*second)

index_array1 = M1.i
time_array1 = M1.t
index_array2 = M2.i
time_array2 = M2.t

np.save(spike_record_path +'index_array1_test', index_array1)
np.save(spike_record_path +'time_array1_test', time_array1)
np.save(spike_record_path +'index_array2_test', index_array2)
np.save(spike_record_path +'time_array2_test', time_array2)
