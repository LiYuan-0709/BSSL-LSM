#RC
from brian2 import *
import brian2 as b2
from brian2.monitors import statemonitor
import numpy as np
from struct import unpack 


start_scope()
#set_device('cpp_standalone', directory = './RC_simulation/output')
## setup for C++ compilation preferences
#codegen.cpp_compiler = 'gcc' #Compiler to use (uses default if empty);Should be gcc or msvc.
#prefs.devices.cpp_standalone.extra_make_args_unix = ['-j']
#prefs.devices.cpp_standalone.openmp_threads = 1
##-----------------------------------------------------------------------------------------
## basic simulation setup
##-----------------------------------------------------------------------------------------
b2.defaultclock.dt = 0.1 * b2.ms # 0.1
b2.core.default_float_dtype = float32 ### can be chaged to float32 in brian 2.2
b2.core.default_integer_dtype = int32

start_scope()
indices = np.load('./input_spike_train_index0001.npy')
times = np.load('./input_spike_train_time0001.npy')

print('load data successful!')

Gs = SpikeGeneratorGroup(64*64,indices,times*ms)

#loihi LIF 神经元
## standard neuron parameters
v_rest_excite = 0
v_rest_inhibite = 0
v_reset_excite = 0
v_reset_inhibite = 0
v_thresh_excite = 1.25*2**13
v_thresh_inhibite = 1.5*2**13 # -40
t_refrac_excite = 5.0 * b2.ms # 5.0
t_refrac_inhibite = 2.0 * b2.ms # 2.0
## spike conditon
v_thresh_excite_str = 'v > v_thresh_excite'
v_thresh_inhibite_str = 'v > v_thresh_inhibite'
## excitatory neuron membrane dynamics
neuron_eqs_excite = '''
        rnd_v = int(sign(v)*floor(abs(v*(32/2**12)))) : 1
        rnd_I = int(sign(I)*floor(abs(I*(128/2**12)))) : 1
        rrnd_I = int(sign(I)*floor(abs(I*(1/2**3)))) : 1
        dv/dt = -floor(rnd_v/(0.1*ms)) + floor(rrnd_I/(0.1*ms)): 1 (unless refractory)
        dI/dt = -floor(rnd_I/(0.1*ms)): 1
        '''
neuron_eqs_inhibite = '''
        rnd_v = int(sign(v)*floor(abs(v*(32/2**12)))) : 1
        rnd_I = int(sign(I)*floor(abs(I*(128/2**12)))) : 1
        rrnd_I = int(sign(I)*floor(abs(I*(1/2**3)))) : 1
        dv/dt = -floor(rnd_v/(0.1*ms)) + floor(rrnd_I/(0.1*ms)): 1 (unless refractory)
        dI/dt = -floor(rnd_I/(0.1*ms)): 1
        '''
## time
## reset dynamics for excitatory and inhibitory neuron
reset_excite = 'v = v_reset_excite; I = 0'
reset_inhibite = 'v = v_reset_inhibite; I = 0'
## synapse conductance model
model_synapse_base = 'w : 1'
spike_pre_excite = ' I += w*2**7'
spike_pre_inhibite = 'I -= w*2**7'


G1 = NeuronGroup(274, neuron_eqs_excite, threshold = v_thresh_excite_str, refractory = t_refrac_excite, reset = reset_excite)
G2 = NeuronGroup(69, neuron_eqs_inhibite, threshold = v_thresh_inhibite_str, refractory = t_refrac_inhibite, reset = reset_inhibite)
G1.I = 0
G2.I = 0

input_e_i = np.load('./simulation_archive/random_connection/index_i_IN_RC00e.npy')
input_e_j = np.load('./simulation_archive/random_connection/index_j_IN_RC00e.npy')
ee_i = np.load('./simulation_archive/random_connection/index_i_RC00e_RC00e.npy')
ee_j = np.load('./simulation_archive/random_connection/index_j_RC00e_RC00e.npy')
ei_i = np.load('./simulation_archive/random_connection/index_i_RC00e_RC00i.npy')
ei_j = np.load('./simulation_archive/random_connection/index_j_RC00e_RC00i.npy')
ie_i = np.load('./simulation_archive/random_connection/index_i_RC00i_RC00e.npy')
ie_j = np.load('./simulation_archive/random_connection/index_j_RC00i_RC00e.npy')
ii_i = np.load('./simulation_archive/random_connection/index_i_RC00i_RC00i.npy')
ii_j = np.load('./simulation_archive/random_connection/index_j_RC00i_RC00i.npy')


on_pre = spike_pre_excite
on_post = '' 
S5 = Synapses(Gs,G1,model = model_synapse_base, on_pre=on_pre,on_post = on_post)
S5.connect(i = input_e_i,j = input_e_j)
S5.w = np.load('./simulation_archive/random_connection/weight_learned_IN_RC00e.npy')


on_pre = spike_pre_excite
on_post = '' 
S1 = Synapses(G1,G1,model = model_synapse_base, on_pre = on_pre, on_post = on_post)
S1.connect(i = ee_i,j = ee_j)
S1.w = np.load('./simulation_archive/random_connection/weight_learned_RC00e_RC00e.npy')


on_pre = spike_pre_excite
on_post = '' 
S2 = Synapses(G1,G2,model = model_synapse_base, on_pre = on_pre, on_post = on_post)
S2.connect(i = ei_i,j = ei_j)
S2.w = np.load('./simulation_archive/random_connection/weight_learned_RC00e_RC00i.npy')


on_pre = spike_pre_inhibite
on_post = '' 
S3 = Synapses(G2,G1,model = model_synapse_base, on_pre = on_pre, on_post = on_post)
S3.connect(i = ie_i,j = ie_j)
S3.w = np.load('./simulation_archive/random_connection/weight_learned_RC00i_RC00e.npy')

on_pre = spike_pre_inhibite
on_post = '' 
S4 = Synapses(G2,G2,model = model_synapse_base, on_pre = on_pre, on_post = on_post)
S4.connect(i = ii_i,j = ii_j)
S4.w = np.load('./simulation_archive/random_connection/weight_learned_RC00i_RC00i.npy')

M1 = SpikeMonitor(G1)
M2 = SpikeMonitor(G2)
V1 = StateMonitor(G1,'v',record=True)
V2 = StateMonitor(G2,'v',record=True)
I1 = StateMonitor(G1,'I',record=True)
I2 = StateMonitor(G2,'I',record=True)
print(scheduling_summary())

run(800*ms,report = 'text',report_period = 600*second)

np.save('./recorder/indexe',np.array(M1.i))
np.save('./recorder/timee',np.array(M1.t))
np.save('./recorder/indexi',np.array(M2.i))
np.save('./recorder/timei',np.array(M2.t))
np.save('./recorder/ev',np.array(V1.v))
np.save('./recorder/iv',np.array(V2.v))
np.save('./recorder/ei',np.array(I1.I))
np.save('./recorder/ii',np.array(I2.I))

print('done')

