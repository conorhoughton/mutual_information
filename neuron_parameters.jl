
v_t=-55*mv::Float64
v_r=-70*mv::Float64
e_l=-70*mv::Float64

tau_m=12*ms::Float64
tau_ref=2*ms::Float64


input_max=35*mv::Float64
#sigma=8*mv::Float64

lasts=30*ms::Float64

dt=0.1*ms::Float64

neuron_parameter_names=["v_t","v_r","e_l","tau_m","tau_ref","input_max","lasts","dt"]
neuron_parameters=Any[v_t,v_r,e_l,tau_m,tau_ref,input_max,lasts,dt]
