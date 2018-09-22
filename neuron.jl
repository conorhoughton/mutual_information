
mutable struct Neuron

    v_t::Float64
    v_r::Float64
    e_l::Float64

    tau_m::Float64

    tau_ref::Float64

    voltage::Float64
    ref_time::Float64

    function Neuron(v_t,v_r,e_l,tau_m,tau_ref)
        new(v_t,v_r,e_l,tau_m,tau_ref,e_l,0.0)
    end

end

function update_neuron!(neuron::Neuron,input::Float64,dt::Float64)

    if neuron.ref_time>0
        neuron.ref_time-=dt
        return false
    end

    dv=dt*(neuron.e_l-neuron.voltage+input)/neuron.tau_m
    neuron.voltage+=dv

    if neuron.voltage>neuron.v_t
        neuron.voltage=neuron.v_r
        neuron.ref_time=neuron.tau_ref
        return true
    end

    false

end
