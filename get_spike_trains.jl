
include("./metric.jl")
include("./neuron.jl")
include("./uniform_variable_input.jl")

function get_spike_trains(neuron_parameters::Array{Float64},input_parameters::Array{Float64},mu::Float64,dt::Float64,time_interval::Float64)

    (v_t,v_r,e_l,tau_m,tau_ref)=neuron_parameters
    (max,lasts)=input_parameters

    neuron1=Neuron(v_t,v_r,e_l,tau_m,tau_ref)
    neuron2=Neuron(v_t,v_r,e_l,tau_m,tau_ref)

    variable_input =Uniform_Variable_Input(max,lasts)
    variable_input1=Uniform_Variable_Input(max,lasts)
    variable_input2=Uniform_Variable_Input(max,lasts)

    t=0*sec::Float64
    
    t_final=t+time_interval

    spike_train1=Float64[]
    spike_train2=Float64[]

    while t<=t_final
        this_input=get_input!(variable_input,dt)
        this_input1=mu*get_input!(variable_input1,dt)+(1-mu)*this_input
        this_input2=mu*get_input!(variable_input2,dt)+(1-mu)*(max-this_input)
        if update_neuron!(neuron1,this_input1,dt)
            push!(spike_train1,t)
        end
        if update_neuron!(neuron2,this_input2,dt)
            push!(spike_train2,t)
        end
        t+=dt
    end

    [spike_train1,spike_train2]

end
