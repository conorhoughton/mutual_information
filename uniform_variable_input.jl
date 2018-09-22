
mutable struct Uniform_Variable_Input

    max::Float64

    lasts::Float64
    

    value::Float64
    new_change::Float64


    function Uniform_Variable_Input(max::Float64,lasts::Float64)        
        this_new_change=randexp()*lasts
        this_value=max*rand()
        new(max,lasts,this_value,this_new_change)
    end

end

function get_input!(variable_input::Uniform_Variable_Input,dt::Float64)

    variable_input.new_change-=dt

    if variable_input.new_change<=0
        variable_input.new_change=randexp()*variable_input.lasts
        variable_input.value=variable_input.max*rand()
    end

    return variable_input.value

end
