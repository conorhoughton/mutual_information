
mutable struct Variable_Input

    average::Float64
    sigma::Float64

    lasts::Float64
    

    value::Float64
    new_change::Float64


    function Variable_Input(average,sigma,lasts)        
        this_new_change=randexp()*lasts
        this_value=average+randn()*sigma
        new(average,sigma,lasts,this_value,this_new_change)
    end

end

function get_input!(variable_input::Variable_Input,dt::Float64)

    variable_input.new_change-=dt

    if variable_input.new_change<=0
        variable_input.new_change=randexp()*variable_input.lasts
        variable_input.value=variable_input.average+randn()*variable_input.sigma
    end

    return variable_input.value

end
