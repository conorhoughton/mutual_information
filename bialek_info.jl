#!/cm/shared/languages/Julia-0.6.2/julia-0.6.2/bin/julia
path_name="/panfs/panasas01/cosc/cscjh/test_information_formula/"
include("/panfs/panasas01/cosc/cscjh/test_information_formula/get_spike_trains.jl")
include("/panfs/panasas01/cosc/cscjh/test_information_formula/trains_to_word.jl")
include("/panfs/panasas01/cosc/cscjh/test_information_formula/information_from_dict.jl")
include("/panfs/panasas01/cosc/cscjh/test_information_formula/neuron_parameters.jl")
include("/panfs/panasas01/cosc/cscjh/test_information_formula/save.jl")

foldername=Foldername()

#mu=1.0 corresponds to independent
word_length=30::Int64
letter_length=1*ms

train_length=200000*sec::Float64

trials_n=20

mu=0.3

run_parameter_names=["mu","word_length","letter_length","train_length","trials_n"]
run_parameters=Any["varies",word_length,letter_length,train_length,trials_n]

save_to_log(foldername,vcat(neuron_parameters,run_parameters),vcat(neuron_parameter_names,run_parameter_names),"bialek")

small_file=open(string(foldername.name,"/info.dat"),"w")
big_file=open(string(foldername.name,"/all_data.dat"),"w")

key_file=open(string(foldername.name,"/README"),"w")

write(key_file,"info.dat:  mu average_info_over_trials\n")
write(key_file,"all_data.dat:  mu info_for_each_trial\n")


while train_length<1500000*sec
    #while mu<=1.0

#    sigma_prime=sigma/sqrt(mu^2+(1-mu)^2)
    
    info_av=Float64[]
    h_av=Float64[]

    for _ in 1:trials_n

        spike_trains=get_spike_trains([v_t,v_r,e_l,tau_m,tau_ref],[input_max,lasts],mu,dt,train_length)

        frequency_table=trains_to_word(spike_trains[1],spike_trains[2],word_length,letter_length,train_length)

        info=information_from_dict(frequency_table.table)/(word_length*letter_length)

        frequency_table_shuffled=trains_to_word_shuffled(spike_trains[1],spike_trains[2],word_length,letter_length,train_length)
        info_shuffled=information_from_dict(frequency_table_shuffled.table)/(word_length*letter_length)

        push!(info_av,info-info_shuffled)
#        push!(info_av,info)	

    end
    
    av=mean(info_av)

    println(mu," ",word_length," ",train_length," ",av)

    write(small_file,"$mu $word_length $train_length $av\n")

    write(big_file,"$mu $word_length $train_length $info_av\n")


    flush(small_file)
    flush(big_file)

    train_length+=100000*sec

 
end

close(small_file)
close(big_file)

comment_to_log()
