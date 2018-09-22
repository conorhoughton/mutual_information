
include("./new_info_header.jl")


# creates a folder name based on time and date
foldername=Foldername()

#mu=1.0 corresponds to independent
#compared to the paper the variable mu corresponds to 1-mu

#run parameters
mu="varies"
window_length=45*ms::Float64
tau=15*ms
trials_n=100
train_length=200*sec::Float64

#saves stuff about the current run
run_parameter_names=["mu","window_length","h","tau","train_length","trials_n"]
run_parameters=Any[mu,window_length,"calculated",tau,train_length,trials_n]


save_to_log(foldername,vcat(neuron_parameters,run_parameters),vcat(neuron_parameter_names,run_parameter_names),"new")

small_file=open(string(foldername.name,"/info.dat"),"w")
big_file=open(string(foldername.name,"/all_data.dat"),"w")

key_file=open(string(foldername.name,"/README"),"w")

write(key_file,"info.dat:  mu average_info_over_trials\n")
write(key_file,"all_data.dat:  mu info_for_each_trial h_value_for_each_trial\n")

close(key_file)



#@profile begin

mu=0.0::Float64

while mu<=1.0

    #upper bound on h
    biggest_h=convert(Int64,floor(2*train_length))	

    old_h=biggest_h

    info_av=Float64[]
    h_av=Float64[]

    
    for trial_c in 1:trials_n
       
        #makes two fictive spike trains - spike_trains[1] and spike_trains[2]
        spike_trains=get_spike_trains([v_t,v_r,e_l,tau_m,tau_ref],[input_max,lasts],mu,dt,train_length)
        
        #splits the trains into fragments and sorts the distances
        #retains the biggest_h nearest points around each point

        fragments=chop_train(spike_trains[1],window_length,train_length)

        points_1=get_and_sort_distances(fragments,tau,biggest_h)

        fragments=chop_train(spike_trains[2],window_length,train_length)

    	points_2=get_and_sort_distances(fragments,tau,biggest_h)

	fragments=length(fragments)

	spike_train=0

        #calculates estimated mutual information, given h_new 

        function correct_info(h_new)
    
            info=information_from_matrix(points_1,points_2,h_new,h_new,1)
            
            info-background(fragments,h_new)

        end
        

        #golden mean search for best h

        phi=(1.0+sqrt(5.0))/2.0

	stride=min(2*old_h,fragments,biggest_h)

        a=10
        b=stride
        
        c = convert(Int64,floor(b-(b- a)/phi))
        d = convert(Int64,floor(a + (b- a)/phi))
        
        info_c=correct_info(c)
        info_d=correct_info(d)

        
        while abs(d-c)>2

            if info_c>info_d
                b=d
                d=c
                c = convert(Int64,floor(b-(b- a)/phi))                
                info_d=info_c
                info_c=correct_info(c)                

            else
                a=c
                c=d
                d = convert(Int64,floor(a + (b- a)/phi))
                info_c=info_d
                info_d=correct_info(d)                

            end

        end
        
        
        h=convert(Int64,floor((a+b)/2))

	old_h=h

        info_best=correct_info(h)

        push!(h_av,h)
        push!(info_av,info_best)

    end
    
    av=mean(info_av)

    write(small_file,"$mu $window_length $train_length $av\n")

    write(big_file,"$mu $window_length $train_length $info_av $h_av\n")

    flush(small_file)
    flush(big_file)

    mu+=0.1

end

close(small_file)
close(big_file)

#end

#Profile.print()


