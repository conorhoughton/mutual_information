
include("./new_info_header.jl")

include("./vp_metric.jl")

foldername=Foldername()

#mu=1.0 corresponds to independent

mu=0.0

window_length=45*ms::Float64

tau=15*ms

q=2.0/tau

trials_n=100

train_length=200*sec::Float64

run_parameter_names=["mu","window_length","h","big_h_stride","small_h_stride","tau","train_length","trials_n"]
run_parameters=Any[mu,window_length,"not in use","not in use","not in use",tau,"varies",trials_n]

save_to_log(foldername,vcat(neuron_parameters,run_parameters),vcat(neuron_parameter_names,run_parameter_names),"new")

small_file=open(string(foldername.name,"/info.dat"),"w")
big_file=open(string(foldername.name,"/all_data.dat"),"w")

key_file=open(string(foldername.name,"/README"),"w")

write(key_file,"info.dat:  mu average_info_over_trials\n")
write(key_file,"all_data.dat:  mu info_for_each_trial h_value_for_each_trial\n")

close(key_file)

#@profile begin

#while train_length<=700*sec
#while window_length<=150*ms
while mu<1.0

#    train_length=window_length/ms

    old_h=convert(Int64,floor(2*train_length))	

    biggest_h=old_h

    info_av=Float64[]
    h_av=Float64[]
    
    for trial_c in 1:trials_n
       
        spike_trains=get_spike_trains([v_t,v_r,e_l,tau_m,tau_ref],[input_max,lasts],mu,dt,train_length)
        

        fragments=chop_train(spike_trains[1],window_length,train_length)

        points_1=vp_get_and_sort_distances(fragments,q,biggest_h)

        fragments=chop_train(spike_trains[2],window_length,train_length)

    	points_2=vp_get_and_sort_distances(fragments,q,biggest_h)

	fragments=length(fragments)

	spike_train=0

        function correct_info(h_new)
    
            info=information_from_matrix(points_1,points_2,h_new,h_new,1)
            
            info-background(fragments,h_new)
        end
        
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

    println(mu," ",window_length," ",train_length," ",av/log(2))

    write(small_file,"$mu $window_length $train_length $av\n")

    write(big_file,"$mu $window_length $train_length $info_av $h_av\n")

    flush(small_file)
    flush(big_file)

    mu+=0.1

#    window_length+=10*ms

#    train_length+=50*sec

end

close(small_file)
close(big_file)

#end

#Profile.print()


#comment_to_log()
