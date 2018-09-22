

function chop_train(train::Array{Float64,1},window_length::Float64,end_time::Float64)

    next=1

    time=0.0::Float64

    fragments=Array{Array{Float64,1},1}()

    while  time<end_time
        this_fragment=Array{Float64,1}()
        while next<=length(train) && train[next]<time+window_length
            push!(this_fragment,train[next]-time)
            next+=1
        end
        push!(fragments,this_fragment)
        time+=window_length
    end

    fragments

end
