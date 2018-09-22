
function information_from_dict(table::Dict{Array{Int64},Int64})

    function make_marginal(i::Int64)

        marginal=Dict{Int64,Int64}()

        for key in keys(table)
            marginal[key[i]]=get(marginal,key[i],0)+table[key]
        end

        marginal
            
    end

    marginal1=make_marginal(1)
    marginal2=make_marginal(2)

    total=0

    for key in keys(table)
        total+=table[key]
    end
    
    information=0.0::Float64

    for key in keys(table)

        p12=table[key]/total
        p1=marginal1[key[1]]/total
        p2=marginal2[key[2]]/total

        information+=p12*log(p12/(p1*p2))
    end

    if information<0
        println("negative information f/p")
        println(table)
    end



    information

end
