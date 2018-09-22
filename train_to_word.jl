
struct Frequency_Table

    table::Dict{Int64,Int64}

    function Frequency_Table()
        new(Dict{Int64,Int64}())
    end

end

function array_to_int(word::Array{Int64})
    place=1
    total=0
    for w in word
        if w!=0
            total+=place
            end
        place*=2
    end

    total

end

function train_to_word(spike_train::Array{Float64},word_length::Int64,letter_length::Float64)

    function advance()
        shift!(word)
        if time>=spike_train[next]
            push!(word,1)
            next+=1
         else
            push!(word,0)
         end
    end


    table=Frequency_Table()

    next=1

    word=[0::Int64 for _ in 1:word_length]

    time=letter_length

    for _ in 1:word_length
        advance()
        time+=letter_length
    end

    while next<=length(spike_train)
        array_place=array_to_int(word)
        table.table[array_place]=get(table.table,array_place,0)+1
        advance()
        println(word," ",array_place)
        time+=letter_length
    end

    table

end

    
