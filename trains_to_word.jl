include("./chop_train.jl")

struct Frequency_Table

    table::Dict{Array{Int64},Int64}

    function Frequency_Table()
        new(Dict{Array{Int64},Int64}())
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

function trains_to_word(spike_train1::Array{Float64},spike_train2::Array{Float64},word_length::Int64,letter_length::Float64)

    function advance(i::Int64)

        shift!(word[i])        
        if next[i]<=length(spike_train[i]) && time[i]>=spike_train[i][next[i]]
            push!(word[i],1)
            next[i]+=1
         else
            push!(word[i],0)
         end
    end


    table=Frequency_Table()

    next=[1,1]

    spike_train=Dict{Int64,Array{Float64}}(1=>spike_train1,2=>spike_train2)

    word1=[0::Int64 for _ in 1:word_length]
    word2=[0::Int64 for _ in 1:word_length]

    word=Dict{Int64,Array{Int64}}(1=>word1,2=>word2)

    time=[letter_length,letter_length]
    d_time=[letter_length,letter_length]

    for _ in 1:word_length
        advance(1)
        advance(2)
        time+=d_time
    end

    while next[1]<=length(spike_train[1]) || next[2]<=length(spike_train[2])  
        array_place=[array_to_int(word[1]),array_to_int(word[2])]
        table.table[array_place]=get(table.table,array_place,0)+1
        advance(1)
        advance(2)
        time+=d_time
    end

    table

end



function trains_to_word(spike_train1::Array{Float64},spike_train2::Array{Float64},word_length::Int64,letter_length::Float64,end_time::Float64)

    function fragment_to_word(u::Int64,i::Int64)

        word=[0::Int64 for _ in 1:word_length]

        for next in 1:length(fragments[u][i])
            word[convert(Int64,1+floor(fragments[u][i][next]/letter_length))]=1
        end

        word
            

    end


    table=Frequency_Table()

    fragments1=chop_train(spike_train1,word_length*letter_length,end_time)
    fragments2=chop_train(spike_train2,word_length*letter_length,end_time)


    fragments=Dict{Int64,Array{Array{Float64,1},1}}(1=>fragments1,2=>fragments2)

    for i in 1:length(fragments1)
        array_place=[array_to_int(fragment_to_word(1,i)),array_to_int(fragment_to_word(2,i))]
        table.table[array_place]=get(table.table,array_place,0)+1
    end

    table

end



function trains_to_word_shuffled(spike_train1::Array{Float64},spike_train2::Array{Float64},word_length::Int64,letter_length::Float64,end_time::Float64)

    function fragment_to_word(u::Int64,i::Int64)

        word=[0::Int64 for _ in 1:word_length]

        for next in 1:length(fragments[u][i])
            word[convert(Int64,1+floor(fragments[u][i][next]/letter_length))]=1
        end

        word
            

    end


    table=Frequency_Table()

    fragments1=chop_train(spike_train1,word_length*letter_length,end_time)
    fragments2=shuffle!(chop_train(spike_train2,word_length*letter_length,end_time))

    fragments=Dict{Int64,Array{Array{Float64,1},1}}(1=>fragments1,2=>fragments2)

    for i in 1:length(fragments1)
        array_place=[array_to_int(fragment_to_word(1,i)),array_to_int(fragment_to_word(2,i))]
        table.table[array_place]=get(table.table,array_place,0)+1
    end

    table

end


    
