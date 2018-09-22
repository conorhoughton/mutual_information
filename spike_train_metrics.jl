
function traditional_metric(u_train::Array{Float64},v_train::Array{Float64},tau::Float64)

    trad_square_term(u_train,tau)+trad_square_term(v_train,tau)-2*trad_cross_term(u_train,v_train,tau)

end

function trad_square_term(u_train::Array{Float64},tau::Float64)

    u_square=0

    for ui in u_train
        for uj in u_train
            u_square+=exp(-abs(ui-uj)/tau)
        end
    end

    u_square

end
    
function trad_cross_term(u_train::Array{Float64},v_train::Array{Float64},tau::Float64)

    cross=0

    for ui in u_train
        for vj in v_train
            cross+=exp(-abs(ui-vj)/tau)
        end
    end

    cross

end

struct Trad_Spike_Train
    train::Array{Float64}
    square_term::Float64
    
    function Trad_Spike_Train(train,tau)
        new(train,trad_square_term(train,tau))
    end
end

function traditional_matrix(trains::Array{Array{Float64,1},1},tau::Float64)

    n=length(trains)

    
    spike_trains=[Trad_Spike_Train(train,tau) for train in trains]

    matrix=zeros(n,n)

    for i in 1:n
        for j in i+1:n
            distance=spike_trains[i].square_term+spike_trains[j].square_term-2*trad_cross_term(spike_trains[i].train,spike_trains[j].train,tau)
            matrix[i,j]=distance
            matrix[j,i]=distance
        end
    end

    matrix

end


function markage(train::Array{Float64},tau)

    n=length(train)

    markage_vector=zeros(n)

    for i in 2:n
        markage_vector[i]=(markage_vector[i-1]+1)*exp(-(train[i]-train[i-1])/tau)
    end

    markage_vector

end

struct Spike_Train

    train::Array{Float64}
    markage_vector::Array{Float64}
    square_term::Float64
    length::Int64

    function Spike_Train(train,tau)
        markage_vector=markage(train,tau)
        new(train,markage_vector,2*sum(markage_vector)+length(train),length(train))
    end

end

function cross_term_uv(u_train::Spike_Train,v_train::Spike_Train,tau)

    big_j=1
    start_i=1

    while start_i<u_train.length && u_train.train[start_i]< v_train.train[big_j]
        start_i+=1
    end

    uv=0

    for i in start_i:u_train.length
        while big_j<v_train.length && v_train.train[big_j+1] <= u_train.train[i]
            big_j+=1
        end
        m=v_train.markage_vector[big_j]
        if v_train.train[big_j] <= u_train.train[i]
            if u_train.train[i] != v_train.train[big_j]  
                uv+=(1+m)*exp(-abs(u_train.train[i]-v_train.train[big_j])/tau)
            else
                uv+=(1.0/2+m)
            end
        end
    end

    2*uv

end


function new_metric(u_train::Array{Float64},v_train::Array{Float64},tau::Float64)

    u_spike_train=Spike_Train(u_train,tau)
    v_spike_train=Spike_Train(v_train,tau)

    u_spike_train.square_term+v_spike_train.square_term-cross_term_uv(u_spike_train,v_spike_train,tau)-cross_term_uv(v_spike_train,u_spike_train,tau)

end


function new_distance(u_train::Spike_Train,v_train::Spike_Train,tau::Float64)
    if u_train.length*v_train.length==0
        return u_train.length+v_train.length
    end

    u_train.square_term+v_train.square_term-cross_term_uv(u_train,v_train,tau)-cross_term_uv(v_train,u_train,tau)
end


function new_matrix(trains::Array{Array{Float64,1},1},tau::Float64)

    n=length(trains)    
    spike_trains=[Spike_Train(train,tau) for train in trains]

    matrix=zeros(n,n)

    for i in 1:n
        for j in i+1:n
            distance=new_distance(spike_trains[i],spike_trains[j],tau)
            matrix[i,j]=distance
            matrix[j,i]=distance
        end
    end

    matrix

end

function sort_distances(distances::Array{Float64,2})
    sort_distances(distances,size(distances)[1])
end


function sort_distances(distances::Array{Float64,2},h::Int64)

    function points(i::Int64)
        lhs=[ [j,d] for (j,d) in enumerate(distances[i,1:i-1]) ]
        rhs=[[j+i,d] for (j,d) in enumerate(distances[i,i+1:end]) ]
        excised=vcat(lhs,rhs)
        Int64[convert(Int64,x[1]) for x in sort!(shuffle(excised),by = x -> x[2])]
    end


    function new_points(i::Int64)
        Int64[x[1] for x in sort!(shuffle([a for a in enumerate(distances[i,:])]),by = x -> x[2]) if x[1]!=i][1:h]
    end


    points_vector=Vector{Vector{Int64}}(size(distances)[1])

    for i in 1:size(distances)[1]
        points_vector[i]=new_points(i)
    end

    points_vector

end


function get_and_sort_distances(trains::Array{Array{Float64,1},1},tau::Float64,h::Int64)


    function new_points(i::Int64)
        Int64[x[1] for x in sort!(shuffle([a for a in enumerate(distances[:])]),by = x -> x[2]) if x[1]!=i][1:h]
    end



    n=length(trains)    
    spike_trains=[Spike_Train(train,tau) for train in trains]

    distances=Vector{Float64}(n)
    points_vector=Vector{Vector{Int64}}(size(distances)[1])

    for i in 1:n
        for j in 1:n
            if i==j
                distances[j]=0.0
            else
                distances[j]=new_distance(spike_trains[i],spike_trains[j],tau)
            end
        end

        points_vector[i]=new_points(i)

    end

    points_vector

end



function rate_distance_matrix(rates::Array{Float64})

    n=length(rates)

    
    matrix=zeros(n,n)

    for i in 1:n
        for j in i+1:n
            distance=abs(rates[i]-rates[j])
            matrix[i,j]=distance
            matrix[j,i]=distance
        end
    end

    matrix

end


        


