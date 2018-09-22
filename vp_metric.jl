


function vp_get_and_sort_distances(trains::Array{Array{Float64,1},1},q::Float64,h::Int64)

    function new_points(i::Int64)
        Int64[x[1] for x in sort!(shuffle([a for a in enumerate(distances[:])]),by = x -> x[2]) if x[1]!=i][1:h]
    end

    n=length(trains)    

    distances=Vector{Float64}(n)
    points_vector=Vector{Vector{Int64}}(size(distances)[1])

    for i in 1:n
        for j in 1:n
            if i==j
                distances[j]=0.0
            else
                distances[j]=vp_distance(trains[i],trains[j],q)
            end
        end

        points_vector[i]=new_points(i)

    end

    points_vector

end

function vp_distance(u_train::Vector{Float64},v_train::Vector{Float64},q::Float64)

    u_train_length=length(u_train)
    v_train_length=length(v_train)

    if u_train_length*v_train_length==0
        return convert(Float64,u_train_length+v_train_length)
    end
    
    function get(g::Array{Float64},i::Int64,j::Int64)
        g[i+1,j+1]
    end

    
    function set(g::Array{Float64},i::Int64,j::Int64,new_value::Float64)
        g[i+1,j+1]=new_value
    end


    g=zeros(Float64,u_train_length+1,v_train_length+1)

    for i in 1:u_train_length
        set(g,i,0,convert(Float64,i))   
    end

    for j in 1:v_train_length
        set(g,0,j,convert(Float64,j))   
    end

    for j in 1:v_train_length
        for i in 1:u_train_length
            set(g,i,j,minimum([get(g,i-1,j-1)+q*abs(u_train[i]-v_train[j]),get(g,i-1,j)+1.0,get(g,i,j-1)+1.0]))
        end
    end

    get(g,u_train_length,v_train_length)

end

v=Float64[0.0,1.4]
u=Float64[0.0,1.0,2.0]


q=1.0::Float64

println(vp_distance(u,v,q))
