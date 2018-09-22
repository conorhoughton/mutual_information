
function information_from_matrix(u_distances::Array{Float64,2},v_distances::Array{Float64,2},u_h::Int64,v_h::Int64)
    information_from_matrix(u_distances,v_distances,u_h,v_h,1)
end

function information_from_matrix(u_distances::Array{Float64,2},v_distances::Array{Float64,2},u_h::Int64,v_h::Int64,leave_out::Int64)

    n=size(u_distances)[1]

    h=[u_h,v_h]

    if n!= size(u_distances)[2] || n!=size(v_distances)[1] || n!=size(v_distances)[2]
        println("distance matrix size f/p")
    end

    function points(u::Int64,i::Int64)
        lhs=[ [j,d] for (j,d) in enumerate(distances[u][i,1:i-1]) ]
        rhs=[[j+i,d] for (j,d) in enumerate(distances[u][i,i+1:end]) ]
        excised=vcat(lhs,rhs)
        [convert(Int64,x[1]) for x in sort!(shuffle(excised),by = x -> x[2])[1:h[u]]]
    end

    information=0.0::Float64

    distances=(u_distances,v_distances)

    total_added=0


    for i in 1:n
        
        v=intersect(points(1,i),points(2,i))
        
        hash_v=leave_out+length(v)
        
        if hash_v>0
            information+=log((n+leave_out-1)*hash_v/(u_h*v_h))
            total_added+=1
        end
        
    end
    

    information/total_added

end


function information_from_matrix(u_points::Vector{Vector{Int64}},v_points::Vector{Vector{Int64}},u_h::Int64,v_h::Int64,leave_out::Int64)


   function convert_to_sets(points,h)
     points_set=Vector{Set{Int64}}(length(points))	
     for (i,p) in enumerate(points)
         points_set[i]=Set{Int64}(a for a in p[1:h])
     end
     points_set
    end

    u_sets=convert_to_sets(u_points,u_h)
    v_sets=convert_to_sets(v_points,v_h)
    
    n=size(u_points)[1]

    information=0.0::Float64

    total_added=0::Int64


    for i in 1:n
                                 
        v=intersect(u_sets[i],v_sets[i]) 
            
        hash_v=leave_out+length(v)

        if hash_v>0
            information+=log((n+leave_out-1)*hash_v/(u_h*v_h))
            total_added+=1
        end

    end
    
    information/total_added

end


function j_from_matrix(u_distances::Array{Float64,2},v_distances::Array{Float64,2},h::Int64)

    n=size(u_distances)[1]

    if n!= size(u_distances)[2] || n!=size(v_distances)[1] || n!=size(v_distances)[2]
        println("distance matrix size f/p")
    end

    function points(u::Int64,i::Int64,this_h)
        excised=vcat(distances[u][i,1:i-1],distances[u][i,i+1:end])
        [convert(Int64,x[1]) for x in sort!(shuffle([[i,d] for (i,d) in enumerate(excised)]),by = x -> x[2])[1:this_h]]
    end

    big_j=0
    
    distances=(u_distances,v_distances)

    total_points=0

    for i in 1:n
        
        v=intersect(points(1,i,h-1),points(2,i,h-1))
        hash_v=1+length(v)


        big_j+=-n*hash_v/h^2*log(n*hash_v/h^2)
 
    end

    big_j/n

end


        

function calculate_coefficient(n::Int64,h::Int64,r::Int64)

    value=1.0::Float64

    if r==0
        for counter in 0:h-1
            value*=(n-h-counter)/(n-counter)
         end
        return value
    end

    # (h r) (n-h h-r) / (n h)
    # (h ... r terms) (n-h  (h-r) terms)    (h . . . 1)
    # (r ... 1)     (h-r . . . 1)         (n ..  h terms)
    #

    for counter in 0:h-r-1
        value*= ((n-h-counter)/(n-counter)) * ((h-counter)/(h-r-counter)) 
    end

    for counter in 0:r-1
        value*= ((h-counter)/(n-h+r-counter))
    end

    value

end


function background(n::Int64,h::Int64)

    info=0.0::Float64

    for r in 0:h-1
        info+= calculate_coefficient(n-1,h-1,r)*log( n*(r+1)/(h^2))
    end
    
    info

end


function background_leave_out(n::Int64,h::Int64)

    info=0.0::Float64

    for r in 1:h
        info+= calculate_coefficient(n-1,h,r)*log( (n-1)*r/(h^2))
    end

    info

end
