
struct Foldername

    name::String

    function Foldername()
        time=Dates.format(now(),"yyyy-mm-dd-HH-MM-SS")
        run(`mkdir ./results/$time/`)
        new("./results/$time")
    end

end

function copy_source(foldername::Foldername,filename::String)
    name=foldername.name
    run(`cp $filename ./$name/`)
end

function save_to_log(foldername::Foldername,parameters::Array{Any},parameter_names::Array{String},run_type::String)

    file=open("./results/log.txt","a")
    write(file,"\n")
    write(file,foldername.name,"\n")
    write(file,run_type,"\n")

    for i in 1:length(parameters)
        write(file,parameter_names[i]," ",string(parameters[i]),"\n")
    end

    close(file)

end

function comment_to_log()
    
    println("comment: ")
    message=readline(STDIN)

    file=open("./results/log.txt","a")
    write(file,"$message\n")

end
