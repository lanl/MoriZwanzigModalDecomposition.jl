using Mmap


function load_data_s8(f_path, nx, nz, nt_samp)
    """
    Data might be double precision, and byteswaped from the data transfer.
    Check byteswap by seeing if there are any values that are very large
        (e.g. 1e30) (TODO automate this...)
    """
    #f_path in path to binary file
    #nx number of grid points in x direction
    #nz number of grid points in z
    #nt_samp number of samples in time

    f = open(f_path, "r");
    F = bswap.(Mmap.mmap(f, Vector{Float64}, nt_samp * nz * nx))
    F = reshape(F, (nz, nx, nt_samp)); #F[:,:,t]'
    close(f)
    println("Field data loaded: size(F) = ", size(F))
    return F
end

function obtain_snapshot_matrix(f_path, nx, nz, nt_samp, t_skip)
    #t_skip used for: Δt_data = t_skip * Δt_sample
    F_full = load_data_s8(f_path, nx, nz, nt_samp)[:,:,1:t_skip:end];
    x0 = 0.15; xf = 0.55; nx = size(F_full)[2]; nz = size(F_full)[1];
    nt_samp = size(F_full)[3];
    X = reshape(F_full, (nx * nz, nt_samp))
    return X
end


# function read_binary_file(f_path, nx, nz, nt_samp)
#     #f_path in path to binary file
#     #nx number of grid points in x direction
#     #nz number of grid points in z
#     #nt_samp number of samples in time
#     file = open(f_path, "r");

#     println("reading binary data and performaing bit swap")
#     p = bswap.(Mmap.mmap(file, Vector{Float64}, nx * nz * nt_samp))
#     p = reshape(p, (nx, nz, nt_samp)); #F[:,:,t]
#     # println("mirror and stack to obtain full pressure field")
#     # p = permutedims(hcat(p, reverse(p, dims=2)), (2,1,3));
#     return p
# end