using Plots, LaTeXStrings, LinearAlgebra, NPZ
using Statistics


#--------- Include files
include("plot_functions.jl")
include("./src/compute_obervability_amplitudes_optimized.jl")
include("./src/main_mz_algorithm.jl")
include("./src/mzmd_modes.jl")


#--------- Load data
re_num = 9
method="companion"
a_method = "x0";
sample = 1;

nz = 129; nx = 4600; 
nt_samp = 3200; 
dt = 2.0e-7;

#----------------------- ADD DATA HERE
X = load_data();
println("type of X = ", typeof(X));


#----------------------- mean subtract and select training data
X_mean = mean(X, dims=2);
X = X .- X_mean;
m = size(X, 1);

t_skip = 1;
m = size(X, 1);
t_end = 2400 #total n steps is 751
t_pred = 100
T_train = ((sample):(t_end+sample-1))[1:t_skip:end];
T_test = ((t_end+sample):(t_end + t_pred + sample))[1:t_skip:end];
time_train = dt * T_train
time_test = dt * t_skip * T_test;
t_all = 0:dt:(dt*(t_end + t_pred +1));
X_train = X[:,T_train];
X_test = X[:,T_test];
X = nothing;

#-------- Set params
#number of memory terms:
# n_ks = 21
n_ks = 10;
#number of snapshots used in fitting
t_win = size(T_train, 1) - n_ks - 1;
r = 100;


#-------- Compute MZ on POD basis observables
X1 = X_train[:, 1:t_win]; #Used to obtain Ur on 1:twin
X0 = X_train[:, 1];

#compute svd based observables (X_proj: states projected onto pod modes)
#method of snapshot more efficient for tall skinny X
# S, Ur, X_proj = svd_method_of_snapshots(X_train, r, subtractmean=true)
U, Sigma, V = svd(X_train);
sigmas_ = diagm(Sigma);
X_proj = sigmas_[1:r,1:r] * V[:,1:r]';
Ur = U[:, 1:r];

#initial condtions with memory for obtaining predictions
X0r = X_proj[:, 1:n_ks];

function obtain_mz_operators()
    #compute two time covariance matrices
    Cov = obtain_C(X_proj, t_win, n_ks);
    #compute MZ operators with Mori projection
    M, Ω = obtain_ker(Cov, n_ks);
    println("----------------------------");
    println("Two time covariance matrices: size(C)) = ", size(Cov));
    println("Markovian operator: size(M)  = ", size(M));
    println("All mz operators: size(Ω)  = ", size(Ω));
    return Cov, M, Ω
end
Cov, M, Ω = obtain_mz_operators()
plotting_mem_norm(Ω);

C = form_companion(Ω, r, n_ks);


function mzmd_modes_reduced_amps_final(C, Ur, r, n_ks)
    #compute modes and amplitudes from C
    Λc, Vc = eigen(C);
    #mzmd modes:
    Φ_mz = Ur*Vc[1:r,:];
    #use initial conditions with k memory terms:
    # X0_mem = X_train[:, 1:n_ks];
    X0_mem = X_test[:, 1:n_ks];
    #compute amplitudes of modes from observable space
    Z0 = Ur' * X0_mem;
    z0 = zeros(n_ks*r)
    for k in 1:n_ks
        # z0[((k-1)*r+1):k*r] = Z0[:,k]
        z0[((k-1)*r+1):k*r] = Z0[:,(n_ks - k + 1)]
    end
    a = pinv(Vc)*z0;
    # a = pinv(Vc[1:r,:])*Z0[:,1];
    # a = pinv(Φ_mz)*X_train[:,1];
    amp = zeros(r*n_ks);
    for i in 1:r*n_ks
        amp[i] = norm(Φ_mz[:,i]*a[i]);
    end
    #sort according to largest amplitude:
    ind = sortperm(abs.(amp), rev=true)
    return Λc[ind], Φ_mz[:, ind], a[ind], amp[ind], Vc[:, ind]
end

Λc, Φ_mz, a, amp, Vc = mzmd_modes_reduced_amps_final(C, Ur, r, n_ks) #fast and accurate algorithm

prim_inst_indx, secd_inst_indx, thrd_inst_indx = [1,3,5]

function plot_c_spectrum2(Λc)
    gr(size=(570,570), xtickfontsize=12, ytickfontsize=12, xguidefontsize=20, yguidefontsize=14, legendfontsize=12,
        dpi=300, grid=(:y, :gray, :solid, 1, 0.4), palette=cgrad(:plasma, 3, categorical = true));
    function circle_shape(h,k,r)
        θ=LinRange(0,2*pi,500);
        return h .+ r*sin.(θ), k .+ r*cos.(θ)
    end
    plt = scatter(Λc, linewidth=2.0, ms = 4.0, color = "blue", legend=false, markerstrokewidth=0, grid = true, framestyle = :box)
    scatter!([Λc[prim_inst_indx]], ms = 6.0, color="deepskyblue3", markershape=:hexagon, label=L"f: \textrm{Fundamental}")
    scatter!([Λc[secd_inst_indx+1]], ms = 7.0, color="green", markershape=:circle, label=L"1^{st} \textrm{Harmonic}")
    scatter!([Λc[thrd_inst_indx]], ms = 6.0, color="gold4", markershape=:square, label=L"2^{nd} \textrm{Harmonic}")
    plot!(circle_shape(0,0,1.0), linewidth=1.5, color="black", linestyle=:dash)
    if n_ks>1
        title!(L"\textrm{MZMD ~ Eigenvalues }", titlefont=16)
    else
        title!(L"\textrm{DMD ~ Eigenvalues }", titlefont=16)
    end
    xlabel!(L"\textrm{Re}(\lambda)", xtickfontsize=10, xguidefontsize=14)
    ylabel!(L"\textrm{Im}(\lambda)", ytickfontsize=10, yguidefontsize=14)
    return plt
end

function plot_c_spectrum_sort(Λc)
    gr(size=(570,570), xtickfontsize=12, ytickfontsize=12, xguidefontsize=20, yguidefontsize=14, legendfontsize=12,
        dpi=300, grid=(:y, :gray, :solid, 1, 0.4), palette=cgrad(:plasma, 3, categorical = true));
    function circle_shape(h,k,r)
        θ=LinRange(0,2*pi,500);
        return h .+ r*sin.(θ), k .+ r*cos.(θ)
    end
    
    plt = scatter(Λc[1:r], linewidth=2.0, ms = 4, color = "blue", legend=false, markerstrokewidth=0, grid = true, framestyle = :box)
    scatter!(Λc[(1+r):2r], linewidth=2.0, ms = 4, color = "cyan1", legend=false, markerstrokewidth=0, grid = true, framestyle = :box)
    scatter!(Λc[(1+2r):end], linewidth=2.0, ms = 2, color = "cyan4", legend=false, grid = true, markerstrokewidth=0, framestyle = :box)
    scatter!([Λc[prim_inst_indx+1]], ms = 6.0, color="deepskyblue3", markershape=:hexagon, label=L"f: \textrm{Fundamental}")
    scatter!([Λc[secd_inst_indx+1]], ms = 7.0, color="green", markershape=:circle, label=L"1^{st} \textrm{Harmonic}")
    scatter!([Λc[thrd_inst_indx+1]], ms = 6.0, color="gold4", markershape=:square, label=L"2^{nd} \textrm{Harmonic}")
    scatter!([Λc[r+1]], ms = 9.0, color="orange", markershape=:star, markerstrokewidth=0, label=L"\textrm{Dominant ~ memory ~ mode}")
    scatter!(Λc[(2+r):2r], linewidth=2.0, ms = 4, color = "brown", legend=false, markerstrokewidth=0, grid = true, framestyle = :box)
    scatter!(Λc[(1+2r):end], linewidth=2.0, ms = 2, color = "cyan4", legend=false, grid = true, markerstrokewidth=0, framestyle = :box)
    plot!(circle_shape(0,0,1.0), linewidth=1.5, color="black", linestyle=:dash)
    if n_ks>1
        title!(L"\textrm{MZMD ~ Eigenvalues }", titlefont=16)
    else
        title!(L"\textrm{DMD ~ Eigenvalues }", titlefont=16)
    end
    xlabel!(L"\textrm{Re}(\lambda)", xtickfontsize=10, xguidefontsize=14)
    ylabel!(L"\textrm{Im}(\lambda)", ytickfontsize=10, yguidefontsize=14)
    return plt
end


function amp_log(lam, a, dt, method, ylim_)
    gr(size=(570,450), xtickfontsize=12, ytickfontsize=12, xguidefontsize=20, yguidefontsize=14, legendfontsize=12,
    dpi=300, grid=(:y, :gray, :solid, 1, 0.4), palette=cgrad(:plasma, 3, categorical = true));
    r_2 = floor(Int, r/2);
    freqs = (imag.(log.(lam))/(2*pi*dt))./1000
    max_a = maximum(abs.(a))
    println("prim instability freq = ", abs.(real(freqs))[1])
    freqs1 = freqs[1:r]; a1 = a[1:r];
    if n_ks>1
        plt = plot([abs.(real(freqs))[prim_inst_indx], abs.(real(freqs))[prim_inst_indx]], [ylim_, (abs.(a)/max_a)[prim_inst_indx]], 
                    lw=2, color="blue", xlims=(0, 2600), yaxis=:log, label=L"\textrm{MZMD}(1 \leq i \leq r)")
    else
        plt = plot([abs.(real(freqs))[prim_inst_indx], abs.(real(freqs))[prim_inst_indx]], [ylim_, (abs.(a)/max_a)[prim_inst_indx]], 
                    lw=2, color="blue", xlims=(0, 2600), yaxis=:log, label=L"\textrm{DMD}")
    end
    if n_ks>1
        a1 = a[1:r]; a2 = a[(r+1):2r]; a3_ = a[(2r+1):end]; 
        lam_1 = lam[1:r]; lam_2 = lam[(r+1):2r]; lam_3 = lam[(2r+1):3r];
        freqs1 = freqs[1:r]; freqs2 = freqs[(r+1):2r]; freqs3_ = freqs[(2r+1):end];
        for i in 2:length(real(freqs2))
            plot!([abs.(real(freqs2))[i], abs.(real(freqs2))[i]], [ylim_, (abs.(a2)/max_a)[i]], lw=2, 
                        color="brown", yscale=:log10, ylims=(1e-4, 1.25), minorgrid=true, label="")
        end
        for i in 2:length(real(freqs3_))
            plot!([abs.(real(freqs3_))[i], abs.(real(freqs3_))[i]], [ylim_, (abs.(a3_)/max_a)[i]], lw=1, 
                        color="cyan4", yscale=:log10, ylims=(1e-4, 1.5), minorgrid=true, label="")
        end
        plot!([abs.(real(freqs2))[1], abs.(real(freqs2))[1]], [ylim_, (abs.(a2)/max_a)[1]], lw=2, 
                        color="brown", yscale=:log10, ylims=(1e-4, 1.25), minorgrid=true, label=L"\textrm{MZMD}(r+1 \leq i \leq 2r)")
        plot!([abs.(real(freqs3_))[1], abs.(real(freqs3_))[1]], [ylim_, (abs.(a3_)/max_a)[1]], lw=1, 
                        color="cyan4", yscale=:log10, ylims=(1e-4, 1.5), minorgrid=true, legendfontsize=11, label=L"\textrm{MZMD}(2r+1 \leq i \leq kr)")
        title!(L"\textrm{Amplitude ~ vs. ~ frequency ~ MZMD ~ }", titlefont=16)
    else
        title!(L"\textrm{Amplitude ~ vs. ~ frequency ~ DMD ~ }", titlefont=16)
    end
    scatter!([abs.(real(freqs))[prim_inst_indx]], [(abs.(a)/max_a)[prim_inst_indx]], ms = 6.0, color="deepskyblue3", markershape=:hexagon, label=L"f: \textrm{Fundamental}")
    scatter!([abs.(real(freqs))[secd_inst_indx]], [(abs.(a)/max_a)[secd_inst_indx]], ms = 7.0, color="green", markershape=:circle, label=L"1^{st} \textrm{Harmonic}")
    scatter!([abs.(real(freqs))[thrd_inst_indx]], [(abs.(a)/max_a)[thrd_inst_indx]], ms = 6.0, color="gold4", markershape=:square, label=L"2^{nd} \textrm{Harmonic}")

    # Add vertical lines from each point to the x-axis
    for i in 1:length(real(freqs1))
        plot!([abs.(real(freqs1))[i], abs.(real(freqs1))[i]], [ylim_, (abs.(a1)/max_a)[i]], lw=2, color="blue", yscale=:log10, ylims=(1e-4, 1.5), minorgrid=true, label="")
    end
    if n_ks > 1
        scatter!([abs.(real(freqs))[r+3]], [(abs.(a)/max_a)[r+1]], ms = 9.0, color="orange", markershape=:star, markerstrokewidth=0, label=L"\textrm{Dominant ~ memory ~ mode}")
    end

    xlabel!(L"\textrm{Frequency ~ (kHz)}", xtickfontsize=10, xguidefontsize=14)
    ylabel!(L"\textrm{Normalized ~ Amplitude}", ytickfontsize=10, yguidefontsize=14)
    display(plt)
    savefig(plt, "./figures/$(method)_log_spectrum_vs_ampr$(r)_k$(n_ks)_tend$(t_end)_re$(re_num).png")
    return plt
end


plt = amp_log(Λc, amp, dt, "mzmd", 5e-7)
display(plt)


if n_ks>1
    plt = plot_c_spectrum_sort(Λc[1:end])
    savefig(plt, "./figures/mzmd_spectrum_r$(r)_k$(n_ks)_re$(re_num).png")
    display(plt)
end

if n_ks==1
    plt = plot_c_spectrum2(Λc)
    display(plt)
    savefig(plt, "./figures/dmd_spectrum_r$(r)_k$(n_ks)_re$(re_num).png")
end


function select_save_dominant_modes(idx_mode)
    println(amp[idx_mode]/maximum(abs.(amp)));
    Φ = 2 .* reshape(Φ_mz[:, idx_mode]/maximum(real.(Φ_mz[:, idx_mode])), (nx, nz)); 
    # Φ = reshape(amp[idx_mode] .* Φ_mz[:, idx_mode], (nx, nz));
    Φ_full = permutedims(hcat(Φ, reverse(Φ, dims=2)), (2, 1));
    npzwrite("./modes/phi_mz_mode_i$(idx_mode)_r$(r)_k$(n_ks).npy", Φ_full)
end

select_save_dominant_modes(prim_inst_indx)
select_save_dominant_modes(secd_inst_indx)
select_save_dominant_modes(thrd_inst_indx)

select_save_dominant_modes(prim_inst_indx+1)
select_save_dominant_modes(secd_inst_indx+1)
select_save_dominant_modes(thrd_inst_indx+1)


select_save_dominant_modes(r+1)




function count_number_periodic_modes(lam)
    num_close_to_one = count(x->abs.(x)>0.96, lam)
    if n_ks==1
        println("number of dmd modes that are periodic = ", num_close_to_one) 
    else
        println("number of mzmd modes that are periodic = ", num_close_to_one) 
    end
end
count_number_periodic_modes(Λc)

function find_dominant_memory_term(lam, amp)
    #first sort the eigenvalues to count number of periodic modes
    ind_e = sortperm(abs.(Λc), rev=true);
    lam_sort = Λc[ind_e];
    amp_sort = amp[ind_e];
    num_modes_periodic = length(lam_sort[abs.(lam_sort) .> 0.96]);
    println("number of mzmd modes that are periodic = ", num_modes_periodic)
    #Look at dominant transient (with biggest e.val)
    println(abs.(lam_sort[r+1]))
    println(amp_sort[r+1])
    return lam_sort, amp_sort, Φ_mz[:, ind_e][:, r+1]
end
if n_ks > 1
    lam_sort_ev, amp_sort_ev, dom_memory_mode = find_dominant_memory_term(Λc, amp);
    function select_save_dominant_memory_mode(dom_memory_mode)
        Φ = 2 .* reshape(dom_memory_mode/maximum(real.(dom_memory_mode)), (nx, nz)); 
        Φ_full = permutedims(hcat(Φ, reverse(Φ, dims=2)), (2, 1));
        npzwrite("./modes/phi_mz_mode_mem_mode_r$(r)_k$(n_ks).npy", Φ_full)
    end
    select_save_dominant_memory_mode(dom_memory_mode)
else
    lam_sort_ev, amp_sort_ev = Λc, amp;
end
find_dominant_memory_term(Λc, amp)


function plot_c_spectrum_sort_ev(Λc)
    gr(size=(570,570), xtickfontsize=12, ytickfontsize=12, xguidefontsize=20, yguidefontsize=14, legendfontsize=12,
        dpi=300, grid=(:y, :gray, :solid, 1, 0.4), palette=cgrad(:plasma, 3, categorical = true));
    function circle_shape(h,k,r)
        θ=LinRange(0,2*pi,500);
        return h .+ r*sin.(θ), k .+ r*cos.(θ)
    end
    plt = scatter(Λc[1:r], linewidth=2.0, ms = 4, color = "blue", legend=false, markerstrokewidth=0, grid = true, framestyle = :box)
    scatter!(Λc[(1+r):2r], linewidth=2.0, ms = 4, color = "cyan1", legend=false, markerstrokewidth=0, grid = true, framestyle = :box)
    scatter!(Λc[(1+2r):end], linewidth=2.0, ms = 2, color = "cyan4", legend=false, grid = true, markerstrokewidth=0, framestyle = :box)
    scatter!([Λc[prim_inst_indx]], ms = 6.0, color="deepskyblue3", markershape=:hexagon, label=L"f: \textrm{Fundamental}")
    scatter!([Λc[secd_inst_indx]], ms = 7.0, color="green", markershape=:circle, label=L"1^{st} \textrm{Harmonic}")
    scatter!([Λc[thrd_inst_indx]], ms = 6.0, color="gold4", markershape=:square, label=L"2^{nd} \textrm{Harmonic}")
    scatter!([Λc[r+1]], ms = 9.0, color="orange", markershape=:star, markerstrokewidth=0, label=L"\textrm{Dominant ~ memory ~ mode}")
    scatter!(Λc[(2+r):2r], linewidth=2.0, ms = 4, color = "brown", legend=false, markerstrokewidth=0, grid = true, framestyle = :box)
    scatter!(Λc[(1+2r):end], linewidth=2.0, ms = 2, color = "cyan4", legend=false, grid = true, markerstrokewidth=0, framestyle = :box)
    plot!(circle_shape(0,0,1.0), linewidth=1.5, color="black", linestyle=:dash)
    if n_ks>1
        title!(L"\textrm{MZMD ~ Eigenvalues }", titlefont=16)
    else
        title!(L"\textrm{DMD ~ Eigenvalues }", titlefont=16)
    end
    xlabel!(L"\textrm{Re}(\lambda)", xtickfontsize=10, xguidefontsize=14)
    ylabel!(L"\textrm{Im}(\lambda)", ytickfontsize=10, yguidefontsize=14)
    return plt
end

if n_ks>1
    plt = plot_c_spectrum_sort_ev(lam_sort_ev)
    savefig(plt, "./figures/mzmd_spectrum_sort_ev_r$(r)_k$(n_ks)_re$(re_num).png")
    display(plt)
end



function amp_log_sort_ev(lam, a, dt, method, ylim_)
    gr(size=(570,450), xtickfontsize=12, ytickfontsize=12, xguidefontsize=20, yguidefontsize=14, legendfontsize=12,
    dpi=300, grid=(:y, :gray, :solid, 1, 0.4), palette=cgrad(:plasma, 3, categorical = true));
    r_2 = floor(Int, r/2);
    freqs = (imag.(log.(lam))/(2*pi*dt))./1000
    max_a = maximum(abs.(a))
    println("prim instability freq = ", abs.(real(freqs))[prim_inst_indx])
    freqs1 = freqs[1:r]; a1 = a[1:r];
    if n_ks>1
        plt = plot([abs.(real(freqs))[prim_inst_indx], abs.(real(freqs))[prim_inst_indx]], [ylim_, (abs.(a)/max_a)[prim_inst_indx]], 
                    lw=2, color="blue", xlims=(0, 2600), yaxis=:log, label=L"\textrm{MZMD}(1 \leq i \leq r)")
    else
        plt = plot([abs.(real(freqs))[prim_inst_indx], abs.(real(freqs))[prim_inst_indx]], [ylim_, (abs.(a)/max_a)[prim_inst_indx]], 
                    lw=2, color="blue", xlims=(0, 2600), yaxis=:log, label=L"\textrm{DMD}")
    end
    if n_ks>1
        a1 = a[1:r]; a2 = a[(r+1):2r]; a3_ = a[(2r+1):end]; 
        lam_1 = lam[1:r]; lam_2 = lam[(r+1):2r]; lam_3 = lam[(2r+1):3r];
        freqs1 = freqs[1:r]; freqs2 = freqs[(r+1):2r]; freqs3_ = freqs[(2r+1):end];
        for i in 1:length(real(freqs1))
            plot!([abs.(real(freqs1))[i], abs.(real(freqs1))[i]], [ylim_, (abs.(a1)/max_a)[i]], lw=2, color="blue", yscale=:log10, ylims=(1e-4, 1.5), minorgrid=true, label="")
        end
        for i in 2:length(real(freqs2))
            plot!([abs.(real(freqs2))[i], abs.(real(freqs2))[i]], [ylim_, (abs.(a2)/max_a)[i]], lw=2, 
                        color="brown", yscale=:log10, ylims=(1e-4, 1.25), minorgrid=true, label="")
        end
        for i in 2:length(real(freqs3_))
            plot!([abs.(real(freqs3_))[i], abs.(real(freqs3_))[i]], [ylim_, (abs.(a3_)/max_a)[i]], lw=1, 
                        color="cyan4", yscale=:log10, ylims=(1e-4, 1.5), minorgrid=true, label="")
        end
        plot!([abs.(real(freqs2))[1], abs.(real(freqs2))[1]], [ylim_, (abs.(a2)/max_a)[1]], lw=2, 
                        color="brown", yscale=:log10, ylims=(1e-4, 1.25), minorgrid=true, label=L"\textrm{MZMD}(r+1 \leq i \leq 2r)")
        plot!([abs.(real(freqs3_))[1], abs.(real(freqs3_))[1]], [ylim_, (abs.(a3_)/max_a)[1]], lw=1, 
                        color="cyan4", yscale=:log10, ylims=(1e-4, 1.5), minorgrid=true, legendfontsize=11, label=L"\textrm{MZMD}(2r+1 \leq i \leq kr)")
        title!(L"\textrm{Amplitude ~ vs. ~ frequency ~ MZMD ~ }", titlefont=16)
    else
        title!(L"\textrm{Amplitude ~ vs. ~ frequency ~ DMD ~ }", titlefont=16)
    end
    scatter!([abs.(real(freqs))[prim_inst_indx]], [(abs.(a)/max_a)[prim_inst_indx]], ms = 6.0, color="deepskyblue3", markershape=:hexagon, label=L"f: \textrm{Fundamental}")
    scatter!([abs.(real(freqs))[secd_inst_indx]], [(abs.(a)/max_a)[secd_inst_indx]], ms = 7.0, color="green", markershape=:circle, label=L"1^{st} \textrm{Harmonic}")
    scatter!([abs.(real(freqs))[thrd_inst_indx]], [(abs.(a)/max_a)[thrd_inst_indx]], ms = 6.0, color="gold4", markershape=:square, label=L"2^{nd} \textrm{Harmonic}")

    # Add vertical lines from each point to the x-axis
    if n_ks==1
        for i in 1:length(real(freqs1))
            plot!([abs.(real(freqs1))[i], abs.(real(freqs1))[i]], [ylim_, (abs.(a1)/max_a)[i]], lw=2, color="blue", yscale=:log10, ylims=(1e-4, 1.5), minorgrid=true, label="")
        end
    end
    if n_ks > 1
        scatter!([abs.(real(freqs))[r+1]], [(abs.(a)/max_a)[r+1]], ms = 9.0, color="orange", markershape=:star, markerstrokewidth=0, label=L"\textrm{Dominant ~ memory ~ mode}")
    end
    xlabel!(L"\textrm{Frequency ~ (kHz)}", xtickfontsize=10, xguidefontsize=14)
    ylabel!(L"\textrm{Normalized ~ Amplitude}", ytickfontsize=10, yguidefontsize=14)
    display(plt)
    savefig(plt, "./figures/$(method)_log_spectrum_sort_ev_vs_ampr$(r)_k$(n_ks)_tend$(t_end)_re$(re_num).png")
    return plt
end

amp_log_sort_ev(lam_sort_ev, amp_sort_ev, dt, "mzmd", 5e-7)

if n_ks==1
    npzwrite("./evs_descending_dmd_r$(r).npy", lam_sort_ev)
else
    npzwrite("./evs_descending_mzmd_r$(r)_k$(n_ks).npy", lam_sort_ev)
end
