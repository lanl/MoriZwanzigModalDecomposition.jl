using Plots, LaTeXStrings, LinearAlgebra, NPZ
using Statistics


#--------- Include files
include("plot_functions.jl")
include("./src/compute_obervability_amplitudes_optimized.jl")
include("./src/main_mz_algorithm.jl")
include("./src/mzmd_modes.jl")


#--------- Load data
# re_num = 9;
re_num = 200
method="companion"
a_method = "x0";
sample = 1;
println("running e.method $(method) and amp.method $(a_method) eigendecomp computing errors over r, k, d")

#--------- Load data
t_skip = 1;
dt = t_skip*0.2;
vort_path = "../data/vort_all_re200_t145.npy"
X = npzread(vort_path) #T=1500

X_mean = mean(X, dims=2);
X = X .- X_mean;

m = size(X, 1);
t_end = 140 #total n steps is 751
t_pred = 2
T_train = ((sample):(t_end+sample-1))[1:t_skip:end];
T_test = ((t_end+sample):(t_end + t_pred + sample))[1:t_skip:end];
time_train = dt * T_train
time_test = dt * t_skip * T_test;
t_all = 0:dt:(dt*(t_end + t_pred +1));
X_train = X[:,T_train];
X_train = X_train .+ 0.25*maximum(X_train) .* randn(size(X_train))
X_test = X[:,T_test];


t_ = 20
t_r = round((dt*(t_-2)), digits=2)

#number of memory terms:
n_ks = 6
#number of snapshots used in fitting
t_win = size(T_train, 1) - n_ks - 1;
r = 20;


#-------- Compute MZ on POD basis observables
X1 = X_train[:, 1:t_win]; #Used to obtain Ur on 1:twin
X0 = X_train[:, 1];

#compute svd based observables (X_proj: states projected onto pod modes)
#method of snapshot more efficient for tall skinny X
S, Ur, X_proj = svd_method_of_snapshots(X_train, r, subtractmean=true)

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
plt = plotting_mem_norm(Ω)
display(plt)

C = form_companion(Ω, r, n_ks);


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
    scatter!([Λc[1]], ms = 6.0, color="deepskyblue3", markershape=:hexagon, label=L"f: \textrm{Fundamental}")
    scatter!([Λc[3]], ms = 7.0, color="green", markershape=:circle, label=L"1^{st} \textrm{Harmonic}")
    scatter!([Λc[5]], ms = 6.0, color="gold4", markershape=:square, label=L"2^{nd} \textrm{Harmonic}")
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


function plot_c_spectrum2(Λc)
    gr(size=(570,570), xtickfontsize=12, ytickfontsize=12, xguidefontsize=20, yguidefontsize=14, legendfontsize=12,
        dpi=300, grid=(:y, :gray, :solid, 1, 0.4), palette=cgrad(:plasma, 3, categorical = true));
    function circle_shape(h,k,r)
        θ=LinRange(0,2*pi,500);
        return h .+ r*sin.(θ), k .+ r*cos.(θ)
    end
    plt = scatter(Λc, linewidth=2.0, ms = 4.0, color = "black", legend=false, grid = true, framestyle = :box)
    scatter!([Λc[5]], ms = 6.0, color = "blue", markershape=:diamond)
    # scatter!([Λc[6]], ms = 6.0, color = "blue", markershape=:diamond)
    scatter!([Λc[1]], ms = 6.0, color = "red", markershape=:star5)
    scatter!([Λc[2]], ms = 6.0, color = "red", markershape=:star5)
    scatter!([Λc[3]], ms = 6.0, color = "green", markershape=:hexagon)
    scatter!([Λc[4]], ms = 6.0, color = "green", markershape=:hexagon)
    plot!(circle_shape(0,0,1.0), linewidth=1.5, color="black", linestyle=:dash)
    if n_ks>1
        title!(L"\textrm{MZMD ~ Eigenvalues }", titlefont=22)
    else
        title!(L"\textrm{DMD ~ Eigenvalues }", titlefont=22)
    end
    xlabel!(L"\textrm{Re}(\lambda)", xtickfontsize=12, xguidefontsize=16)
    ylabel!(L"\textrm{Im}(\lambda)", ytickfontsize=12, yguidefontsize=16)
    return plt
end


function mzmd_modes_reduced_amps_compute(C, Ur, r, n_ks)
    #compute modes and amplitudes from C
    Λc, Vc = eigen(C);
    #mzmd modes:
    Φ_mz = Ur*Vc[1:r,:];
    #use initial conditions with k memory terms:
    X0_mem = X[:, 1:n_ks];
    #compute amplitudes of modes from observable space
    Z0 = Ur' * X0_mem;
    z0 = zeros(n_ks*r)
    for k in 1:n_ks
        # z0[((k-1)*r+1):k*r] = Z0[:,k]
        z0[((k-1)*r+1):k*r] = Z0[:,(n_ks - k + 1)]
    end
    a = pinv(Vc)*z0;
    amp = zeros(r*n_ks);
    for i in 1:r*n_ks
        amp[i] = norm(Φ_mz[:,i]*a[i]);
    end
    #sort according to largest amplitude:
    ind = sortperm(abs.(amp), rev=true)
    return Λc[ind], Φ_mz[:, ind], a[ind], amp[ind], Vc
end

Λc, Φ_mz, a, amp = mzmd_modes_reduced_amps_compute(C, Ur, r, n_ks) #fast and accurate algorithm

# plt = plot_c_spectrum2(Λc)
# display(plt)

plt = plot_c_spectrum_sort(Λc)
display(plt)

function plot_amplitude_vs_frequency_select_amps_mzmd(lam, a, dt, method)
    gr(size=(570,450), xtickfontsize=12, ytickfontsize=12, xguidefontsize=20, yguidefontsize=14, legendfontsize=12,
        dpi=300, grid=(:y, :gray, :solid, 1, 0.4), palette=cgrad(:plasma, 3, categorical = true));
    r_2 = floor(Int, r/2);
    freqs = imag.(log.(lam))/(2*pi*dt);
    # freqs = imag.(log.(lam))/(dt);
    # plt = scatter(abs.(real(freqs)), abs.(a)/maximum(abs.(a)), ms=3.0, color="black", legend=false, ylims=(0, 1+0.1))
    if n_ks>1
        plt = plot([abs.(real(freqs))[1], abs.(real(freqs))[1]], [0, (abs.(a)/maximum(abs.(a)))[1]], lw=2, color="blue", xlims=(0, 1.3), ylims=(0, 1+0.15), label=L"\textrm{MZMD}")
    else
        plt = plot([abs.(real(freqs))[1], abs.(real(freqs))[1]], [0, (abs.(a)/maximum(abs.(a)))[1]], lw=2, color="blue", xlims=(0, 1.3), ylims=(0, 1+0.15), label=L"\textrm{DMD}")
    end
    scatter!([abs.(real(freqs))[1]], [(abs.(a)/maximum(abs.(a)))[1]], ms = 6.0, color = "red", markershape=:star5, label=false, legend=false)
    scatter!([abs.(real(freqs))[5]], [(abs.(a)/maximum(abs.(a)))[5]], ms = 6.0, color = "blue", markershape=:diamond, label=false, legend=false)
    scatter!([abs.(real(freqs))[3]], [(abs.(a)/maximum(abs.(a)))[3]], ms = 6.0, color = "green", markershape=:hexagon, label=false, legend=false)
    vline!([0.1967], label=L"f_s: \textrm{shedding ~ freq}", lw=2, linestyle=:dash, color=:green, legend=true)
    vline!([0.3934], label=L"1^{st} \textrm{higher ~ harmonic}", lw=1.5, linestyle=:dashdot, color=:green, legend=true)
    vline!([0.590], label=L"2^{nd} \textrm{higher ~ harmonic}", lw=1.5, linestyle=:dashdotdot, color=:green, legend=true)
    # Add vertical lines from each point to the x-axis
    for i in 1:length(real(freqs))
        plot!([abs.(real(freqs))[i], abs.(real(freqs))[i]], [0, (abs.(a)/maximum(abs.(a)))[i]], lw=2, color="blue", label="")
    end
    if n_ks>1
        title!(L"\textrm{Amplitude ~ vs. ~ frequency ~ MZMD ~ }", titlefont=22)
    else
        title!(L"\textrm{Amplitude ~ vs. ~ frequency ~ DMD ~ }", titlefont=22)
    end
    xlabel!(L"\textrm{Frequency ~ (Hz)}", xtickfontsize=12, xguidefontsize=16)
    ylabel!(L"\textrm{Normalized ~ Amplitude}", ytickfontsize=12, yguidefontsize=16)
    display(plt)
    # savefig(plt, "./figures/$(method)_spectrum_vs_ampr$(r)_k$(n_ks)_tend$(t_end)_re$(re_num).png")
    return plt
end


function amp_log(lam, a, dt, method, ylim_)
    gr(size=(570,450), xtickfontsize=12, ytickfontsize=12, xguidefontsize=20, yguidefontsize=14, legendfontsize=12,
    dpi=300, grid=(:y, :gray, :solid, 1, 0.4), palette=cgrad(:plasma, 3, categorical = true));
    r_2 = floor(Int, r/2);
    freqs = (imag.(log.(lam))/(2*pi*dt))
    max_a = maximum(abs.(a))
    println("prim instability freq = ", abs.(real(freqs))[1])
    freqs1 = freqs[1:r]; a1 = a[1:r];
    prim_inst_indx = 1; secd_inst_indx = 3; thrd_inst_indx = 5;
    if n_ks>1spe
        plt = plot([abs.(real(freqs))[prim_inst_indx], abs.(real(freqs))[prim_inst_indx]], [ylim_, (abs.(a)/max_a)[prim_inst_indx]], 
                    lw=2, color="blue", xlims=(0, 1.3), yaxis=:log, label=L"\textrm{MZMD}(1 \leq i \leq r)")
    else
        plt = plot([abs.(real(freqs))[prim_inst_indx], abs.(real(freqs))[prim_inst_indx]], [ylim_, (abs.(a)/max_a)[prim_inst_indx]], 
                    lw=2, color="blue", xlims=(0, 1.3), yaxis=:log, label=L"\textrm{DMD}")
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

    vline!([0.1967], lw=1, label=false, linestyle=:dash, color=:red, legend=true)
    vline!([2*0.1967], lw=1, label=false, linestyle=:dash, color=:red, legend=true)
    vline!([3*0.1967], lw=1, label=L"\textrm{DNS ~ harmonics}", linestyle=:dash, color=:red, legend=true)
    # Add vertical lines from each point to the x-axis
    for i in 1:length(real(freqs1))
        plot!([abs.(real(freqs1))[i], abs.(real(freqs1))[i]], [ylim_, (abs.(a1)/max_a)[i]], lw=2, color="blue", yscale=:log10, ylims=(1e-4, 1.5), minorgrid=true, label="")
    end
    scatter!([abs.(real(freqs))[r+1]], [(abs.(a)/max_a)[r+1]], ms = 9.0, color="orange", markershape=:star, markerstrokewidth=0, label=L"\textrm{Dominant ~ memory ~ mode}")

    xlabel!(L"\textrm{Frequency ~ (kHz)}", xtickfontsize=10, xguidefontsize=14)
    ylabel!(L"\textrm{Normalized ~ Amplitude}", ytickfontsize=10, yguidefontsize=14)
    display(plt)
    # savefig(plt, "./figures/$(method)_log_spectrum_vs_ampr$(r)_k$(n_ks)_tend$(t_end)_re$(re_num).png")
    return plt
end
plt = amp_log(Λc, amp, dt, "mzmd", 5e-7)
display(plt)

# plt = plot_amplitude_vs_frequency_select_amps_mzmd(Λc, amp, dt, "mzmd")

a_norm = abs.(amp)/maximum(abs.(amp));
# println("a_norm dominant = ", a_norm[1])
# println("a_norm first hh = ", a_norm[5])
# println("a_norm second hh = ", a_norm[3])

# function select_save_dominant_modes(idx_mode)
#     Φ = reshape(Φ_mz[:, idx_mode]/norm(Φ_mz[:, idx_mode]), (199, 449)); 
#     println("i = ", idx_mode, " amp = ", a_norm[idx_mode])
#     plt = plot_mzmd_modes(Φ_mz, idx_mode)
#     display(plt)
#     npzwrite("./modes/phi_mz_mode_i$(idx_mode)_r$(r)_k$(n_ks).npy", Φ)
# end

# select_save_dominant_modes(1)
# select_save_dominant_modes(5)
# select_save_dominant_modes(3)


#check the error (residual) of projecting onto pod modes
# println("checking residual of projecting onto pod modes")
# v_ = randn(size(Ur, 1));
# error_check = norm((I - Ur * Ur') * v_)/norm(v_)
# println("||(I - Ur * Ur') * v_||/||v_|| = ", error_check)