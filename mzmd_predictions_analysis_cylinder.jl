using MoriZwanzigModalDecomposition
using Plots, LaTeXStrings, LinearAlgebra, NPZ
using Statistics, Measures


#--------- Include files
include("./src/plot_functions2.jl")
include("./src/compute_obervability_amplitudes_optimized.jl")
include("./src/main_mz_algorithm.jl")
include("./src/mzmd_modes.jl")


#--------- Load data
re_num = 200;
vort_path = "./data/vort_all_re200_t145.npy"

method="companion"
#select amplitude method:
# a_method = "lsa";
a_method = "x0";
println("running eigen.method $(method) and amp.method $(a_method) eigendecomp")
X = npzread(vort_path);
X_mean = mean(X, dims=2);
X = X .- X_mean;


# gx = load_grid_file(nx)
m = size(X, 1);
#time window for training
dt = 0.2;
sample = 1;
t_skip = 1;
t_end = 130 #total n steps is 751
t_pred = 12
T_train = ((sample):(t_end+sample-1))[1:t_skip:end];
T_test = ((t_end+sample):(t_end + t_pred + sample))[1:t_skip:end];
time_train = dt * T_train
time_test = dt * t_skip * T_test;
t_all = 0:dt:(dt*(t_end + t_pred +1));
X_train = X[:,T_train];
X_test = X[:,T_test];

#-------- Set params
#number of memory terms:
# n_ks = 12
n_ks = 5;
#number of snapshots used in fitting
t_win = size(T_train, 1) - n_ks - 1;
r = 12;


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
plotting_mem_norm(Ω)

function obtain_omega_norms(Ω)
    Ω_norms = zeros(n_ks)
    r_ = size(Ω, 2);
    I_ = I(r_);
    for i in 1:n_ks
        if i==1
            Ω_norms[i] = norm(Ω[i,:,:] - I_)
        else
            Ω_norms[i] = norm(Ω[i,:,:])
        end
    end
    return Ω_norms
end
Ω_norms = obtain_omega_norms(Ω);
plt = plot(Ω_norms, yaxis=:log)
display(plt)

C = form_companion(Ω, r, n_ks);

#----------------------------------------------------------------
                    #Analysis
#----------------------------------------------------------------

function simulate_modes(Ur, evects, a, r, λ, dt, t)
    a_length = length(a);
    t_length = length(t);
    time_dynamics = complex(zeros(a_length, t_length));
    ts = (0:t_length-1) * dt;
    omega = log.(λ)/dt;
    for i in 1 : t_length
        time_dynamics[:,i] = (a.*exp.(ts[i] .* omega));
    end
    X_pred = evects[1:r,:] * time_dynamics;
    Xmz_pred = Ur * X_pred;
    return real.(Xmz_pred)
end

function predict_mzmd(amp_method, t_all, X_proj, C, Ur, r, d, n_ks, T)
    if amp_method=="x0"
        # Λc, a_mz, Vc = mzmd_modes_reduced_amps_pred(C, X, Ur, r, n_ks);
        Λc, Vc, a_mz, Ur = compute_modes_amplitudes_time_delay_observables(X, r, n_ks, d, T, method)
        solution_modes = simulate_modes(Ur, Vc, a_mz, r, Λc, dt, t_all);
        a = a_mz;
        amps = zeros(r*n_ks);
        Φ_mz = Ur*Vc[1:r,:];
        for i in 1:r*n_ks
            amps[i] = norm(Φ_mz[:,i]*a[i]);
        end
        Q = Vc;
    end
    if amp_method=="lsa"
        Λc, Vc = eigen(C);
        e_vects = Vc[1:r,:];
        a, amps, Q = compute_amplitudes_observability_matrix(e_vects, Λc, X_proj, Ur, m);
        solution_modes = simulate_modes(Ur, Q, a, r, Λc, dt, t_all);
    end
    return Λc, Vc, a, amps, Q, solution_modes
end

function predict_dmd(amp_method, t_all, X0r, M, Ur, r)
    if amp_method=="x0"
        lambda, e_vects, a, amps = compute_markovian_modes_reduced(M, X0r[:, 1], Ur, r)
        solution_modes = simulate_modes(Ur, e_vects, a, r, lambda, dt, t_all);
    end
    if amp_method=="lsa"
        lambda, e_vects = eigen(M);
        a, amps, Q = compute_amplitudes_observability_matrix(e_vects, Λ, X_proj, Ur, m);
        solution_modes = simulate_modes(Ur, Q, a, r, Λ, dt, t_all);
    end
    return lambda, e_vects, a, amps, solution_modes
end

# construct time delay obserables in projected (x_proj) space. Use this to find operators and e.vects
function compute_modes_amplitudes_time_delay_observables(X, r, n_ks, n_td, T, method)
    #inputs: r number of observables (pod modes)
            #n_ks number of mz memory terms
            #n_td number of time delay embeddings
    # S, Ur, X_proj = svd_method_of_snapshots(X[:,T_train], r, subtractmean=true)
    # Ur = U[:, 1:r];
    X_proj = Ur' * X[:, T_train];
    # %% Create time delay observable matrix
    t_g = T + n_ks + 1;
    t_gtilde = length(1:(1+t_g - n_td));
    G_tilde=zeros(n_td*r, t_gtilde);
    for i in 1:n_td
        G_tilde[(i-1)*r+1:i*r,:] = X_proj[:,i:(i+t_g-n_td)];
    end
    #initial conditions for z0 in companion 
    g0_test = Ur'*X_test[:, 1:(n_ks+n_td+1)];
    G0_tilde_test = zeros(n_td*r, n_ks);
    for i in 1:n_td
        G0_tilde_test[(i-1)*r+1:i*r,:] = g0_test[:,i:(i+n_ks-1)];
    end
    function obtain_mz_operators_(G, t_win, n_ks)
        #compute two time covariance matrices
        Cov = obtain_C(G, t_win, n_ks);
        #compute MZ operators with Mori projection
        _, Ω = obtain_ker(Cov, n_ks);
        println("----------------------------");
        return Ω
    end
    twin_tilde = t_gtilde - n_ks - 1;
    if n_ks==1 #hodmd (i.e. no mz memory)
        # #TODO SVD 2
        U1, Sigma1, T1 = svd(G_tilde); 
        sigmas1=diagm(Sigma1); 
        # Second Spatial dimension reduction
        r2 = n_td*r
        if r2 >= size(U1, 2)
            r2 = size(U1, 2); #this will break companion structure, but is a limit of this method!
        end
        U1=U1[:,1:r2];
        #svd low rank approximation of G_tilde:
        hatG=sigmas1[1:r2,1:r2]*T1[:,1:r2]';
        # #TODO SVD 3
        K1 = size(hatG, 2);
        tildeU1, tildeSigma, tildeV1 = svd(hatG[:,1:K1-1]);
        tildesigmas = diagm(tildeSigma)
        Ω_tilde=hatG[:,2:K1]*tildeV1*inv(tildesigmas)*tildeU1';
        lam, e_vects = eigen(Ω_tilde);
        Q=U1*e_vects;
        if a_method=="lsa"
            a, Q = compute_amplitudes_observability_matrix_aq(Q, lam, G_tilde);
        else
            a, Q = compute_amplitudes_given_ic_z0_hodmd(Ur, X_test[:, 1:n_td], Q, n_td, r)
        end
    else
        Ω_tilde = obtain_mz_operators_(G_tilde, twin_tilde, n_ks);
        if method=="nep"
            lam, e_vects = compute_nonlinear_eigen(Ω_tilde); 
        else
            r2 = size(Ω_tilde, 2);
            C = form_companion(Ω_tilde, r2, n_ks);
            lam, Vc = eigen(C);
        end
        if a_method=="lsa"
            e_vects = Vc[1:r2,:];
            a, Q = compute_amplitudes_observability_matrix_aq(e_vects, lam, G_tilde);
        else
            a, Q = compute_amplitudes_given_ic_z0(G0_tilde_test, Vc, n_ks, n_td, r)
        end
    end
    return lam, Q, a, Ur
end

d = 1;
Λc, Vc, a1, amps, Q, solution_modes = predict_mzmd("x0", T_train, X_proj, C, Ur, r, d, n_ks, t_win);
# amp_norm = zeros(r*n_ks);
# for i in 1:r*n_ks
#     amp_norm[i] = norm(Φ_mz[:,i]*amps[i]);
# end

# lambda, e_vects, a_m, amps_m, X_dmd = predict_dmd("x0", 1:751, X0r, M, Ur, r);

#number of time delays included (d = 1 is no time delays)
d = 1;
Λc, Q, a, Ur = compute_modes_amplitudes_time_delay_observables(X, r, n_ks, d, t_win, method);


if n_ks>1
    plt = plot_c_spectrum(Λc)
    display(plt)
    # savefig(plt, "./figures/mzmd_spectrum_r$(r)_k$(n_ks).png")
end

if n_ks==1
    plt = plot_dmd_spectrum(Λc)
    display(plt)
    # savefig(plt, "./figures/dmd_spectrum_r$(r)_k$(n_ks).png")
end


# Λc, Φ_mz, a, amp = mzmd_modes(C, X0)
Λc, Φ_mz, a, amp = mzmd_modes_reduced_amps(X_train, C, Ur, r, n_ks) #fast and accurate algorithm


function plot_amplitude_vs_frequency_select_amps_mzmd(lam, a, dt, method)
    gr(size=(570,450), xtickfontsize=12, ytickfontsize=12, xguidefontsize=20, yguidefontsize=14, legendfontsize=12,
        dpi=300, grid=(:y, :gray, :solid, 1, 0.4), palette=cgrad(:plasma, 3, categorical = true));

    r_2 = floor(Int, r/2);
    freqs = imag.(log.(lam))/(2*pi*dt);
    # freqs = imag.(log.(lam))/(dt);
    plt = scatter(abs.(real(freqs)), abs.(a)/maximum(abs.(a)), ms=4.0, color="black", legend=false, ylims=(0, 1+0.05))
    if n_ks>1
        title!(L"\textrm{Amplitude ~ vs. ~ frequency ~ MZMD ~ }", titlefont=22)
    else
        title!(L"\textrm{Amplitude ~ vs. ~ frequency ~ DMD ~ }", titlefont=22)
    end
    xlabel!(L"\textrm{Im}(\omega) \textrm{: Frequency ~ in ~ Hz}", xtickfontsize=12, xguidefontsize=16)
    ylabel!(L"\textrm{Amplitude: ~ } ||a_n * Φ_{n}||", ytickfontsize=12, yguidefontsize=16)
    display(plt)
    # savefig(plt, "./figures/$(method)_spectrum_vs_ampr$(r)_d$(d)_tend$(t_end).png")
    return plt
end

plt = plot_amplitude_vs_frequency_select_amps_mzmd(Λc, amp, dt, "mzmd")
display(plt)


#simulating predictions
out_path = "./prediction_compare_all_full_r$(r)_k$(n_ks).mov"
animate_field(solution_modes .+ X_mean, method, out_path)




#other methods to try:
# #compute modes and spectrum of MZMD
# Λc_red, Φ_mz, a_mz_red, amp_mz_red, Vc = mzmd_modes_reduced_amps(C, Ur, r, n_ks);
# amp_norm = zeros(r*n_ks);
# for i in 1:r*n_ks
#     amp_norm[i] = norm(Φ_mz[:,i]*amps[i]);
# end
# plt = plot_amplitude_vs_frequency_select_amps(Λc_red, norm(Q)*sqrt(m)*amp_norm/10, dt, "mzmd")
# display(plt)

