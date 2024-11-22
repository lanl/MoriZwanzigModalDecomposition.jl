using Plots, LaTeXStrings, LinearAlgebra, NPZ
using Statistics
using Colors, ColorSchemes


""" 
Comparing MZMD errors
    - Measuring convergence rates of modes
"""

include("./src/main_mz_algorithm.jl")
include("./src/compute_obervability_amplitudes_optimized.jl")
include("./plot_functions.jl")
include("./src/mzmd_modes.jl")

# include("./hodmd_rtilde_comp.jl")
re_num = 600
method="companion"
a_method = "x0";
sample = 1;
println("running e.method $(method) and amp.method $(a_method) eigendecomp computing errors over r, k, d")


#--------- Load data
t_skip = 2;
dt = t_skip*0.2;
if re_num==600
    vort_path = "/Users/l352947/mori_zwanzig/modal_analysis_mzmd/mori_zwanzig/mzmd_code_release/data/vort_all_re600_t5100.npy"
    # vort_path = "/Users/l352947/mori_zwanzig/mori_zwanzig/mzmd_code_release/data/vort_all_re600_t5100.npy"
    X = npzread(vort_path)[:, 300:5000];
else
    vort_path = "/Users/l352947/mori_zwanzig/cylinder_mzmd_analysis/data/vort_all_re200.npy";
    X = npzread(vort_path) #T=1500
end
X_mean = mean(X, dims=2);
X = X .- X_mean;
# X = X .+ 0.2*maximum(X) * randn(size(X))


m = size(X, 1);
t_end = 4000 #total n steps is 751
t_pred = 100
T_train = ((sample):(t_end+sample-1))[1:t_skip:end];
T_test = ((t_end+sample):(t_end + t_pred + sample))[1:t_skip:end];
time_train = dt * T_train
time_test = dt * t_skip * T_test;
t_all = 0:dt:(dt*(t_end + t_pred +1));
X_train = X[:,T_train];
X_test = X[:,T_test];
X_train = X_train .+ 0.15*maximum(X_train) .* randn(size(X_train));

n_ks = 8;
#number of snapshots used in fitting
t_win = size(T_train, 1) - n_ks - 1;
r = 20;
# num_modes = r*n_ks; #all
num_modes = 6;



#-------- Compute MZ on POD basis observables
X1 = X_train[:, 1:t_win]; #Used to obtain Ur on 1:twin
X0 = X_train[:, 1];

# set this on first run
U, Sigma, V = svd(X_train);
sigmas_ = diagm(Sigma);
X_proj = sigmas_[1:r,1:r] * V[:,1:r]';
Ur = U[:, 1:r];

function select_dominant_modes(a, Ur, Vc, r, n_ks, lam, num_modes)
    Φ_mz = Ur*Vc[1:r,:];
    nmodes = size(Φ_mz, 2);
    amp = zeros(nmodes);
    for i in 1:nmodes
        amp[i] = norm(Φ_mz[:,i]*a[i]);
    end
    #sort according to largest amplitude:
    nmodes2 = minimum([size(Φ_mz, 2), num_modes]);
    ind = sortperm(abs.(amp), rev=true)
    a_dom = a[ind][1:nmodes2];
    Vc_dom = Vc[1:r, ind][:, 1:nmodes2];
    lam_dom = lam[ind][1:nmodes2];
    return a_dom, Vc_dom, lam_dom
end


function convergence_mzmd_modes(t_range, n_ks)
    n_t = length(t_range)
    modes_t = zeros(n_t, r, num_modes);
    modesdmd_t = zeros(n_t, r, r);
    function obtain_mz_operators(X_proj, t_win, n_ks)
        #compute two time covariance matrices
        Cov = obtain_C(X_proj, t_win, n_ks);
        #compute MZ operators with Mori projection
        M, Ω = obtain_ker(Cov, n_ks);
        return Ω
    end
    t_idx = 1;
    for t in t_range
        T_train = 1:t
        t_win = size(T_train, 1) - n_ks - 1;
        X_proj_ = X_proj[:, T_train];
        println("t = ", t)
        Ω = obtain_mz_operators(X_proj_, t_win, n_ks);
        C = form_companion(Ω, r, n_ks);
        Λc, Vc = eigen(C);
        Z0 = Ur'*X_test[:, 1:(n_ks+1)];
        z0 = zeros(n_ks*r)
        for k in 1:n_ks
            # z0[((k-1)*r+1):k*r] = Z0[:,k]
            z0[((k-1)*r+1):k*r] = Z0[:,(n_ks - k + 1)]
        end
        a = pinv(Vc)*z0;
        Λc, a, Vc[1:r, :], Ur
        a_dom, Vc_dom, lam_dom = select_dominant_modes(a, Ur, Vc[1:r,:], r, n_ks, Λc, num_modes);
        nmodes = minimum([size(Vc_dom, 2), num_modes])
        for m in 1:nmodes
            modes_t[t_idx, :, m] = abs.(a_dom[m].*Vc_dom[:, m]);
        end
        lam_dmd, phi_dmd = eigen(Ω[1,:,:]);
        x0 = Ur'*X_test[:, 1];
        admd = pinv(phi_dmd)*x0;
        admd_dom, Vdmd_dom, lamdmd_dom = select_dominant_modes(admd, Ur, phi_dmd, r, n_ks, Λc, r);
        for m in 1:r
            modesdmd_t[t_idx, :, m] = abs.(admd_dom[m].*Vdmd_dom[:, m]);
        end   
        t_idx+=1;
    end
    return modes_t, modesdmd_t
end


function hodmd_dominant_modes(d, U, Sigma, T, J, K, T_train, r1, r2; subtractmean::Bool = false)
    # %% STEP 1: SVD of the original data
    sigmas=diagm(Sigma); #create diagonal matrix from vector
    n = size(Sigma, 1); #number of singular values
    NormS=norm(sigmas,2);
    # %% Spatial complexity: kk
    kk = r1
    U=U[:,1:kk];
    # %% reduced snapshots matrix
    hatT=sigmas[1:kk,1:kk]*T[T_train,1:kk]';
    N=size(hatT, 1);
    # %% Create the modified snapshot matrix
    tildeT=zeros(d*N, K-d+1);
    for ppp=1:d
        tildeT[(ppp-1)*N+1:ppp*N,:]=hatT[:,ppp:ppp+K-d];
    end
    # %% Dimension reduction 2
    U1,Sigma1,T1 = svd(tildeT); 
    sigmas1=diagm(Sigma1); 
    n = size(sigmas1, 1);
    NormS1=norm(sigmas1,2);
    # Second Spatial dimension reduction
    kk1 = r2 #d*r1
    if r2 > size(U1, 2)
        kk1 = size(U1, 2); #this will break companion structure, but is a limit of this method!
    end
    U1=U1[:,1:kk1];
    hatT1=sigmas1[1:kk1,1:kk1]*T1[:,1:kk1]';
    # %% Reduced modified snapshot matrix (SVD Reduction 3)
    K1 = size(hatT1, 2);
    tildeU1,tildeSigma,tildeU2 =svd(hatT1[:,1:K1-1]);
    tildesigmas = diagm(tildeSigma)
    tildeR=hatT1[:,2:K1]*tildeU2*inv(tildesigmas)*tildeU1';
    lam, e_vects = eigen(tildeR);
    Q=U1*e_vects;
    a, Q = compute_amplitudes_given_ic_z0_hodmd(Ur, X_test[:, 1:n_ks], Q, n_ks, r)
    a_dom, Vc_dom, lam_dom = select_dominant_modes(a, Ur, Q, r, 1, lam, num_modes);
    return a_dom, Vc_dom, lam_dom
end

function convergence_hodmd_dom_modes(t_range, d, r)
    n_t = length(t_range)
    modes = zeros(n_t, r, num_modes);
    t_idx = 1;
    for t in t_range
        T_train = 1:t
        X_train = X[:,T_train];
        J,K = size(X_train);
        println("t = ", t)
        a_dom, phi, _ = hodmd_dominant_modes(d, U, Sigma, V, J, K, T_train, r, d*r);
        nmodes = minimum([size(phi, 2), num_modes])
        for m in 1:nmodes
            modes[t_idx, :, m] = abs.(a_dom[m].*phi[:, m]);
        end
        t_idx+=1;
    end
    return modes
end

function compute_convergence_mzmd_modes(modes_t)
    nts = size(modes_t, 1);
    conv = zeros(nts-1);
    for i in 1:(nts-1)
        conv[i] = norm(modes_t[i+1,:,:] .- modes_t[i,:,:])/norm(modes_t[2,:,:]);
    end
    return conv
end

function compute_convergence_hodmd_modes(modes)
    nts = size(modes, 1);
    conv = zeros(nts-1);
    for i in 1:(nts-1)
        conv[i] = norm(modes[i+1,:,:] - modes[i,:,:])/norm(modes[1,:,:]);
    end
    return conv
end

t_range = n_ks*r:10:1800
# t_range = r:10:1500
t_range_mz = r:10:1800

modes_t, modesdmd_t = convergence_mzmd_modes(t_range_mz, n_ks);
modes = convergence_hodmd_dom_modes(t_range, n_ks, r)

conv_mz = compute_convergence_mzmd_modes(modes_t)
conv_dmd = compute_convergence_mzmd_modes(modesdmd_t)
conv_hodmd = compute_convergence_hodmd_modes(modes)


function make_directory(path::String)
    if !isdir(path)  # Check if the directory already exists
        mkdir(path)
        println("Directory created at: $path")
    else
        println("Directory already exists at: $path")
    end
end

# dir_path = "convergence_modes_noise_dom_dt$(t_skip)_te$(t_end)_re$(re_num)_nm$(num_modes)"
# make_directory(dir_path)

# npzwrite("$(dir_path)/conv_dom_modes_mz_r$(r)_k$(n_ks).npy", conv_mz)
# npzwrite("$(dir_path)/conv_dom_modes_hodmd_r$(r)_k$(n_ks).npy", conv_hodmd)
# npzwrite("$(dir_path)/conv_dom_modes_dmd_r$(r)_k$(n_ks).npy", conv_dmd)


function plot_convergence(t_range, t_rangemz, mz_oms, hodmd_R, dmd_conv)
	gr(size=(570,450), xtickfontsize=12, ytickfontsize=12, xguidefontsize=20, yguidefontsize=14, legendfontsize=12,
		dpi=300, grid=(:y, :gray, :solid, 1, 0.4), palette=cgrad(:plasma, 3, categorical = true));
    plt = plot(t_range, hodmd_R, fillalpha=.4, ms = 7.0, linestyle=:dash, linewidth=1.5, color="black", 
                yaxis=:log, legendfontsize=12, left_margin=2mm, bottom_margin=2mm, label=L"\textbf{HODMD}")
    plot!(t_rangemz, mz_oms, fillalpha=.4, ms = 7.0, linewidth=1.5, color="blue", 
                yaxis=:log, legendfontsize=12, left_margin=2mm, bottom_margin=2mm, label=L"\textbf{MZMD}")
    # plot!(t_rangemz, dmd_conv, fillalpha=.4, ms = 7.0, linestyle=:dashdot, linewidth=1.5, color="green", 
    #             yaxis=:log, legendfontsize=15, left_margin=2mm, bottom_margin=2mm, label=L"\textbf{DMD}")
    if num_modes==r*n_ks
        title!(L"\textrm{Convergence ~ of ~ modes}", titlefont=20)
    else
        title!(L"\textrm{Convergence ~ of ~ dominant ~ modes}", titlefont=20)
    end
    xlabel!(L"\textrm{Number ~ of ~ Snapshots}", xtickfontsize=14, xguidefontsize=16)
    ylabel!(L"||\Phi_n - \Phi_{n-1}||/||\Phi_1||", ytickfontsize=14, yguidefontsize=16)
    savefig(plt, "./figures/convergence_of_modes_noise_r$(r)_k$(n_ks).png")
    return plt
end


plt = plot_convergence(t_range[1:(size(conv_hodmd, 1)-2)], t_range_mz[1:(size(conv_mz, 1)-2)], conv_mz[2:end-1], conv_hodmd[1:end-2], conv_dmd[1:end-2])
display(plt)

