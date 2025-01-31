using Plots, LaTeXStrings, LinearAlgebra, NPZ
using Statistics
using Colors, ColorSchemes



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
X = npzread(vort_path)

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


#number of memory terms:
n_ks = 3
#number of snapshots used in fitting
t_win = size(T_train, 1) - n_ks - 1;
r = 20;

#--------- Load data
#check data:
nt_samp = t_end; t_skip = 1;
dt = 2.0e-7*t_skip;

U, Sigma, V = svd(X[:, 1:nt_samp]);
sigmas_ = diagm(Sigma);
X_proj = sigmas_[1:r,1:r] * V[:,1:r]';
println(size(X_proj))
# X = nothing;

function convergence_mz_operators(t_range, n_ks)
    n_t = length(t_range)
    omega_t = zeros(n_t, n_ks, r, r);
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
        omega_t[t_idx, :, :, :] = Ω;
        t_idx+=1;
    end
    return omega_t
end



function compute_congergence_omega(omega_t)
    nts = size(omega_t, 1);
    conv = zeros(nts);
    for i in 2:nts
        conv[i] = norm(omega_t[i,:,:,:] .- omega_t[i-1,:,:,:])/norm(omega_t[1,:,:,:]);
    end
    return conv
end


t_range_mz = r:5:(nt_samp-1)

omega_t = convergence_mz_operators(t_range_mz, n_ks);
conv_mz = compute_congergence_omega(omega_t);


npzwrite("./conv_mz_r$(r)_k$(n_ks).npy", conv_mz)


function plot_convergence(t_rangemz, mz_oms)
	gr(size=(800,600)
    , xtickfontsize=14, ytickfontsize=14, 
        xguidefontsize=20, yguidefontsize=20, legendfontsize=14,
		dpi=200, grid=(:y, :gray, :solid, 1, 0.4))

    plt = plot(t_rangemz, mz_oms, fillalpha=.4, ms = 7.0, linewidth=2.5, color="blue", 
                yaxis=:log, legendfontsize=15, left_margin=10mm, bottom_margin=8mm, 
                yscale=:log10, ylims=(1e-4, 5), minorgrid=true, label=L"\textbf{MZMD}")

    title!(L"\textbf{Convergence ~ of ~ Companion ~ Matrix}", titlefont=20)
    xlabel!(L"\textbf{Number ~ of ~ Snapshots}", xtickfontsize=14, xguidefontsize=16)
    ylabel!(L"||C_n - C_{n-1}||", ytickfontsize=14, yguidefontsize=16)
    # savefig(plt, "./figures/convergence_of_companions_r$(r)_k$(n_ks).png")
    return plt
end

plt = plot_convergence(t_range_mz[2:end], conv_mz[2:end])
display(plt)







































# function convergence_mz_operators(t_range, n_ks)
#     n_t = length(t_range)
#     omega_t = zeros(n_t, n_ks, r, r);
#     function obtain_mz_operators(X_proj, t_win, n_ks)
#         #compute two time covariance matrices
#         Cov = obtain_C(X_proj, t_win, n_ks);
#         #compute MZ operators with Mori projection
#         M, Ω = obtain_ker(Cov, n_ks);
#         return Ω
#     end
#     t_idx = 1;
#     for t in t_range
#         T_train = 1:t
#         t_win = size(T_train, 1) - n_ks - 1;
#         X_train = X[:,T_train];
#         println("t = ", t)
#         S, Ur, X_proj = svd_method_of_snapshots(X_train, r, subtractmean=true)
#         Ω = obtain_mz_operators(X_proj, t_win, n_ks);
#         omega_t[t_idx, :, :, :] = Ω;
#         t_idx+=1;
#     end
#     return omega_t
# end


# using LinearAlgebra

# function DMDd_reduced_Rtilde_in_g(d, U, Sigma, T, J, K, T_train, r1, r2; subtractmean::Bool = false)
#     # %% STEP 1: SVD of the original data
#     sigmas=diagm(Sigma); #create diagonal matrix from vector
#     n = size(Sigma, 1); #number of singular values
#     NormS=norm(sigmas,2);
#     # %% Spatial complexity: kk
#     kk = r1
#     U=U[:,1:kk];
#     # %% reduced snapshots matrix
#     hatT=sigmas[1:kk,1:kk]*T[:,1:kk]';
#     N=size(hatT, 1);
#     # %% Create the modified snapshot matrix
#     tildeT=zeros(d*N, K-d+1);
#     for ppp=1:d
#         tildeT[(ppp-1)*N+1:ppp*N,:]=hatT[:,ppp:ppp+K-d];
#     end
#     # %% Dimension reduction 2
#     U1,Sigma1,T1 = svd(tildeT); 
#     sigmas1=diagm(Sigma1); 
#     n = size(sigmas1, 1);
#     NormS1=norm(sigmas1,2);
#     # Second Spatial dimension reduction
#     kk1 = r2 #d*r1
#     if r2 >= T_train[end]
#         kk1 = size(U1, 2); #this will break companion structure, but is a limit of this method!
#     end
#     U1=U1[:,1:kk1];
#     hatT1=sigmas1[1:kk1,1:kk1]*T1[:,1:kk1]';

#     # %% Reduced modified snapshot matrix (SVD Reduction 3)
#     K1 = size(hatT1, 2);
#     tildeU1,tildeSigma,tildeU2 =svd(hatT1[:,1:K1-1]);
#     tildesigmas = diagm(tildeSigma)
#     tildeR=hatT1[:,2:K1]*tildeU2*inv(tildesigmas)*tildeU1';
#     lam, e_vects = eigen(tildeR);
#     Q=U1*e_vects;
#     R_hat = Q*diagm(lam)*pinv(Q);
#     return R_hat
# end




# function convergence_hodmd_operators(t_range, d, r)
#     n_t = length(t_range)
#     R_t = complex.(zeros(n_t, d*r, d*r));
#     t_idx = 1;
#     for t in t_range
#         T_train = 1:t
#         X_train = X[:,T_train];
#         J,K = size(X_train);
#         println("t = ", t)
#         U, Sigma, V = svd(X_train)
#         Rtil = DMDd_reduced_Rtilde_in_g(d, U, Sigma, V, J, K, T_train, r, d*r);
#         # println(size(Rtil))
#         R_t[t_idx, :, :] = Rtil
#         t_idx+=1;
#     end
#     return R_t
# end

# function compute_congergence_omega(omega_t)
#     nts = size(omega_t, 1);
#     conv = zeros(nts);
#     for i in 2:nts
#         conv[i] = norm(omega_t[i,2:end,:,:] - omega_t[i-1,2:end,:,:])/norm(omega_t[1,2:end,:,:]);
#     end
#     return conv
# end

# function compute_congergence_R(R_t)
#     nts = size(R_t, 1);
#     conv = zeros(nts);
#     for i in 2:nts
#         conv[i] = norm(R_t[i,:,:] - R_t[i-1,:,:])/norm(R_t[1,:,:]);
#     end
#     return conv
# end

# t_range = n_ks*r:10:750
# omega_t = convergence_mz_operators(t_range, n_ks);
# Rt = convergence_hodmd_operators(t_range, n_ks, r)

# conv_mz = compute_congergence_omega(omega_t)
# conv_hodmd = compute_congergence_R(Rt)

# npzwrite("./conv2_mz_r$(r)_k$(n_ks).npy", conv_mz)
# npzwrite("./conv2_hodmd_r$(r)_k$(n_ks).npy", conv_hodmd)



# function plot_convergence(t_range, mz_oms, hodmd_R)
# 	gr(size=(800,600)
#     , xtickfontsize=14, ytickfontsize=14, 
#         xguidefontsize=20, yguidefontsize=20, legendfontsize=14,
# 		dpi=200, grid=(:y, :gray, :solid, 1, 0.4))

#     plt = plot(t_range, hodmd_R, fillalpha=.4, ms = 7.0, linestyle=:dash, linewidth=2.5, color="black", 
#                 yaxis=:log, legendfontsize=15, left_margin=10mm, bottom_margin=8mm, label=L"\textbf{HODMD}")

#     plot!(t_range, mz_oms, fillalpha=.4, ms = 7.0, linewidth=2.5, color="blue", 
#                 yaxis=:log, legendfontsize=15, left_margin=10mm, bottom_margin=8mm, label=L"\textbf{MZMD}")

#     title!(L"\textbf{Convergence ~ of ~ Companion ~ Matrix}", titlefont=20)
#     xlabel!(L"\textbf{Number ~ of ~ Snapshots}", xtickfontsize=14, xguidefontsize=16)
#     ylabel!(L"||C_n - C_{n-1}||", ytickfontsize=14, yguidefontsize=16)
#     savefig(plt, "./figures/convergence2_of_companions_r$(r)_k$(n_ks).png")
#     return plt
# end

# plt = plot_convergence(t_range[2:end], conv_mz[2:end], conv_hodmd[2:end])
# display(plt)

