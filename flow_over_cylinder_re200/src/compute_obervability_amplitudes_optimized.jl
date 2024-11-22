using LinearAlgebra

function compute_amplitudes_observability_matrix_aq(evects, evals, x_proj)
    #inputs: e.vects and e.vals of companion matrix.
    #outputs: amplitudes obtained from optimized dmd
    Q = evects;
    num_modes = size(Q, 2);
    #normalize modes
    for m=1:num_modes
        NormQ=Q[:,m];
        Q[:,m]= Q[:,m]/norm(NormQ,2);
    end
    M = diagm(evals); #matrix of evals given as vector
    r, rk = size(Q);
    nobs, T = size(x_proj);
    b = zeros(nobs*T);
    L = complex.(zeros(r*T, rk))
    aa = I(rk)
    for i in 1:T
        b[1+(i-1)*nobs:i*nobs] = x_proj[:, i]
        L[1+(i-1)*nobs:i*nobs,:] = Q*aa;
        aa=aa*M;
    end
    Ur,Sigmar,Vr = svd(L);
    Sigmar = diagm(Sigmar)
    a=Vr*(Sigmar\(Ur'*b));
    return a, Q
end


function compute_amplitudes_observability_matrix(evects, evals, x_proj, U, num_states)
    #inputs: e.vects and e.vals of companion matrix.
    #outputs: amplitudes obtained from optimized dmd
    Q = evects;
    num_modes = size(Q, 2);
    #normalize modes
    for m=1:num_modes
        NormQ=Q[:,m];
        Q[:,m]= Q[:,m]/norm(NormQ,2);
    end
    M = diagm(evals); #matrix of evals given as vector
    r, rk = size(Q);
    nobs, T = size(x_proj);
    b = zeros(nobs*T);
    L = complex.(zeros(r*T, rk))
    aa = I(rk)
    for i in 1:T
        b[1+(i-1)*nobs:i*nobs] = x_proj[:, i]
        L[1+(i-1)*nobs:i*nobs,:] = Q*aa;
        aa=aa*M;
    end
    Ur,Sigmar,Vr = svd(L);
    Sigmar = diagm(Sigmar)
    a=Vr*(Sigmar\(Ur'*b));
    # a = L\b;
    #compute amplitudes for plotting:
    # Φ = U*Q;
    # amplitude = zeros(rk);
    # for i in 1:rk
    #     amplitude[i] = norm(Φ[:,i]*a[i]);
    # end
    u=complex.(zeros(r, rk));
    for m=1:num_modes
        u[:,m]=a[m]*Q[:,m];
    end
    amplitude=zeros(num_modes);
    for m=1:num_modes
        aca=U*u[:,m];
        amplitude[m]=norm(aca,2)/sqrt(num_states);
    end
    return a, amplitude, Q
end

function compute_amplitudes_given_ic_z0_hodmd(Ur, x0_mem, evects, n_td, r)
    Z0 = Ur' * x0_mem;
    z0 = zeros(n_td*r)
    # println(size(Z0))
    # println(size(z0))
    for k in 1:n_td
        # z0[((k-1)*r+1):k*r] = Z0[:,(n_td-k+1)]
        z0[((k-1)*r+1):k*r] = Z0[:, k]  #hodmd order
    end
    # println(size(pinv(evects)))
    a = pinv(evects)*z0;
    # a = inv(evects)*z0;
    # Q = evects[1:r, :];
    # #TODO why does this cause larger errors?
    Q = evects[(n_td-1)*r+1:n_td*r, :];
    return a, Q
end


# function compute_amplitudes_given_ic_z0_hodmd(g0, evects, n_td, r)
#     #inputs: e.vects and e.vals of companion matrix.
#     #outputs: amplitudes obtained from optimized dmd
#     a = pinv(evects)*g0;
#     # a = inv(evects)*z0;
#     # Q = evects[1:r*n_td, :];
#     Q=evects[(n_td-1)*r+1:n_td*r,:];
#     return a, Q
# end

function compute_amplitudes_given_ic_z0(g0_mem, evects, n_ks, n_td, r)
    #inputs: e.vects and e.vals of companion matrix.
    #outputs: amplitudes obtained from optimized dmd
    #mzmd modes:
    z0 = zeros(n_td*n_ks*r);
    for k in 1:n_ks
        z0[((k-1)*r*n_td+1):(k*r*n_td)] = g0_mem[:,(n_ks-k+1)];
        # z0[((k-1)*r*n_td+1):(k*r*n_td)] = g0_mem[:, k];
    end
    a = pinv(evects)*z0;
    # a = inv(evects)*z0;
    Q = evects[1:r*n_td, :];
    return a, Q
end

function compute_amplitudes_given_ic_x0(Ur, X0, evects, d, r)
    #inputs: e.vects and e.vals of companion matrix.
    #outputs: amplitudes obtained from optimized dmd
    #mzmd modes:
    x0 = Ur' * X0;
    a = pinv(evects)*x0;
    Q = evects[1:d*r, :];
    return a, Q
end



function compute_amplitudes_given_ic_z0_lsa(z0, evects, evals, U, num_states)
    #inputs: e.vects and e.vals of companion matrix.
    #outputs: amplitudes obtained from optimized dmd
    Q = evects[1:r, :]; #equivalent to P0*Vc
    # Q = evects;
    num_modes = size(Q, 2);
    # Phi_mz = Ur*Q; #must be done before normalizing
    #normalize modes
    for m=1:num_modes
        NormQ=Q[:,m];
        Q[:,m]= Q[:,m]/norm(NormQ,2);
    end
    M = diagm(evals); #matrix of evals given as vector
    r_, rk = size(Q);
    Z0 = U'*z0;
    nobs, t_ic= size(Z0);
    b = zeros(r_*t_ic);
    L = complex.(zeros(r_*t_ic, rk))
    aa = I(rk)
    println("size L = ", size(L))
    println("size b = ", size(b))
    println("size Z = ", size(Z0))
    for i in 1:t_ic
        b[1+(i-1)*nobs:i*nobs] = Z0[:, i]
        L[1+(i-1)*nobs:i*nobs,:] = Q*aa;
        aa=aa*M;
    end
    a = L\b;
    # Phi_mz = Ur*Q;
    # a = Phi_mz\X0;
    #compute amplitudes for plotting:
    u=complex.(zeros(r_, rk));
    for m=1:num_modes
        u[:,m]=a[m]*Q[:,m];
    end
    amplitude=zeros(num_modes);
    for m=1:num_modes
        aca=U*u[:,m];
        amplitude[m]=norm(aca,2)/sqrt(num_states);
    end
    return a, amplitude, Q
end

function compute_amplitudes_given_ic_x0(X0, evects, U, num_states)
    #inputs: e.vects and e.vals of companion matrix.
    #outputs: amplitudes obtained from optimized dmd
    Q = evects;
    num_modes = size(Q, 2);
    Phi_mz = U*Q; #must be done before normalizing
    println("size Phi =", size(Phi_mz))
    r_, rk = size(Q);
    a = Phi_mz\X0;
    #compute amplitudes for plotting:
    u=complex.(zeros(r_, rk));
    for m=1:num_modes
        u[:,m]=a[m]*Q[:,m];
    end
    amplitude=zeros(num_modes);
    for m=1:num_modes
        aca=U*u[:,m];
        amplitude[m]=norm(aca,2)/sqrt(num_states);
    end
    return a, amplitude, Q
end
