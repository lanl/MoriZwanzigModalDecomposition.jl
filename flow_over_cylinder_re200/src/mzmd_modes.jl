using LinearAlgebra

"""
Computing markovian (DMD) modes and modes of MZ with memory: MZMD
    -Forms block companion matrix and extracts modes and spectrum.
    -Computes predictions with preselected modes
"""

#Markovian modes: (equivalent to DMD modes)
function compute_markovian_modes(M, Ur, X0, r)
    lambda, W_r = eigen(M);
    Φ_markovian = Ur * W_r;
    a_m = Φ_markovian\X0; #ampliltude of modes
    amp = zeros(r)
    for i in 1:r
        amp[i] = norm(Φ_markovian[:,i]*a_m[i]);
    end
    ind = sortperm(abs.(amp), rev=true)
    return lambda[ind], Φ_markovian[:, ind], a_m[ind]
end



function form_companion(Ω, r, n_ks)
    #form block companion matrix
    C_r1 = zeros(r, n_ks*r) #row 1 of C
    for i in 1:n_ks
        C_r1[:, ((i-1)*r+1):i*r] = Ω[i,:,:]
    end
    if n_ks > 1
        eye = I((n_ks)*r) .+ zeros((n_ks)*r,(n_ks)*r)
        eye_sub = eye[1:((n_ks-1)*r), 1:((n_ks)*r)]
    end
    if n_ks > 1
        C = vcat(C_r1, eye_sub)
    else 
        C = C_r1
    end
    return C
end

function form_companion_memory_only(Ω, r, n_ks)
    #form block companion matrix
    C_r1 = zeros(r, n_ks*r) #row 1 of C
    for i in 2:n_ks
        C_r1[:, ((i-1)*r+1):i*r] = Ω[i,:,:]
    end
    if n_ks > 1
        eye = I((n_ks)*r) .+ zeros((n_ks)*r,(n_ks)*r)
        eye_sub = eye[1:((n_ks-1)*r), 1:((n_ks)*r)]
    end
    if n_ks > 1
        C = vcat(C_r1, eye_sub)
    else 
        C = C_r1
    end
    return C
end

function mzmd_modes(C, X0)
    #compute modes and amplitudes from C
    Λc, Vc = eigen(C);
    #mzmd modes:
    Φ_mz = Ur*Vc[1:r,:];
    #compute amplitudes of modes
    a = Φ_mz\X0;
    amp = zeros(r*n_ks)
    for i in 1:r*n_ks
        amp[i] = norm(Φ_mz[:,i]*a[i]);
    end
    #sort according to largest amplitude:
    ind = sortperm(abs.(amp), rev=true)
    return Λc[ind], Φ_mz[:, ind], a[ind], amp[ind]
end

function mzmd_modes_reduced_amps(C, Ur, r, n_ks)
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



function mzmd_modes_full_amps(C)
    #NOTE this is more computationally expensive than
    #mzmd_modes_reduced_amps and they are equaivlent.

    #X0 needs to include first n_ks snapshots in X
    X0_mem = X[:, 1:n_ks];
    #compute modes and amplitudes from C_g
    Λc, Vc = eigen(C);
    # Λc, Vc = eigen(C, sortby=x->abs.(x))
    #mzmd modes:
    Φ_mz = Ur*Vc[1:r,:];
    #set i.c. and compute amplitudes of modes
    z0 = zeros(n_ks*m)
    for k in 1:n_ks
        z0[((k-1)*m+1):k*m] = X0_mem[:,k]
    end
    #construct: full Φ
    ty = typeof(Φ_mz[1,1]);
    Φ = zeros(ty, m*n_ks, r*n_ks)
    for i in 1:n_ks
        # Φ[((i-1)*m+1):i*m, :] = Ur*(Vc[((i-1)*r+1):i*r, :] * diagm(1 ./(Λc.^(i-1))))
        Φ[((i-1)*m+1):i*m, :] = Φ_mz * diagm(1 ./(Λc.^(i-1)))
    end
    a = (pinv(Φ)*z0)
    amp = zeros(r*n_ks)
    for i in 1:r*n_ks
        amp[i] = norm(Φ_mz[:,i]*a[i]);
    end
    #sort according to largest amplitude:
    ind = sortperm(abs.(amp), rev=true)
    return Λc[ind], Φ_mz[:, ind], a[ind], amp[ind]
end


function compute_prediction_modes(Φi, ai, λi, tpred)
    #eigen-reconstruction with dominant modes Φi
    #amplitude: ai, e.val: λi, prediction time steps: tpred
    solution = zeros(size(Φi,1), tpred)
    for t in 1:tpred
        solution[:,t] = real.(Φi * (ai.* (λi.^t)));
    end
    return solution
end


