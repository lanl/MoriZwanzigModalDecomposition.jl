using LinearAlgebra
using Printf
# using ProgressBars

function compute_mz(g,h)

    M = size(g,1)
    N = size(g,2)
    C = zeros(M,M,h+2)

    # @printf("number of BLAS threads: %i\n",LinearAlgebra.BLAS.get_num_threads())

    for k=1:h+2
        # @printf("computing C(%i)...\n",k-1)
        temp = zeros(M,M)
        # for l in ProgressBar(1:N-k+1)
        for l in 1:(N-k+1)
            #@printf("\tloop iter #%i of %i)\n",l,N-k+1)
            LinearAlgebra.BLAS.ger!(1.0,g[:,l+k-1],g[:,l],temp)
            #temp+=g[:,l+k-1]*g[:,l]'
        end
        C[:,:,k] = temp./(N-k+1)
    end

    # print("computing C0 inverse...\n")
    Cinv = inv(C[:,:,1])

    Om = zeros(M,M,h+1)

    # print("computing Om(0)...\n")
    Om[:,:,1] = C[:,:,2]*Cinv

    for k=2:h+1
        # @printf("computing Om(%i)...\n",k-1)
        temp = zeros(M,M)
        for l=1:k-1
            temp += Om[:,:,l]*C[:,:,k-l+1]
        end
        Om[:,:,k] = (C[:,:,k+1]-temp)*Cinv
    end

    return Om,C
end
