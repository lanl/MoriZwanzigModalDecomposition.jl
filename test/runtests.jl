using MZMD
using Test


@testset "MZMD.jl" begin
    # Write your tests here.
    s, u, xp = svd_method_of_snapshots(randn(100, 10), 5)
    @test size(s, 1)==10
    @test size(u,1)==100
    @test size(xp, 1)==5
    cov = obtain_C(randn(100, 10), 5, 3)
    @test size(cov, 1)==4
    @test size(cov, 2)==100
    m_, Ω_ = obtain_ker(cov, 3)
    @test size(m_, 1)==100
    @test size(Ω_, 1)==3
    @test size(Ω_, 2)==100
end
