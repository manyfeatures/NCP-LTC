#using Plots
using Parameters
using Flux
#using StatsBase # for random choice
#import Random
#Random.seed!(1234)

include("./wirings.jl")
using .wirings

# Data
in_features = 2
out_features = 1
N = 48

π_32 = Float32(π)
t = range(0.0f0, stop = 3π_32, length = N)
sin_t = sin.(t)
cos_t = cos.(t)
data_x = reshape(hcat(sin_t, cos_t), (1, N, 2))
data_y = reshape(sin.(range(0.0f0, stop = 6π_32, length = N)), (1, N, 1))

println(size(data_x))
println(size(data_y))

# Plots
#plot(data_x[:,:,1]')
#plot!(data_x[:,:,2]')
#plot!(data_y[1,:,1])

wiring = FullyConnected(8)
LTC = Flux.Recur(LTCCell(wiring, in_features), rand(1,8)) # toy state
LTC(data_x)

grads = Flux.gradient(params(LTC)) do
    @show sum(LTC(data_x))
end

for p in grads
    println(p)
end
