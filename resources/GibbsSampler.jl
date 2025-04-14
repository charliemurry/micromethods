# A Gibbs sampler of a Bivariate Normal Distribution
# This is just a toy example, it would be much more efficient to just draw
# from the multivariate normal directly.

using Plots
using Statistics
using Distributions
using Random
using LinearAlgebra

# Set random seed (0 would be random based on clock)
Random.seed!(0)

# Set parameters
mu = [0.0, 2.0]
sig = [3.0, 3.0]
rho = 0.3

# Setup simulation parameters
animate = true
if animate
    T = 100
    afterburn = 50
else
    T = 10000
    afterburn = 1000
end

# Initialize sample storage
samp = zeros(T, 2)

# Set initial values
start = [40.0, 40.0]
samp[1,:] = start

# Create figure for animation
if animate
    plt = scatter([], [], 
        xlim=(-15, 50), 
        ylim=(-15, 50),
        markersize=2,
        label="Samples")
end

# Gibbs sampling loop
for t in 2:T
    # Draw first element conditional on previous draw
    samp[t,1] = rand(Normal(
        mu[1] + rho*(sig[1]/sig[2])*(samp[t-1,2] - mu[2]),
        sqrt(sig[1]^2*(1 - rho^2))
    ))
    
    # Draw second element conditional on first element
    samp[t,2] = rand(Normal(
        mu[2] + rho*(sig[2]/sig[1])*(samp[t,1] - mu[1]),
        sqrt(sig[2]^2*(1 - rho^2))
    ))
    
    if animate
        scatter!(plt, samp[1:t,1], samp[1:t,2], 
            markersize=2, 
            label="")
        display(plt)
        sleep(0.05)
    end
end

# Compute moments
m_samp = mean(samp[afterburn:T,:], dims=1)
sd_samp = std(samp[afterburn:T,:], dims=1)
corr_samp = cor(samp[afterburn:T,:])

# Final scatter plot
p1 = scatter(samp[:,1], samp[:,2],
    markersize=2,
    title="Draws from the Gibbs Sampler",
    label="Samples")

# Compute autocorrelation of first parameter
dsamp = samp[:,1] .- m_samp[1]
denom = dot(dsamp, dsamp)
acor = zeros(25)

for k in 1:25
    cov = dot(dsamp[1+k:end], dsamp[1:end-k])
    acor[k] = cov/denom
end

# Plot autocorrelation
p2 = bar(1:25, acor,
    title="Autocorrelation Function of First Parameter",
    label="Autocorrelation")

# Time series plot
p3 = plot(samp[1:min(T,2000),1],
    title="Time Series Trace of First Parameter",
    label="Parameter Value")

# Display all plots
plot(p1, p2, p3, layout=(3,1), size=(800,1200))

# Save the samples
# using JLD2
# @save "simpGibbsSamp.jld2" samp