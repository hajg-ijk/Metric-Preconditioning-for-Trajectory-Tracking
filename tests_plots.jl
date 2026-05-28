using Manifolds, ManifoldsBase, LieGroups, ManifoldDiff
using LinearAlgebra, Statistics, RecursiveArrayTools
using DifferentialEquations, Dierckx
using DifferentiationInterface, ForwardDiff, FiniteDifferences
using OrdinaryDiffEqRosenbrock: Rodas4P
using ADTypes: AutoFiniteDiff
using GLMakie, Printf, NPZ
using StaticArrays
using Plots
using CSV, DataFrames

# Define groups, points, basis and vectors
SE3 = SpecialEuclideanGroup(3)
se3 = LieAlgebra(SE3)
SO3 = SpecialOrthogonalGroup(3)
so3 = LieAlgebra(SO3)

e1 = @SVector [1.0, 0.0, 0.0]
e2 = @SVector [0.0, 1.0, 0.0]
e3 = @SVector [0.0, 0.0, 1.0]

include("dynamics.jl")
include("gain_sweep.jl")
include("manifolds_helpers.jl")
include("plotting.jl")
include("sampling.jl")
include("vector_field.jl")
#
# Metric modifier
λ = 1.0
λ_metric = 1.0

preconditioner_scale = 1.0
preconditioner_correction_max = 30.0
preconditioner_debug = true

ω0 = 0.0
ωd_field(G, R) = hat(so3, [0.0, 0.0, ω0])  # constant rotation about z-axis
function X(G::SpecialEuclideanGroup, g) 
  R = g[G, :Rotation]
  q = g[G, :Translation]
  Ω = ωd_field(G, R)  
  # NOTE: For now the vector field does not depend on the rotational component of g
  return vcat(hcat(Ω, corridor_flow_direction_3d(q[1], q[2], q[3])), zeros(4)')
end
μ(G, g) = flat(G, g, λ * X(G, g))
#
# Load reference trajectory (assume maze_reference.npz exists)
T, qd, vd, ad, jd, sd, ref_data = load_reference_npz("maze_reference.npz")

# Initial conditions 
R0 = @SMatrix [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
q0 = SVector{3, Float64}(qd(0.0) + [0.25, -0.05, 0.0])  # perturbed initial position
v0 = @SVector zeros(3) # initial velocity
Ω0 = @SVector zeros(3)
#
# Control choices
feedforward_type = :euclidean
f_max = 10.0
u_max = 100.0

# Gains returned by `main_gain_run.jl` for the exact integral curve tracking
kR_e = 5.75 
kq_e = 5.75 
kv_e = 7.525 
kΩ_e = 2.575
kR_p = 5.75
kq_p = 5.75 
kv_p = 5.05 
kΩ_p = 5.05

plot_tracking_comparison(
  T,
  R0, q0, v0, Ω0,
  qd, vd, ad, jd, sd,
  kR_e, kq_e, kv_e, kΩ_e,
  kR_p, kq_p, kv_p, kΩ_p;
  feedforward_type=feedforward_type, 
  f_max=f_max,
  u_max=u_max,
  xmin=-2.0, 
  xmax=2.0, 
  ymin=-2.5, 
  ymax=2.0, 
  zmin=0.0, 
  zmax=2.0,
  show_maze=true, 
  show_μ_field=true,
  write=false
)
#
# --------------------------------------------------------------
# Gains returned by `main_gain_run.jl` 
# for the tracking of a curve not integral to the vector field
# Uncomment to plot this simulation
# --------------------------------------------------------------
#
# kR_e = 20.0 
# kq_e = 5.75 
# kv_e = 5.05 
# kΩ_e = 10.0
# kR_p = 15.25 
# kq_p = 10.5 
# kv_p = 10.0 
# kΩ_p = 7.525
#
# T, qd, vd, ad, jd, sd, ref_data = load_reference_npz("inexact_reference.npz")
#
# plot_tracking_comparison(
#   T,
#   R0, q0, v0, Ω0,
#   qd, vd, ad, jd, sd,
#   kR_e, kq_e, kv_e, kΩ_e,
#   kR_p, kq_p, kv_p, kΩ_p;
#   feedforward_type=feedforward_type, 
#   f_max=f_max,
#   u_max=u_max,
#   xmin=-2.0, 
#   xmax=2.0, 
#   ymin=-2.5, 
#   ymax=2.0, 
#   zmin=0.0, 
#   zmax=2.0,
#   show_maze=true, 
#   show_μ_field=true,
#   write=false
# )
# --------------------------------------------------------------
# Gain search with automatic plotting
# Uncomment to run this simulation
# The parameters T, qd, vd, ad, jd, sd depend on the loaded 
# reference trajectory
# --------------------------------------------------------------
#
# Gain grid
# gain_criterion = :sum
# kR_vals = range(1.0, 20.0, length=5)
# kq_vals = range(1.0, 20.0, length=5)
# kv_vals = range(0.1, 10.0, length=5)
# kΩ_vals = range(0.1, 10.0, length=5)

# Auto-select gains and plot comparison
# result = auto_select_and_plot_tracking_comparison(
#   T, 
#   R0, q0, v0, Ω0, 
#   qd, vd, ad, jd, sd,
#   kR_vals, kq_vals, kv_vals, kΩ_vals;
#   feedforward_type=feedforward_type, 
#   f_max=f_max,
#   u_max=u_max,
#   gain_criterion=gain_criterion, 
#   gain_weights=(1.0, 1.0, 1.0, 1.0),
#   xmin=-2.0, 
#   xmax=2.0, 
#   ymin=-2.5, 
#   ymax=2.0, 
#   zmin=0.0, 
#   zmax=2.0,
#   show_maze=true, 
#   show_μ_field=true,
#   trajectory="exact_f50_u250_",
#   write=true,
# )
#
# Plot gain sweep comparison
# plot_gain_sweep_comparison(
#   T, 
#   R0, q0, v0, Ω0, 
#   qd, vd, ad, jd, sd,
#   kR_vals, kq_vals, kv_vals, kΩ_vals;
#   feedforward_type=feedforward_type, 
#   f_max=f_max,
#   u_max=u_max,
#   gain_criterion=gain_criterion, 
#   gain_weights=(1.0, 1.0, 1.0, 1.0)
# )
