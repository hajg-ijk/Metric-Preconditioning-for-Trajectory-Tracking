using Manifolds, ManifoldsBase, LieGroups, ManifoldDiff
using LinearAlgebra, Statistics, RecursiveArrayTools
using DifferentialEquations, Dierckx
using DifferentiationInterface, ForwardDiff, FiniteDifferences
using GLMakie, Printf, NPZ
using StaticArrays
using Plots
using ADTypes: AutoFiniteDiff
using CSV, DataFrames

SE3 = SpecialEuclideanGroup(3)
se3 = LieAlgebra(SE3)
SO3 = SpecialOrthogonalGroup(3)
so3 = LieAlgebra(SO3)
Atlas_SE3 = get_default_atlas(SE3)
chart_index_SE3_e = Manifolds.get_chart_index(SE3, Atlas_SE3, identity_element(SE3))
induced_basis_e = induced_basis(SE3, Atlas_SE3, chart_index_SE3_e)

e1 = @SVector [1.0, 0.0, 0.0]
e2 = @SVector [0.0, 1.0, 0.0]
e3 = @SVector [0.0, 0.0, 1.0]

include("diagnostics.jl")
include("dynamics.jl")
include("gain_sweep.jl")
include("manifolds_helpers.jl")
include("plotting.jl")
include("sampling.jl")
include("vector_field.jl")

λ = 1.0
λ_metric = 1.0
preconditioner_scale = 1.0
preconditioner_correction_max = 30.0
preconditioner_debug = true
ω0 = 0.0

ωd_field(G, R) = hat(so3, [0.0, 0.0, ω0])

function X(G::SpecialEuclideanGroup, g)
    R = g[G, :Rotation]
    q = g[G, :Translation]
    Ω = ωd_field(G, R)
    return vcat(hcat(Ω, corridor_flow_direction_3d(q[1], q[2], q[3])), zeros(4)')
end

μ(G, g) = flat(G, g, λ * X(G, g))

R0 = @SMatrix [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
v0 = @SVector zeros(3)
Ω0 = @SVector zeros(3)

feedforward_type = :euclidean
gain_criterion = :control

kR_vals = range(1.0, 20.0, length=5)
kq_vals = range(1.0, 20.0, length=5)
kv_vals = range(0.1, 10.0, length=5)
kΩ_vals = range(0.1, 10.0, length=5)

shared_kwargs = (
    feedforward_type = feedforward_type,
    gain_criterion   = gain_criterion,
    gain_weights     = (1.0, 1.0, 1.0, 1.0),
    xmin=-2.0, xmax=2.0,
    ymin=-2.5, ymax=2.0,
    zmin=0.0,  zmax=2.0,
    show_maze=true,
    show_μ_field=true,
    write=true,
)

cases = [
    # (npz file,            q0 perturbation,       f_max,   u_max,   trajectory label)
    ("maze_reference.npz",    [0.25, -0.05, 0.0],  10.0,    100.0,   "exact_f10_u100"),
    ("inexact_reference.npz", [0.25, -0.05, 0.0],  10.0,    100.0,   "inexact_f10_u100"),
]

for (npz, q0_perturb, f_max, u_max, traj_label) in cases
  @info "Running case: $traj_label"

  T, qd, vd, ad, jd, sd, _ = load_reference_npz(npz)
  q0 = SVector{3,Float64}(qd(0.0) + q0_perturb)

  sweep_and_save_gains(
    T, 
    R0, q0, v0, Ω0, 
    qd, vd, ad, jd, sd,
    kR_vals, kq_vals, kv_vals, kΩ_vals;
    feedforward_type=feedforward_type, f_max=f_max, u_max=u_max,
    gain_criterion=gain_criterion, trajectory=traj_label
  )

  GC.gc()
end
