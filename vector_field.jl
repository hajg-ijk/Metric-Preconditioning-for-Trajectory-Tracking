# ============================================================
# Maze drawing
# ============================================================

function plot_maze_3d!(ax; color=:black, linewidth=3)
  segments = [
    ((0.0, -2.0, 0.0), (0.0, 0.0, 0.0)),
    ((1.0, -2.0, 0.0), (1.0, 0.5, 0.0)),
    ((1.0, 0.5, 0.0), (-0.5, 0.5, 0.0)),
    ((0.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
    ((-1.0, 0.0, 0.0), (-1.0, 1.5, 0.0)),
    ((-0.5, 0.5, 0.0), (-0.5, 1.5, 0.0)),
  ]

  for (p1, p2) in segments
    lines!(ax, [p1[1], p2[1]], [p1[2], p2[2]], [0.0, 0.0], color=color, linewidth=linewidth)
    lines!(ax, [p1[1], p1[1]], [p1[2], p1[2]], [0.0, 2.0], color=color, linewidth=linewidth)
    lines!(ax, [p2[1], p2[1]], [p2[2], p2[2]], [0.0, 2.0], color=color, linewidth=linewidth)
    lines!(ax, [p1[1], p2[1]], [p1[2], p2[2]], [2.0, 2.0], color=color, linewidth=linewidth)
  end
end

# ============================================================
# Better-conditioned smooth maze vector field
# ============================================================
# Design goals:
#   1. bounded, smooth field;
#   2. integral curves follow the corridor centerline
#      (0.5,-1.8) -> (0.5,0.25) -> (-0.75,0.25) -> (-0.75,1.25);
#   3. no hard wall-repulsion singularity and no unit-normalization singularity;
#   4. modest divergence: weak transverse centering plus smooth gates;
#   5. nonconstant tangential speed and a very small z-wave, so the reference
#      acceleration does not vanish on long straight pieces.

# Stable logistic and C^∞-like smooth window.
logistic(z) = 1.0 / (1.0 + exp(-clamp(z, -60.0, 60.0)))

smooth_window(z, zmin, zmax, gate) =
  logistic((z - zmin) / gate) * logistic((zmax - z) / gate)

# Smoothly saturate a vector without unit-normalizing it.
"""
    smooth_saturate_2d(U, V; vmax=1.15, ε=1e-8) -> (U_sat, V_sat)
 
Smoothly limit the magnitude of a 2D vector `(U, V)` to at most `vmax` using
a `tanh`-based saturation, without hard unit normalization.
 
# Arguments
- `U, V::Real`: Vector components.
- `vmax::Float64`: Saturation ceiling.
- `ε::Float64`: Regularization for near-zero norm.
 
# Returns
Two `Float64` values: saturated components.
"""
function smooth_saturate_2d(U, V; vmax=1.15, ε=1e-8)
  n = sqrt(U^2 + V^2 + ε^2)
  scale = vmax * tanh(n / vmax) / n
  return scale * U, scale * V
end

"""
    smooth_saturate_3d(U, V, W; vmax=1.15, ε=1e-8) -> (U_sat, V_sat, W_sat)
 
3D analogue of `smooth_saturate_2d`.
 
# Returns
Three `Float64` values: saturated components.
"""
function smooth_saturate_3d(U, V, W; vmax=1.15, ε=1e-8)
  n = sqrt(U^2 + V^2 + W^2 + ε^2)
  scale = vmax * tanh(n / vmax) / n
  return scale * U, scale * V, scale * W
end

# Mildly varying tangential speed. This is deliberate: on the centerline of a
# straight corridor, a strictly constant velocity field gives zero material
# acceleration, which makes b3d = ad/||ad|| ill-conditioned in the driftless
# reduced model.
"""
    tangential_speed(s, smin, smax; base_speed=0.90, speed_slope=0.10) -> Float64
 
Return a mildly varying scalar speed along a corridor as a function of
along-corridor coordinate `s`, using a `tanh` modulation. The non-constant
speed ensures the reference acceleration does not vanish on straight segments,
keeping `b3d = ad/‖ad‖` well-conditioned.
 
# Arguments
- `s::Real`: Along-corridor coordinate.
- `smin, smax::Real`: Corridor extent.
- `base_speed, speed_slope::Float64`: Base speed and modulation amplitude.
 
# Returns
`Float64`: speed value.
"""
function tangential_speed(s, smin, smax; base_speed=0.90, speed_slope=0.10)
  mid = 0.5 * (smin + smax)
  scale = max(0.5 * (smax - smin), 1e-6)
  return base_speed * (1.0 + speed_slope * tanh((s - mid) / scale))
end

# The three corridor primitives are expressed in local coordinates:
#   rho = transverse offset from the corridor centerline,
#   s   = along-corridor coordinate increasing toward the goal.
# Each local field is approximately Hamiltonian/tangential plus a weak centering
# term -center_gain*rho*n. The centering term is what makes nearby integral
# curves stay inside the corridor; keeping center_gain small keeps divergence
# moderate.
"""
    conditioned_maze_vector_field_2d(x, y; sigma=0.38, gate=0.28,
                                      center_gain=0.16, base_speed=0.90,
                                      speed_slope=0.10, vmax=1.15, ε=1e-8)
    -> (U, V, phase, Wsum)
 
Core 2D maze vector field. Blends three corridor primitives (right vertical,
horizontal, left vertical) with Gaussian weights. Each primitive applies weak
transverse centering and mildly varying tangential speed.
 
# Arguments
- `x, y::Real`: Evaluation point.
- `sigma::Float64`: Gaussian corridor width.
- `gate::Float64`: Logistic gate sharpness for along-corridor extent.
- `center_gain::Float64`: Transverse centering gain.
- `base_speed, speed_slope, vmax, ε`: Speed and saturation parameters.
 
# Returns
- `U, V::Float64`: 2D velocity components (already saturated).
- `phase::Float64`: Blended corridor phase (used for z-wave in 3D).
- `Wsum::Float64`: Sum of corridor weights (useful for diagnostics).
"""
function conditioned_maze_vector_field_2d(
  x, y;
  σ=nothing,
  sigma=0.38,
  gate=0.28,
  center_gain=0.16,
  base_speed=0.90,
  speed_slope=0.10,
  vmax=1.15,
  ε=1e-8,
  eps=nothing,
)
  width = σ === nothing ? sigma : σ
  ϵ = eps === nothing ? ε : eps

  # Right/bottom vertical corridor: center x = 0.5, upward flow.
  ρr = x - 0.50
  sr = y
  wr = exp(-(ρr / width)^2) * smooth_window(y, -1.95, 0.34, gate)
  Ur = -center_gain * ρr
  Vr = tangential_speed(sr, -1.95, 0.34; base_speed=base_speed, speed_slope=speed_slope)
  φr = (sr + 1.95) / (0.34 + 1.95)

  # Middle horizontal corridor: center y = 0.25, leftward flow.
  ρt = y - 0.25
  st = 0.60 - x       # increases as x moves left
  wt = exp(-(ρt / width)^2) * smooth_window(x, -0.88, 0.62, gate)
  Ut = -tangential_speed(st, 0.0, 1.50; base_speed=base_speed, speed_slope=speed_slope)
  Vt = -center_gain * ρt
  φt = 1.0 + st / 1.50

  # Left vertical corridor: center x = -0.75, upward flow.
  ρl = x + 0.75
  sl = y
  wl = exp(-(ρl / width)^2) * smooth_window(y, 0.16, 1.42, gate)
  Ul = -center_gain * ρl
  Vl = tangential_speed(sl, 0.16, 1.42; base_speed=base_speed, speed_slope=speed_slope)
  φl = 2.0 + (sl - 0.16) / (1.42 - 0.16)

  Wsum = wr + wt + wl + ϵ
  Uraw = (wr * Ur + wt * Ut + wl * Ul) / Wsum
  Vraw = (wr * Vr + wt * Vt + wl * Vl) / Wsum
  phase = (wr * φr + wt * φt + wl * φl) / Wsum

  U, V = smooth_saturate_2d(Uraw, Vraw; vmax=vmax, ε=ϵ)
  return U, V, phase, Wsum
end

# Backwards-compatible public 2D functions.
# Unlike the old implementation, this does not normalize to unit length and does
# not add hard wall repulsion. The field is already bounded by smooth saturation.
"""
    corridor_flow(x, y; kwargs...) -> Vector{Float64}
 
Public 2D wrapper for `conditioned_maze_vector_field_2d`. Returns the 2D
velocity vector `[U, V]` without normalization.
 
# Returns
`Vector{Float64}` of length 2.
"""
function corridor_flow(x, y; kwargs...)
  U, V, _, _ = conditioned_maze_vector_field_2d(x, y; kwargs...)
  return [U, V]
end

"""
    maze_vector_field(x, y; sigma=0.38, ε=1e-8, flow_strength=1.0,
                       repulsion_strength=0.0, gate=0.28, center_gain=0.16,
                       base_speed=0.90, speed_slope=0.10, vmax=1.15)
    -> Vector{Float64}
 
Backwards-compatible 2D maze field wrapper. The `repulsion_strength` argument
is accepted but ignored (wall-repulsion was removed for lower divergence).
 
# Returns
`Vector{Float64}` of length 2.
"""
function maze_vector_field(
  x, y;
  σ=nothing,
  sigma=0.38,
  ε=1e-8,
  eps=nothing,
  flow_strength=1.0,
  repulsion_strength=0.0,
  gate=0.28,
  center_gain=0.16,
  base_speed=0.90,
  speed_slope=0.10,
  vmax=1.15,
)
  # repulsion_strength is intentionally ignored. The old wall-repulsion term was
  # a high-divergence source and created very sharp spatial derivatives.
  ϵ = eps === nothing ? ε : eps
  width = σ === nothing ? sigma : σ
  U, V, _, _ = conditioned_maze_vector_field_2d(
    x, y;
    sigma=width,
    gate=gate,
    center_gain=center_gain,
    base_speed=flow_strength * base_speed,
    speed_slope=speed_slope,
    vmax=vmax,
    ε=ϵ,
  )
  return [U, V]
end

# 3D field used both for reference generation and for μ-shaping.
# The z component is small and bounded. It is included for conditioning: it
# removes the exact zero-acceleration stretches that occur with a planar,
# constant-speed straight-corridor field. Set z_wave_amp=0.0 if a
# strictly planar reference is needed.
"""
    corridor_flow_direction_3d(x, y, z; sigma=0.38, ε=1e-8,
                                flow_strength=1.0, repulsion_strength=0.0,
                                gate=0.28, center_gain=0.16,
                                base_speed=0.90, speed_slope=0.10, vmax=1.15,
                                z0=1.0, z_restore=0.04, z_wave_amp=0.035,
                                z_wave_freq=2π, z_phase_shift=π/4,
                                z_weight=nothing, normalize_output=false)
    -> Vector{Float64}
 
Production 3D corridor vector field used for reference generation and
μ-shaping. Adds a small z-component combining a sinusoidal wave (to avoid
zero-acceleration straight segments) and a restoring term toward `z0`.
 
# Arguments
- `x, y, z::Real`: 3D evaluation point.
- `sigma, gate, center_gain, base_speed, speed_slope, vmax, ε`: 2D field
  parameters.
- `z0::Float64`: Target z-height for restoration.
- `z_restore::Float64`: Gain for z-restoration toward `z0`.
- `z_wave_amp::Float64`: Amplitude of z sinusoidal wave.
- `z_wave_freq::Float64`: Frequency of z wave (in phase units).
- `z_phase_shift::Float64`: Phase offset of z wave.
- `z_weight`: Legacy parameter; adds to `z_restore` if provided.
- `normalize_output::Bool`: If `true`, unit-normalize the output.
 
# Returns
`Vector{Float64}` of length 3.
"""
function corridor_flow_direction_3d(
  x, y, z;
  σ=nothing,
  sigma=0.38,
  ε=1e-8,
  eps=nothing,
  flow_strength=1.0,
  repulsion_strength=0.0,
  gate=0.28,
  center_gain=0.16,
  base_speed=0.90,
  speed_slope=0.10,
  vmax=1.15,
  z0=1.0,
  z_restore=0.04,
  z_wave_amp=0.035,
  z_wave_freq=2π,
  z_phase_shift=π / 4,
  z_weight=nothing,
  normalize_output=false,
)
  ϵ = eps === nothing ? ε : eps
  width = σ === nothing ? sigma : σ

  U, V, phase, _ = conditioned_maze_vector_field_2d(
    x, y;
    sigma=width,
    gate=gate,
    center_gain=center_gain,
    base_speed=flow_strength * base_speed,
    speed_slope=speed_slope,
    vmax=vmax,
    ε=ϵ,
  )

  # If older code passes z_weight, interpret it as an additional weak restoring
  # gain toward z0, not as W = z_weight*z, which can make z drift away.
  restore = z_weight === nothing ? z_restore : z_restore + z_weight
  W = z_wave_amp * sin(z_wave_freq * phase + z_phase_shift) - restore * (z - z0)

  U, V, W = smooth_saturate_3d(U, V, W; vmax=vmax, ε=ϵ)

  if normalize_output
    n = sqrt(U^2 + V^2 + W^2 + ϵ^2)
    return [U / n, V / n, W / n]
  else
    return [U, V, W]
  end
end

# ============================================================
# Diagnostics for the vector field itself
# ============================================================

"""
    finite_difference_divergence_2d(x, y; h=1e-4, kwargs...) -> Float64
 
Estimate the 2D divergence of `maze_vector_field` at `(x, y)` using central
finite differences with step `h`.
 
# Returns
`Float64`: ∂U/∂x + ∂V/∂y.
"""
function finite_difference_divergence_2d(x, y; h=1e-4, kwargs...)
  vp_x = maze_vector_field(x + h, y; kwargs...)
  vm_x = maze_vector_field(x - h, y; kwargs...)
  vp_y = maze_vector_field(x, y + h; kwargs...)
  vm_y = maze_vector_field(x, y - h; kwargs...)
  return (vp_x[1] - vm_x[1]) / (2h) + (vp_y[2] - vm_y[2]) / (2h)
end

"""
    material_acceleration_3d(x, y, z; h=1e-4, kwargs...) -> Vector{Float64}
 
Compute the material (Lie) derivative of the 3D corridor flow:
    a = (∇V) V
via finite differences.
 
# Returns
`Vector{Float64}` of length 3: material acceleration at `(x, y, z)`.
"""
function material_acceleration_3d(x, y, z; h=1e-4, kwargs...)
  v = corridor_flow_direction_3d(x, y, z; kwargs...)
  J = zeros(3, 3)
  for j in 1:3
    ep = zeros(3)
    ep[j] = h
    vp = corridor_flow_direction_3d(x + ep[1], y + ep[2], z + ep[3]; kwargs...)
    vm = corridor_flow_direction_3d(x - ep[1], y - ep[2], z - ep[3]; kwargs...)
    J[:, j] .= (vp .- vm) ./ (2h)
  end
  return J * v
end

"""
    vector_field_conditioning_report(; nx=40, ny=40, z=1.0, kwargs...) -> NamedTuple
 
Evaluate conditioning statistics of the 3D corridor field over a 2D grid at
height `z`, reporting field norm, divergence, and material acceleration.
 
# Arguments
- `nx, ny::Int`: Grid resolution.
- `z::Float64`: Evaluation height.
- `kwargs...`: Forwarded to `corridor_flow_direction_3d` and `maze_vector_field`.
 
# Returns
`NamedTuple` with fields:
`max_norm, mean_norm, max_abs_divergence, mean_abs_divergence,
min_material_acceleration, mean_material_acceleration, max_material_acceleration`.
"""
function vector_field_conditioning_report(; nx=40, ny=40, z=1.0, kwargs...)
  xs = range(-1.15, 0.85, length=nx)
  ys = range(-1.90, 1.35, length=ny)

  norms = Float64[]
  divs = Float64[]
  accs = Float64[]

  for x in xs, y in ys
    v = corridor_flow_direction_3d(x, y, z; kwargs...)
    push!(norms, sqrt(sum(abs2, v)))
    push!(divs, finite_difference_divergence_2d(x, y; kwargs...))
    a = material_acceleration_3d(x, y, z; kwargs...)
    push!(accs, sqrt(sum(abs2, a)))
  end

  return (
    max_norm=maximum(norms),
    mean_norm=mean(norms),
    max_abs_divergence=maximum(abs.(divs)),
    mean_abs_divergence=mean(abs.(divs)),
    min_material_acceleration=minimum(accs),
    mean_material_acceleration=mean(accs),
    max_material_acceleration=maximum(accs),
  )
end

# ============================================================
# Reference trajectory loading and optional rebuilding
# ============================================================

"""
    load_reference_npz(filename="maze_reference.npz")
    -> (T, qd, vd, ad, jd, sd, data)
 
Load a reference trajectory from an `.npz` file and fit quintic (or lower
order if insufficient samples) Dierckx splines to the x, y, z coordinates.
Returns callable functions for position through snap.
 
# Arguments
- `filename::String`: Path to the `.npz` file.
 
# Returns
- `T::Float64`: End time.
- `qd, vd, ad, jd, sd`: Callables `t::Float64 -> Vector{Float64}` for
  position, velocity, acceleration, jerk, and snap.
- `data::Dict`: Raw dictionary from `npzread`.
"""
function load_reference_npz(filename="maze_reference.npz")
  data = npzread(filename)

  # Truncate the last entry. This avoids endpoint evaluation pathologies in
  # Dierckx for some generated files.
  t = data["t"][1:end-1]
  x = data["x"][1:end-1]
  y = data["y"][1:end-1]
  z = data["z"][1:end-1]

  # Use a quintic spline when enough samples are present. This makes jerk and
  # snap diagnostics substantially less artificial than fitting a lower-order
  # spline and then differentiating it four times.
  k = length(t) >= 6 ? 5 : max(1, length(t) - 1)
  sx = Spline1D(t, x; k=k)
  sy = Spline1D(t, y; k=k)
  sz = Spline1D(t, z; k=k)

  T = t[end]

  qd(t) = [sx(t), sy(t), sz(t)]
  vd(t) = [
    Dierckx.derivative(sx, t, 1),
    Dierckx.derivative(sy, t, 1),
    Dierckx.derivative(sz, t, 1),
  ]
  ad(t) = [
    Dierckx.derivative(sx, t, 2),
    Dierckx.derivative(sy, t, 2),
    Dierckx.derivative(sz, t, 2),
  ]
  jd(t) = [
    Dierckx.derivative(sx, t, 3),
    Dierckx.derivative(sy, t, 3),
    Dierckx.derivative(sz, t, 3),
  ]
  sd(t) = [
    Dierckx.derivative(sx, t, 4),
    Dierckx.derivative(sy, t, 4),
    Dierckx.derivative(sz, t, 4),
  ]

  return T, qd, vd, ad, jd, sd, data
end

# Optional reference builder. Use this to regenerate maze_reference.npz after
# replacing the vector field. It assumes DifferentialEquations/NPZ are already
# loaded by the caller, as in existing scripts.
"""
    conditioned_reference_rhs!(dq, q, p, t)
 
ODE right-hand side for the conditioned reference trajectory (in-place).
Sets `dq = speed * corridor_flow_direction_3d(q...)`.
 
# Arguments
- `dq`: Output derivative (mutated).
- `q::Vector`: Current state `[x, y, z]`.
- `p`: Parameter vector `[speed]`.
- `t`: Current time.
 
# Returns
`nothing`.
"""
function conditioned_reference_rhs!(dq, q, p, t)
  speed = p[1]
  v = corridor_flow_direction_3d(q[1], q[2], q[3])
  dq[1] = speed * v[1]
  dq[2] = speed * v[2]
  dq[3] = speed * v[3]
  return nothing
end

"""
    build_conditioned_reference(q_start, q_goal; speed=0.70, T_max=8.0,
                                  n_samples=1600, out_file="maze_reference_conditioned.npz")
    -> (t_ref, q_ref)
 
Generate a reference trajectory by integrating `corridor_flow_direction_3d`
from `q_start` toward `q_goal` (terminating within 0.04 in the xy-plane of
the goal). Saves to an `.npz` file with both old and new schemas for
compatibility with `load_reference_npz`.
 
# Arguments
- `q_start, q_goal::Vector{Float64}`: Start/goal in ℝ³.
- `speed::Float64`: Integration speed.
- `T_max::Float64`: Maximum integration time.
- `n_samples::Int`: Number of save points.
- `out_file::String`: Output file path.
 
# Returns
- `t_ref::Vector{Float64}`: Time vector.
- `q_ref::Matrix{Float64}`: Trajectory `(n_samples, 3)`.
"""
function build_conditioned_reference(
  q_start=[0.50, -1.80, 1.0],
  q_goal=[-0.75, 1.25, 1.0];
  speed=0.70,
  T_max=8.0,
  n_samples=1600,
  out_file="maze_reference_conditioned.npz",
)
  p = [speed]
  prob = ODEProblem(conditioned_reference_rhs!, q_start, (0.0, T_max), p)

  condition(u, t, integrator) = norm(u[1:2] - q_goal[1:2]) - 0.04
  affect!(integrator) = terminate!(integrator)
  cb = ContinuousCallback(condition, affect!)

  sol = solve(
    prob,
    Tsit5();
    callback=cb,
    saveat=range(0.0, T_max, length=n_samples),
    reltol=1e-8,
    abstol=1e-10,
    maxiters=1_000_000,
  )

  t_ref = sol.t
  q_ref = hcat(sol.u...)'

  vector_field_name = collect(UInt8, codeunits("conditioned_centerline_low_divergence"))

npzwrite(
    out_file;
    t=t_ref,

    # Old schema expected by load_reference_npz:
    x=q_ref[:, 1],
    y=q_ref[:, 2],
    z=q_ref[:, 3],

    # New convenient matrix schema:
    q=q_ref,

    q_start=collect(q_start),
    q_goal=collect(q_goal),
    speed=[speed],
    vector_field_name=vector_field_name,
)

  println("Saved conditioned reference trajectory to $out_file")
  println("T_end = $(t_ref[end]), samples = $(length(t_ref))")
  return t_ref, q_ref
end

# Fields used by the preconditioning code.
vd_field(M, q) = corridor_flow_direction_3d(q[1], q[2], q[3])
ad_field(M, q; backend=AutoFiniteDifferences(central_fdm(5, 1))) =
  DifferentiationInterface.jacobian(p -> vd_field(M, p), backend, q) * vd_field(M, q)
