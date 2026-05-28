# Helper functions
@inline function pack_state(R, q, v, Ω)
  @SVector [R[1,1], R[1,2], R[1,3],
            R[2,1], R[2,2], R[2,3],
            R[3,1], R[3,2], R[3,3],
            q[1], q[2], q[3],
            v[1], v[2], v[3],
            Ω[1], Ω[2], Ω[3]]
end
#
@inline function unpack_state(u::SVector{18})
  R = @SMatrix [u[1]  u[2]  u[3];
                u[4]  u[5]  u[6];
                u[7]  u[8]  u[9]]
  q = @SVector [u[10], u[11], u[12]]
  v = @SVector [u[13], u[14], u[15]]
  Ω = @SVector [u[16], u[17], u[18]]
  return R, q, v, Ω
end
#
function robust_frame_from_b3(b3)
  candidates = (e1, e2, e3)
  dots = [abs(dot(a, b3)) for a in candidates]
  a = candidates[argmin(dots)]

  b1_raw = a - dot(a, b3) * b3
  b1 = b1_raw / norm(b1_raw)
  b2 = cross(b3, b1)

  return b1, b2, hcat(b1, b2, b3)
end
#
function controls_and_desired_rotation(
  t, 
  R, q, v, Ω, 
  qd, vd, ad, jd, sd,
  kR, kq, kv, kΩ; 
  feedforward_type=:none,
  Ω̇=zeros(3),
  atol=1e-7,
  nu_tol=1e-2,
  return_intermediates=false,
  nu_preconditioning_world=SVector{3,Float64}(0.0, 0.0, 0.0),
  freeze_desired_attitude_derivatives=false,
)
  # Position and velocity errors
  eq = q - qd(t)
  ev = R*v - vd(t)
  @assert !any(isnan, eq) "eq is NaN at t=$t: q=$q, qd=$(qd(t))"
  @assert !any(isnan, ev) "ev is NaN at t=$t: v=$v, vd=$(vd(t))"

  # Base Euclidean virtual acceleration. In preconditioned mode we add
  # nu_preconditioning_world before constructing Rd, so the thrust axis is
  # selected from the preconditioned virtual acceleration.
  nu_base = feedforward_type == :none ? -kq * eq - kv * ev : -kq * eq - kv * ev + ad(t)
  nu_bias = SVector{3,Float64}(
      nu_preconditioning_world[1],
      nu_preconditioning_world[2],
      nu_preconditioning_world[3],
  )
  nu = nu_base + nu_bias

  norm_nu = norm(nu)

  # if t < 1e-8
  #   @show t
  #   @show eq
  #   @show ev
  #   @show nu_base
  #   @show nu_bias
  #   @show nu
  #   @show norm_nu
  #   @show feedforward_type
  # end

  if norm_nu < nu_tol
    b3 = R * e3
    Rd = R
    Ωd = zeros(3)
    Ω̇d = zeros(3)

    eR = 1/2 * vee(so3, (Rd' * R - R' * Rd))
    eΩ = Ω

    f = dot(nu, R * e3)

    u = -kR * eR - kΩ * eΩ + Ω × (𝕁 * Ω)

    return return_intermediates ? (
        f, u, Rd,
        (
            nu=nu,
            nu_base=nu_base,
            nu_bias=nu_bias,
            norm_nu=norm_nu,
            b3=b3,
            Rd=Rd,
            eR=eR,
            eΩ=eΩ,
            Ωd=Ωd,
            Ω̇d=Ω̇d,
        )
    ) : (f, u, Rd)
  end
  # @assert norm(nu) > nu_tol "nu is near-zero at t=$t: nu=$nu, eq=$eq, ev=$ev"

  # Thrust control
  f = dot(nu, R * e3)

  b3 = norm_nu < nu_tol ? R * e3 : nu/norm_nu
  a = abs(dot(b3, e1)) < 0.9 ? e1 : e2
  b1, b2, Rd = robust_frame_from_b3(b3)
  # Rotation error
  eR = 1/2 * vee(so3, (Rd' * R - R' * Rd))
  @assert !any(isnan, eR) "eR is NaN at t=$t: eR=$eR, b1=$b1, b2=$b2, b3=$b3"

  # if t < 1e-8
  #   @show b3
  #   @show R * e3
  #   @show dot(R * e3, b3)
  #   @show eR
  #   @show norm(eR)
  # end

  Ω̂ = hat(so3, Ω)
  Ṙ = R * Ω̂

  # When the preconditioner changes nu inside the ODE RHS, the exact
  # derivatives of Rd would require differentiating the preconditioning
  # correction. For the allocation experiment, treat Rd quasi-statically:
  # use the preconditioned thrust direction, but do not feed forward Ωd, Ω̇d.
  if freeze_desired_attitude_derivatives
    Ωd = zero(Ω)
    Ω̇d = zero(Ω)
    eΩ = Ω
    u = -kR * eR - kΩ * eΩ + Ω × (𝕁 * Ω)

    @assert !any(isnan, f) "f is NaN at t=$t: f=$f"
    @assert !any(isnan, u) "u is NaN at t=$t: u=$u"

    if return_intermediates
      return f, u, Rd, (
        v̇=zero(v),
        Ṙ=Ṙ,
        Ṙd=zeros(3,3),
        b3=b3,
        ḃ3=zeros(3),
        nu=nu,
        nu_base=nu_base,
        nu_bias=nu_bias,
        norm_nu=norm_nu,
        Ωd=Ωd,
        Ω̇d=Ω̇d,
      )
    else
      return f, u, Rd
    end
  end

  Ṙ = R * Ω̂
  v̇ = - Ω × v + f * e3 # from the RHS of the dynamics
  ev̇ = R*v̇ + Ṙ*v - ad(t)
  nu̇ = feedforward_type == :none ? - kq * ev - kv * ev̇ : -kq * ev - kv * ev̇ + jd(t)
  ḃ3 = norm_nu < nu_tol ? Ṙ * e3 : nu̇ /norm_nu - nu * dot(nu, nu̇)/norm_nu^3
  ḃ1 = isapprox(b3, e1; atol=atol) ? zeros(3) : (e1 × ḃ3 - b1 * dot(b1, e1 × ḃ3))/norm(e1 × b3)
  ḃ2 = ḃ3 × b1 + b3 × ḃ1
  Ṙd = reduce(hcat, [ḃ1, ḃ2, ḃ3])
  Ω̂d = Rd' * Ṙd
  Ωd = vee(so3, Ω̂d)

  # Angular velocity error
  eΩ = Ω - R' * Rd * Ωd
  @assert !any(isnan, eΩ) "eΩ is NaN at t=$t: eΩ=$eΩ, ḃ1=$ḃ1, ḃ2=$ḃ2, ḃ3=$ḃ3, eR=$eR, b1=$b1, b2=$b2, b3=$b3, |nu|=$(norm(nu)), ev=$ev, eq=$eq"

  f_dot = dot(nu̇, R * e3) + dot(nu, (R * hat(so3, Ω)) * e3)
  # NOTE: Ω̇ is given as a lagged approximation from the previous step
  v̈ = - Ω̇ × v - Ω × v̇ + f_dot * e3
  R̈ = Ṙ * Ω̂ + R * hat(so3, Ω̇)
  ev̈ = 2*Ṙ*v̇ + R*v̈ + R̈*v - jd(t)
  nü = feedforward_type == :none ? - kq * ev̇ - kv * ev̈ : -kq * ev̇ - kv * ev̈ + sd(t)
  b̈3 = norm_nu < nu_tol ? (Ṙ * Ω̂ + R * hat(so3, Ω̇)) * e3 : nü/norm_nu - (2 * nu̇ * dot(nu, nu̇) + nu * (norm(nu̇)^2 + dot(nu, nü)))/norm_nu^3 + 3 * nu * dot(nu, nu̇)^2/norm_nu^5
  b̈1 = isapprox(b3, e1; atol=atol) ? zeros(3) : (e1 × b̈3 - ḃ1 * dot(b1, e1 × ḃ3) - b1 * (dot(ḃ1, e1 × ḃ3) + dot(b1, e1 × b̈3)))/norm(e1 × b3 ) - (e1 × ḃ3 - b1 * dot(b1, e1 × ḃ3)) * dot(b1, e1 × ḃ3)/norm(e1 × b3)^2
  b̈2 = b̈3 × b1 + 2 * ḃ3 × ḃ1 + b3 × b̈1

  # @show t, q, v, Ω
  # @show qd(0.0), vd(0.0), ad(0.0)
  # @show eq, ev, nu, f
  # @show v̇, ev̇, nu̇, ḃ3
  # @show v̈, ev̈, nü, b̈3

  R̈d = reduce(hcat, [b̈1, b̈2, b̈3])
  @assert !any(isnan, R̈d) "R̈d is NaN at t=$t: R̈d=$R̈d, b̈1=$b̈1, b̈2=$b̈2, b̈3=$b̈3, v̈=$v̈, nü=$nü, ev̈=$ev̈, |nu|=$(norm(nu)), ev=$ev, eq=$eq"
  Ω̇d = vee(so3, Rd' * R̈d - Ω̂d^2)
  
  u = - kR * eR - kΩ * eΩ + Ω × (𝕁 * Ω) - 𝕁 * (hat(so3, Ω) * R' * Rd * Ωd - R' * Rd * Ω̇d)

  @assert !any(isnan, f) "f is NaN at t=$t: f=$f"
  @assert !any(isnan, u) "u is NaN at t=$t: u=$u, Ω̇=$(Ω̇), Ω̇d=$(Ω̇d)"

  # if isapprox(t, 0.0) || isapprox(t, 0.1) || isapprox(t, 0.5)
  #   @show t, norm(eR), norm(eΩ), norm(eq), norm(ev), norm(Ωd), f, nu, nu̇, b3, ḃ3
  # end

  if return_intermediates
    return f, u, Rd, (v̇=v̇, Ṙ=Ṙ, Ṙd=Ṙd, b3=b3, ḃ3=ḃ3, nu=nu, norm_nu=norm_nu)
  else
    return f, u, Rd
  end
end

function clamp_vector_norm(x, xmax)
    xmax === nothing && return x

    if x isa Number
        isfinite(x) || return zero(x)
        return abs(x) <= xmax ? x : sign(x) * xmax
    end

    all(isfinite, x) || return zero(x)

    n = norm(x)
    if !isfinite(n) || n <= 1e-14
        return zero(x)
    elseif n <= xmax
        return x
    else
        return (xmax / n) * x
    end
end

function saturate_vector(u, u_max=nothing)
    u_max === nothing && return u
    n = norm(u)
    n <= u_max || n <= 1e-15 ? u : (u_max / n) * u
end
#
# -----------------------------------------------------------------------------
# Translational output-space metric preconditioner
# -----------------------------------------------------------------------------
# This implements the R^3 output metric
#   h(q) = I - (λ^2/(1 + λ^2 ||V(q)||^2)) V(q)V(q)'
# where V(q) = corridor_flow_direction_3d(q...).
# The correction returned is Γ_h(q)(qdot,qdot) in world coordinates.
# It intentionally avoids the SE(3) atlas/local_metric/difference_tensor path for now.

function _metric_lambda()
    if isdefined(Main, :λ_metric)
        return Float64(getfield(Main, :λ_metric))
    elseif isdefined(Main, :λ)
        return Float64(getfield(Main, :λ))
    else
        return 0.0
    end
end

function _preconditioner_scale()
    if isdefined(Main, :preconditioner_scale)
        return Float64(getfield(Main, :preconditioner_scale))
    else
        # Conservative default.  Increase only after checking the attitude loop.
        return 0.02
    end
end

function _preconditioner_correction_max()
    if isdefined(Main, :preconditioner_correction_max)
        return Float64(getfield(Main, :preconditioner_correction_max))
    else
        return 0.5
    end
end

function _preconditioner_fd_step()
    if isdefined(Main, :preconditioner_fd_step)
        return Float64(getfield(Main, :preconditioner_fd_step))
    else
        return 1e-5
    end
end

function _vf_world(q)
    V = corridor_flow_direction_3d(q[1], q[2], q[3])
    V = SVector{3,Float64}(V[1], V[2], V[3])

    if !all(isfinite, V)
        return SVector{3,Float64}(0.0, 0.0, 0.0)
    end

    return V
end

function translational_metric_matrix(q; λ_metric=_metric_lambda())
    q = SVector{3,Float64}(q[1], q[2], q[3])
    V = _vf_world(q)

    β = λ_metric^2
    s = dot(V, V)

    I3 = @SMatrix [
        1.0 0.0 0.0
        0.0 1.0 0.0
        0.0 0.0 1.0
    ]

    if !isfinite(s)
        return I3
    end

    return I3 - (β / (1.0 + β * s)) * (V * V')
end

function translational_inverse_metric_matrix(q; λ_metric=_metric_lambda())
    q = SVector{3,Float64}(q[1], q[2], q[3])
    V = _vf_world(q)

    β = λ_metric^2

    I3 = @SMatrix [
        1.0 0.0 0.0
        0.0 1.0 0.0
        0.0 0.0 1.0
    ]

    return I3 + β * (V * V')
end

function translational_connection_correction(
    q,
    qdot;
    λ_metric=_metric_lambda(),
    hfd=_preconditioner_fd_step(),
)
    q = SVector{3,Float64}(q[1], q[2], q[3])
    qdot = SVector{3,Float64}(qdot[1], qdot[2], qdot[3])

    if !all(isfinite, q) || !all(isfinite, qdot)
        return SVector{3,Float64}(0.0, 0.0, 0.0)
    end

    Hinv = translational_inverse_metric_matrix(q; λ_metric=λ_metric)

    dH = zeros(3, 3, 3)

    for a in 1:3
        ea = zeros(3)
        ea[a] = 1.0
        ea = SVector{3,Float64}(ea)

        Hp = translational_metric_matrix(q + hfd * ea; λ_metric=λ_metric)
        Hm = translational_metric_matrix(q - hfd * ea; λ_metric=λ_metric)

        dH[:, :, a] .= (Hp - Hm) / (2hfd)
    end

    Γ = zeros(3, 3, 3)

    for k in 1:3, i in 1:3, j in 1:3
        Γ[k, i, j] = 0.5 * sum(
            Hinv[k, ℓ] * (dH[j, ℓ, i] + dH[i, ℓ, j] - dH[i, j, ℓ])
            for ℓ in 1:3
        )
    end

    T = zeros(3)

    for k in 1:3
        T[k] = sum(Γ[k, i, j] * qdot[i] * qdot[j] for i in 1:3, j in 1:3)
    end

    if !all(isfinite, T)
        return SVector{3,Float64}(0.0, 0.0, 0.0)
    end

    return SVector{3,Float64}(T)
end

function _maybe_print_translational_preconditioner_debug(t, q, qd_t, q̇_world, Tq, Tq_clamped, γ_pre, λ_metric)
    debug = isdefined(Main, :preconditioner_debug) ? Bool(getfield(Main, :preconditioner_debug)) : false
    debug || return nothing

    nT = norm(Tq)
    nTc = norm(Tq_clamped)
    if nT > 1.0 || nTc > 1.0
        V_now = _vf_world(q)
        V_ref = _vf_world(qd_t)
#         @printf("""
# [translational output-space preconditioner]
# t = %.6f
# λ_metric = %.6e
# γ_pre = %.6e
# q = %s
# qd = %s
# q̇_world = %s
# V(q) = %s
# ||V(q)|| = %.6e
# V(qd) = %s
# ||V(qd)|| = %.6e
# Tq = %s
# ||Tq|| = %.6e
# Tq_clamped = %s
# ||Tq_clamped|| = %.6e
# ν_bias = %s
#
# """,
#             t,
#             λ_metric,
#             γ_pre,
#             string(q),
#             string(qd_t),
#             string(q̇_world),
#             string(V_now),
#             norm(V_now),
#             string(V_ref),
#             norm(V_ref),
#             string(Tq),
#             norm(Tq),
#             string(Tq_clamped),
#             norm(Tq_clamped),
#             string(-γ_pre * Tq_clamped),
#         )
    end

    return nothing
end

function physical_control(
  t, 
  R, q, v, Ω, 
  qd, vd, ad, jd, sd,
  kR, kq, kv, kΩ; 
  mode=:euclidean, 
  feedforward_type=:none, 
  f_max=nothing,
  u_max=nothing,
  atol=1e-7,
  nu_tol=1e-7,
  Ω̇=zeros(3)
)
  if mode == :euclidean
    f_phys, u_phys, Rd, intermediates = controls_and_desired_rotation(
      t,
      R, q, v, Ω,
      qd, vd, ad, jd, sd,
      kR, kq, kv, kΩ;
      feedforward_type=feedforward_type,
      atol=atol,
      nu_tol=nu_tol,
      Ω̇=Ω̇,
      return_intermediates=true,
    )

  elseif mode == :preconditioned
    # Output-space preconditioning:
    # Compute Γ_h(q)(qdot,qdot) for the metric h on R^3, then insert it into
    # the virtual translational acceleration before Rd is built.
    # This does NOT use the full SE(3) difference_tensor path.
    q̇_world = R * v

    λ_metric = _metric_lambda()
    γ_pre = _preconditioner_scale()
    correction_max = _preconditioner_correction_max()

    # Feedback term coming from the intrinsic acceleration in the modified metric
    Tqd = translational_connection_correction(
        qd(t),
        vd(t);
        λ_metric=λ_metric,
        hfd=_preconditioner_fd_step(),
    )
    Tqd_clamped = clamp_vector_norm(Tqd, correction_max)

    # Preconditioner
    Tq = translational_connection_correction(
        q,
        q̇_world;
        λ_metric=λ_metric,
        hfd=_preconditioner_fd_step(),
    )

    Tq_clamped = clamp_vector_norm(Tq, correction_max)

    nu_preconditioning_world = γ_pre * (Tqd_clamped - Tq_clamped)

    _maybe_print_translational_preconditioner_debug(
        t,
        q,
        qd(t),
        q̇_world,
        Tq,
        Tq_clamped,
        γ_pre,
        λ_metric,
    )

    f_phys, u_phys, Rd, intermediates = controls_and_desired_rotation(
      t,
      R, q, v, Ω,
      qd, vd, ad, jd, sd,
      kR, kq, kv, kΩ;
      feedforward_type=feedforward_type,
      atol=atol,
      nu_tol=nu_tol,
      Ω̇=Ω̇,
      return_intermediates=true,
      nu_preconditioning_world=nu_preconditioning_world,
      # Keep the usual desired-attitude derivative pipeline.
      # This guarantees that when the preconditioning correction is zero
      # (λ=0 or preconditioner_scale=0), preconditioned mode reduces to the
      # Euclidean controller.  The derivatives currently ignore the time
      # derivative of nu_preconditioning_world, so for large corrections this
      # is still an approximation, but it is the correct invariance baseline.
      freeze_desired_attitude_derivatives=false,
    )

  else
      error("The kwarg `mode` must be either :euclidean or :preconditioned")
  end

  return saturate_vector(f_phys, f_max), saturate_vector(u_phys, u_max)
end
#
function closed_loop_rhs(state, parameters, t)
  # Assuming state = ArrayPartition(R, q, v, Ω), with the components of g ∈ SE(3) given by R ∈ SO(3), q ∈ ℝ³, and velocities v, Ω ∈ ℝ³
  # make sure 𝕁, so3 and e3 are either defined in the global scope, of passed as function parameters
  #
  # @assert !any(isnan, state.x[2]) "q is NaN at t=$t"
  # @assert !any(isnan, state.x[3]) "v is NaN at t=$t, v=$(state.x[3])"
  R, q, v, Ω = unpack_state(state) #state.x

  Ω̇_prev, qd, vd, ad, jd, sd, kR, kq, kv, kΩ, mode, feedforward_type, f_max, u_max, atol, nu_tol = parameters

  f, u = physical_control(
    t,
    R, q, v, Ω, 
    qd, vd, ad, jd, sd, 
    kR, kq, kv, kΩ; 
    mode=mode, 
    feedforward_type=feedforward_type, 
    f_max=f_max, 
    u_max=u_max,
    atol=atol,
    nu_tol=nu_tol,
    Ω̇=Ω̇_prev.x # De-Ref parameter
  )

  Ω̂ = SMatrix{3, 3}(hat(so3, Ω))
  
  # Dynamics
  Ṙ = R * Ω̂
  q̇ = R * v
  v̇ = - Ω × v + f * e3
  Ω̇ = inv(𝕁) * ((𝕁 * Ω) × Ω) + u 

  # Keep memory of Ω̇
  parameters[1].x = Ω̇

  return pack_state(Ṙ, q̇, v̇, Ω̇)
end
function simulate_tracking(
  T, 
  R0, q0, v0, Ω0, 
  qd, vd, ad, jd, sd,
  kR, kq, kv, kΩ; 
  mode=:euclidean, 
  feedforward_type=:none, 
  f_max=nothing,
  u_max=nothing,
  rtol=1e-7, 
  atol=1e-9, 
  nu_tol=1e-3,
  Ω̇0 = Ref(SVector{3,Float64}(0.0, 0.0, 0.0)), # Pass a reference for in-solver updates
  max_step=0.02,
)
  initial_state = pack_state(R0, q0, v0, Ω0)
  parameters = (
    Ω̇0,
    qd, vd, ad, jd, sd,
    kR, kq, kv, kΩ, 
    mode, 
    feedforward_type, 
    f_max, 
    u_max,
    atol,
    nu_tol
  )

  problem = ODEProblem(closed_loop_rhs, initial_state, (0.0, T), parameters)
  solution = solve(
    problem,
    AutoTsit5(Rodas4P(autodiff=false));
    reltol=rtol,
    abstol=atol,
    dtmax=max_step,
    save_everystep=true
  )

  return solution
end


