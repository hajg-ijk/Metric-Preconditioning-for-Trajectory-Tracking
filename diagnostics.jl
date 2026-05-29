"""
    sanity_check_exact_tracking(T, qd, vd, ad, jd, sd)
 
Run a quick closed-loop simulation for both `:euclidean` and `:preconditioned`
modes with hardcoded gains, and print position/velocity tracking errors.
Intended as a smoke test to verify the controller pipeline.
 
# Arguments
- `T::Float64`: Simulation horizon.
- `qd, vd, ad, jd, sd`: Functions `t -> ℝ³` for desired position, velocity,
  acceleration, jerk, and snap respectively.
 
# Returns
`nothing` (prints results to stdout).
"""
function sanity_check_exact_tracking(
  T, 
  qd, vd, ad, jd, sd, 
)
  kR = 8.81
  kq = 70
  kv = 24
  kΩ = 2.54

  R0 = @SMatrix [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
  q0 = SVector{3, Float64}(qd(0.0)...)
  v0 = SVector{3, Float64}(vd(0.0)...)
  Ω0 = @SVector zeros(3)

  sol_e = simulate_tracking(
    T,  
    R0, q0, v0, Ω0, 
    qd, vd, ad, jd, sd,
    kR, kq, kv, kΩ; 
    mode=:euclidean, 
    feedforward_type=:none, 
    f_max=nothing,
    u_max=nothing,
    rtol=1e-8, 
    atol=1e-8, 
    nu_tol=1e-4,
    max_step=1e-2
  )

  data_e = sample_solution(
    sol_e, 
    qd, vd, ad, jd, sd, 
    kR, kq, kv, kΩ;
    mode=:euclidean, 
    feedforward_type=:none, 
    f_max=nothing,
    u_max=nothing,
    n_samples=1000
  )

  @printf("\n=== Exact tracking sanity check ===\n")
  @printf("Solver success: %s\n", sol_e.retcode)
  @printf("Max position error: %.6e\n", maximum(data_e.pos_err))
  @printf("Max velocity error: %.6e\n", maximum(data_e.vel_err))
  @printf("Final position error: %.6e\n", data_e.pos_err[end])
  @printf("Final velocity error: %.6e\n", data_e.vel_err[end])

  sol_p = simulate_tracking(
    0.001,  
    R0, q0, v0, Ω0, 
    qd, vd, ad, jd, sd,
    kR, kq, kv, kΩ; 
    mode=:preconditioned, 
    feedforward_type=:none, 
    f_max=nothing,
    u_max=nothing,
    rtol=1e-8, 
    atol=1e-8, 
    nu_tol=1e-4,
    max_step=1e-2
  )

  data_p = sample_solution(
    sol_p, 
    qd, vd, ad, jd, sd, 
    kR, kq, kv, kΩ;
    mode=:preconditioned, 
    feedforward_type=:none, 
    f_max=nothing,
    u_max=nothing,
    n_samples=1000
  )

  @printf("\n=== Preconditioned tracking sanity check ===\n")
  @printf("Solver success: %s\n", sol_p.retcode)
  @printf("Max position error: %.6e\n", maximum(data_p.pos_err))
  @printf("Max velocity error: %.6e\n", maximum(data_p.vel_err))
  @printf("Final position error: %.6e\n", data_p.pos_err[end])
  @printf("Final velocity error: %.6e\n", data_p.vel_err[end])

end

"""
    reference_diagnostics(T, qd, vd, ad, jd, sd; n=2000, eps=1e-8, top_k=8)
    -> NamedTuple
 
Compute and print a comprehensive conditioning report for a reference
trajectory, including derivative norms, space-curve curvature, conditioning of
the desired thrust axis `b3d = ν/‖ν‖`, and conditioning of the auxiliary body
axis `b1 = normalize(e1 × b3)`.
 
# Arguments
- `T::Float64`: Trajectory duration.
- `qd, vd, ad, jd, sd`: Reference trajectory functions `t -> ℝ³`.
- `n::Int`: Number of evaluation points. Default `2000`.
- `eps::Float64`: Small regularization constant. Default `1e-8`.
- `top_k::Int`: Number of worst-case time instants to print per metric. Default `8`.
 
# Returns
`NamedTuple` with fields:
- `t`, `vnorm`, `anorm`, `jnorm`, `snorm`: Time vector and norms of derivatives.
- `curvature`, `radius`: Space-curve curvature κ and turn radius 1/κ.
- `nunorm`: Norm of the virtual acceleration `ν_ref = ad(t)`.
- `jerk_over_nu`, `snap_over_nu`, `jerk2_over_nu2`: Conditioning ratios.
- `b1_denom`, `b3_dot_e1`: Quantities measuring proximity to the `b1` singularity.
"""
function reference_diagnostics(T, qd, vd, ad, jd, sd; n=2000, eps=1e-8, top_k=8)
  ts = collect(range(0.0, T, length=n))

  vnorm = [norm(vd(t)) for t in ts]
  anorm = [norm(ad(t)) for t in ts]
  jnorm = [norm(jd(t)) for t in ts]
  snorm = [norm(sd(t)) for t in ts]

  # Curvature diagnostic for a space curve:
  # κ = ||v × a|| / ||v||^3.
  # For nearly constant speed, large κ means sharp turns.
  curvature = [
    norm(cross(vd(t), ad(t))) / max(norm(vd(t))^3, eps)
    for t in ts
  ]

  radius = [
    curvature[i] > eps ? 1.0 / curvature[i] : Inf
    for i in eachindex(curvature)
  ]

  # Nominal exact-tracking thrust/acceleration direction.
  # In driftless reduced model q̈ = f R e3, exact feedforward tracking gives ν_ref = ad(t).
  # If ||ad|| is small, b3d = ν/||ν|| is ill-conditioned.
  nunorm = anorm

  # These ratios estimate how badly derivatives of b3 = ν/||ν|| can amplify.
  # Roughly: ḃ3 contains jd / ||ad||, and b̈3 contains sd / ||ad|| plus quadratic jerk terms.
  jerk_over_nu = [
    jnorm[i] / max(nunorm[i], eps)
    for i in eachindex(ts)
  ]

  snap_over_nu = [
    snorm[i] / max(nunorm[i], eps)
    for i in eachindex(ts)
  ]

  jerk2_over_nu2 = [
    jnorm[i]^2 / max(nunorm[i]^2, eps)
    for i in eachindex(ts)
  ]

  # Check singularity of current b1 choice:
  # b1 = (e1 × b3)/||e1 × b3||.
  # This is singular when b3 ≈ +e1 OR b3 ≈ -e1.
  b1_denom = Float64[]
  b3_dot_e1 = Float64[]
  for t in ts
    ν = ad(t)
    if norm(ν) > eps
      b3 = ν / norm(ν)
      push!(b1_denom, norm(cross(e1, b3)))
      push!(b3_dot_e1, dot(e1, b3))
    else
      push!(b1_denom, NaN)
      push!(b3_dot_e1, NaN)
    end
  end

  valid_b1 = filter(!isnan, b1_denom)

  @printf("\n=== Reference diagnostics ===\n")
  @printf("T = %.6f\n", T)

  @printf("\n--- Basic derivative magnitudes ---\n")
  @printf("||vd(0)|| = %.6e\n", norm(vd(0.0)))
  @printf("Min ||vd(t)|| = %.6e\n", minimum(vnorm))
  @printf("Max ||vd(t)|| = %.6e\n", maximum(vnorm))
  @printf("Mean ||vd(t)|| = %.6e\n", mean(vnorm))

  @printf("Min ||ad(t)|| = %.6e\n", minimum(anorm))
  @printf("Max ||ad(t)|| = %.6e\n", maximum(anorm))
  @printf("Mean ||ad(t)|| = %.6e\n", mean(anorm))

  @printf("Max ||jd(t)|| = %.6e\n", maximum(jnorm))
  @printf("Mean ||jd(t)|| = %.6e\n", mean(jnorm))

  @printf("Max ||sd(t)|| = %.6e\n", maximum(snorm))
  @printf("Mean ||sd(t)|| = %.6e\n", mean(snorm))

  @printf("\n--- Geometric conditioning of qd(t) ---\n")
  @printf("Max curvature κ = %.6e\n", maximum(curvature))
  @printf("Min turn radius 1/κ = %.6e\n", minimum(radius))
  @printf("Mean curvature κ = %.6e\n", mean(curvature))

  @printf("\n--- Conditioning of b3d = ν/||ν|| with ν_ref = ad(t) ---\n")
  @printf("Min ||ν_ref|| = min ||ad(t)|| = %.6e\n", minimum(nunorm))
  @printf("Max ||jd|| / ||ν_ref|| = %.6e\n", maximum(jerk_over_nu))
  @printf("Max ||sd|| / ||ν_ref|| = %.6e\n", maximum(snap_over_nu))
  @printf("Max ||jd||^2 / ||ν_ref||^2 = %.6e\n", maximum(jerk2_over_nu2))

  if !isempty(valid_b1)
    @printf("\n--- Conditioning of b1 = normalize(e1 × b3) ---\n")
    @printf("Min ||e1 × b3|| = %.6e\n", minimum(valid_b1))
    @printf("Min |dot(e1,b3)| distance from ±e1 = %.6e\n", 1.0 - maximum(abs.(filter(!isnan, b3_dot_e1))))
  end

  function print_worst(label, values; largest=true)
    idx = sortperm(values; rev=largest)
    @printf("\nWorst %d times for %s:\n", min(top_k, length(idx)), label)
    @printf("    t          value        ||v||        ||a||        ||j||        ||s||\n")
    for k in 1:min(top_k, length(idx))
      i = idx[k]
      @printf(
        "%10.6f  %11.4e  %11.4e  %11.4e  %11.4e  %11.4e\n",
        ts[i], values[i], vnorm[i], anorm[i], jnorm[i], snorm[i]
      )
    end
  end

  print_worst("large ||sd||", snorm; largest=true)
  print_worst("large ||sd|| / ||ν_ref||", snap_over_nu; largest=true)
  print_worst("small ||ν_ref||", nunorm; largest=false)
  print_worst("large curvature", curvature; largest=true)

  return (
    t=ts,
    vnorm=vnorm,
    anorm=anorm,
    jnorm=jnorm,
    snorm=snorm,
    curvature=curvature,
    radius=radius,
    nunorm=nunorm,
    jerk_over_nu=jerk_over_nu,
    snap_over_nu=snap_over_nu,
    jerk2_over_nu2=jerk2_over_nu2,
    b1_denom=b1_denom,
    b3_dot_e1=b3_dot_e1,
  )
end
