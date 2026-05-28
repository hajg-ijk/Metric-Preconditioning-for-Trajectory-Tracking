function sample_solution(
  sol, 
  qd, vd, ad, jd, sd, 
  kR, kq, kv, kΩ;
  mode=:euclidean, 
  feedforward_type=:none, 
  f_max=nothing,
  u_max=nothing,
  n_samples=100
)
  if !(SciMLBase.successful_retcode(sol))
    error("ODE solve failed: $(sol.retcode)")
  end

  t = range(sol.t[1], sol.t[end], length=n_samples)
  Z = [unpack_state(sol(ti)) for ti in t]
  
  R = [z[1] for z in Z]
  q = reduce(hcat, [z[2] for z in Z])'
  v = reduce(hcat, [z[3] for z in Z])'
  Ω = reduce(hcat, [z[4] for z in Z])'

  q_ref = zeros(size(q))
  v_ref = zeros(size(v))
  a_ref = zeros(size(v))
  j_ref = zeros(size(v))
  s_ref = zeros(size(v))
  f = zeros(length(t))
  u = zeros(size(v))

  for (i, ti) in enumerate(t)
    q_ref[i, :] = SVector{3}(qd(ti))
    v_ref[i, :] = SVector{3}(vd(ti))
    a_ref[i, :] = SVector{3}(ad(ti))
    j_ref[i, :] = SVector{3}(jd(ti))
    s_ref[i, :] = SVector{3}(sd(ti))

    f[i], u[i, :] = physical_control(
      ti, 
      R[i], q[i, :], v[i, :], Ω[i, :],
      qd, vd, ad, jd, sd,
      kR, kq, kv, kΩ; 
      mode=mode, 
      feedforward_type=feedforward_type, 
      f_max=f_max,
      u_max=u_max
    )
  end

  pos_err = [norm(q[i, :] - q_ref[i, :]) for i in 1:size(q, 1)]
  vel_err = [norm(R[i] * v[i, :] - v_ref[i, :]) for i in 1:size(v, 1)]
  u_norm = [norm(u[i, :]) for i in 1:size(u, 1)]

  return (
    t=t, 
    q=q, 
    v=v, 
    q_ref=q_ref, 
    v_ref=v_ref, 
    a_ref=a_ref, 
    j_ref=j_ref, 
    s_ref=s_ref,
    f=f,
    u=u,
    pos_err=pos_err, 
    vel_err=vel_err, 
    u_norm=u_norm
  )
end

function weighted_rms(values, t; power=2.0, eps=0.05)
  T = t[end] - t[1]
  T <= 0 && return sqrt(mean(values.^2))

  tau = (t .- t[1]) / T
  w = eps .+ tau.^power

  # Trapezoidal rule approximation
  num = sum(w[1:end-1] .* values[1:end-1].^2 .* diff(t))    
  den = sum(w[1:end-1] .* diff(t))
  return sqrt(num / den)
end

function evaluate_tracking(
  sol, 
  qd, vd, ad, jd, sd, 
  kR, kq, kv, kΩ;
  mode=:euclidean, 
  feedforward_type=:none, 
  f_max=nothing,
  u_max=nothing,
  final_pos_tol=0.08, 
  final_vel_tol=0.10,
  weighted_rms_pos_tol=0.20, 
  weighted_rms_vel_tol=0.25,
  weight_power=2.0, 
  weight_eps=0.05
)
  if !(SciMLBase.successful_retcode(sol))
      return false, Inf
  end

  data = sample_solution(
    sol, 
    qd, vd, ad, jd, sd, 
    kR, kq, kv, kΩ;
    mode=mode, 
    feedforward_type=feedforward_type, 
    f_max=f_max,
    u_max=u_max,
    n_samples=100
  )

  t = data.t
  pos_err = data.pos_err
  vel_err = data.vel_err

  final_pos = pos_err[end]
  final_vel = vel_err[end]

  wrms_pos = weighted_rms(pos_err, t; power=weight_power, eps=weight_eps)
  wrms_vel = weighted_rms(vel_err, t; power=weight_power, eps=weight_eps)

  success = (
    isfinite(final_pos) && isfinite(final_vel) &&
    isfinite(wrms_pos) && isfinite(wrms_vel) &&
    final_pos <= final_pos_tol && final_vel <= final_vel_tol &&
    wrms_pos <= weighted_rms_pos_tol && wrms_vel <= weighted_rms_vel_tol
  )

  score = 2.0 * wrms_pos + 1.0 * wrms_vel + 2.0 * final_pos + 1.0 * final_vel

  return success, score
end
