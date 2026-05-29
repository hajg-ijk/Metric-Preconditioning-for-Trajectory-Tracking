"""
    gain_sweep(
      T, 
      R0, q0, v0, Ω0, 
      qd, vd, ad, jd, sd,
      kR_vals, kq_vals, kv_vals, kΩ_vals;
      mode=:euclidean, feedforward_type=:none,
      f_max=nothing, u_max=nothing
    )
    -> (success, score, controls)
 
Sequential (single-threaded) gain grid search. Same logic as
`parallel_gain_sweep` but also returns the sampled control signals for each
combination.
 
# Arguments
Same as `parallel_gain_sweep`.
 
# Returns
- `success::BitArray{4}`, `score::Array{Float64,4}`: As in `parallel_gain_sweep`.
- `controls::Array{Float64,5}`: Shape `(nR, nq, nv, nΩ, 100, 4)`. Each `[h,i,j,k,:,:]`
  holds the concatenated `[f, u]` signals over 100 time samples.
"""
function gain_sweep(
  T, 
  R0, q0, v0, Ω0,
  qd, vd, ad, jd, sd, 
  kR_vals, kq_vals, kv_vals, kΩ_vals;
  mode=:euclidean, 
  feedforward_type=:none, 
  f_max=nothing,
  u_max=nothing
)
  success = falses(length(kR_vals), length(kq_vals), length(kv_vals), length(kΩ_vals))
  score = fill(Inf, length(kR_vals), length(kq_vals), length(kv_vals), length(kΩ_vals))

  # The default number of samples for `data` is 100, and we concatenate f with u
  controls = zeros(length(kR_vals), length(kq_vals), length(kv_vals), length(kΩ_vals), 100, 4) 

  for (h, kR) in enumerate(kR_vals)
    for (i, kq) in enumerate(kq_vals)
      for (j, kv) in enumerate(kv_vals)
        for (k, kΩ) in enumerate(kΩ_vals)
          sol = simulate_tracking(
            T,
            R0, q0, v0, Ω0, 
            qd, vd, ad, jd, sd,
            kR, kq, kv, kΩ; 
            mode=mode, 
            feedforward_type=feedforward_type, 
            f_max=f_max,
            u_max=u_max
          )

          data = sample_solution(
            sol, 
            qd, vd, ad, jd, sd, 
            kR, kq, kv, kΩ; 
            mode=mode, 
            feedforward_type=feedforward_type, 
            f_max=f_max,
            u_max=u_max
          )

          ok, val = evaluate_tracking(
            sol, 
            qd, vd, ad, jd, sd, 
            kR, kq, kv, kΩ;
            mode=mode,
            feedforward_type=feedforward_type, 
            f_max=f_max,
            u_max=u_max
          )
          success[h, i, j, k] = ok
          score[h, i, j, k] = val

          controls[h, i, j, k, :, :] = hcat(data.f, data.u)
        end
      end
    end
  end
  return success, score, controls
end

"""
    select_minimal_successful_gain(
      success, score, controls,
      kR_vals, kq_vals, kv_vals, kΩ_vals;
      criterion=:sum, weights=(1,1,1,1),
      allow_best_failure=true
    )
    -> NamedTuple
 
From the gain sweep results, select the gain combination that is (a) successful
and (b) minimizes a user-specified cost criterion. Falls back to the
lowest-score failing combination if no successes exist and `allow_best_failure`.
 
# Arguments
- `success, score, controls`: Outputs of `gain_sweep`.
- `kR_vals, kq_vals, kv_vals, kΩ_vals`: Gain grids (for index lookup).
- `criterion::Symbol`: One of `:sum`, `:norm2`, `:max`, `:control`.
- `weights`: Relative weights `(wR, wq, wv, wΩ)` applied before the criterion.
- `allow_best_failure::Bool`: If `true`, return best failing result instead of
  erroring when no success exists.
 
# Returns
`NamedTuple` with fields `kR, kq, kv, kΩ, cost, score, h, i, j, k, criterion, successful`.
"""
function select_minimal_successful_gain(
  success, score, controls, kR_vals, kq_vals, kv_vals, kΩ_vals;
  criterion=:sum, 
  weights=(1.0, 1.0, 1.0, 1.0),
  allow_best_failure=true
)
  candidates = findall(success)

  if isempty(candidates)
    score_min = argmin(score)
    h = score_min[1]
    i = score_min[2]
    j = score_min[3]
    k = score_min[4]
    if !allow_best_failure
      error(
        "No successful gains found. Best failing gains: kR = $(kR_vals[h]), kq = $(kq_vals[i]), kv = $(kv_vals[j]), kΩ = $(kΩ_vals[k]), score = $(score[h, i, j, k])"
      )
    end
    return (
      kR=kR_vals[h], 
      kq=kq_vals[i], 
      kv=kv_vals[j], 
      kΩ=kΩ_vals[k], 
      cost=NaN, 
      score=score[h, i, j, k],
      h=h, i=i, j=j, k=k, 
      criterion="best_failure", 
      successful=false
    )
  end

  wR, wq, wv, wΩ = weights
  best = nothing
  candidates = [c.I for c in candidates]
  for (h, i, j, k) in candidates
    kR = kR_vals[h]
    kq = kq_vals[i]
    kv = kv_vals[j]
    kΩ = kΩ_vals[k]
    control_slice = controls[h, i, j, k, :, :]
    n = size(control_slice)[1]
    cost = criterion == :sum ? wR * kR + wq * kq + wv * kv + wΩ * kΩ :
      criterion == :norm2 ? norm([wR * kR, wq * kq, wv * kv, wΩ * kΩ]) :
      criterion == :max ? max(wR * kR, wq * kq, wv * kv, wΩ * kΩ) :
      criterion == :control ? norm(control_slice)^2 + 0.25 * maximum([norm(control_slice[l, :])^2 for l in 1:n]) :
      error("Criterion must be :sum, :norm2, :max, or :control")

    current = (
      cost=cost, 
      score=score[h, i, j, k], 
      kR=kR, 
      kq=kq, 
      kv=kv, 
      kΩ=kΩ,
      h=h, i=i, j=j, k=k
    )
    best = best === nothing ? current : (current.cost < best.cost ? current : best)
  end

  return (
    kR=best.kR, 
    kq=best.kq, 
    kv=best.kv, 
    kΩ=best.kΩ, 
    cost=best.cost, 
    score=best.score,
    h=best.h, 
    i=best.i, 
    j=best.j, 
    k=best.k, 
    criterion=criterion, 
    successful=true
  )
end
#
"""
    sweep_and_save_gains(
      T, 
      R0, q0, v0, Ω0, 
      qd, vd, ad, jd, sd,
      kR_vals, kq_vals, kv_vals, kΩ_vals;
      feedforward_type=:none, f_max=nothing, u_max=nothing,
      gain_criterion=:sum, gain_weights=(1,1,1,1),
      trajectory="run_"
    )
 
Run `gain_sweep` for both `:euclidean` and `:preconditioned` modes, select the
best gains, re-simulate with the selected gains, and write four CSV files per
mode: gains, trajectory (xy), and errors/controls.
 
# Arguments
- `T, R0, q0, v0, Ω0, qd, vd, ad, jd, sd`: Simulation setup.
- `kR_vals, kq_vals, kv_vals, kΩ_vals`: Gain grids.
- `feedforward_type, f_max, u_max`: Controller options.
- `gain_criterion, gain_weights`: Passed to `select_minimal_successful_gain`.
- `trajectory::String`: Prefix for output CSV filenames.
 
# Returns
`nothing` (side-effects: prints summary, creates a "data" folder and writes 
          CSV files in there).
"""
function sweep_and_save_gains(
  T, R0, q0, v0, Ω0,
  qd, vd, ad, jd, sd,
  kR_vals, kq_vals, kv_vals, kΩ_vals;
  feedforward_type=:none, f_max=nothing, u_max=nothing,
  gain_criterion=:sum, gain_weights=(1.0, 1.0, 1.0, 1.0),
  trajectory="run_"
)
  for (mode, label) in ((:euclidean, "euclidean"), (:preconditioned, "preconditioned"))
    success, score, controls = gain_sweep(
      T, R0, q0, v0, Ω0, qd, vd, ad, jd, sd,
      kR_vals, kq_vals, kv_vals, kΩ_vals;
      mode=mode, feedforward_type=feedforward_type, f_max=f_max, u_max=u_max
    )

    @printf("\n[%s] %s successes: %d/%d\n", trajectory, label, sum(success), length(success))

    best_idx = argmin(score)
    selected = select_minimal_successful_gain(
      success, score, controls, kR_vals, kq_vals, kv_vals, kΩ_vals;
      criterion=gain_criterion, weights=gain_weights
    )

    @printf("[%s] Best score: %.6f at kR=%.4f, kq=%.4f, kv=%.4f, kΩ=%.4f\n",
      trajectory, score[best_idx],
      kR_vals[best_idx[1]], kq_vals[best_idx[2]], kv_vals[best_idx[3]], kΩ_vals[best_idx[4]])
    @printf("[%s] Selected:   kR=%.4f, kq=%.4f, kv=%.4f, kΩ=%.4f, cost=%.4f, score=%.6f\n",
      trajectory, selected.kR, selected.kq, selected.kv, selected.kΩ, selected.cost, selected.score)

    df = DataFrame(
      type  = ["best_score",               "selected"],
      kR    = [kR_vals[best_idx[1]],        selected.kR],
      kq    = [kq_vals[best_idx[2]],        selected.kq],
      kv    = [kv_vals[best_idx[3]],        selected.kv],
      kΩ    = [kΩ_vals[best_idx[4]],        selected.kΩ],
      score = [score[best_idx],             selected.score],
      cost  = [NaN,                         selected.cost],
    )
    cd(@__DIR__)
    mkpath("data")
    CSV.write("data/$(trajectory)_$(label)_gains.csv", df)

    solution = simulate_tracking(
      T,
      R0, q0, v0, Ω0, 
      qd, vd, ad, jd, sd,
      selected.kR, selected.kq, selected.kv, selected.kΩ; 
      mode=mode, 
      feedforward_type=feedforward_type, 
      f_max=f_max,
      u_max=u_max
    )

    data = sample_solution(
      solution, 
      qd, vd, ad, jd, sd, 
      selected.kR, selected.kq, selected.kv, selected.kΩ; 
      mode=mode, 
      feedforward_type=feedforward_type, 
      f_max=f_max,
      u_max=u_max
    )

    CSV.write("data/$(trajectory)_$(label)_trajectory.csv", DataFrame(
      q_ref_x = data.q_ref[:, 1],
      q_ref_y = data.q_ref[:, 2],
      q_x   = data.q[:, 1],
      q_y   = data.q[:, 2],
    ))

    CSV.write("data/$(trajectory)_$(label)_errors_control.csv", DataFrame(
      t         = data.t,
      pos_err = data.pos_err,
      vel_err = data.vel_err,
      f  = data.f,
      u_norm  = data.u_norm,
    ))
  end
end
