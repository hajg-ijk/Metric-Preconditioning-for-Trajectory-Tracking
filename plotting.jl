"""
    plot_tracking_comparison(
      T, 
      R0, q0, v0, Ω0,
      qd, vd, ad, jd, sd,
      kR_e, kq_e, kv_e, kΩ_e,
      kR_p, kq_p, kv_p, kΩ_p;
      feedforward_type=:none, f_max=nothing, u_max=nothing,
      xmin=-2, xmax=2, ymin=-2.5, ymax=2, zmin=0, zmax=2,
      show_maze=true, show_μ_field=true,
      trajectory="exact_", write=true
    )
    -> (sol_e, sol_p, data_e, data_p)
 
Simulate both Euclidean and preconditioned controllers, produce a 4-panel
GLMakie figure (trajectory, position error, control magnitude, velocity error),
and optionally write trajectory and error CSVs.
 
# Arguments
- `T, R0, q0, v0, Ω0`: Simulation horizon and initial conditions.
- `qd, vd, ad, jd, sd`: Reference callables.
- `kR_e, kq_e, kv_e, kΩ_e`: Euclidean controller gains.
- `kR_p, kq_p, kv_p, kΩ_p`: Preconditioned controller gains.
- `feedforward_type, f_max, u_max`: Passed to `simulate_tracking`.
- `xmin, xmax, ymin, ymax, zmin, zmax`: Vector field plot extent.
- `show_maze::Bool`: Draw the maze walls.
- `show_μ_field::Bool`: Overlay the corridor vector field arrows.
- `trajectory::String`: Prefix for output CSV filenames.
- `write::Bool`: Whether to write CSV output files.
 
# Returns
- `sol_e, sol_p`: ODE solutions for Euclidean and preconditioned controllers.
- `data_e, data_p`: Sampled data NamedTuples from `sample_solution`.
"""
function plot_tracking_comparison(
  T,
  R0, q0, v0, Ω0,
  qd, vd, ad, jd, sd,
  kR_e, kq_e, kv_e, kΩ_e,
  kR_p, kq_p, kv_p, kΩ_p;
  feedforward_type=:none, 
  f_max=nothing,
  u_max=nothing,
  xmin=-2.0, 
  xmax=2.0, 
  ymin=-2.5, 
  ymax=2.0, 
  zmin=0.0, 
  zmax=2.0,
  show_maze=true, 
  show_μ_field=true,
  trajectory="exact_",
  write=true
)
  # Simulate both controllers
  # No preconditioning
  sol_e = simulate_tracking(
    T,
    R0, q0, v0, Ω0, 
    qd, vd, ad, jd, sd,
    kR_e, kq_e, kv_e, kΩ_e; 
    mode=:euclidean, 
    feedforward_type=feedforward_type, 
    f_max=f_max,
    u_max=u_max
  )
  # Preconditioned solution
  sol_p = simulate_tracking(
    T,
    R0, q0, v0, Ω0, 
    qd, vd, ad, jd, sd,
    kR_p, kq_p, kv_p, kΩ_p; 
    mode=:preconditioned, 
    feedforward_type=feedforward_type, 
    f_max=f_max,
    u_max=u_max
  )

  data_e = sample_solution(
    sol_e, 
    qd, vd, ad, jd, sd, 
    kR_e, kq_e, kv_e, kΩ_e;
    mode=:euclidean, 
    feedforward_type=feedforward_type, 
    f_max=f_max,
    u_max=u_max
  )
  data_p = sample_solution(
    sol_p, 
    qd, vd, ad, jd, sd, 
    kR_p, kq_p, kv_p, kΩ_p;
    mode=:preconditioned, 
    feedforward_type=feedforward_type, 
    f_max=f_max,
    u_max=u_max
  )

  # Create figure
  fig = Figure(size=(1600, 500))
  axs = [
    Axis(
      fig[1, i], 
      title=i == 1 ? "Tracking in the Maze" :
            i == 2 ? "Position Tracking Error" :
            i == 3 ? "Control Magnitude" :
                    "Velocity Tracking Error"
      # aspect=i == 1 ? :data : :auto
    ) for i in 1:4
  ]

  # Plot maze and vector field (3D)
  ax = axs[1]
  show_maze && plot_maze_3d!(ax)
  if show_μ_field
    # Compute and plot vector field (at z=1 for visibility)
    nx, ny, nz = 10, 10, 1
    xs = range(xmin, xmax, length=nx)
    ys = range(ymin, ymax, length=ny)
    zs = [1.0]  # Plot at z=1
    X = [x for x in xs, y in ys, z in zs]
    Y = [y for x in xs, y in ys, z in zs]
    Z = [z for x in xs, y in ys, z in zs]
    U = zeros(size(X)...)
    V = zeros(size(X)...)
    W = zeros(size(X)...)
    for (i, x) in enumerate(xs), (j, y) in enumerate(ys), (k, z) in enumerate(zs)
      v = corridor_flow_direction_3d(x, y, z)
      U[i, j, k] = v[1]
      V[i, j, k] = v[2]
      W[i, j, k] = v[3]
    end
    GLMakie.arrows3d!(
      ax, 
      vec(X), 
      vec(Y), 
      vec(Z), 
      vec(U), 
      vec(V), 
      vec(W),
      lengthscale=0.1, 
      color=:blue, 
      alpha=0.5
    )
    write && CSV.write("vector_field.csv", DataFrame(
      x = vec(X),
      y = vec(Y),
      u = vec(U),
      v = vec(V)
    ))
  end

  # Plot trajectories (project to xy-plane for clarity)
  GLMakie.lines!(
    ax, 
    data_e.q_ref[:, 1], 
    data_e.q_ref[:, 2], 
    fill(zmin, size(data_e.q_ref, 1)),
    color=:black, 
    linestyle=:dash, 
    linewidth=2, 
    label="Desired trajectory"
  )
  GLMakie.lines!(
    ax, 
    data_e.q[:, 1], 
    data_e.q[:, 2], 
    fill(1.0, size(data_e.q, 1)),
    color=:red, 
    linewidth=2, 
    label="Euclidean PD: kR=$(kR_e), kq=$(kq_e), kv=$(kv_e), kΩ=$(kΩ_e)"
  )
  GLMakie.lines!(
    ax, 
    data_p.q[:, 1], 
    data_p.q[:, 2], 
    fill(1.0, size(data_p.q, 1)),
    color=:blue, 
    linewidth=2, 
    label="Preconditioned PD: kR=$(kR_p), kq=$(kq_p), kv=$(kv_p), kΩ=$(kΩ_p)"
  )
  GLMakie.scatter!(
    ax, 
    [q0[1]], 
    [q0[2]], 
    [1.0], 
    color=:black, 
    markersize=10, 
    label="Initial state"
  )

  ax.xlabel = "x"
  ax.ylabel = "y"
  # ax.zlabel = "z"
  GLMakie.axislegend(ax)

  # Position error plot
  ax = axs[2]
  GLMakie.lines!(
    ax, 
    data_e.t, 
    data_e.pos_err, 
    color=:red, 
    linewidth=2,
    label="Euclidean PD"
  )
  GLMakie.lines!(
    ax, 
    data_p.t, 
    data_p.pos_err, 
    color=:blue, 
    linewidth=2,
    label="Preconditioned PD"
  )
  ax.xlabel = "t"
  ax.ylabel = "||q - q_d||"
  GLMakie.axislegend(ax)

  # Control magnitude plot
  ax = axs[3]
  GLMakie.lines!(
    ax, 
    data_e.t, 
    data_e.u_norm, 
    color=:red, 
    linewidth=2,
    label="Euclidean PD"
  )
  GLMakie.lines!(
    ax, 
    data_p.t, 
    data_p.u_norm, 
    color=:blue, 
    linewidth=2,
    label="Preconditioned PD"
  )
  ax.xlabel = "t"
  ax.ylabel = "||u||"
  GLMakie.axislegend(ax)
  #
  # Velocity error plot
  ax = axs[4]
  GLMakie.lines!(
    ax, 
    data_e.t, 
    data_e.vel_err, 
    color=:red, 
    linewidth=2,
    label="Euclidean PD"
  )
  GLMakie.lines!(
    ax, 
    data_p.t, 
    data_p.vel_err, 
    color=:blue, 
    linewidth=2,
    label="Preconditioned PD"
  )
  ax.xlabel = "t"
  ax.ylabel = "||R*v - v_d||"
  GLMakie.axislegend(ax)

  write && CSV.write(trajectory * "trajectory.csv", DataFrame(
    q_ref_x = data_e.q_ref[:, 1],
    q_ref_y = data_e.q_ref[:, 2],
    q_e_x   = data_e.q[:, 1],
    q_e_y   = data_e.q[:, 2],
    q_p_x   = data_p.q[:, 1],
    q_p_y   = data_p.q[:, 2]
  ))

  write && CSV.write(trajectory * "errors_controls.csv", DataFrame(
    t         = data_e.t,
    pos_err_e = data_e.pos_err,
    pos_err_p = data_p.pos_err,
    u_norm_e  = data_e.u_norm,
    u_norm_p  = data_p.u_norm,
    vel_err_e = data_e.vel_err,
    vel_err_p = data_p.vel_err,
  ))

  display(fig)
  return sol_e, sol_p, data_e, data_p
end
#
"""
    auto_select_and_plot_tracking_comparison(
      T, 
      R0, q0, v0, Ω0, 
      qd, vd, ad, jd, sd,
      kR_vals, kq_vals, kv_vals, kΩ_vals;
      feedforward_type=:none, f_max=nothing,
      u_max=nothing, gain_criterion=:sum,
      gain_weights=(1,1,1,1),
      xmin, xmax, ymin, ymax, zmin, zmax,
      show_maze=true, show_μ_field=true,
      trajectory="exact_", write=false
    )
    -> NamedTuple
 
Run `gain_sweep` for both modes, select the best gains via
`select_minimal_successful_gain`, print a summary, and call
`plot_tracking_comparison` with the selected gains.
 
# Arguments
- `T, R0, q0, v0, Ω0, qd, vd, ad, jd, sd`: Simulation setup.
- `kR_vals, kq_vals, kv_vals, kΩ_vals`: Gain grids.
- `feedforward_type, f_max, u_max, gain_criterion, gain_weights`: Controller options.
- `xmin, xmax, ymin, ymax, zmin, zmax, show_maze, show_μ_field, trajectory, write`:
  Forwarded to `plot_tracking_comparison`.
 
# Returns
`NamedTuple` with fields `euclidean, preconditioned, success_e, success_p,
score_e, score_p`.
"""
function auto_select_and_plot_tracking_comparison(
  T,
  R0, q0, v0, Ω0, 
  qd, vd, ad, jd, sd,
  kR_vals, kq_vals, kv_vals, kΩ_vals;
  feedforward_type=:none, 
  f_max=nothing,
  u_max=nothing,
  gain_criterion=:sum, 
  gain_weights=(1.0, 1.0, 1.0, 1.0),
  xmin=-2.0, 
  xmax=2.0, 
  ymin=-2.5, 
  ymax=2.0, 
  zmin=0.0, 
  zmax=2.0,
  show_maze=true, 
  show_μ_field=true,
  trajectory="exact_",
  write=false
)
  # Euclidean PD sweep
  success_e, score_e = gain_sweep(
    T,
    R0, q0, v0, Ω0, 
    qd, vd, ad, jd, sd, 
    kR_vals, kq_vals, kv_vals, kΩ_vals;
    mode=:euclidean, 
    feedforward_type=feedforward_type, 
    f_max=f_max,
    u_max=u_max
  )

  # Preconditioned PD sweep
  success_p, score_p = gain_sweep(
    T,
    R0, q0, v0, Ω0, 
    qd, vd, ad, jd, sd, 
    kR_vals, kq_vals, kv_vals, kΩ_vals;
    mode=:preconditioned, 
    feedforward_type=feedforward_type, 
    f_max=f_max,
    u_max=u_max
  )

  @printf("\nEuclidean successes: %d/%d\n", sum(success_e), length(success_e))
  @printf("Preconditioned successes: %d/%d\n", sum(success_p), length(success_p))

  min_e = argmin(score_e)
  h_e = min_e[1] 
  i_e = min_e[2] 
  j_e = min_e[3] 
  k_e = min_e[4] 
  min_p = argmin(score_p)
  h_p = min_p[1] 
  i_p = min_p[2] 
  j_p = min_p[3] 
  k_p = min_p[4] 
  @printf(
    "Best Euclidean score: %.6f at kR=%.4f, kq=%.4f, kv=%.4f, kΩ=%.4f\n", 
    score_e[h_e, i_e, j_e, k_e], 
    kR_vals[h_e], 
    kq_vals[i_e], 
    kv_vals[j_e],
    kΩ_vals[k_e],
  )
  @printf(
    "Best Preconditioned score: %.6f at kR=%.4f, kq=%.4f, kv=%.4f, kΩ=%.4f\n", 
    score_p[h_p, i_p, j_p, k_p], 
    kR_vals[h_p],
    kq_vals[i_p], 
    kv_vals[j_p],
    kΩ_vals[k_p],
  )

  choice_e = select_minimal_successful_gain(
    success_e, score_e, 
    kR_vals, kq_vals, kv_vals, kΩ_vals;
    criterion=gain_criterion, 
    weights=gain_weights
  )
  choice_p = select_minimal_successful_gain(
    success_p, score_p, 
    kR_vals, kq_vals, kv_vals, kΩ_vals;
    criterion=gain_criterion, 
    weights=gain_weights
  )

  @printf("\nSelected gains from successful grid points:\n")
  @printf(
    "  Euclidean PD: kR = %.4f, kq = %.4f, kv = %.4f, kΩ = %.4f, cost = %.4f, score = %.6f\n",
    choice_e.kR,       
    choice_e.kq, 
    choice_e.kv, 
    choice_e.kΩ, 
    choice_e.cost, 
    choice_e.score
  )
  @printf(
    "  Preconditioned PD: kR = %.4f, kq = %.4f, kv = %.4f, kΩ = %.4f, cost = %.4f, score = %.6f\n",
    choice_p.kR,      
    choice_p.kq, 
    choice_p.kv, 
    choice_p.kΩ,
    choice_p.cost, 
    choice_p.score
  )

  if choice_p.cost < choice_e.cost
    @printf("  -> Preconditioned PD achieved the smallest minimal gain cost on this grid.\n")
  elseif choice_p.cost > choice_e.cost
    @printf("  -> Euclidean PD achieved the smallest minimal gain cost on this grid.\n")
  else
    @printf("  -> The two controllers tied on the chosen gain cost criterion.\n")
  end

  plot_tracking_comparison(
    T, 
    R0, q0, v0, Ω0,
    qd, vd, ad, jd, sd,
    choice_e.kR, choice_e.kq, choice_e.kv, choice_e.kΩ,
    choice_p.kR, choice_p.kq, choice_p.kv, choice_p.kΩ;
    feedforward_type=feedforward_type, 
    f_max=f_max,
    u_max=u_max,
    xmin=xmin, 
    xmax=xmax, 
    ymin=ymin, 
    ymax=ymax, 
    zmin=zmin, 
    zmax=zmax,
    show_maze=show_maze, 
    show_μ_field=show_μ_field,
    trajectory=trajectory,
    write=write
  )

  return (
    euclidean=choice_e, 
    preconditioned=choice_p,
    success_e=success_e, 
    success_p=success_p,
    score_e=score_e, 
    score_p=score_p
  )
end
#
"""
    plot_gain_sweep_comparison(
      T, 
      R0, q0, v0, Ω0, 
      qd, vd, ad, jd, sd,
      kR_vals, kq_vals, kv_vals, kΩ_vals;
      feedforward_type=:none, f_max=nothing, u_max=nothing,
      gain_criterion=:sum, gain_weights=(1,1,1,1)
    )
 
Run gain sweeps for both modes and display a 3-panel GLMakie heatmap figure
showing: Euclidean success, preconditioned success, and their difference, as
functions of `kv` and `kq`.
 
# Returns
`nothing` (displays figure).
"""
function plot_gain_sweep_comparison(
  T,
  R0, q0, v0, Ω0, 
  qd, vd, ad, jd, sd,
  kR_vals, kq_vals, kv_vals, kΩ_vals;
  feedforward_type=:none, 
  f_max=nothing,
  u_max=nothing,
  gain_criterion=:sum, 
  gain_weights=(1.0, 1.0, 1.0, 1.0)
)
  success_e, score_e = gain_sweep(
    T,
    R0, q0, v0, Ω0, 
    qd, vd, ad, jd, sd, 
    kR_vals, kq_vals, kv_vals, kΩ_vals;
    mode=:euclidean, 
    feedforward_type=feedforward_type, 
    f_max=f_max,
    u_max=u_max
  )
  success_p, score_p = gain_sweep(
    T,
    R0, q0, v0, Ω0, 
    qd, vd, ad, jd, sd, 
    kR_vals, kq_vals, kv_vals, kΩ_vals;
    mode=:preconditioned, 
    feedforward_type=feedforward_type, 
    f_max=f_max,
    u_max=u_max
  )

  choice_e = select_minimal_successful_gain(
    success_e, score_e, 
    kR_vals, kq_vals, kv_vals, kΩ_vals;
    criterion=gain_criterion, 
    weights=gain_weights
  )
  choice_p = select_minimal_successful_gain(
    success_p, score_p, 
    kR_vals, kq_vals, kv_vals, kΩ_vals;
    criterion=gain_criterion, 
    weights=gain_weights
  )

  diff = Int.(success_p) .- Int.(success_e)
  # TODO: kR and kΩ too
  extent = (kv_vals[1], kv_vals[end], kq_vals[1], kq_vals[end])

  fig = Figure(size=(1400, 450))
  axs = [
    Axis(
      fig[1, i], 
      title=i == 1 ? "Euclidean PD success" :
            i == 2 ? "Preconditioned PD success" :
                     "Preconditioned - Euclidean"
    ) for i in 1:3
  ]

  GLMakie.heatmap!(axs[1], kv_vals, kq_vals, success_e, colormap=:binary, colorrange=(0, 1))
  GLMakie.scatter!(axs[1], [choice_e.kv], [choice_e.kq], marker=:xcross, color=:red, markersize=20)
  axs[1].xlabel = "kv"
  axs[1].ylabel = "kq"

  GLMakie.heatmap!(axs[2], kv_vals, kq_vals, success_p, colormap=:binary, colorrange=(0, 1))
  GLMakie.scatter!(axs[2], [choice_p.kv], [choice_p.kq], marker=:xcross, color=:red, markersize=20)
  axs[2].xlabel = "kv"
  axs[2].ylabel = "kq"

  GLMakie.heatmap!(axs[3], kv_vals, kq_vals, diff, colormap=:balance, colorrange=(-1, 1))
  axs[3].xlabel = "kv"
  axs[3].ylabel = "kq"

  display(fig)
end

"""
    animate_trajectories(euclidean_file, preconditioned_file;
                          output="trajectory.gif", framerate=90, sleep_time=nothing)
 
Read two trajectory CSV files (columns `q_x, q_y, q_ref_x, q_ref_y`) and
produce an animation of the two controlled trajectories against the reference.
 
# Arguments
- `euclidean_file, preconditioned_file::String`: Paths to CSV files.
- `output::String`: Output GIF filename.
- `framerate::Int`: Frames per second for the GIF.
- `sleep_time`: If not `nothing`, display interactively with `sleep` between
  frames instead of recording.
 
# Returns
`nothing`.
"""
function animate_trajectories(
  euclidean_file, preconditioned_file;
  output="trajectory.gif", framerate=90, sleep_time=nothing
)

    df_euc  = CSV.read(euclidean_file, DataFrame)
    df_pre  = CSV.read(preconditioned_file, DataFrame)

    n = min(nrow(df_euc), nrow(df_pre))

    fig = Figure()
    ax  = Axis(fig[1,1], aspect=DataAspect())

    lines!(ax, df_euc.q_ref_x, df_euc.q_ref_y, color=:black, linewidth=2, label="reference")

    idx = Observable(1)

    for (df, col, pcol, lab) in [
        (df_euc, :red, :red,     "euclidean"),
        (df_pre, :blue, :blue, "preconditioned"),
    ]
        tx = @lift df.q_x[1:$idx]
        ty = @lift df.q_y[1:$idx]
        px = @lift [df.q_x[$idx]]
        py = @lift [df.q_y[$idx]]
        lines!(ax, tx, ty, color=col, linewidth=1.5, label=lab)
        GLMakie.scatter!(ax, px, py, color=pcol, markersize=10)
    end

    axislegend(ax)

    if isnothing(sleep_time)
        record(fig, output, 1:n; framerate=framerate) do i
            idx[] = i
        end
    else
        display(fig)
        for i in 1:n
            idx[] = i
            sleep(sleep_time)
        end
    end
end
