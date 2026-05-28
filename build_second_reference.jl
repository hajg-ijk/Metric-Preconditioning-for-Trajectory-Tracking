using GLMakie
using DifferentialEquations
using LinearAlgebra
using NPZ

# ============================================================
# 1) Maze drawing in 3D
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
        # Draw the vertical walls (from z=0 to z=2)
        lines!(ax, [p1[1], p2[1]], [p1[2], p2[2]], [0.0, 0.0], color=color, linewidth=linewidth)
        lines!(ax, [p1[1], p1[1]], [p1[2], p1[2]], [0.0, 2.0], color=color, linewidth=linewidth)
        lines!(ax, [p2[1], p2[1]], [p2[2], p2[2]], [0.0, 2.0], color=color, linewidth=linewidth)
        lines!(ax, [p1[1], p2[1]], [p1[2], p2[2]], [2.0, 2.0], color=color, linewidth=linewidth)
    end
end

# ============================================================
# 2) 3D Vector Field
# ============================================================
logistic(z) = 1.0 / (1.0 + exp(-z))

smooth_window(z, zmin, zmax, s) = logistic((z - zmin) / s) * logistic((zmax - z) / s)

function corridor_field_raw(x, y; sigma=0.22, gate=0.12, center_gain=2.5)
    w_r = exp(-((x - 0.50) / sigma)^2) * smooth_window(y, -2.05, 0.45, gate)
    w_t = exp(-((y - 0.25) / sigma)^2) * smooth_window(x, -0.90, 0.60, gate)
    w_l = exp(-((x + 0.75) / sigma)^2) * smooth_window(y, 0.05, 1.40, gate)

    v_r = [-center_gain * (x - 0.50), 1.0]
    v_t = [-1.0, -center_gain * (y - 0.25)]
    v_l = [-center_gain * (x + 0.75), 1.0]

    raw = w_r * v_r + w_t * v_t + w_l * v_l
    return raw
end

function corridor_flow_direction(x, y; sigma=0.22, gate=0.12, center_gain=2.5, eps=1e-8)
    raw = corridor_field_raw(x, y; sigma=sigma, gate=gate, center_gain=center_gain)
    n = sqrt(dot(raw, raw) + eps^2)
    return raw / n
end

# 3D version: Add a z-component (e.g., W = 0.1 * z)
function corridor_flow_direction_3d(x, y, z; sigma=0.22, gate=0.12, center_gain=2.5, eps=1e-8, z_weight=0.0)
    v_2d = corridor_flow_direction(x, y; sigma=sigma, gate=gate, center_gain=center_gain)
    W = z_weight * z  # Example: z-component scales with z
    U, V = v_2d
    norm = sqrt(U^2 + V^2 + W^2 + eps^2)
    return [U / norm, V / norm, W / norm]
end

# ============================================================
# 3) Compute 3D Vector Field on a Grid
# ============================================================
function compute_3d_vector_field(xmin, xmax, ymin, ymax, zmin, zmax; nx=10, ny=10, nz=10)
    xs = range(xmin, xmax, length=nx)
    ys = range(ymin, ymax, length=ny)
    zs = range(zmin, zmax, length=nz)

    # Preallocate flattened arrays
    X = zeros(nx * ny * nz)
    Y = zeros(nx * ny * nz)
    Z = zeros(nx * ny * nz)
    U = zeros(nx * ny * nz)
    V = zeros(nx * ny * nz)
    W = zeros(nx * ny * nz)

    idx = 1
    for x in xs, y in ys, z in zs
        X[idx] = x
        Y[idx] = y
        Z[idx] = z
        v = corridor_flow_direction_3d(x, y, z)
        U[idx] = v[1]
        V[idx] = v[2]
        W[idx] = v[3]
        idx += 1
    end

    return X, Y, Z, U, V, W
end

# ============================================================
# 4) ODE Solver for Reference Trajectory
# ============================================================
function reference_rhs!(dq, q, p, t)
    x, y, z = q
    speed, sigma, gate, center_gain = p
    v = corridor_flow_direction_3d(x, y, z; sigma=sigma, gate=gate, center_gain=center_gain)
    dq[1] = speed * v[1]
    dq[2] = speed * v[2]
    dq[3] = speed * v[3]
end

function build_reference(
    q_start=[0.50, -1.80, 1.0],
    q_goal=[-0.75, 1.25, 1.0];
    speed=0.9,
    sigma=0.22,
    gate=0.12,
    center_gain=2.5,
    T_max=8.0,
    n_samples=1200,
    out_file="inexact_reference.npz"
)
    p = [speed, sigma, gate, center_gain]
    prob = ODEProblem(reference_rhs!, q_start, (0.0, T_max), p)

    # Terminate when close to q_goal
    condition(u, t, integrator) = norm(u - q_goal) - 0.03
    affect!(integrator) = terminate!(integrator)
    cb = ContinuousCallback(condition, affect!)

    sol = solve(prob, Tsit5(), callback=cb, saveat=range(0.0, T_max, length=n_samples),
                reltol=1e-8, abstol=1e-10, maxiters=1e6)

    t_ref = sol.t
    q_ref = hcat(sol.u...)'  # Transpose to get (n_samples, 3)
    
    npzwrite(
      out_file,
      t=t_ref,
      x=q_ref[:, 1],
      y=q_ref[:, 2],
      z=q_ref[:, 3],
      q_start=q_start,
      q_goal=q_goal,
      speed=speed,
      sigma=sigma,
      gate=gate,
      center_gain=center_gain,
    )
    println("Saved reference trajectory to $out_file")
    println("T_end = $(t_ref[end]), samples = $n_samples")
    return t_ref, q_ref
end

# ============================================================
# 5) Main Function
# ============================================================
function main()
    # Compute reference trajectory
    q_start = [0.50, -1.80, 1.0]
    q_goal = [-0.75, 1.25, 1.0]
    t_ref, q_ref = build_reference(q_start, q_goal)

    # Create a GLMakie figure
    fig = Figure(size = (1000, 800))
    ax = Axis3(fig[1, 1],
               title = "3D Maze with Vector Field and Reference Trajectory",
               aspect = :data,
               xlabel = "x",
               ylabel = "y",
               zlabel = "z")

    # Plot the maze
    plot_maze_3d!(ax)

    # Plot the 3D vector field
    X, Y, Z, U, V, W = compute_3d_vector_field(-2.0, 2.0, -2.5, 2.0, 0.8, 1.2, nx=10, ny=10, nz=10)
    arrows3d!(ax, X, Y, Z, U, V, W,
              lengthscale = 0.1,  # Size of arrowheads
              color = :blue,    # Color of arrows
              alpha = 0.7,      # Transparency
              )

    # Plot the reference trajectory
    lines!(ax, q_ref[:, 1], q_ref[:, 2], q_ref[:, 3],
           color = :red,
           linewidth = 2,
           label = "Reference trajectory")

    # Mark start and goal points
    GLMakie.scatter!(ax, [q_start[1]], [q_start[2]], [q_start[3]],
             color = :green,
             markersize = 10,
             label = "Start")
    GLMakie.scatter!(ax, [q_goal[1]], [q_goal[2]], [q_goal[3]],
             color = :purple,
             markersize = 10,
             label = "Goal")

    # Add a legend
    axislegend(ax)

    # Display the figure
    display(fig)
end

# Run the main function
main()

