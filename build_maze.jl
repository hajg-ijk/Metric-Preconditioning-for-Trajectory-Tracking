using GLMakie
using DifferentialEquations
using LinearAlgebra
using Statistics
using NPZ
using Dierckx
using DifferentiationInterface
using FiniteDifferences
using OrdinaryDiffEq: Tsit5

include("vector_field.jl")

function main()
    q_start = [0.50, -1.80, 1.0]
    q_goal  = [-0.75, 1.25, 1.0]

    t_ref, q_ref = build_conditioned_reference(
        q_start,
        q_goal;
        speed=1.0,
        T_max=10.0,
        n_samples=1600,
        out_file="maze_reference.npz",
    )

    println(vector_field_conditioning_report())

    fig = Figure(size=(1000, 800))
    ax = Axis3(
        fig[1, 1],
        title="Conditioned maze vector field and reference trajectory",
        aspect=:data,
        xlabel="x",
        ylabel="y",
        zlabel="z",
    )

    plot_maze_3d!(ax)

    lines!(
        ax,
        q_ref[:, 1],
        q_ref[:, 2],
        q_ref[:, 3],
        color=:red,
        linewidth=3,
        label="Conditioned reference trajectory",
    )

    GLMakie.scatter!(
        ax,
        [q_start[1]], [q_start[2]], [q_start[3]],
        color=:green,
        markersize=12,
        label="Start",
    )

    GLMakie.scatter!(
        ax,
        [q_goal[1]], [q_goal[2]], [q_goal[3]],
        color=:purple,
        markersize=12,
        label="Goal",
    )

    axislegend(ax)
    display(fig)

    return t_ref, q_ref
end

main()
