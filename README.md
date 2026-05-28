# Metric-Preconditioning-for-Trajectory-Tracking

This is a collection of Julia scripts for running the simulations from the paper "Metric Preconditioning for Trajectory Tracking" by Jacob Goodman and Hajg Jasa.

After starting a Julia REPL from the folder with the present code, run

`]activate .`

and then 

`include("script-name")` 

to run the chosen script file.

The script `build_maze.jl` constructs a nominal trajectory from the vector field `corridor_flow_direction_3d` designed in `vector_field.jl` to navigate a pre-defined maze and with which the metric is modified.
The script saves the data as a .npz file. 
This file must be compiled once before running other files.

`build_second_reference.jl` constructs a second nominal trajectory that is not an integral curve of the original vector field `corridor_flow_direction_3d` used for modifying the original Riemannian metric.
The results are saved again as a .npz file.
This script must also be run once before running `main_gain_run.jl`.

The main file for obtaining optimal gains is `main_gain_run.jl`.
It imports the previously built nominal trajectories and simulates the dynamics written in the `dynamics.jl` file. 
The dynamics rest on the `manifold_helpers.jl` file, used to compute the Levi-Civita connection, the differentials of one-form `μ` used to modify the original Riemannian metric of `SE(3)`, and the difference tensor between the old and new Levi-Civita connections.
This is also where the inertia matrix `𝕁` is defined globally. 
The main gain script then writes the obtained gains, as well as the simulated trajectories, position and velocity errors, and control magnitudes in .csv files.
One simulation is run with a standard PD control, and another is run with a PD control plus metric preconditioning. 
In both cases, a grid search is performed over possible gain values, and the performance is evaluated on a cost function that penalizes a weighted sum of the L2 and sup norms of the controls. This is chosen with the keyword argument `gain_criterion = :control`. Other choices include `:sum`, `norm2`, and `max`, to minimize a weighted sum of the gains, the 2 norm of the vector of gains, or the maximum weighted gain.
The gain set that minimizes the chosen cost and successfully solves the tracking problem is kept and used for simulations. 
The utility functions that are used to run the gain search are contained in the file `gain_sweep.jl`.

Within the `main_gain_run.jl` script, `λ` determines the strength of the length contraction in the direction of the vector field `corridor_flow_direction_3d`. 
`R0`, `q0`, `v0`, and `Ω0` are the initial rotation, position, velocity, and attitude, respectively. 
`feedforward_type` determines the type of control scheme implemented; `:none` refers to the standard PD control, `:euclidean` refers to a more stable type where the body acceleration is added to the control. 
`mode` determines whether to add the preconditioner or not: `:euclidean` will function without a preconditioner, while `:preconditioned` will add the geometric preconditioner.
`u_max` is the control saturation threshold. `nothing` means that controls can grow unbounded. 
`kR_vals`, `kq_vals`, `kv_vals`, and `kΩ_vals` determine the domain for the gains grid search.

The script `plotting.jl` contains plotting utilities, with the defaults plotting the maze, vector field, loaded reference trajectory, simulated trajectories, position and velocity errors, as well as body torque control magnitude.
There is a function that can animate the trajectories, given .csv files from which to read the data.
It is possible to write the data to .csv files, but the default is not to do so.

The file `test_plots.jl` contains an example of how to plot simulated trajectories with fixed gains. The hard-coded gains are the ones obtained by the `main_gain_run.jl` script, but they can be changed at will. 
Optionally, one may uncomment the last section of this script to run an automated gain search (with the criteria selected with the keyword argument `gain_criterion` as above) and plot of the results.

The file `diagnostics.jl` contains some sanity checks and reference diagnostics that may come in handy when debugging.
