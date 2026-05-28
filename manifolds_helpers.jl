# For SE(3)
function make_SE3_point(R, q)
  return vcat(hcat(R, q), [0.0 0.0 0.0 1.0])
end
# Overload `identity_element`
function LieGroups.identity_element(::LieGroup{ℝ, LeftSemidirectProductGroupOperation{MatrixMultiplicationGroupOperation, AdditionGroupOperation, LeftMultiplicationGroupAction, ActionActsOnRight}, ProductManifold{ℝ, Tuple{Rotations{ManifoldsBase.TypeParameter{Tuple{3}}}, Euclidean{ℝ, ManifoldsBase.TypeParameter{Tuple{3}}}}}}, ::Type{Int64})
  return identity_element(SE3, Matrix{Int64})
end
function LieGroups.identity_element(::LieGroup{ℝ, LeftSemidirectProductGroupOperation{MatrixMultiplicationGroupOperation, AdditionGroupOperation, LeftMultiplicationGroupAction, ActionActsOnRight}, ProductManifold{ℝ, Tuple{Rotations{ManifoldsBase.TypeParameter{Tuple{3}}}, Euclidean{ℝ, ManifoldsBase.TypeParameter{Tuple{3}}}}}})
  return identity_element(SE3, Matrix{Float64})
end
# Fix Manifolds.jl bug? This at least makes it so that 
# the functions get_parameters and get_coordinates_induced_basis 
# produce a reasonable output
function Manifolds.submanifold_components(p::Matrix)
  return ManifoldsBase.submanifold_components(SE3, p)
end
function Manifolds.get_parameters(SE3::AbstractLieGroup, Atlas_SE3::RetractionAtlas, chart_index_SE3, g::ArrayPartition)
  R = g[SE3, :Rotation]
  q = g[SE3, :Translation]
  g0 = vcat(hcat(R, q), [0.0 0.0 0.0 1.0])
  return get_parameters(SE3, Atlas_SE3, chart_index_SE3, g0)
end
#
# Inertia matrix, set globally 
global 𝕁 = SMatrix{3, 3, Float64}(diagm([0.082, 0.0845, 0.1377]))
#
# NOTE: the following `inner` function, and all subsequent ones, depend on the global 𝕁
function LieGroups.inner(se3::LieAlgebra{ℝ, <:LieGroups.SpecialEuclideanGroupOperation, <:SpecialEuclideanGroup}, Xe, Ye; J=𝕁)
  so3 = LieAlgebra(SpecialOrthogonalGroup(3))
  Ω1 = vee(so3, Xe[se3, :Rotation]) 
  v1 = Xe[se3, :Translation]
  Ω2 = vee(so3, Ye[se3, :Rotation]) 
  v2 = Ye[se3, :Translation]
  return Ω1' * J * Ω2 + v1' * v2
end
function Manifolds.inner(SE3::SpecialEuclideanGroup, g, Xg, Yg; J=𝕁)
  # Pull back the vectors to the identity_element
  Xe = diff_left_compose(SE3, g, inv(SE3, g), Xg)
  Ye = diff_left_compose(SE3, g, inv(SE3, g), Yg)
  return LieGroups.inner(LieAlgebra(SE3), Xe, Ye; J=J)
end
function Manifolds.inner(SE3::SpecialEuclideanGroup, A::AbstractAtlas, chart_index, g_coords, Xg_coords, Yg_coords; J=𝕁)
  ib = induced_basis(SE3, A, chart_index)
  g = get_point(SE3, A, chart_index, g_coords)
  Xg = get_vector(SE3, g, Xg_coords, ib) 
  Yg = get_vector(SE3, g, Yg_coords, ib) 
  return Manifolds.inner(SE3, g, Xg, Yg; J=J)
end
#
function levi_civita_connection_analytical(SE3, g, X, Y; J=𝕁)
  # Step 1: Left-translate g, X, Y to the identity
  g_inv = inv(SE3, g)
  # X_e = g_inv * X  # Left-translated X at e (still in se(3))
  # Y_e = g_inv * Y  # Left-translated Y at e

  X_e = diff_left_compose(SE3, g, g_inv, X)
  Y_e = diff_left_compose(SE3, g, g_inv, Y)

  # Step 2: Extract angular/translational parts at e
  Ω_X = vee(so3, X_e[se3, :Rotation])
  v_X = X_e[se3, :Translation]
  Ω_Y = vee(so3, Y_e[se3, :Rotation])
  v_Y = Y_e[se3, :Translation]

  # Step 3: Compute connection at identity (e)
  # For SE(3), the Levi-Civita connection at e is:
  # ∇_X Y = (1/2)[X, Y] + (1/2)(ad_X^† Y + ad_Y^† X)
  # where:
  #   [X, Y] = (Ω_X × Ω_Y, Ω_X × v_Y - Ω_Y × v_X)  (Lie bracket)
  #   ad_X^† Y = (J⁻¹(J Ω_Y × Ω_X + v_Y × (J Ω_X) - v_X × (J Ω_Y)), v_X × Ω_Y)

  # Lie bracket part
  Ω_bracket = Ω_X × Ω_Y
  v_bracket = Ω_X × v_Y - Ω_Y × v_X

  # ad_X^† Y part
  Ω_ad_XY = J \ (J * Ω_Y × Ω_X + v_Y × (J * Ω_X) - v_X × (J * Ω_Y))
  v_ad_XY = v_X × Ω_Y

  # ad_Y^† X part
  Ω_ad_YX = J \ (J * Ω_X × Ω_Y + v_X × (J * Ω_Y) - v_Y × (J * Ω_X))
  v_ad_YX = v_Y × Ω_X

  # Sum at identity
  Ω_rot_e = 0.5 * Ω_bracket + 0.5 * (Ω_ad_XY + Ω_ad_YX)
  v_trans_e = 0.5 * v_bracket + 0.5 * (v_ad_XY + v_ad_YX)

  # Reconstruct as se(3) matrix at e
  ∇XY_e = vcat(hcat(hat(so3, Ω_rot_e), v_trans_e), zeros(1, 4))

  # Step 4: Left-translate back to g
  return diff_left_compose(SE3, identity_element(SE3, typeof(g)), g, ∇XY_e)
end
#
# -----------------------------------------------------------------------------
# Direct dμ implementation for the current translational SE(3) preconditioner
# -----------------------------------------------------------------------------
# The old chart-based computation of (i_X dμ)^♯ used local_metric and
# inverse_local_metric. That path is numerically unstable for this control loop.
# These routines specialize dμ to the current field
#
#     μ_g(δg) = λ V(q) ⋅ δq = λ (R'V(q)) ⋅ v_body,
#
# where V(q) = corridor_flow_direction_3d(q...).
#
# In left-trivialized SE(3) coordinates X=(Ω_X,v_X), Y=(Ω_Y,v_Y), the rotational
# terms cancel in dμ and
#
#     dμ(X,Y) = λ [(B v_X)⋅v_Y - (B v_Y)⋅v_X],   B = R' DV(q) R.
#
# Consequently (i_X dμ)^♯ is translational in body coordinates:
#
#     (i_X dμ)^♯ = (0, λ (B-B') v_X).

function se3_body_twist(G::SpecialEuclideanGroup, g, Xg)
  Xe = diff_left_compose(G, g, inv(G, g), Xg)
  Ω = vee(so3, Xe[se3, :Rotation])
  v = Xe[se3, :Translation]
  return SVector{3,Float64}(Ω[1], Ω[2], Ω[3]), SVector{3,Float64}(v[1], v[2], v[3])
end

function se3_tangent_from_body_twist(G::SpecialEuclideanGroup, g, Ω, v)
  R = g[G, :Rotation]
  Ωs = SVector{3,Float64}(Ω[1], Ω[2], Ω[3])
  vs = SVector{3,Float64}(v[1], v[2], v[3])
  return vcat(hcat(R * hat(so3, Ωs), R * vs), zeros(1, 4))
end

function corridor_vec3(q)
  V = corridor_flow_direction_3d(q[1], q[2], q[3])
  return SVector{3,Float64}(V[1], V[2], V[3])
end

function corridor_jacobian_fd(q; h=1e-5)
  qs = SVector{3,Float64}(q[1], q[2], q[3])
  J = zeros(3, 3)

  for i in 1:3
    ei = zeros(3)
    ei[i] = 1.0
    eis = SVector{3,Float64}(ei[1], ei[2], ei[3])

    Vp = corridor_vec3(qs + h * eis)
    Vm = corridor_vec3(qs - h * eis)

    J[:, i] .= (Vp - Vm) / (2h)
  end

  return SMatrix{3,3,Float64}(J)
end

function infer_translational_λ(G::SpecialEuclideanGroup, g, μ; fallback=nothing)
  R = g[G, :Rotation]
  q = g[G, :Translation]
  Vb = R' * corridor_vec3(q)

  μsharp = sharp(G, g, μ(G, g))
  _, vμ = se3_body_twist(G, g, μsharp)

  denom = dot(Vb, Vb)
  if isfinite(denom) && denom > 1e-14
    λ_eff = dot(vμ, Vb) / denom
    if isfinite(λ_eff)
      return λ_eff
    end
  end

  if fallback !== nothing
    return fallback
  elseif isdefined(Main, :λ)
    val = getfield(Main, :λ)
    return Float64(val)
  else
    return 1.0
  end
end

function dμ_translational_SE3(
  G::SpecialEuclideanGroup,
  g,
  Xg,
  Yg;
  λ_metric,
  jacobian_h=1e-5,
)
  R = g[G, :Rotation]
  q = g[G, :Translation]

  _, vx = se3_body_twist(G, g, Xg)
  _, vy = se3_body_twist(G, g, Yg)

  DV = corridor_jacobian_fd(q; h=jacobian_h)
  B = R' * DV * R

  val = λ_metric * (dot(B * vx, vy) - dot(B * vy, vx))
  return isfinite(val) ? val : 0.0
end

function sharp_i_dμ_translational_SE3(
  G::SpecialEuclideanGroup,
  g,
  Xg;
  λ_metric,
  jacobian_h=1e-5,
)
  R = g[G, :Rotation]
  q = g[G, :Translation]

  _, vx = se3_body_twist(G, g, Xg)

  DV = corridor_jacobian_fd(q; h=jacobian_h)
  B = R' * DV * R

  αv = λ_metric * ((B - B') * vx)

  if !all(isfinite, αv)
    return zero(Xg)
  end

  return se3_tangent_from_body_twist(
    G,
    g,
    SVector{3,Float64}(0.0, 0.0, 0.0),
    SVector{3,Float64}(αv[1], αv[2], αv[3]),
  )
end

function difference_tensor(
  G::SpecialEuclideanGroup, A::AbstractAtlas, chart_index, g, μ, Xg, Yg; 
  backend=AutoForwardDiff(),
  atol=1e-9,
  jacobian_h=1e-5,
)
  μX = μ(G, g)(Xg)
  μY = μ(G, g)(Yg)

  if !isfinite(μX) || !isfinite(μY)
    return zero(Xg)
  end

  μ♯ = sharp(G, g, μ(G, g))

  if !all(isfinite, μ♯)
    return zero(Xg)
  end

  nμ = norm(G, g, μ♯)^2
  if !isfinite(nμ) || nμ < 1e-6
    return zero(Xg)
  end

  Λ = 1.0 + nμ
  nμ_reg = max(nμ, 1e-6)

  μ_perp = μX * μY / Λ * μ♯ - μY / 2 * Xg - μX / 2 * Yg
  if !all(isfinite, μ_perp)
    return zero(Xg)
  end

  λ_eff = infer_translational_λ(G, g, μ)
  if !isfinite(λ_eff)
    return zero(Xg)
  end

  # Direct, chart-free dμ terms for μ = λ V(q)^♭.
  diff_μXY = dμ_translational_SE3(G, g, Xg, Yg; λ_metric=λ_eff, jacobian_h=jacobian_h)
  diff_μμ  = dμ_translational_SE3(G, g, μ_perp, μ♯; λ_metric=λ_eff, jacobian_h=jacobian_h)
  diff_μ♯  = sharp_i_dμ_translational_SE3(G, g, μ_perp; λ_metric=λ_eff, jacobian_h=jacobian_h)

  lX   = levi_civita_connection_analytical(G, g, Xg, μ♯)
  lY   = levi_civita_connection_analytical(G, g, Yg, μ♯)
  lμ♯  = levi_civita_connection_analytical(G, g, μ♯, μ♯)
  μlμ♯ = μ(G, g)(lμ♯)

  if !all(isfinite, lX) || !all(isfinite, lY) || !all(isfinite, lμ♯) || !isfinite(μlμ♯)
    return zero(Xg)
  end

  term1 = (
    1.0 / Λ * inner(G, g, μY * lX + μX * lY, μ♯)
    - inner(G, g, lX, Yg)
  ) * μ♯

  term2 = (
    0.5 * diff_μXY
    - μX * μY / Λ * μlμ♯ / nμ_reg
    + Λ * diff_μμ / nμ_reg
  ) * μ♯

  term3 = -μX * μY / Λ^2 * (
    lμ♯ - μlμ♯ / nμ_reg * μ♯
  )

  # Restored term 4 using direct, chart-free sharp(i_{μ_perp} dμ).
  term4 = 1.0 / Λ * diff_μ♯ - diff_μμ / nμ_reg * μ♯

  T = term1 + term2 + term3 + term4

  if !all(isfinite, T)
    return zero(Xg)
  end

  return T
end

function print_difference_tensor_term_breakdown(
  G::SpecialEuclideanGroup,
  A::AbstractAtlas,
  chart_index,
  g,
  μ,
  Xg,
  Yg;
  backend=AutoForwardDiff(),
  jacobian_h=1e-5,
)
  μX = μ(G, g)(Xg)
  μY = μ(G, g)(Yg)
  μ♯ = sharp(G, g, μ(G, g))

  nμ = norm(G, g, μ♯)^2
  if !isfinite(nμ) || nμ < 1e-6
    @printf("[DIFFERENCE TENSOR TERM BREAKDOWN] skipped: nμ = %.6e\n", nμ)
    return nothing
  end

  Λ = 1.0 + nμ
  nμ_reg = max(nμ, 1e-6)
  μ_perp = μX * μY / Λ * μ♯ - μY / 2 * Xg - μX / 2 * Yg

  λ_eff = infer_translational_λ(G, g, μ)

  diff_μXY = dμ_translational_SE3(G, g, Xg, Yg; λ_metric=λ_eff, jacobian_h=jacobian_h)
  diff_μμ  = dμ_translational_SE3(G, g, μ_perp, μ♯; λ_metric=λ_eff, jacobian_h=jacobian_h)
  diff_μ♯  = sharp_i_dμ_translational_SE3(G, g, μ_perp; λ_metric=λ_eff, jacobian_h=jacobian_h)

  lX   = levi_civita_connection_analytical(G, g, Xg, μ♯)
  lY   = levi_civita_connection_analytical(G, g, Yg, μ♯)
  lμ♯  = levi_civita_connection_analytical(G, g, μ♯, μ♯)
  μlμ♯ = μ(G, g)(lμ♯)

  term1 = (
    1.0 / Λ * inner(G, g, μY * lX + μX * lY, μ♯)
    - inner(G, g, lX, Yg)
  ) * μ♯

  term2a = (0.5 * diff_μXY) * μ♯
  term2b = (-μX * μY / Λ * μlμ♯ / nμ_reg) * μ♯
  term2c = (Λ * diff_μμ / nμ_reg) * μ♯
  term3 = -μX * μY / Λ^2 * (lμ♯ - μlμ♯ / nμ_reg * μ♯)
  term4a = (1.0 / Λ) * diff_μ♯
  term4b = -(diff_μμ / nμ_reg) * μ♯

  terms = (
    term1=term1,
    term2a=term2a,
    term2b=term2b,
    term2c=term2c,
    term3=term3,
    term4a=term4a,
    term4b=term4b,
    total=term1 + term2a + term2b + term2c + term3 + term4a + term4b,
  )

  @printf("""
[DIFFERENCE TENSOR TERM BREAKDOWN -- DIRECT dμ]
μX = %.6e
μY = %.6e
Λ = %.6e
nμ = %.6e
sqrt(nμ) = %.6e
λ_eff = %.6e
diff_μXY = %.6e
diff_μμ = %.6e
μlμ♯ = %.6e
||diff_μ♯|| = %.6e

""",
    μX,
    μY,
    Λ,
    nμ,
    sqrt(nμ),
    λ_eff,
    diff_μXY,
    diff_μμ,
    μlμ♯,
    norm(G, g, diff_μ♯),
  )

  for name in keys(terms)
    Zg = terms[name]
    Ze = diff_left_compose(G, g, inv(G, g), Zg)

    Ω_part = vee(so3, Ze[se3, :Rotation])
    v_part = Ze[se3, :Translation]

    @printf(
      "%-8s  ||rot|| = %.6e   ||trans|| = %.6e   trans = %s\n",
      string(name),
      norm(Ω_part),
      norm(v_part),
      string(v_part),
    )
  end

  println()

  return terms
end
#
# ------------------------------------------------------------------------------------------
# LEGACY CODE, seemingly unstable for these dynamics
# ------------------------------------------------------------------------------------------
# Compute the Levi-Civita connection (in coordinates)
# Assume Y(M, p) is a function that returns the coordinates of Y in chart_index at any point near p
function partial_derivatives_vector_field(
  M::AbstractManifold, A, chart_index, p, Y, k; 
  backend=AutoForwardDiff()
)
  # Finite difference approximation of ∂_i Y^k at p
  Yk_coord = p_coords -> Manifolds.get_coordinates_induced_basis(
    M, 
    get_point(M, A, chart_index, p_coords), 
    Y(M, get_point(M, A, chart_index, p_coords)), 
    induced_basis(M, A, chart_index)
  )[k]
  return DifferentiationInterface.gradient(Yk_coord, backend, get_parameters(M, A, chart_index, p))
end
#
function levi_civita_connection(
  M::AbstractManifold, A::AbstractAtlas, chart_index, p, X, Y; 
  backend=AutoForwardDiff()
)
  ib = induced_basis(M, A, chart_index)
  p_coords = get_parameters(M, A, chart_index, p)
  # Evaluate the vector field X at the point p (intrinsic) and then take its coordinates in the chart_index "chart_index"
  X_coords = Manifolds.get_coordinates_induced_basis(M, p, X(M, p), ib)
  Y_coords = Manifolds.get_coordinates_induced_basis(M, p, Y(M, p), ib)

  Γ = christoffel_symbols_second(M, A, chart_index, p_coords; backend=backend)

  n = length(X_coords)
  sum = term1 = term2 = zeros(n)
  for k in 1:n
    term1 = X_coords' * partial_derivatives_vector_field(M, A, chart_index, p, Y, k; backend=backend)
    term2 = X_coords' * Γ[k, :, :] * Y_coords  
    sum[k] = term1 + term2
    term1 = zeros(n)
    term2 = zeros(n)
  end
  L = zeros(representation_size(M))#size(X(M, p)))
  # For some reason the allocating version of Manifolds.get_vector_induced_basis doesn't work
  # So we use the inplace variant
  Manifolds.get_vector_induced_basis!(M, L, p, sum, ib)
  return L 
end
#
# For abstract Lie groups
# Y is a vector field, i.e. a function Y(G, g) that returns a vector tangent to G at g
function partial_derivatives_vector_field(
  G::AbstractLieGroup, A, chart_index, g, Y, k; 
  backend=AutoForwardDiff()
)
  # Finite difference approximation of ∂_i Y^k at g
  Yk_coord = g_coords -> get_coordinates(
    G, get_point(G, A, chart_index, g_coords), 
    Y(G, get_point(G, A, chart_index, g_coords)), 
    induced_basis(G, A, chart_index).A.basis
  )[k]  
  return DifferentiationInterface.gradient(Yk_coord, backend, get_parameters(G, A, chart_index, g))
end
function levi_civita_connection(
  G::AbstractLieGroup, A::AbstractAtlas, chart_index, g, X, Y; 
  backend=AutoForwardDiff()
)
  ib = induced_basis(G, A, chart_index)
  p_coords = get_parameters(G, A, chart_index, g)
  # Evaluate the vector field X at the point g (intrinsic)  
  # Then take its coordinates in the chart_index "chart_index"
  # On LieGroups, Manifold.get_coordinates_induced_basis produced coordinates 
  # That do not map back to the original vector representation of X(G, g)
  # Once brought back with Manifolds.get_vector_induced_basis, 
  # Whereas get_coordinates and get_vector actually are mutual inverses for LieGroups!
  X_coords = get_coordinates(G, g, X(G, g), ib.A.basis)
  Y_coords = get_coordinates(G, g, Y(G, g), ib.A.basis)

  Γ = christoffel_symbols_second(G, A, chart_index, p_coords; backend=backend)

  n = length(X_coords)
  sum = term1 = term2 = zeros(n)
  for k in 1:n
    term1 = X_coords' * partial_derivatives_vector_field(G, A, chart_index, g, Y, k; backend=backend)
    term2 = X_coords' * Γ[k, :, :] * Y_coords  
    sum[k] = term1 + term2
    term1 = zeros(n)
    term2 = zeros(n)
  end
  return get_vector(G, g, sum, ib.A.basis) 
end
# For Lie groups
# Just like before, μ is assumed to be a function with signature μ(G, g) returning a vector of components of the 1-form μ
# Where μ = X♭ for some X
function partial_derivatives_one_form(
  G::AbstractLieGroup, A, chart_index, g, μ, k; 
  backend=AutoForwardDiff()
)
  # Finite difference approximation of ∂_i μ_k at g
  μk_coord = g_coords -> (
    local_metric(G, A, chart_index, g_coords) * get_coordinates(
      G, get_point(G, A, chart_index, g_coords), μ(G, get_point(G, A, chart_index, g_coords)).X, induced_basis(G, A, chart_index).A.basis
    )
  )[k]  
  return DifferentiationInterface.gradient(μk_coord, backend, get_parameters(G, A, chart_index, g))
end
#
function differential_one_form(
  G::AbstractLieGroup, A, chart_index, g, μ; 
  backend=AutoForwardDiff()
)
  n = length(get_coordinates(G, g, μ(G, g).X))
  ∂μ_coords = zeros(n, n)
  dμ_coords = zeros(n, n)
  # For a 1-form μ, (dμ)_ij = ∂_i μ_j - ∂_j μ_i
  for k in 1:n
    ∂μ_coords[k, :] = partial_derivatives_one_form(G, A, chart_index, g, μ, k; backend=backend)
    for j in 1:k
      dμ_coords[k, j] += ∂μ_coords[k, j]
      # Since j ≤ k
      dμ_coords[k, j] -= ∂μ_coords[j, k]
    end
  end
  # Anti-symmetrize 
  dμ_coords -= dμ_coords'
  return dμ_coords
end
#
# To evaluate the 1-form μ, Xg and Yg are assumed to be tangent vectors at g
function evaluate_differential_one_form(
  G::AbstractLieGroup, A, chart_index, g, μ, Xg, Yg; 
  backend=AutoForwardDiff()
)
  ib = induced_basis(G, A, chart_index)
  dμ = differential_one_form(G, A, chart_index, g, μ; backend=backend)
  Xg_coords = get_coordinates(G, g, Xg, ib.A.basis)
  Yg_coords = get_coordinates(G, g, Yg, ib.A.basis)
  return Xg_coords' * dμ * Yg_coords
end
#
function sharp_differential_one_form(
  G::AbstractLieGroup,
  A,
  chart_index,
  g,
  μ,
  Xg;
  backend=AutoForwardDiff(),
  cond_max=1e8,
  sharp_max=1e3,
)
  ib = induced_basis(G, A, chart_index)

  dμ = differential_one_form(G, A, chart_index, g, μ; backend=backend)
  Xg_coords = get_coordinates(G, g, Xg, ib.A.basis)

  α_row = Xg_coords' * dμ
  α = vec(α_row')

  params = get_parameters(G, A, chart_index, g)
  Gloc = local_metric(G, A, chart_index, params)

  if !all(isfinite, params) || !all(isfinite, Gloc) || !all(isfinite, α)
    @printf("""
[sharp_differential_one_form failure: nonfinite input]
all finite params = %s
all finite Gloc  = %s
all finite α     = %s
params = %s
||Xg_coords|| = %.6e
||dμ|| = %.6e
||α|| = %.6e

""",
      string(all(isfinite, params)),
      string(all(isfinite, Gloc)),
      string(all(isfinite, α)),
      string(params),
      norm(Xg_coords),
      all(isfinite, dμ) ? opnorm(dμ) : NaN,
      all(isfinite, α) ? norm(α) : NaN,
    )

    return zero(Xg)
  end

  Gloc_sym = Symmetric(0.5 * (Gloc + Gloc'))

  vals = eigvals(Gloc_sym)
  λmin = minimum(abs.(vals))
  λmax = maximum(abs.(vals))
  κ = λmax / λmin

  if !isfinite(κ) || λmin < 1e-10 || κ > cond_max
      @printf("""
[sharp_differential_one_form warning: bad local metric]
λmin = %.6e
λmax = %.6e
cond = %.6e
params = %s
||Xg_coords|| = %.6e
||dμ|| = %.6e
||α|| = %.6e

""",
      λmin,
      λmax,
      κ,
      string(params),
      norm(Xg_coords),
      opnorm(dμ),
      norm(α),
    )

    return zero(Xg)
  end

  sharp_coords = Gloc_sym \ α

  sharp_norm = norm(sharp_coords)

  if !all(isfinite, sharp_coords) || sharp_norm > sharp_max
      @printf("""
[sharp_differential_one_form warning: large sharp]
||sharp_coords|| = %.6e
sharp_coords = %s
λmin = %.6e
λmax = %.6e
cond = %.6e
||α|| = %.6e

""",
      sharp_norm,
      string(sharp_coords),
      λmin,
      λmax,
      κ,
      norm(α),
    )

    if !all(isfinite, sharp_coords)
      return zero(Xg)
    end

    sharp_coords .= (sharp_max / sharp_norm) .* sharp_coords
  end

  return get_vector(G, g, sharp_coords, ib.A.basis)
end

function safe_dμ(M, p, Xp, Yp)
  try
    val = evaluate_differential_one_form(
      M, A, chart_index, p, μ, Xp, Yp;
      backend=backend,
    )

    if val isa Number && isfinite(val)
      return val
    else
      return 0.0
    end
  catch err
    return 0.0
  end
end
#
# Here μ, X, and Y are fields, i.e functions with signature (G, g)
function difference_tensor(
  G::AbstractLieGroup, A::AbstractAtlas, chart_index, g, μ, X, Y; 
  backend=AutoForwardDiff(),
  atol=1e-9,
)
  Xg = X(G, g)
  Yg = Y(G, g)

  # Delegate to the concrete tangent-vector overload when available.
  return difference_tensor(G, A, chart_index, g, μ, Xg, Yg; backend=backend, atol=atol)
end

