# using Pkg
# Pkg.activate("/g/data/hh5/tmp/ab2462/julia_work/OOC_LES")
using Statistics, Printf

using Oceananigans
using Oceananigans.Fields
using Oceananigans.TurbulenceClosures
using Oceananigans.OutputWriters
using Oceananigans.Diagnostics
using Oceananigans.Utils

using Oceananigans.Advection: cell_advection_timescale

""" Friction velocity. See equation (16) of Vreugdenhil & Taylor (2018). """
function uτ(model, Uavg, U_wall, n)
    Nz, Hz, Δz = model.grid.Nz, model.grid.Hz, model.grid.Δzᵃᵃᶜ
    ν = model.closure[n].ν

    compute!(Uavg)
    U = Array(interior(Uavg))  # Exclude average of halo region.

    # Use a finite difference to calculate dU/dz at the top and bottom walls.
    # The distance between the center of the cell adjacent to the wall and the
    # wall itself is Δz/2.
    uτ²_top    = ν * abs(U_wall - U[Nz]) / (Δz/2)  # Top wall    where u = +U_wall
    uτ²_bottom = ν * abs(U[1] + U_wall)  / (Δz/2)  # Bottom wall where u = -U_wall

    uτ_top, uτ_bottom = √uτ²_top, √uτ²_bottom

    return uτ_top, uτ_bottom
end

""" Heat flux at the wall. See equation (16) of Vreugdenhil & Taylor (2018). """
function q_wall(model, Tavg, Θ_wall, n)
    Nz, Hz, Δz = model.grid.Nz, model.grid.Hz, model.grid.Δzᵃᵃᶜ
    # TODO: interface function for extracting diffusivity?
    κ = model.closure[n].κ.T

    compute!(Tavg)
    Θ = Array(interior(Tavg)) # Exclude average of halo region.

    # Use a finite difference to calculate dθ/dz at the top and bottomtom walls.
    # The distance between the center of the cell adjacent to the wall and the
    # wall itself is Δz/2.
    q_wall_top    = κ * abs(Θ[1] - Θ_wall)   / (Δz/2)  # Top wall    where Θ = +Θ_wall
    q_wall_bottom = κ * abs(-Θ_wall - Θ[Nz]) / (Δz/2)  # Bottom wall where Θ = -Θ_wall

    return q_wall_top, q_wall_bottom
end

struct FrictionReynoldsNumber{H, U}
    Uavg :: H
    U_wall :: U
    n_scalar :: Int
end

struct NusseltNumber{H, T}
    Tavg :: H
    Θ_wall :: T
    n_scalar :: Int
end

""" Friction Reynolds number. See equation (20) of Vreugdenhil & Taylor (2018). """
function (Reτ::FrictionReynoldsNumber)(model)
    ν = model.closure[Reτ.n_scalar].ν
    h = model.grid.Lz / 2
    uτ_top, uτ_bottom = uτ(model, Reτ.Uavg, Reτ.U_wall, Reτ.n_scalar)

    return h * uτ_top / ν, h * uτ_bottom / ν
end

""" Nusselt number. See equation (20) of Vreugdenhil & Taylor (2018). """
function (Nu::NusseltNumber)(model)
    κ = model.closure[Nu.n_scalar].κ.T
    h = model.grid.Lz / 2

    q_wall_top, q_wall_bottom = q_wall(model, Nu.Tavg, Nu.Θ_wall, Nu.n_scalar)

    return (q_wall_top * h)/(κ * Nu.Θ_wall), (q_wall_bottom * h)/(κ * Nu.Θ_wall)
end

"""
    simulate_stratified_couette_flow(; Nxy, Nz, h=1, U_wall=1, Re=4250, Pr=0.7,
                                     Ri, Ni=10, end_time=1000)

Simulate stratified plane Couette flow with `Nxy` grid cells in each horizontal
direction, `Nz` grid cells in the vertical, in a domain of size (4πh, 2πh, 2h),
with wall velocities of `U_wall` at the top and -`U_wall` at the bottom, at a Reynolds
number `Re, Prandtl number `Pr`, and Richardson number `Ri`.

`Ni` is the number of "intermediate" time steps taken at a time before printing a progress
statement and updating the time step.
"""
# function simulate_stratified_couette_flow(; Nxy, Nz, arch=GPU(), h=1, U_wall=1,
#                                           Re=4250, Pr=0.7, Ri, Ni=50, end_time=1000)

Nxy, Nz = 128, 128
arch=GPU()
h=1
U_wall=1
Re=4250
Pr=0.7
Ri=0.01
Ni=10
end_time=1000


#####
##### Computed parameters
#####

ν = U_wall * h / Re    # From Re = U_wall h / ν
Θ_wall = Ri * U_wall^2 / h  # From Ri = L Θ_wall / U_wall²
κ = ν / Pr             # From Pr = ν / κ

#####
##### Impose boundary conditions
#####

Nh = 5

σ1 = 2
# z_faces(k) = - h * (1 - tanh(σ1 * (k - 1) / Nz) / tanh(σ1))
z_faces(k) = - (1 - tanh(σ1 * (k - 1) / 64 - σ1) / tanh(σ1))

grid =  RectilinearGrid(arch, size = (Nxy, Nxy, Nz), x = (0, 4π*h), y = (0, 2π*h), z = z_faces, #(-2h, 0), #
                        topology=(Oceananigans.Periodic, Oceananigans.Periodic, Bounded),
                        halo = (Nh, Nh, Nh))

@info "Build a grid:"
@show grid

Tbcs = FieldBoundaryConditions(top = ValueBoundaryCondition(Θ_wall),
                                bottom = ValueBoundaryCondition(-Θ_wall))

ubcs = FieldBoundaryConditions(top = ValueBoundaryCondition(U_wall),
                                bottom = ValueBoundaryCondition(-U_wall))

vbcs = FieldBoundaryConditions(top = ValueBoundaryCondition(0.0),
                                bottom = ValueBoundaryCondition(0.0))

#####
##### Non-dimensional model setup
#####

equation_of_state = LinearEquationOfState(thermal_expansion=1.0e-4, haline_contraction=0.0)
buoyancy = SeawaterBuoyancy(; equation_of_state)

vitd = VerticallyImplicitTimeDiscretization()
molecular_diffusivity = ScalarDiffusivity(vitd, ν=ν, κ=κ)
closure = (AnisotropicMinimumDissipation(vitd), molecular_diffusivity)

model = NonhydrostaticModel(; grid, buoyancy,
                            advection = WENO(grid, order=9),
                            timestepper = :RungeKutta3,
                            tracers = (:T, :S),
                            closure = closure,
                            boundary_conditions = (u=ubcs, v=vbcs, T=Tbcs))

                            
@info "Constructed a model"
@show model

#####
##### Set initial conditions
#####

# Add a bit of surface-concentrated noise to the initial condition
parabola(z) = (cos(z/model.grid.Lz - 0.68 * π) / 0.465 + 2.1506)
ε(σ, z) = σ * randn() * z/model.grid.Lz * (1 + z/model.grid.Lz) #* parabola(z)

# We add a sinusoidal initial condition to u to encourage instability.
T₀(x, y, z) = 2Θ_wall * (1/2 + z/model.grid.Lz) * (1 + ε(2e-1, z))
u₀(x, y, z) = 2U_wall * (1/2 + z/model.grid.Lz) * (1 + ε(2e-1, z)) * (1 + 0.2*sin(4π/model.grid.Lx * x))
# u₀(x, y, z) =  U_wall * parabola(z) * (1 + ε(5e-2, z)) * (1 + 5e-2*sin(4π/model.grid.Lx * x))
v₀(x, y, z) = ε(2e-1, z)
w₀(x, y, z) = ε(2e-1, z)

set!(model, u=u₀, v=v₀, w=w₀, T=T₀)

n_amd = findfirst(c -> c isa AnisotropicMinimumDissipation, model.closure)

n_scalar = findfirst(c -> c isa ScalarDiffusivity, model.closure)

Reτ = FrictionReynoldsNumber(Uavg, U_wall, n_scalar)
Nu = NusseltNumber(Tavg, Θ_wall, n_scalar)

#####
##### Time stepping
#####

simulation = Simulation(model, Δt=0.0001, stop_time=end_time)

wizard = TimeStepWizard(cfl=0.05, max_change=1.1, max_Δt=0.05)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(Ni))

# We will ramp up the CFL used by the adaptive time step wizard during spin up.
cfl(t) = min(0.05t, 0.2)

wall_clock = time_ns()

function progress_message(simulation)
    model = simulation.model
    u, v, w = model.velocities

    wizard.cfl = cfl(model.clock.time)

    CFL = simulation.Δt / cell_advection_timescale(model)
    
    # Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, umax: (%.1e, %.1e, %.1e) ms⁻¹, CFL: %.2e, wall time: %s\n",
                   model.clock.iteration,
                   prettytime(model.clock.time),
                   prettytime(simulation.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w), CFL,
                   prettytime(1e-9 * (time_ns() - wall_clock))
                  )
        @info msg
    return nothing
end

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(300))

dir = "./"
file_name = "stratified_couette_flow2.nc"

fields = Dict("u" => model.velocities.u, 
              "v" => model.velocities.v, 
              "w" => model.velocities.w, 
              "T" => model.tracers.T, 
            #   "S" => model.tracers.S
)

simulation.output_writers[:_2D_writer] =
        NetCDFOutputWriter(model, fields, filename=dir * file_name, schedule=TimeInterval(5), indices=(1, :, :))


@info "Output files gererated"

run!(simulation)
