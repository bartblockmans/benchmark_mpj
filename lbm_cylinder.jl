# lbm_cylinder.jl
# =============================================================================
# Julia Program: Lattice Boltzmann Method (D2Q9) - Fluid Flow Past a Fixed Cylinder
# =============================================================================
#
# This program simulates 2D fluid flow past a circular cylinder using the Lattice 
# Boltzmann Method with a D2Q9 lattice model. The simulation computes the evolution 
# of fluid particle distribution functions and visualizes the resulting flow field 
# including vorticity and streamlines.
#
# Key Features:
# - D2Q9 lattice model with 9 velocity directions
# - Reynolds number based flow simulation (Re = 200)
# - Cylinder boundary condition handling with bounce-back method
# - Periodic boundary conditions for domain edges
# - Real-time visualization of vorticity and streamlines
# - Organized output to dedicated 'julia' subdirectory
# - Optimized simulation parameters for faster execution
# - Performance mode: disable visualization for computational benchmarking
#
# What This Code Does:
# ====================
# 1. Sets up a 400x100 computational domain with a cylinder at position (70, 50)
# 2. Implements the LBM algorithm with streaming and collision steps
# 3. Applies proper boundary conditions (no-slip at cylinder, periodic at edges)
# 4. Computes macroscopic variables (density, velocity) from distribution functions
# 5. Visualizes the flow field showing vorticity patterns and streamlines (optional)
# 6. Saves output images every 2000 steps for analysis (optional)
#
# Physics Background:
# ==================
# The Lattice Boltzmann Method is a computational fluid dynamics technique that:
# - Models fluid as discrete particles moving on a regular lattice
# - Uses distribution functions to represent particle populations
# - Replaces the Navier-Stokes equations with simple collision and streaming rules
# - Naturally handles complex boundaries and multiphase flows
# - Provides accurate results for incompressible flows at moderate Reynolds numbers
#
# This simulation specifically studies the classic problem of flow past a cylinder,
# which exhibits phenomena like boundary layer separation, vortex shedding, and
# wake formation - fundamental concepts in fluid dynamics.
#
# Author: BB
# Modified: Enhanced structure and documentation for educational use
# Language: Julia (with CairoMakie for visualization)

using CairoMakie
using GeometryBasics: Point2f, Circle

# Change to the directory containing this script
cd(@__DIR__)

# =============================================================================
# PARAMETER INITIALIZATION
# =============================================================================
function initialize_parameters()
    """
    Initialize all simulation parameters including domain size, lattice properties,
    cylinder geometry, and fluid properties.
    
    Returns a NamedTuple containing all simulation parameters for easy access.
    """
    # Fluid domain dimensions
    MAX_X = 400
    MAX_Y = 100

    # D2Q9 lattice model parameters
    # 9 velocity directions: center (1), cardinal directions (2-5), diagonal directions (6-9)
    LATTICE_NUM = 9
    CX = [0, 1, 0, -1, 0, 1, -1, -1, 1]      # x-velocity components
    CY = [0, 0, 1,  0, -1, 1,  1, -1,-1]     # y-velocity components
    WEIGHTS = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]  # lattice weights
    
    # Opposite direction indices for bounce-back boundary conditions
    # Note: Julia uses 1-based indexing, so we convert from Python's 0-based OPP
    # Python OPP (0-based): [0,3,4,1,2,7,8,5,6] -> Julia (1-based): [1,4,5,2,3,8,9,6,7]
    OPP = [1, 4, 5, 2, 3, 8, 9, 6, 7]

    # Cylinder geometry and position
    POSITION_OX = 70    # x-coordinate of cylinder center
    POSITION_OY = 50    # y-coordinate of cylinder center
    RADIUS      = 20    # cylinder radius

    # Fluid properties and Reynolds number
    REYNOLDS = 200      # Reynolds number for the flow
    U_MAX    = 0.1      # maximum inlet velocity
    kinematic_viscosity = U_MAX * 2 * RADIUS / REYNOLDS
    relaxation_time     = 3.0 * kinematic_viscosity + 0.5

    # Simulation control parameters
    MAX_STEP    = 20001    # total number of time steps
    OUTPUT_STEP = 2000     # frequency of output visualization
    PICTURE_NUM = 1        # counter for saved images
    
    # Visualization control flags
    # Set to false for performance benchmarking (no visualization or file I/O)
    # Set to true for normal operation with visualization and output
    VISUALIZE = true
    
    # Clean visualization parameter: removes ticks, labels, title and minimizes margins
    # Set to true for clean, publication-ready images without visual clutter
    NOTICKS = false

    # Return all parameters as a NamedTuple for clean access
    return (; MAX_X, MAX_Y, LATTICE_NUM, CX, CY, WEIGHTS, OPP,
            POSITION_OX, POSITION_OY, RADIUS, REYNOLDS, U_MAX,
            kinematic_viscosity, relaxation_time, MAX_STEP, OUTPUT_STEP, PICTURE_NUM, VISUALIZE, NOTICKS)
end

# =============================================================================
# HELPER FUNCTIONS AND SETUP
# =============================================================================
function create_cylinder_mask(MAX_X, MAX_Y, POSITION_OX, POSITION_OY, RADIUS)
    """
    Create a boolean mask representing the cylinder in the computational domain.
    
    This function creates a 2D grid and identifies which grid points lie inside
    the circular cylinder using the distance formula.
    """
    # Create coordinate grids (Julia uses 1-based indexing)
    x = repeat(reshape(0:MAX_X-1, 1, MAX_X), MAX_Y, 1)
    y = repeat(reshape(0:MAX_Y-1, MAX_Y, 1), 1, MAX_X)
    
    # Return boolean mask: true for points inside cylinder, false otherwise
    return (x .- POSITION_OX).^2 .+ (y .- POSITION_OY).^2 .<= RADIUS^2
end

function initialize_flow_field(MAX_Y, MAX_X, LATTICE_NUM, U_MAX, CX, CY, WEIGHTS)
    """
    Initialize the flow field with uniform density and inlet velocity.
    Sets up the initial distribution functions F based on equilibrium conditions.
    
    This function creates the initial state of the simulation where:
    - Density is uniform (rho = 1) throughout the domain
    - Velocity is zero except at inlet/outlet boundaries
    - Distribution functions are set to equilibrium values
    """
    # Initialize density and velocity fields
    rho = ones(Float64, MAX_Y, MAX_X)           # uniform density field
    ux  = zeros(Float64, MAX_Y, MAX_X)         # zero velocity initially
    uy  = zeros(Float64, MAX_Y, MAX_X)         # zero velocity initially
    
    # Set inlet and outlet velocities (left and right boundaries)
    ux[:, 1]   .= U_MAX    # left boundary (inlet)
    ux[:, end] .= U_MAX    # right boundary (outlet)

    # Initialize distribution functions F using equilibrium distribution
    # This sets up the initial particle populations for each velocity direction
    F = zeros(Float64, MAX_Y, MAX_X, LATTICE_NUM)
    for i in 1:LATTICE_NUM
        cx = CX[i]; cy = CY[i]; w = WEIGHTS[i]
        cu = cx .* ux .+ cy .* uy  # velocity component in this direction
        # Equilibrium distribution function (Chapman-Enskog expansion)
        F[:, :, i] .= rho .* w .* (1 .+ 3 .* cu .+ 4.5 .* cu.^2 .- 1.5 .* (ux.^2 .+ uy.^2))
    end
    return rho, ux, uy, F
end

# =============================================================================
# LBM CORE ALGORITHM STEPS
# =============================================================================
function apply_periodic_boundary_conditions!(F)
    """
    Apply periodic boundary conditions to the distribution functions.
    
    This ensures particles leaving one edge of the domain re-enter from the 
    opposite edge, creating a periodic flow field. This is essential for:
    - Maintaining mass conservation
    - Avoiding artificial boundary effects
    - Creating an infinite domain approximation
    
    Note: The '!' in the function name indicates this modifies F in-place.
    """
    # X-direction periodic boundaries (left/right edges)
    F[:, 1,  [2, 6, 9]] .= F[:, end, [2, 6, 9]]   # left edge gets from right (dirs 2,6,9)
    F[:, end,[4, 7, 8]] .= F[:, 1,   [4, 7, 8]]   # right edge gets from left (dirs 4,7,8)
    
    # Y-direction periodic boundaries (top/bottom edges)
    F[1,  :,  [3, 6, 7]] .= F[end, :, [3, 6, 7]]  # bottom edge gets from top (dirs 3,6,7)
    F[end, :,  [5, 8, 9]] .= F[1,  :, [5, 8, 9]]  # top edge gets from bottom (dirs 5,8,9)
    
    return F
end

function streaming_step!(F, CX, CY)
    """
    Perform the streaming step of the LBM algorithm.
    
    In this step, particles move along their velocity directions to neighboring 
    lattice sites. This is the advection part of the LBM that handles:
    - Particle movement across the lattice
    - Information propagation through the domain
    - Spatial discretization of the fluid flow
    
    Note: The '!' in the function name indicates this modifies F in-place.
    """
    for i in eachindex(CX)
        # Use circshift to move particles: (rows, cols) = (y, x) directions
        F[:, :, i] = circshift(F[:, :, i], (CY[i], CX[i]))
    end
    return F
end

function handle_cylinder_boundary(F, cylinder, CX, CY, OPP)
    """
    Handle the no-slip boundary condition at the cylinder surface.
    
    Implements the bounce-back method where particles hitting the cylinder
    reverse their direction. This creates a solid wall effect by:
    - Identifying incoming particles (moving toward cylinder surface)
    - Reversing their velocity direction (bounce-back)
    - Maintaining mass conservation at the boundary
    
    The bounce-back method is a simple and effective way to implement
    no-slip boundary conditions in LBM simulations.
    """
    cylinderF = copy(F)  # Create a copy to avoid modifying the original
    
    for i in 2:length(CX)  # skip rest particle (i=1, which has zero velocity)
        # Find incoming particles: inside cylinder but moving toward cylinder surface
        roll_x = circshift(cylinder, (0, -CX[i]))      # shift cylinder mask in x-direction
        roll_y = circshift(cylinder, (-CY[i], 0))      # shift cylinder mask in y-direction
        incoming_particles = cylinder .& .!(roll_x .& roll_y)

        # Use @views for efficient memory access and modification
        @views begin
            Fi   = view(cylinderF, :, :, i)           # current direction
            Fopp = view(F,          :, :, OPP[i])     # opposite direction
            Fi[incoming_particles] .= Fopp[incoming_particles]  # bounce back
        end
    end
    return cylinderF
end

function compute_macroscopic_variables(F, cylinder, CX, CY)
    """
    Compute macroscopic fluid variables (density and velocity) from the
    distribution functions using moment integrals.
    
    This function converts the microscopic particle distribution functions
    into macroscopic quantities that we can observe and analyze:
    - Density: sum of all distribution functions (mass conservation)
    - Velocity: momentum divided by density (momentum conservation)
    
    These macroscopic variables represent the actual fluid properties
    and are used for visualization and analysis.
    """
    # Density: sum of all distribution functions (zeroth moment)
    rho = dropdims(sum(F, dims=3), dims=3)
    
    # Velocity: momentum divided by density (first moment)
    ux  = dropdims(sum(F .* reshape(CX, 1, 1, :), dims=3), dims=3) ./ rho
    uy  = dropdims(sum(F .* reshape(CY, 1, 1, :), dims=3), dims=3) ./ rho

    # Set velocity to zero inside the cylinder (no-slip condition)
    ux[cylinder] .= 0.0
    uy[cylinder] .= 0.0
    
    return rho, ux, uy
end

function collision_step!(F, rho, ux, uy, CX, CY, WEIGHTS, relaxation_time)
    """
    Perform the collision step using the BGK (Bhatnagar-Gross-Krook) approximation.
    
    In this step, distribution functions relax toward their equilibrium values.
    The BGK approximation simplifies the collision operator by assuming:
    - Single relaxation time for all distribution functions
    - Linear relaxation toward equilibrium
    - Local equilibrium based on current macroscopic variables
    
    This is the collision part of the LBM that handles viscosity and
    drives the system toward thermodynamic equilibrium.
    """
    # Compute equilibrium distribution functions
    Feq = zeros(size(F))
    for i in 1:length(CX)
        cx = CX[i]; cy = CY[i]; w = WEIGHTS[i]
        cu = cx .* ux .+ cy .* uy  # velocity component in this direction
        # Equilibrium distribution function (Chapman-Enskog expansion)
        Feq[:, :, i] .= rho .* w .* (1 .+ 3 .* cu .+ 4.5 .* cu.^2 .- 1.5 .* (ux.^2 .+ uy.^2))
    end
    
    # BGK collision: relax toward equilibrium
    F .+= .- (1/relaxation_time) .* (F .- Feq)
    
    return F
end

function apply_inflow_outflow_boundary_conditions!(F, rho, ux, CX, WEIGHTS, U_MAX)
    """
    Apply velocity boundary conditions at inlet and outlet.
    
    Maintains constant velocity at domain boundaries by:
    - Setting boundary velocities to U_MAX
    - Updating distribution functions using equilibrium distribution
    - Ensuring mass and momentum conservation at boundaries
    
    This creates a constant flow rate into and out of the domain,
    which is essential for maintaining the steady-state flow conditions.
    """
    # Set boundary velocities
    ux[:, 1]   .= U_MAX    # left boundary (inlet)
    ux[:, end] .= U_MAX    # right boundary (outlet)
    
    # Update distribution functions at boundaries using equilibrium distribution
    for i in 1:length(CX)
        cx = CX[i]; w = WEIGHTS[i]
        
        # Inlet boundary (left edge)
        cuL = cx .* ux[:, 1]
        F[:, 1,   i] .= rho[:, 1]   .* w .* (1 .+ 3 .* cuL .+ 4.5 .* cuL.^2 .- 1.5 .* (ux[:, 1].^2))
        
        # Outlet boundary (right edge)
        cuR = cx .* ux[:, end]
        F[:, end, i] .= rho[:, end] .* w .* (1 .+ 3 .* cuR .+ 4.5 .* cuR.^2 .- 1.5 .* (ux[:, end].^2))
    end
    
    return F, ux
end

# =============================================================================
# VISUALIZATION AND OUTPUT
# =============================================================================
function visualize_flow_field(step, ux, uy, cylinder, POSITION_OX, POSITION_OY, RADIUS,
    MAX_Y, MAX_X, PICTURE_NUM, NOTICKS)
    """
    Visualize the flow field by plotting vorticity and streamlines.
    
    This function creates comprehensive visualizations showing:
    - Vorticity field (curl of velocity) using a diverging colormap
    - Streamlines colored by velocity magnitude
    - Cylinder outline for reference
    - Clear title and proper axis limits (unless NOTICKS is true)
    
    The visualization helps understand the flow physics including:
    - Vortex formation and shedding
    - Flow separation patterns
    - Wake development behind the cylinder
    
    When NOTICKS is true, creates clean images without ticks, labels, title,
    and with minimized margins for publication use.
    """
    # Compute vorticity (discrete curl of velocity field)
    # This measures the local rotation of the fluid
    vorticity = (circshift(uy, (0, -1)) .- circshift(uy, (0, 1))) .-
                (circshift(ux, (-1, 0)) .- circshift(ux, (1, 0)))
    vorticity[cylinder] .= NaN  # hide vorticity inside cylinder

    # Create the figure using CairoMakie (Julia's plotting library)
    # Use 'size' (not 'resolution') to avoid Makie deprecation warnings
    if NOTICKS
        # Clean layout: minimize margins, no ticks/labels/title
        fig = Figure(size = (900, 250), figure_padding = (0, 0, 0, 0))
        ax  = Axis(fig[1, 1], aspect = DataAspect(),
                   xticksvisible = false, yticksvisible = false, # hide ticks
                   xlabelvisible = false, ylabelvisible = false, # hide labels
                   titlevisible = false, # hide title
                   xticklabelsvisible = false, yticklabelsvisible = false, # hide axis values
                   leftspinevisible = true, rightspinevisible = true, 
                   topspinevisible = true, bottomspinevisible = true) # keep border
    else
        # Normal layout: standard margins, ticks, labels, title
        fig = Figure(size = (900, 250))
        ax  = Axis(fig[1, 1], aspect = DataAspect(),
                   title = "Vorticity and streamlines at step $step")
    end

    # Bilinear interpolation function for smooth velocity field sampling
    # This provides better streamline visualization by interpolating between grid points
    bilin(u, x, y) = begin
        ny, nx = size(u)
        x = clamp(x, 0, nx-1); y = clamp(y, 0, ny-1)
        j0 = Int(floor(x)) + 1; i0 = Int(floor(y)) + 1
        j1 = min(j0 + 1, nx);   i1 = min(i0 + 1, ny)
        tx = x - (j0 - 1);      ty = y - (i0 - 1)
        v00 = u[i0, j0]; v10 = u[i1, j0]; v01 = u[i0, j1]; v11 = u[i1, j1]
        (1-ty)*((1-tx)*v00 + tx*v01) + ty*((1-tx)*v10 + tx*v11)
    end

    # Vector field function for streamline generation
    # Returns velocity vector at any point (x,y), with zero velocity inside cylinder
    f = (x,y) -> begin
        # Stop streamlines inside the cylinder by returning zero velocity
        iy = clamp(Int(round(y)) + 1, 1, size(cylinder,1))
        ix = clamp(Int(round(x)) + 1, 1, size(cylinder,2))
        cylinder[iy, ix] ? Point2f(0,0) : Point2f(bilin(ux, x, y), bilin(uy, x, y))
    end

    # Create vorticity heatmap with coordinates to emulate origin="lower"
    # Red-Blue diverging colormap shows positive/negative vorticity clearly
    heatmap!(ax, 0:MAX_X-1, 0:MAX_Y-1, vorticity';
             colormap = :RdBu, colorrange = (-0.02, 0.02))

    # Add cylinder outline — ensure Float types to avoid Circle{Int64} errors
    poly!(ax, Circle(Point2f(Float32(POSITION_OX), Float32(POSITION_OY)), Float32(RADIUS));
             color = :black, strokecolor = :black, strokewidth = 1)

    # Generate streamlines using the vector field function
    # This shows the flow patterns and direction throughout the domain
    streamplot!(ax, f, 0..(MAX_X-1), 0..(MAX_Y-1);
                color     = v -> hypot(v[1], v[2]),  # color by velocity magnitude
                colormap  = :cool,                     # cool colormap for streamlines
                linewidth = 0.8,                      # line thickness
                density   = 0.35,                     # streamline density (0.25–0.6)
                gridsize  = (120, 40),                # sampling resolution
                arrow_size = 6)                       # arrowhead size for direction

    # Set axis limits to match the computational domain
    xlims!(ax, 0, MAX_X-1); ylims!(ax, 0, MAX_Y-1)

    # Create output directory and save the figure
    outdir = joinpath(@__DIR__, "julia")
    mkpath(outdir)
    save(joinpath(outdir, "lattice_boltzmann_$(lpad(PICTURE_NUM, 4, '0'))_julia.png"), fig)

    return PICTURE_NUM + 1
end

# =============================================================================
# MAIN SIMULATION FUNCTION
# =============================================================================
function main()
    """
    Main simulation loop implementing the Lattice Boltzmann Method.
    
    This function orchestrates the complete LBM simulation by:
    1. Setting up the computational domain and initial conditions
    2. Running the main time-stepping loop with 7 key algorithm steps
    3. Generating visualizations at regular intervals (optional)
    4. Saving results to the 'julia' output directory (optional)
    
    The simulation follows the classic LBM algorithm:
    - Initialize → Stream → Collide → Repeat
    - With proper boundary condition handling at each step
    - Regular output for analysis and visualization (when enabled)
    """
    # Initialize all simulation parameters
    p = initialize_parameters()
    
    # Print simulation parameters for user reference
    @info "Reynolds number = $(p.REYNOLDS)"
    @info "Relaxation time = $(p.relaxation_time)"
    @info "Domain size: $(p.MAX_X) x $(p.MAX_Y)"
    @info "Maximum velocity = $(p.U_MAX)"
    @info "Kinematic viscosity = $(p.kinematic_viscosity)"
    @info "Visualization enabled = $(p.VISUALIZE)"
    if p.VISUALIZE
        @info "Output directory = julia/"
        @info "Clean visualization (NOTICKS) = $(p.NOTICKS)"
    end

    # Create cylinder mask (defines where the solid cylinder is located)
    cylinder = create_cylinder_mask(p.MAX_X, p.MAX_Y, p.POSITION_OX, p.POSITION_OY, p.RADIUS)
    
    # Initialize flow field with uniform density and inlet velocity
    rho, ux, uy, F = initialize_flow_field(p.MAX_Y, p.MAX_X, p.LATTICE_NUM, p.U_MAX, p.CX, p.CY, p.WEIGHTS)

    # Main simulation loop - each iteration represents one time step
    @info "Starting simulation for $(p.MAX_STEP) steps..."
    PICTURE_NUM = p.PICTURE_NUM
    
    for step in 0:(p.MAX_STEP-1)
        if step % 1000 == 0  # Progress indicator every 1000 steps
            @info "Step $step/$(p.MAX_STEP)"
        end

        # LBM Algorithm Steps (executed in sequence each time step):
        
        # 1. Apply periodic boundary conditions
        #    Particles leaving one edge re-enter from the opposite edge
        apply_periodic_boundary_conditions!(F)
        
        # 2. Streaming step: particles move along their velocity directions
        #    This is the advection part of the LBM
        streaming_step!(F, p.CX, p.CY)
        
        # 3. Handle cylinder boundary (no-slip condition)
        #    Particles hitting the cylinder bounce back (bounce-back method)
        cylinderF = handle_cylinder_boundary(F, cylinder, p.CX, p.CY, p.OPP)

        # 4. Compute macroscopic variables (density, velocity)
        #    These are the physical quantities we actually care about
        rho, ux, uy = compute_macroscopic_variables(cylinderF, cylinder, p.CX, p.CY)
        F = cylinderF

        # 5. Collision step: relaxation toward equilibrium
        #    This is the collision part of the LBM (BGK approximation)
        collision_step!(F, rho, ux, uy, p.CX, p.CY, p.WEIGHTS, p.relaxation_time)
        
        # 6. Apply inlet/outlet boundary conditions
        #    Maintains constant velocity at domain boundaries
        F, ux = apply_inflow_outflow_boundary_conditions!(F, rho, ux, p.CX, p.WEIGHTS, p.U_MAX)

        # 7. Visualization and output (every OUTPUT_STEP iterations, only if enabled)
        #    Generates plots showing vorticity and streamlines
        if p.VISUALIZE && step % p.OUTPUT_STEP == 0
            PICTURE_NUM = visualize_flow_field(step, ux, uy, cylinder, p.POSITION_OX, p.POSITION_OY,
                                               p.RADIUS, p.MAX_Y, p.MAX_X, PICTURE_NUM, p.NOTICKS)
        end
    end
    
    @info "Simulation completed successfully!"
    if p.VISUALIZE
        @info "Output images saved in: julia/"
    else
        @info "Performance mode: no visualization or file I/O performed."
    end
end

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================
# Run when called as a script (not when imported as a module)
if isinteractive() || abspath(PROGRAM_FILE) == @__FILE__
    main()
end
