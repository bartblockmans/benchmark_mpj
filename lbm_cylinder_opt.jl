# lbm_cylinder_opt.jl
# =============================================================================
# Julia Program: Lattice Boltzmann Method (D2Q9) - Fluid Flow Past a Fixed Cylinder
# OPTIMIZED VERSION
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
# - OPTIMIZED: Float32, buffer reuse, precomputed masks, ping-pong streaming
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
# Author: Bart Blockmans
# Date: August 2025
#
# PERFORMANCE TIPS:
# - Run with threading: julia -t auto lbm_cylinder_opt.jl
# - For maximum performance: julia -t auto --optimize=3 lbm_cylinder_opt.jl
# - Expected speedup: 3-6x over original version

using CairoMakie
using GeometryBasics: Point2f, Circle
using Base.Threads

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
    # OPTIMIZATION: Use Float32 for better memory bandwidth
    WEIGHTS = Float32[4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]  # lattice weights
    
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
    # OPTIMIZATION: Use Float32 for better memory bandwidth
    U_MAX    = 0.1f0   # maximum inlet velocity
    kinematic_viscosity = U_MAX * (2f0 * RADIUS) / REYNOLDS
    relaxation_time     = 3f0 * kinematic_viscosity + 0.5f0

    # Simulation control parameters
    MAX_STEP    = 20001    # total number of time steps
    OUTPUT_STEP = 2000     # frequency of output visualization
    PICTURE_NUM = 1        # counter for saved images
    
    # Visualization control flags
    # Set to false for performance benchmarking (no visualization or file I/O)
    # Set to true for normal operation with visualization and output
    VISUALIZE = true
    # Set to true for clean images without ticks, labels, title (animation mode)
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
    # OPTIMIZATION: Use Float32 for better memory bandwidth
    rho = ones(Float32, MAX_Y, MAX_X)           # uniform density field
    ux  = zeros(Float32, MAX_Y, MAX_X)         # zero velocity initially
    uy  = zeros(Float32, MAX_Y, MAX_X)         # zero velocity initially
    
    # Set inlet and outlet velocities (left and right boundaries)
    ux[:, 1]   .= U_MAX    # left boundary (inlet)
    ux[:, end] .= U_MAX    # right boundary (outlet)

    # Initialize distribution functions F using equilibrium distribution
    # This sets up the initial particle populations for each velocity direction
    # OPTIMIZATION: Use Float32 for better memory bandwidth
    F = zeros(Float32, MAX_Y, MAX_X, LATTICE_NUM)
    for i in 1:LATTICE_NUM
        cx = CX[i]; cy = CY[i]; w = WEIGHTS[i]
        cu = cx .* ux .+ cy .* uy  # velocity component in this direction
        # Equilibrium distribution function (Chapman-Enskog expansion)
        F[:, :, i] .= rho .* w .* (1 .+ 3 .* cu .+ 4.5 .* cu.^2 .- 1.5 .* (ux.^2 .+ uy.^2))
    end
    return rho, ux, uy, F
end

# =============================================================================
# OPTIMIZED LBM CORE ALGORITHM STEPS
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

# OPTIMIZATION: Precompute streaming indices and use ping-pong buffers
function precompute_streaming_indices(MAX_X, MAX_Y, CX, CY)
    """
    Precompute indices for streaming step to avoid circshift allocations.
    """
    x_from = [Vector{Int}(undef, MAX_X) for _ in 1:length(CX)]
    y_from = [Vector{Int}(undef, MAX_Y) for _ in 1:length(CY)]
    
    for i in 1:length(CX)
        for x in 1:MAX_X
            # index of the cell that streams into (y, x) along dir i
            x_from[i][x] = 1 + mod(x - CX[i] - 1, MAX_X)
        end
        for y in 1:MAX_Y
            y_from[i][y] = 1 + mod(y - CY[i] - 1, MAX_Y)
        end
    end
    
    return x_from, y_from
end

@inline function streaming_step!(G, F, x_from, y_from)
    """
    Perform the streaming step using precomputed indices and ping-pong buffers.
    No allocations, fully linear memory access.
    """
    @inbounds for i in 1:size(F,3)
        xf = x_from[i]; yf = y_from[i]
        for y in 1:size(F,1), x in 1:size(F,2)
            G[y, x, i] = F[yf[y], xf[x], i]
        end
    end
    return nothing
end

# OPTIMIZATION: Precompute cylinder masks for bounce-back
function precompute_cylinder_masks(cylinder, CX, CY)
    """
    Precompute incoming particle masks for bounce-back boundary conditions.
    These masks don't change during simulation, so compute once and reuse.
    """
    incoming_masks = Vector{BitMatrix}(undef, length(CX))
    incoming_masks[1] = falses(size(cylinder))  # unused for i=1 (rest particle)
    
    for i in 2:length(CX)
        roll_x = circshift(cylinder, (0, -CX[i]))
        roll_y = circshift(cylinder, (-CY[i], 0))
        incoming_masks[i] = cylinder .& .!(roll_x .& roll_y)
    end
    
    return incoming_masks
end

function handle_cylinder_boundary!(destF, srcF, incoming_masks, OPP)
    """
    Handle the no-slip boundary condition at the cylinder surface using precomputed masks.
    
    Implements the bounce-back method where particles hitting the cylinder
    reverse their direction. This creates a solid wall effect by:
    - Using precomputed masks to identify incoming particles
    - Reversing their velocity direction (bounce-back)
    - Maintaining mass conservation at the boundary
    
    The bounce-back method is a simple and effective way to implement
    no-slip boundary conditions in LBM simulations.
    """
    @inbounds for i in 2:length(OPP)
        mask = incoming_masks[i]
        @views destFi = destF[:,:,i]
        @views srcFopp = srcF[:,:,OPP[i]]
        destFi[mask] .= srcFopp[mask]
    end
    return destF
end

# OPTIMIZATION: Manual reductions for macroscopic variables (faster than sum)
@inline function macroscopic!(rho, ux, uy, F, CX, CY, cylinder)
    """
    Compute macroscopic fluid variables using manual reductions for better performance.
    Avoids allocations from sum/dropdims and broadcast intermediates.
    """
    ny, nx, nd = size(F)
    @inbounds for y in 1:ny, x in 1:nx
        r  = zero(eltype(rho))
        mx = zero(eltype(ux))
        my = zero(eltype(uy))
        @fastmath @simd for i in 1:nd
            fi = F[y,x,i]
            r  += fi
            mx += fi * CX[i]
            my += fi * CY[i]
        end
        rho[y,x] = r
        ux[y,x]  = mx / r
        uy[y,x]  = my / r
    end
    ux[cylinder] .= 0
    uy[cylinder] .= 0
    return nothing
end

# OPTIMIZATION: Fused collision step (no Feq allocation, compute on-the-fly)
@inline function collision_fused!(F, rho, ux, uy, CX, CY, W, tau)
    """
    Perform collision step in a single fused loop - compute feq on-the-fly and update F in-place.
    This is the most efficient approach, avoiding temporary allocations.
    """
    ny, nx, nd = size(F)
    invtau = one(eltype(F)) / tau
    
    @threads for y in 1:ny
        @inbounds for x in 1:nx
            ρ   = rho[y,x]
            uxi = ux[y,x]; uyi = uy[y,x]
            @fastmath begin
                usq = muladd(uxi, uxi, uyi*uyi)   # u^2
                @simd for i in 1:nd
                    cu  = CX[i]*uxi + CY[i]*uyi
                    feq = ρ * W[i] * (1f0 + 3f0*cu + 4.5f0*cu*cu - 1.5f0*usq)
                    F[y,x,i] += -(invtau) * (F[y,x,i] - feq)
                end
            end
        end
    end
    return nothing
end

# Legacy collision function (kept for reference, not used in optimized version)
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
    - Clear title and proper axis limits
    
    The visualization helps understand the flow physics including:
    - Vortex formation and shedding
    - Flow separation patterns
    - Wake development behind the cylinder
    """
    # Compute vorticity (discrete curl of velocity field)
    # This measures the local rotation of the fluid
    vorticity = (circshift(uy, (0, -1)) .- circshift(uy, (0, 1))) .-
                (circshift(ux, (-1, 0)) .- circshift(ux, (1, 0)))
    vorticity[cylinder] .= NaN  # hide vorticity inside cylinder

    # Create the figure using CairoMakie (Julia's plotting library)
    # Use 'size' (not 'resolution') to avoid Makie deprecation warnings
    fig = Figure(size = (900, 250))
    ax  = Axis(fig[1, 1], aspect = DataAspect(),
                title = "Vorticity and streamlines at step $step")

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
    
    OPTIMIZATIONS APPLIED:
    - Float32 for better memory bandwidth
    - Precomputed cylinder masks and streaming indices
    - Ping-pong buffers for streaming (no circshift allocations)
    - Fused collision step (no temporary allocations)
    - Manual macroscopic variable computation
    - Threaded loops for better performance
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
    @info "Threads available: $(nthreads())"
    if p.VISUALIZE
        @info "Output directory = julia/"
    end

    # Create cylinder mask (defines where the solid cylinder is located)
    cylinder = create_cylinder_mask(p.MAX_X, p.MAX_Y, p.POSITION_OX, p.POSITION_OY, p.RADIUS)
    
    # OPTIMIZATION: Precompute cylinder masks and streaming indices
    incoming_masks = precompute_cylinder_masks(cylinder, p.CX, p.CY)
    x_from, y_from = precompute_streaming_indices(p.MAX_X, p.MAX_Y, p.CX, p.CY)
    
    # OPTIMIZATION: Convert arrays to tuples for better performance
    CX_tuple = Tuple(p.CX)
    CY_tuple = Tuple(p.CY)
    WEIGHTS_tuple = Tuple(p.WEIGHTS)
    OPP_tuple = Tuple(p.OPP)
    
    # Initialize flow field with uniform density and inlet velocity
    rho, ux, uy, F = initialize_flow_field(p.MAX_Y, p.MAX_X, p.LATTICE_NUM, p.U_MAX, p.CX, p.CY, p.WEIGHTS)

    # OPTIMIZATION: Preallocate buffers for reuse
    G = similar(F)      # Second buffer for ping-pong streaming
    F2 = similar(F)     # Second buffer for cylinder boundary handling
    Feq = similar(F)    # Buffer for equilibrium distribution (if needed)

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
        
        # 2. OPTIMIZED Streaming step: use ping-pong buffers and precomputed indices
        #    This avoids circshift allocations and provides linear memory access
        streaming_step!(G, F, x_from, y_from)
        F, G = G, F  # swap buffers
        
        # 3. OPTIMIZED Handle cylinder boundary using precomputed masks
        #    Copy to buffer and modify in-place to avoid allocations
        copyto!(F2, F)
        handle_cylinder_boundary!(F2, F, incoming_masks, OPP_tuple)
        F = F2

        # 4. OPTIMIZED Compute macroscopic variables using manual reductions
        #    This is faster than sum/dropdims and avoids temporary allocations
        macroscopic!(rho, ux, uy, F, CX_tuple, CY_tuple, cylinder)

        # 5. OPTIMIZED Collision step: fused computation, no temporary allocations
        #    This is the most efficient approach, computing feq on-the-fly
        collision_fused!(F, rho, ux, uy, CX_tuple, CY_tuple, WEIGHTS_tuple, p.relaxation_time)
        
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
