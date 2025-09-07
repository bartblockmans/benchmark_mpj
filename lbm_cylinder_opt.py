import matplotlib.pyplot as plt
import numpy as np
import os

"""
Python Program: Lattice Boltzmann Method (D2Q9) - Fluid Flow Past a Fixed Cylinder
OPTIMIZED VERSION
================================================================================

This program simulates 2D fluid flow past a circular cylinder using the Lattice Boltzmann Method
with a D2Q9 lattice model. The simulation computes the evolution of fluid particle distribution
functions and visualizes the resulting flow field including vorticity and streamlines.

Key Features:
- D2Q9 lattice model with 9 velocity directions
- Reynolds number based flow simulation (Re = 200)
- Cylinder boundary condition handling with bounce-back method
- Periodic boundary conditions for domain edges
- Real-time visualization of vorticity and streamlines
- Organized output to dedicated 'python' subdirectory
- OPTIMIZED: Float32, buffer reuse, precomputed masks, slice-based streaming
- Performance mode: disable visualization for computational benchmarking

What This Code Does:
====================
1. Sets up a 400x100 computational domain with a cylinder at position (70, 50)
2. Implements the LBM algorithm with streaming and collision steps
3. Applies proper boundary conditions (no-slip at cylinder, periodic at edges)
4. Computes macroscopic variables (density, velocity) from distribution functions
5. Visualizes the flow field showing vorticity patterns and streamlines (optional)
6. Saves output images every 2000 steps for analysis (optional)

Physics Background:
==================
The Lattice Boltzmann Method is a computational fluid dynamics technique that:
- Models fluid as discrete particles moving on a regular lattice
- Uses distribution functions to represent particle populations
- Replaces the Navier-Stokes equations with simple collision and streaming rules
- Naturally handles complex boundaries and multiphase flows
- Provides accurate results for incompressible flows at moderate Reynolds numbers

This simulation specifically studies the classic problem of flow past a cylinder,
which exhibits phenomena like boundary layer separation, vortex shedding, and
wake formation - fundamental concepts in fluid dynamics.

Author: Bart Blockmans
Date: August 2025

"""

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "python")
os.makedirs(OUT_DIR, exist_ok=True)

def initialize_parameters():
    """
    Initialize all simulation parameters including domain size, lattice properties,
    cylinder geometry, and fluid properties.
    """
    # Fluid domain dimensions
    MAX_X = 400
    MAX_Y = 100
    
    # D2Q9 lattice model parameters
    # 9 velocity directions: center (0), cardinal directions (1-4), diagonal directions (5-8)
    LATTICE_NUM = 9
    # OPTIMIZATION: Use int8 for velocity components and float32 for weights
    CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int8)
    CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int8)
    WEIGHTS = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float32)
    OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int8)
    
    # Cylinder geometry and position
    POSITION_OX = 70    # x-coordinate of cylinder center
    POSITION_OY = 50    # y-coordinate of cylinder center
    RADIUS = 20         # cylinder radius
    
    # Fluid properties and Reynolds number
    REYNOLDS = 200      # Reynolds number for the flow
    # OPTIMIZATION: Use float32 for better memory bandwidth
    U_MAX = np.float32(0.1)        # maximum inlet velocity
    kinematic_viscosity = np.float32(U_MAX * 2 * RADIUS / REYNOLDS)
    relaxation_time = np.float32(3.0) * kinematic_viscosity + np.float32(0.5)
    
    # Simulation control parameters
    MAX_STEP = 20001    # total number of time steps
    OUTPUT_STEP = 2000  # frequency of output visualization
    PICTURE_NUM = 1     # counter for saved images
    
    # Visualization control flags
    # Set to False for performance benchmarking (no visualization or file I/O)
    # Set to True for normal operation with visualization and output
    VISUALIZE = False

    # Set to True for clean images without ticks, labels, title (animation mode)
    NOTICKS = False
    
    return (MAX_X, MAX_Y, LATTICE_NUM, CX, CY, WEIGHTS, OPP, 
            POSITION_OX, POSITION_OY, RADIUS, REYNOLDS, U_MAX, 
            kinematic_viscosity, relaxation_time, MAX_STEP, OUTPUT_STEP, PICTURE_NUM, VISUALIZE, NOTICKS)

def create_cylinder_mask(MAX_X, MAX_Y, POSITION_OX, POSITION_OY, RADIUS):
    """
    Create a boolean mask representing the cylinder in the computational domain.
    """
    x, y = np.meshgrid(range(MAX_X), range(MAX_Y))
    cylinder = (x - POSITION_OX)**2 + (y - POSITION_OY)**2 <= RADIUS**2
    return cylinder

def initialize_flow_field(MAX_Y, MAX_X, LATTICE_NUM, U_MAX, CX, CY, WEIGHTS):
    """
    Initialize the flow field with uniform density and inlet velocity.
    Sets up the initial distribution functions F based on equilibrium conditions.
    OPTIMIZED: All arrays are float32 end-to-end for better performance.
    """
    # Initialize density and velocity fields as float32
    rho = np.ones((MAX_Y, MAX_X), dtype=np.float32)           # uniform density field
    ux = np.zeros((MAX_Y, MAX_X), dtype=np.float32)           # zero velocity initially
    uy = np.zeros((MAX_Y, MAX_X), dtype=np.float32)           # zero velocity initially
    ux[:, 0], ux[:, -1] = U_MAX, U_MAX     # set inlet/outlet velocity
    
    # Initialize distribution functions F using equilibrium distribution as float32
    F = np.zeros((MAX_Y, MAX_X, LATTICE_NUM), dtype=np.float32)
    for i, cx, cy, w in zip(range(LATTICE_NUM), CX, CY, WEIGHTS):
        F[:, :, i] = rho * w * (1 + 3 * (cx * ux + cy * uy) 
                                + 9 * (cx * ux + cy * uy)**2 / 2 
                                - 3 * (ux**2 + uy**2) / 2)
    return rho, ux, uy, F

def apply_periodic_boundary_conditions(F, MAX_Y, MAX_X):
    """
    Apply periodic boundary conditions to the distribution functions.
    This ensures particles leaving one edge of the domain re-enter from the opposite edge.
    """
    # Periodic boundary in x-direction (left/right edges)
    F[:, 0, [1, 5, 8]] = F[:, -1, [1, 5, 8]]    # particles moving right at left edge
    F[:, -1, [3, 6, 7]] = F[:, 0, [3, 6, 7]]    # particles moving left at right edge
    
    # Periodic boundary in y-direction (top/bottom edges)
    F[0, :, [2, 5, 6]] = F[-1, :, [2, 5, 6]]    # particles moving up at bottom edge
    F[-1, :, [4, 7, 8]] = F[0, :, [4, 7, 8]]    # particles moving down at top edge

# OPTIMIZATION: Replace np.roll streaming with slice-based ping-pong (no temporaries)
def streaming_step_into(src, dst, CX, CY, TMP2D):
    """
    dst[y,x,i] = src[y - CY[i], x - CX[i], i] with periodic wrap, no temp arrays.
    This avoids np.roll allocations which create full plane copies per call.
    """
    H, W, Q = src.shape
    for i in range(Q):
        dx, dy = int(CX[i]), int(CY[i])
        # start with a view of src[:,:,i]
        s = src[:, :, i]
        d = dst[:, :, i]
        # shift in x (wrap) into TMP then shift in y (wrap) into d
        # x shift
        if dx == 0:
            np.copyto(TMP2D, s)
        elif dx == 1:
            TMP2D[:, 1:] = s[:, :-1]; TMP2D[:, 0] = s[:, -1]
        elif dx == -1:
            TMP2D[:, :-1] = s[:, 1:]; TMP2D[:, -1] = s[:, 0]
        # y shift
        if dy == 0:
            np.copyto(d, TMP2D)
        elif dy == 1:
            d[1:, :] = TMP2D[:-1, :]; d[0, :] = TMP2D[-1, :]
        elif dy == -1:
            d[:-1, :] = TMP2D[1:, :]; d[-1, :] = TMP2D[0, :]

# OPTIMIZATION: Precompute cylinder bounce-back masks (CORRECTED algorithm)
def precompute_cylinder_masks_corrected(cylinder, LATTICE_NUM, CX, CY):
    """
    Precompute incoming particle masks for bounce-back boundary conditions.
    Uses the CORRECTED algorithm that matches the baseline exactly.
    """
    incoming_masks = [None]*LATTICE_NUM
    incoming_masks[0] = np.zeros_like(cylinder, dtype=bool)  # unused for i=0 (rest particle)
    
    for i in range(1, LATTICE_NUM):
        # CORRECTED: Use the original mask logic that works properly
        roll_x = np.roll(cylinder, -CX[i], axis=1)
        roll_y = np.roll(cylinder, -CY[i], axis=0)
        incoming_masks[i] = cylinder & ~(roll_x & roll_y)
    
    return incoming_masks

# OPTIMIZATION: Unrolled streaming + bounce-back
def streaming_and_bounceback_fused(src, dst, cylinder, LATTICE_NUM, CX, CY, OPP, precomputed_masks, TMP2D):
    """
    OPTIMIZED fused streaming and bounce-back operation.
    Uses unrolled directions to eliminate if/elif branching for better performance.
    """
    H, W, Q = src.shape
    
    # Direction 0: rest particle (no streaming, no bounce-back)
    np.copyto(dst[:, :, 0], src[:, :, 0])
    
    # Direction 1: East (dx=1, dy=0)
    s = src[:, :, 1]
    d = dst[:, :, 1]
    TMP2D[:, 1:] = s[:, :-1]; TMP2D[:, 0] = s[:, -1]  # x shift
    np.copyto(d, TMP2D)  # y shift (dy=0)
    mask = precomputed_masks[1]; d[mask] = src[mask, OPP[1]]  # bounce-back
    
    # Direction 2: North (dx=0, dy=1)
    s = src[:, :, 2]
    d = dst[:, :, 2]
    np.copyto(TMP2D, s)  # x shift (dx=0)
    d[1:, :] = TMP2D[:-1, :]; d[0, :] = TMP2D[-1, :]  # y shift
    mask = precomputed_masks[2]; d[mask] = src[mask, OPP[2]]  # bounce-back
    
    # Direction 3: West (dx=-1, dy=0)
    s = src[:, :, 3]
    d = dst[:, :, 3]
    TMP2D[:, :-1] = s[:, 1:]; TMP2D[:, -1] = s[:, 0]  # x shift
    np.copyto(d, TMP2D)  # y shift (dy=0)
    mask = precomputed_masks[3]; d[mask] = src[mask, OPP[3]]  # bounce-back
    
    # Direction 4: South (dx=0, dy=-1)
    s = src[:, :, 4]
    d = dst[:, :, 4]
    np.copyto(TMP2D, s)  # x shift (dx=0)
    d[:-1, :] = TMP2D[1:, :]; d[-1, :] = TMP2D[0, :]  # y shift
    mask = precomputed_masks[4]; d[mask] = src[mask, OPP[4]]  # bounce-back
    
    # Direction 5: Northeast (dx=1, dy=1)
    s = src[:, :, 5]
    d = dst[:, :, 5]
    TMP2D[:, 1:] = s[:, :-1]; TMP2D[:, 0] = s[:, -1]  # x shift
    d[1:, :] = TMP2D[:-1, :]; d[0, :] = TMP2D[-1, :]  # y shift
    mask = precomputed_masks[5]; d[mask] = src[mask, OPP[5]]  # bounce-back
    
    # Direction 6: Northwest (dx=-1, dy=1)
    s = src[:, :, 6]
    d = dst[:, :, 6]
    TMP2D[:, :-1] = s[:, 1:]; TMP2D[:, -1] = s[:, 0]  # x shift
    d[1:, :] = TMP2D[:-1, :]; d[0, :] = TMP2D[-1, :]  # y shift
    mask = precomputed_masks[6]; d[mask] = src[mask, OPP[6]]  # bounce-back
    
    # Direction 7: Southwest (dx=-1, dy=-1)
    s = src[:, :, 7]
    d = dst[:, :, 7]
    TMP2D[:, :-1] = s[:, 1:]; TMP2D[:, -1] = s[:, 0]  # x shift
    d[:-1, :] = TMP2D[1:, :]; d[-1, :] = TMP2D[0, :]  # y shift
    mask = precomputed_masks[7]; d[mask] = src[mask, OPP[7]]  # bounce-back
    
    # Direction 8: Southeast (dx=1, dy=-1)
    s = src[:, :, 8]
    d = dst[:, :, 8]
    TMP2D[:, 1:] = s[:, :-1]; TMP2D[:, 0] = s[:, -1]  # x shift
    d[:-1, :] = TMP2D[1:, :]; d[-1, :] = TMP2D[0, :]  # y shift
    mask = precomputed_masks[8]; d[mask] = src[mask, OPP[8]]  # bounce-back

def handle_cylinder_boundary_inplace(destF, srcF, cylinder, LATTICE_NUM, CX, CY, OPP, precomputed_masks):
    """
    Handle the no-slip boundary condition at the cylinder surface using CORRECTED bounce-back.
    
    OPTIMIZED VERSION:
    - Uses precomputed masks to avoid repeated np.roll calls
    - Eliminates array copying with direct assignment
    - Vectorized operations for better performance
    
    This function now uses the CORRECTED algorithm that matches the baseline exactly.
    """
    # Direct assignment instead of copy (destF already contains srcF from streaming)
    
    # Apply bounce-back using precomputed masks (much faster)
    for i in range(1, LATTICE_NUM):  # skip the rest particle (i=0)
        mask = precomputed_masks[i]
        destF[mask, i] = srcF[mask, OPP[i]]

# OPTIMIZATION: Fused macroscopic moments computation (single pass)
def compute_macroscopic_variables_optimized(F, cylinder, LATTICE_NUM, CX, CY):
    """
    Compute macroscopic fluid variables using OPTIMIZED single-pass computation.
    Fuses density and momentum calculations into one loop over directions for better performance.
    """
    H, W, Q = F.shape
    rho = np.zeros((H, W), dtype=F.dtype)
    mx = np.zeros((H, W), dtype=F.dtype)  # momentum x
    my = np.zeros((H, W), dtype=F.dtype)  # momentum y
    
    # Single pass: accumulate density and momentum
    for i in range(Q):
        Fi = F[:, :, i]
        rho += Fi
        mx += Fi * CX[i]
        my += Fi * CY[i]
    
    # Compute velocities
    ux = mx / rho
    uy = my / rho
    
    # Set velocity to zero inside the cylinder (no-slip condition)
    ux[cylinder] = 0.0
    uy[cylinder] = 0.0
    
    return rho, ux, uy

# OPTIMIZATION: Collision step in-place per direction, reuse a 2D scratch
def collision_step_inplace(F, rho, ux, uy, LATTICE_NUM, CX, CY, WEIGHTS, c1, c3, c45, c15, invtau, TMP2D):
    """
    Perform collision step in-place using 2D scratch buffer to avoid 3D Feq allocation.
    OPTIMIZED VERSION with precomputed constants for better performance.
    """
    usq = ux*ux + uy*uy  # 2D
    
    for i, cx, cy, w in zip(range(LATTICE_NUM), CX, CY, WEIGHTS):
        cu = cx*ux + cy*uy
        # TMP2D = 1 + 3*cu + 4.5*cu^2 - 1.5*|u|^2   (all in float32)
        np.multiply(cu, cu, out=TMP2D)                 # TMP = cu^2
        TMP2D *= c45
        TMP2D += c3*cu
        TMP2D += c1
        TMP2D -= c15*usq
        # feq = rho * w * TMP2D  (reuse TMP2D)
        TMP2D *= (rho * w)
        # F[:,:,i] += -(1/tau) * (F[:,:,i] - feq)
        np.subtract(F[:, :, i], TMP2D, out=TMP2D)      # TMP = F - feq
        F[:, :, i] += (-invtau) * TMP2D

# OPTIMIZATION: Vectorized inlet/outlet boundary conditions
def apply_inflow_outflow_boundary_conditions_optimized(F, rho, ux, LATTICE_NUM, CX, WEIGHTS, U_MAX):
    """
    Apply velocity boundary conditions at inlet and outlet with OPTIMIZED vectorized operations.
    Computes all 9 directions at once using broadcasting for better performance.
    """
    # Set boundary velocities
    ux[:, 0] = U_MAX
    ux[:, -1] = U_MAX
    
    # Left edge: vectorized across all directions
    usqL = ux[:, 0] * ux[:, 0]  # (H,)
    cuL = ux[:, 0, None] * CX[None, :]  # (H, 9) - broadcast ux with CX
    tmpL = 1.0 + 3.0*cuL + 4.5*cuL**2 - 1.5*usqL[:, None]  # (H, 9)
    F[:, 0, :] = rho[:, 0, None] * WEIGHTS[None, :] * tmpL  # (H, 9)
    
    # Right edge: vectorized across all directions
    usqR = ux[:, -1] * ux[:, -1]  # (H,)
    cuR = ux[:, -1, None] * CX[None, :]  # (H, 9) - broadcast ux with CX
    tmpR = 1.0 + 3.0*cuR + 4.5*cuR**2 - 1.5*usqR[:, None]  # (H, 9)
    F[:, -1, :] = rho[:, -1, None] * WEIGHTS[None, :] * tmpR  # (H, 9)

# Legacy functions (kept for reference, not used in optimized version)
def streaming_step(F, LATTICE_NUM, CX, CY):
    """
    Perform the streaming step of the LBM algorithm.
    Particles move along their velocity directions to neighboring lattice sites.
    """
    for i, cx, cy in zip(range(LATTICE_NUM), CX, CY):
        F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)  # stream in x-direction
        F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)  # stream in y-direction

def handle_cylinder_boundary(F, cylinder, LATTICE_NUM, CX, CY, OPP):
    """
    Handle the no-slip boundary condition at the cylinder surface.
    Implements the bounce-back method where particles hitting the cylinder
    reverse their direction.
    """
    cylinderF = F.copy()
    for i in range(1, LATTICE_NUM):  # skip the rest particle (i=0)
        # Find incoming particles: inside cylinder but moving toward cylinder surface
        incoming_particles = (cylinder & 
                            (np.roll(cylinder, -CX[i], axis=1) & 
                             np.roll(cylinder, -CY[i], axis=0) == False))
        # Bounce back: reverse particle direction
        cylinderF[incoming_particles, i] = F[incoming_particles, OPP[i]]
    return cylinderF

def compute_macroscopic_variables(F, cylinder, LATTICE_NUM, CX, CY):
    """
    Compute macroscopic fluid variables (density and velocity) from the
    distribution functions using moment integrals.
    """
    # Density: sum of all distribution functions
    rho = np.sum(F, axis=2)
    
    # Velocity: momentum divided by density
    ux = np.sum(F * CX, axis=2) / rho
    uy = np.sum(F * CY, axis=2) / rho
    
    # Set velocity to zero inside the cylinder (no-slip condition)
    ux[cylinder] = 0
    uy[cylinder] = 0
    
    return rho, ux, uy

def collision_step(F, rho, ux, uy, LATTICE_NUM, CX, CY, WEIGHTS, relaxation_time):
    """
    Perform the collision step using the BGK (Bhatnagar-Gross-Krook) approximation.
    Distribution functions relax toward their equilibrium values.
    """
    # Compute equilibrium distribution functions
    Feq = np.zeros(F.shape)
    for i, cx, cy, w in zip(range(LATTICE_NUM), CX, CY, WEIGHTS):
        Feq[:, :, i] = rho * w * (1 + 3 * (cx * ux + cy * uy) 
                                  + 9 * (cx * ux + cy * uy)**2 / 2 
                                  - 3 * (ux**2 + uy**2) / 2)
    
    # BGK collision: relax toward equilibrium
    F += -(1 / relaxation_time) * (F - Feq)

def apply_inflow_outflow_boundary_conditions(F, rho, ux, LATTICE_NUM, CX, WEIGHTS, U_MAX):
    """
    Apply velocity boundary conditions at inlet and outlet.
    Maintains constant velocity at domain boundaries.
    """
    # Set boundary velocities
    ux[:, 0], ux[:, -1] = U_MAX, U_MAX
    
    # Update distribution functions at boundaries using equilibrium distribution
    for i, cx, w in zip(range(LATTICE_NUM), CX, WEIGHTS):
        # Inlet boundary (left edge)
        F[:, 0, i] = rho[:, 0] * w * (1 + 3 * (cx * ux[:, 0]) 
                                      + 9 * (cx * ux[:, 0])**2 / 2 
                                      - 3 * (ux[:, 0]**2) / 2)
        # Outlet boundary (right edge)
        F[:, -1, i] = rho[:, -1] * w * (1 + 3 * (cx * ux[:, -1]) 
                                        + 9 * (cx * ux[:, -1])**2 / 2 
                                        - 3 * (ux[:, -1]**2) / 2)

def visualize_flow_field(step, ux, uy, cylinder, POSITION_OX, POSITION_OY, RADIUS, 
                        MAX_Y, MAX_X, PICTURE_NUM, NOTICKS):
    """
    Visualize the flow field by plotting vorticity and streamlines.
    Saves the plot as an image file.
    
    When NOTICKS is True, creates clean images without ticks, labels, title,
    and with minimized margins for publication use.
    """
    # Compute vorticity (curl of velocity field)
    vorticity = ((np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) 
                 - (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)))
    vorticity[cylinder] = np.nan  # hide vorticity inside cylinder
    
    # Create the visualization
    plt.imshow(vorticity, cmap="RdBu", origin="lower", vmin=-0.02, vmax=0.02)
    
    # Add cylinder outline
    plt.gca().add_patch(plt.Circle((POSITION_OX, POSITION_OY), RADIUS, color="black"))
    
    # Add streamlines colored by velocity magnitude
    # Y, X = np.mgrid[0:MAX_Y, 0:MAX_X]
    # speed = np.sqrt(ux**2 + uy**2)
    # plt.streamplot(X, Y, ux, uy, color=speed, linewidth=1, cmap='cool')
    
    # Apply NOTICKS styling if requested
    if NOTICKS:
        # Clean layout: no ticks, labels, title, minimized margins
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('')
        plt.ylabel('')
        plt.title('')
        # Minimize margins while keeping the plot box
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    else:
        # Normal layout: standard title
        plt.title(f'Vorticity and streamlines at step {step}')
    
    # Figure limits
    plt.xlim([0, MAX_X])
    plt.ylim([0, MAX_Y])

    # Set title and save
    plt.savefig(os.path.join(OUT_DIR, f"lattice_boltzmann_{PICTURE_NUM:04d}_python.png"))
    plt.pause(0.01)  # brief pause for real-time visualization
    plt.cla()        # clear current axes for next iteration
    
    return PICTURE_NUM + 1

def main():
    """
    Main simulation loop implementing the Lattice Boltzmann Method.
    
    This function orchestrates the complete LBM simulation by:
    1. Setting up the computational domain and initial conditions
    2. Running the main time-stepping loop with 7 key algorithm steps
    3. Generating visualizations at regular intervals (optional)
    4. Saving results to the 'python' output directory (optional)
    
    OPTIMIZATIONS APPLIED:
    - Float32 for better memory bandwidth
    - Precomputed cylinder masks and streaming indices
    - Slice-based streaming (no np.roll allocations)
    - In-place collision step (no 3D Feq allocation)
    - Optimized macroscopic variable computation
    - Buffer reuse to avoid step-wise allocations
    """
    # Initialize all parameters
    (MAX_X, MAX_Y, LATTICE_NUM, CX, CY, WEIGHTS, OPP, 
     POSITION_OX, POSITION_OY, RADIUS, REYNOLDS, U_MAX, 
     kinematic_viscosity, relaxation_time, MAX_STEP, OUTPUT_STEP, PICTURE_NUM, VISUALIZE, NOTICKS) = initialize_parameters()
    
    # Print simulation parameters for user reference
    print(f"Reynolds number = {REYNOLDS}")
    print(f"Relaxation time = {relaxation_time}")
    print(f"Domain size: {MAX_X} x {MAX_Y}")
    print(f"Maximum velocity = {U_MAX}")
    print(f"Kinematic viscosity = {kinematic_viscosity}")
    print(f"Visualization enabled: {VISUALIZE}")
    if VISUALIZE:
        print(f"Output directory: {OUT_DIR}")
    
    # Create cylinder mask (defines where the solid cylinder is located)
    cylinder = create_cylinder_mask(MAX_X, MAX_Y, POSITION_OX, POSITION_OY, RADIUS)
    
    # OPTIMIZATION: Precompute cylinder masks for corrected bounce-back
    incoming_masks = precompute_cylinder_masks_corrected(cylinder, LATTICE_NUM, CX, CY)
    
    # Initialize flow field with uniform density and inlet velocity
    rho, ux, uy, F = initialize_flow_field(MAX_Y, MAX_X, LATTICE_NUM, U_MAX, CX, CY, WEIGHTS)
    
    # OPTIMIZATION: Preallocate buffers for reuse (no step-wise big allocations)
    Feq = np.empty_like(F)                  # reuse in collision (if needed)
    F2 = np.empty_like(F)                   # ping-pong / bounce-back dest
    TMP2D = np.empty((MAX_Y, MAX_X), np.float32)  # scratch for 2D formulas
    
    # OPTIMIZATION: Precompute constants to avoid repeated type conversion
    F_dtype = F.dtype
    c1 = F_dtype.type(1.0)
    c3 = F_dtype.type(3.0)
    c45 = F_dtype.type(4.5)
    c15 = F_dtype.type(1.5)
    invtau = F_dtype.type(1.0) / relaxation_time
    
    # Precompute WEIGHTS as float32 for boundary conditions
    WEIGHTS_f32 = WEIGHTS.astype(np.float32)
    CX_f32 = CX.astype(np.float32)
    
    # Main simulation loop - each iteration represents one time step
    print(f"Starting simulation for {MAX_STEP} steps...")
    for step in range(MAX_STEP):
        if step % 1000 == 0:  # Progress indicator every 1000 steps
            print(f"Step {step}/{MAX_STEP}")
        
        # LBM Algorithm Steps (executed in sequence each time step):
        
        # 1-3. OPTIMIZED: Fused streaming + bounce-back (includes periodic BC)
        #    Combines streaming, periodic boundary conditions, and bounce-back in one operation
        #    Periodic BC is handled within the streaming kernel for better performance
        streaming_and_bounceback_fused(F, F2, cylinder, LATTICE_NUM, CX, CY, OPP, incoming_masks, TMP2D)
        
        # Swap buffers efficiently (no temporary references)
        F, F2 = F2, F
        
        # 4. Compute macroscopic variables (density, velocity)
        #    These are the physical quantities we actually care about
        rho, ux, uy = compute_macroscopic_variables_optimized(F, cylinder, LATTICE_NUM, CX, CY)
        
        # 5. Collision step: relaxation toward equilibrium
        #    This is the collision part of the LBM (BGK approximation)
        collision_step_inplace(F, rho, ux, uy, LATTICE_NUM, CX, CY, WEIGHTS, c1, c3, c45, c15, invtau, TMP2D)
        
        # 6. Apply inlet/outlet boundary conditions
        #    Maintains constant velocity at domain boundaries
        apply_inflow_outflow_boundary_conditions_optimized(F, rho, ux, LATTICE_NUM, CX_f32, WEIGHTS_f32, U_MAX)
        
        # 7. Visualization and output (every OUTPUT_STEP iterations)
        #    Generates plots showing vorticity and streamlines
        if VISUALIZE and step % OUTPUT_STEP == 0:
            PICTURE_NUM = visualize_flow_field(step, ux, uy, cylinder, POSITION_OX, 
                                             POSITION_OY, RADIUS, MAX_Y, MAX_X, PICTURE_NUM, NOTICKS)
    
    print("Simulation completed successfully!")
    if VISUALIZE:
        print(f"Output images saved in: {OUT_DIR}")
    else:
        print("Performance mode: no visualization or file I/O performed.")

if __name__ == "__main__":
    main()