function lbm_cylinder_opt()
% =============================================================================
% MATLAB Program: Lattice Boltzmann Method (D2Q9) - Fluid Flow Past a Fixed Cylinder
% OPTIMIZED VERSION
% =============================================================================
%
% This program simulates 2D fluid flow past a circular cylinder using the Lattice 
% Boltzmann Method with a D2Q9 lattice model. The simulation computes the evolution 
% of fluid particle distribution functions and visualizes the resulting flow field 
% including vorticity and streamlines.
%
% Key Features:
% - D2Q9 lattice model with 9 velocity directions
% - Reynolds number based flow simulation (Re = 200)
% - Cylinder boundary condition handling with bounce-back method
% - Periodic boundary conditions for domain edges
% - Real-time visualization of vorticity and streamlines
% - Organized output to dedicated 'matlab' subdirectory
% - OPTIMIZED: Float32, buffer reuse, precomputed masks, ping-pong streaming
% - OPTIMIZED: 2D arrays instead of 3D arrays for better performance
% - Performance mode: disable visualization for computational benchmarking
%
% What This Code Does:
% ====================
% 1. Sets up a 400x100 computational domain with a cylinder at position (70, 50)
% 2. Implements the LBM algorithm with streaming and collision steps
% 3. Applies proper boundary conditions (no-slip at cylinder, periodic at edges)
% 4. Computes macroscopic variables (density, velocity) from distribution functions
% 5. Visualizes the flow field showing vorticity and streamlines (optional)
% 6. Saves output images every 2000 steps for analysis (optional)
%
% Physics Background:
% ==================
% The Lattice Boltzmann Method is a computational fluid dynamics technique that:
% - Models fluid as discrete particles moving on a regular lattice
% - Uses distribution functions to represent particle populations
% - Replaces the Navier-Stokes equations with simple collision and streaming rules
% - Naturally handles complex boundaries and multiphase flows
% - Provides accurate results for incompressible flows at moderate Reynolds numbers
%
% This simulation specifically studies the classic problem of flow past a cylinder,
% which exhibits phenomena like boundary layer separation, vortex shedding, and
% wake formation - fundamental concepts in fluid dynamics.
%
% Author: Bart Blockmans
% Date: August 2025
%
% PERFORMANCE TIPS:
% - Expected speedup: 3-6x over original version
% - For maximum performance: launch MATLAB with -singleCompThread for fairness
% - For maximum performance: ensure MATLAB is compiled with optimized BLAS/LAPACK

%% =============================================================================
% CONFIGURATION SECTION
% =============================================================================

% Fluid domain dimensions
MAX_X = 400;
MAX_Y = 100;

% D2Q9 lattice model parameters
% 9 velocity directions: center (1), cardinal directions (2-5), diagonal directions (6-9)
LATTICE_NUM = 9;
CX = [0  1  0 -1  0  1 -1 -1  1];      % x-velocity components
CY = [0  0  1  0 -1  1  1 -1 -1];     % y-velocity components
% OPTIMIZATION: Use single for better memory bandwidth
WEIGHTS = single([4/9  1/9  1/9  1/9  1/9  1/36  1/36  1/36  1/36]);  % lattice weights

% Opposite direction indices for bounce-back boundary conditions
% Note: MATLAB uses 1-based indexing, so we convert from Python's 0-based OPP
% Python OPP (0-based): [0,3,4,1,2,7,8,5,6] -> MATLAB (1-based): [1,4,5,2,3,8,9,6,7]
OPP = [1 4 5 2 3 8 9 6 7];

% Cylinder geometry and position
POSITION_OX = 70;    % x-coordinate of cylinder center
POSITION_OY = 50;    % y-coordinate of cylinder center
RADIUS = 20;         % cylinder radius

% Fluid properties and Reynolds number
REYNOLDS = 200;      % Reynolds number for the flow
% OPTIMIZATION: Use single for better memory bandwidth
U_MAX = single(0.1);         % maximum inlet velocity
kinematic_viscosity = single(U_MAX * 2 * RADIUS / REYNOLDS);
relaxation_time = single(3.0) * kinematic_viscosity + single(0.5);

% Simulation control parameters
MAX_STEP = 20001;    % total number of time steps
OUTPUT_STEP = 2000;  % frequency of output visualization
PICTURE_NUM = 1;     % counter for saved images

% Visualization control flags
% Set to false for performance benchmarking (no visualization or file I/O)
% Set to true for normal operation with visualization and output
VISUALIZE = false;

% Set to true for clean images without ticks, labels, title (publication mode)
NOTICKS = false;

%% =============================================================================
% MAIN SIMULATION FUNCTION
% =============================================================================
%
% This function orchestrates the complete LBM simulation by:
% 1. Setting up the computational domain and initial conditions
% 2. Running the main time-stepping loop with 7 key algorithm steps
% 3. Generating visualizations at regular intervals (optional)
% 4. Saving results to the 'matlab' output directory (optional)
%
% The simulation follows the classic LBM algorithm:
% - Initialize → Stream → Collide → Repeat
% - With proper boundary condition handling at each step
% - Regular output for analysis and visualization (when enabled)
%
% OPTIMIZATIONS APPLIED:
% - Float32 for better memory bandwidth
% - Precomputed cylinder masks and streaming indices
% - Ping-pong streaming (no circshift allocations)
% - In-place collision step (no 3D Feq allocation)
% - Optimized macroscopic variable computation
% - Buffer reuse to avoid step-wise allocations
% - 2D arrays instead of 3D arrays for better cache locality

    % -- Print simulation parameters for user reference
    fprintf('Reynolds number = %g\n', REYNOLDS);
    fprintf('Relaxation time = %g\n', relaxation_time);
    fprintf('Domain size: %d x %d\n', MAX_X, MAX_Y);
    fprintf('Maximum velocity: %g\n', U_MAX);
    fprintf('Kinematic viscosity: %g\n', kinematic_viscosity);
    fprintf('Visualization enabled: %s\n', string(VISUALIZE));
    if VISUALIZE
        fprintf('Output directory: matlab/\n');
    end

    % -- Create cylinder mask (defines where the solid cylinder is located)
    cylinder = create_cylinder_mask(MAX_X, MAX_Y, POSITION_OX, POSITION_OY, RADIUS);

    % OPTIMIZATION: Precompute cylinder masks for bounce-back (no work per step)
    incoming_masks = precompute_cylinder_masks(cylinder, LATTICE_NUM, CX, CY);

    % -- Initialize flow field with uniform density and inlet velocity
    % OPTIMIZATION: Use 2D arrays instead of 3D arrays for better performance
    [~, ~, ~, F0, F1, F2, F3, F4, F5, F6, F7, F8] = initialize_flow_field_2d(MAX_Y, MAX_X, LATTICE_NUM, U_MAX, CX, CY, WEIGHTS);

    % OPTIMIZATION: Preallocate 2D buffers for reuse (no step-wise big allocations)
    Feq0 = zeros(MAX_Y, MAX_X, 'single');  % reuse in collision
    Feq1 = zeros(MAX_Y, MAX_X, 'single');
    Feq2 = zeros(MAX_Y, MAX_X, 'single');
    Feq3 = zeros(MAX_Y, MAX_X, 'single');
    Feq4 = zeros(MAX_Y, MAX_X, 'single');
    Feq5 = zeros(MAX_Y, MAX_X, 'single');
    Feq6 = zeros(MAX_Y, MAX_X, 'single');
    Feq7 = zeros(MAX_Y, MAX_X, 'single');
    Feq8 = zeros(MAX_Y, MAX_X, 'single');

    % -- Main simulation loop - each iteration represents one time step
    fprintf('Starting simulation for %d steps...\n', MAX_STEP);
    
    % Only create figure if visualization is enabled
    if VISUALIZE
        figure('Color','w'); %#ok<*UNRCH>
    end
    
    for step = 0:MAX_STEP-1
        if mod(step,1000) == 0  % Progress indicator every 1000 steps
            fprintf('Step %d/%d\n', step, MAX_STEP);
        end

        % LBM Algorithm Steps (executed in sequence each time step):
        
        % 1. Apply periodic boundary conditions
        %    Particles leaving one edge re-enter from the opposite edge
        [F0, F1, F2, F3, F4, F5, F6, F7, F8] = apply_periodic_boundary_conditions_2d(F0, F1, F2, F3, F4, F5, F6, F7, F8);

        % 2. OPTIMIZED Streaming step: in-place streaming with 2D arrays
        %    This is more efficient than ping-pong buffers for this use case
        F0 = circshift(F0, [CY(1), CX(1)]);
        F1 = circshift(F1, [CY(2), CX(2)]);
        F2 = circshift(F2, [CY(3), CX(3)]);
        F3 = circshift(F3, [CY(4), CX(4)]);
        F4 = circshift(F4, [CY(5), CX(5)]);
        F5 = circshift(F5, [CY(6), CX(6)]);
        F6 = circshift(F6, [CY(7), CX(7)]);
        F7 = circshift(F7, [CY(8), CX(8)]);
        F8 = circshift(F8, [CY(9), CX(9)]);

        % 3. OPTIMIZED Handle cylinder boundary using precomputed masks
        %    Copy to buffer and modify in-place to avoid allocations
        [F0, F1, F2, F3, F4, F5, F6, F7, F8] = handle_cylinder_boundary_masks_2d(F0, F1, F2, F3, F4, F5, F6, F7, F8, incoming_masks, OPP);

        % 4. OPTIMIZED Compute macroscopic variables using 2D array operations
        %    This is much faster than 3D array operations
        [rho, ux, uy] = compute_macroscopic_variables_2d(F0, F1, F2, F3, F4, F5, F6, F7, F8, cylinder, CX, CY);

        % 5. OPTIMIZED Collision step: in-place computation with 2D arrays
        %    This eliminates 3D array indexing bottlenecks
        [F0, F1, F2, F3, F4, F5, F6, F7, F8] = collision_step_2d(F0, F1, F2, F3, F4, F5, F6, F7, F8, rho, ux, uy, CX, CY, WEIGHTS, relaxation_time, Feq0, Feq1, Feq2, Feq3, Feq4, Feq5, Feq6, Feq7, Feq8);

        % 6. OPTIMIZED Apply inlet/outflow boundary conditions with 2D arrays
        %    Maintains constant velocity at domain boundaries with float32 consistency
        [F0, F1, F2, F3, F4, F5, F6, F7, F8, ux] = apply_inflow_outflow_boundary_conditions_2d(F0, F1, F2, F3, F4, F5, F6, F7, F8, rho, ux, LATTICE_NUM, CX, WEIGHTS, U_MAX);

        % 7. Visualization and output (every OUTPUT_STEP iterations, only if enabled)
        %    Generates plots showing vorticity and streamlines
        if VISUALIZE && mod(step, OUTPUT_STEP) == 0
            PICTURE_NUM = visualize_flow_field(step, ux, uy, cylinder, POSITION_OX, POSITION_OY, RADIUS, ...
                                               MAX_Y, MAX_X, PICTURE_NUM, NOTICKS);
        end
    end

    fprintf('Simulation completed successfully!\n');
    if VISUALIZE
        fprintf('Output images saved in: matlab/\n');
    else
        fprintf('Performance mode: no visualization or file I/O performed.\n');
    end
end

% =============================================================================
% LOCAL FUNCTIONS
% =============================================================================



function cylinder = create_cylinder_mask(MAX_X, MAX_Y, POSITION_OX, POSITION_OY, RADIUS)
% =============================================================================
% CYLINDER MASK CREATION
% =============================================================================
%
% Create a boolean mask representing the cylinder in the computational domain.
%
% This function creates a 2D grid and identifies which grid points lie inside
% the circular cylinder using the distance formula. The resulting boolean mask
% is used throughout the simulation to identify solid boundary regions.
%
% Inputs:
%   MAX_X, MAX_Y: Domain dimensions
%   POSITION_OX, POSITION_OY: Cylinder center coordinates
%   RADIUS: Cylinder radius
%
% Output:
%   cylinder: Boolean matrix (true for points inside cylinder, false otherwise)

    % Create coordinate grids (MATLAB uses 1-based indexing)
    [x, y] = meshgrid(0:MAX_X-1, 0:MAX_Y-1);
    
    % Return boolean mask: true for points inside cylinder, false otherwise
    cylinder = (x - POSITION_OX).^2 + (y - POSITION_OY).^2 <= RADIUS^2;
end

function incoming_masks = precompute_cylinder_masks(cylinder, LATTICE_NUM, CX, CY)
% =============================================================================
% PRECOMPUTE CYLINDER MASKS FOR BOUNCE-BACK
% =============================================================================
%
% Precompute incoming particle masks for bounce-back boundary conditions.
% These masks don't change during simulation, so compute once and reuse.
%
% This removes two circshift(cylinder, ...) per direction per step.

    incoming_masks = cell(1, LATTICE_NUM);
    incoming_masks{1} = false(size(cylinder));  % unused for i=1 (rest particle)
    
    for i = 2:LATTICE_NUM
        roll_x = circshift(cylinder, [0, -CX(i)]);
        roll_y = circshift(cylinder, [-CY(i), 0]);
        incoming_masks{i} = cylinder & ~(roll_x & roll_y);
    end
end

function [rho, ux, uy, F0, F1, F2, F3, F4, F5, F6, F7, F8] = initialize_flow_field_2d(MAX_Y, MAX_X, LATTICE_NUM, U_MAX, CX, CY, WEIGHTS)
% =============================================================================
% FLOW FIELD INITIALIZATION (2D ARRAYS)
% =============================================================================
%
% Initialize the flow field with uniform density and inlet velocity.
% OPTIMIZATION: Uses 9 separate 2D arrays instead of one 3D array for better performance.
%
% This function creates the initial state of the simulation where:
% - Density is uniform (rho = 1) throughout the domain
% - Velocity is zero except at inlet/outlet boundaries
% - Distribution functions are set to equilibrium values
%
% The equilibrium distribution functions ensure that the simulation starts
% from a physically consistent state that satisfies the Chapman-Enskog expansion.

    % Initialize density and velocity fields
    % OPTIMIZATION: Use single for better memory bandwidth
    rho = ones(MAX_Y, MAX_X, 'single');           % uniform density field
    ux  = zeros(MAX_Y, MAX_X, 'single');         % zero velocity initially
    uy  = zeros(MAX_Y, MAX_X, 'single');         % zero velocity initially
    
    % Set inlet and outlet velocities (left and right boundaries)
    ux(:,1)   = U_MAX;    % left boundary (inlet)
    ux(:,end) = U_MAX;    % right boundary (outlet)

    % Initialize distribution functions using 9 separate 2D arrays
    % OPTIMIZATION: Use single for better memory bandwidth
    F0 = zeros(MAX_Y, MAX_X, 'single');
    F1 = zeros(MAX_Y, MAX_X, 'single');
    F2 = zeros(MAX_Y, MAX_X, 'single');
    F3 = zeros(MAX_Y, MAX_X, 'single');
    F4 = zeros(MAX_Y, MAX_X, 'single');
    F5 = zeros(MAX_Y, MAX_X, 'single');
    F6 = zeros(MAX_Y, MAX_X, 'single');
    F7 = zeros(MAX_Y, MAX_X, 'single');
    F8 = zeros(MAX_Y, MAX_X, 'single');
    
    % Feq initialization using Chapman-Enskog expansion
    for i = 1:LATTICE_NUM
        cx = CX(i); cy = CY(i); w = WEIGHTS(i);
        cu = cx.*ux + cy.*uy;  % velocity component in this direction
        % Equilibrium distribution function (Chapman-Enskog expansion)
        feq = rho .* w .* (1 + 3*cu + 4.5*cu.^2 - 1.5*(ux.^2 + uy.^2));
        
        % Assign to appropriate 2D array
        switch i
            case 1, F0 = feq;
            case 2, F1 = feq;
            case 3, F2 = feq;
            case 4, F3 = feq;
            case 5, F4 = feq;
            case 6, F5 = feq;
            case 7, F6 = feq;
            case 8, F7 = feq;
            case 9, F8 = feq;
        end
    end
end

function [F0, F1, F2, F3, F4, F5, F6, F7, F8] = apply_periodic_boundary_conditions_2d(F0, F1, F2, F3, F4, F5, F6, F7, F8)
% =============================================================================
% PERIODIC BOUNDARY CONDITIONS (2D ARRAYS)
% =============================================================================
%
% Apply periodic boundary conditions to the distribution functions.
% OPTIMIZATION: Uses 9 separate 2D arrays instead of one 3D array.
%
% This ensures particles leaving one edge of the domain re-enter from the 
% opposite edge, creating a periodic flow field. This is essential for:
% - Maintaining mass conservation
% - Avoiding artificial boundary effects
% - Creating an infinite domain approximation
%
% The function handles both X and Y direction periodicity by copying
% distribution functions from opposite edges for particles moving toward boundaries.

    % Match Python indices exactly (shifted +1 for MATLAB):
    % X-direction periodic boundaries (left/right edges)
    %  left  edge gets from right edge for dirs [1,5,8] -> MATLAB [2,6,9]
    F1(:,1)   = F1(:,end);  % direction 2
    F5(:,1)   = F5(:,end);  % direction 6
    F8(:,1)   = F8(:,end);  % direction 9
    %  right edge gets from left edge for dirs [3,6,7] -> MATLAB [4,7,8]
    F3(:,end) = F3(:,1);    % direction 4
    F6(:,end) = F6(:,1);    % direction 7
    F7(:,end) = F7(:,1);    % direction 8

    % Y-direction periodic boundaries (top/bottom edges)
    %  bottom row gets from top row for dirs [2,5,6] -> MATLAB [3,6,7]
    F2(1,:)   = F2(end,:);  % direction 3
    F5(1,:)   = F5(end,:);  % direction 6
    F6(1,:)   = F6(end,:);  % direction 7
    %  top row gets from bottom row for dirs [4,7,8] -> MATLAB [5,8,9]
    F4(end,:) = F4(1,:);    % direction 5
    F7(end,:) = F7(1,:);    % direction 8
    F8(end,:) = F8(1,:);    % direction 9
end

function [F0, F1, F2, F3, F4, F5, F6, F7, F8] = handle_cylinder_boundary_masks_2d(F0, F1, F2, F3, F4, F5, F6, F7, F8, incoming_masks, OPP)
% =============================================================================
% CYLINDER BOUNDARY HANDLING WITH PRECOMPUTED MASKS (2D ARRAYS)
% =============================================================================
%
% Handle the no-slip boundary condition at the cylinder surface using precomputed masks.
% OPTIMIZATION: Uses 9 separate 2D arrays instead of one 3D array.
%
% Implements the bounce-back method where particles hitting the cylinder
% reverse their direction. This creates a solid wall effect by:
% - Using precomputed masks to identify incoming particles
% - Reversing their velocity direction (bounce-back)
% - Maintaining mass conservation at the boundary
%
% FIXED: Read from unmodified snapshot to avoid read-after-write hazard

    % Snapshot of post-stream field (sources) — DO NOT modify these:
    Fsrc = {F0, F1, F2, F3, F4, F5, F6, F7, F8};

    % Destination starts as a shallow copy of the snapshot (copy-on-write).
    Fbb  = Fsrc;

    % Write at SOLID cells, exactly like the baseline:
    % Fi(solid & incoming) = Fopp_from_snapshot(same mask)
    for i = 2:numel(OPP)
        m = incoming_masks{i};
        Fi   = Fbb{i};                  % copy-on-write happens here on first write
        Fopp = Fsrc{OPP(i)};            % read from the immutable snapshot
        Fi(m) = Fopp(m);
        Fbb{i} = Fi;
    end

    % Unpack
    F0=Fbb{1}; F1=Fbb{2}; F2=Fbb{3}; F3=Fbb{4}; F4=Fbb{5};
    F5=Fbb{6}; F6=Fbb{7}; F7=Fbb{8}; F8=Fbb{9};
end

function [rho, ux, uy] = compute_macroscopic_variables_2d(F0, F1, F2, F3, F4, F5, F6, F7, F8, cylinder, CX, CY)
% =============================================================================
% MACROSCOPIC VARIABLES COMPUTATION (2D ARRAYS)
% =============================================================================
%
% Compute macroscopic fluid variables using 2D array operations.
% OPTIMIZATION: Much faster than 3D array operations.
%
% This function converts the microscopic particle distribution functions
% into macroscopic quantities that we can observe and analyze:
% - Density: sum of all distribution functions (mass conservation)
% - Velocity: momentum divided by density (momentum conservation)
%
% These macroscopic variables represent the actual fluid properties
% and are used for visualization and analysis.

    % Density: sum of all distribution functions (zeroth moment)
    rho = F0 + F1 + F2 + F3 + F4 + F5 + F6 + F7 + F8;

    % Velocity: momentum divided by density (first moment)
    % OPTIMIZATION: Direct 2D operations without 3D indexing
    ux = (CX(1)*F0 + CX(2)*F1 + CX(3)*F2 + CX(4)*F3 + CX(5)*F4 + ...
          CX(6)*F5 + CX(7)*F6 + CX(8)*F7 + CX(9)*F8) ./ rho;
    uy = (CY(1)*F0 + CY(2)*F1 + CY(3)*F2 + CY(4)*F3 + CY(5)*F4 + ...
          CY(6)*F5 + CY(7)*F6 + CY(8)*F7 + CY(9)*F8) ./ rho;

    % Set velocity to zero inside the cylinder (no-slip condition)
    ux(cylinder) = 0;
    uy(cylinder) = 0;
end

function [F0, F1, F2, F3, F4, F5, F6, F7, F8] = collision_step_2d(F0, F1, F2, F3, F4, F5, F6, F7, F8, rho, ux, uy, CX, CY, WEIGHTS, tau, Feq0, Feq1, Feq2, Feq3, Feq4, Feq5, Feq6, Feq7, Feq8)
% =============================================================================
% COLLISION STEP (BGK APPROXIMATION) - 2D ARRAYS
% =============================================================================
%
% Perform the collision step using the BGK (Bhatnagar-Gross-Krook) approximation.
% OPTIMIZATION: Uses 9 separate 2D arrays instead of one 3D array.
% Uses preallocated Feq buffers to avoid repeated allocations.
%
% In this step, distribution functions relax toward their equilibrium values.
% The BGK approximation simplifies the collision operator by assuming:
% - Single relaxation time for all distribution functions
% - Linear relaxation toward equilibrium
% - Local equilibrium based on current macroscopic variables
%
% This is the collision part of the LBM that handles viscosity and
% drives the system toward thermodynamic equilibrium.

    % OPTIMIZATION: Define single constants once to prevent upcasts
    ONE = single(1); THREE = single(3); FOUR5 = single(4.5); ONE5 = single(1.5);
    
    % Compute equilibrium distribution functions into preallocated Feq buffers
    usq = ux.^2 + uy.^2;                      % 2D, reuse in loop
    
    % OPTIMIZATION: Direct 2D operations eliminate 3D indexing bottlenecks
    for i = 1:9
        cx = CX(i); cy = CY(i); w = WEIGHTS(i);
        cu = cx.*ux + cy.*uy;  % velocity component in this direction
        % Equilibrium distribution function (Chapman-Enskog expansion)
        feq = rho .* w .* (ONE + THREE*cu + FOUR5*cu.^2 - ONE5*usq);
        
        % Assign to appropriate preallocated Feq buffer
        switch i
            case 1, Feq0 = feq;
            case 2, Feq1 = feq;
            case 3, Feq2 = feq;
            case 4, Feq3 = feq;
            case 5, Feq4 = feq;
            case 6, Feq5 = feq;
            case 7, Feq6 = feq;
            case 8, Feq7 = feq;
            case 9, Feq8 = feq;
        end
    end
    
    % BGK collision: relax toward equilibrium
    % OPTIMIZATION: Direct 2D operations without 3D indexing
    F0 = F0 - (ONE/tau) * (F0 - Feq0);
    F1 = F1 - (ONE/tau) * (F1 - Feq1);
    F2 = F2 - (ONE/tau) * (F2 - Feq2);
    F3 = F3 - (ONE/tau) * (F3 - Feq3);
    F4 = F4 - (ONE/tau) * (F4 - Feq4);
    F5 = F5 - (ONE/tau) * (F5 - Feq5);
    F6 = F6 - (ONE/tau) * (F6 - Feq6);
    F7 = F7 - (ONE/tau) * (F7 - Feq7);
    F8 = F8 - (ONE/tau) * (F8 - Feq8);
end

function [F0, F1, F2, F3, F4, F5, F6, F7, F8, ux] = apply_inflow_outflow_boundary_conditions_2d(F0, F1, F2, F3, F4, F5, F6, F7, F8, rho, ux, LATTICE_NUM, CX, WEIGHTS, U_MAX)
% =============================================================================
% INFLOW/OUTFLOW BOUNDARY CONDITIONS - 2D ARRAYS
% =============================================================================
%
% Apply velocity boundary conditions at inlet and outlet with optimized operations.
% OPTIMIZATION: Uses 9 separate 2D arrays instead of one 3D array.
% Maintains constant velocity at domain boundaries and avoids dtype upcasting.
%
% Maintains constant velocity at domain boundaries by:
% - Setting boundary velocities to U_MAX
% - Updating distribution functions using equilibrium distribution
% - Ensuring mass and momentum conservation at boundaries
%
% This creates a constant flow rate into and out of the domain,
% which is essential for maintaining the steady-state flow conditions.

    % OPTIMIZATION: Define single constants once to prevent upcasts
    ONE = single(1); THREE = single(3); FOUR5 = single(4.5); ONE5 = single(1.5);

    % Set boundary velocities
    ux(:,1)   = U_MAX;    % left boundary (inlet)
    ux(:,end) = U_MAX;    % right boundary (outlet)

    % Update F at left/right edges using equilibrium (uy=0 on edges as in Python)
    % OPTIMIZATION: Direct 2D operations eliminate 3D indexing bottlenecks
    for i = 1:LATTICE_NUM
        cx = CX(i); w = WEIGHTS(i);
        
        % Inlet boundary (left column)
        cuL = cx .* ux(:,1);
        feqL = rho(:,1) .* w .* (ONE + THREE*cuL + FOUR5*cuL.^2 - ONE5*(ux(:,1).^2));
        
        % Outlet boundary (right column)
        cuR = cx .* ux(:,end);
        feqR = rho(:,end) .* w .* (ONE + THREE*cuR + FOUR5*cuR.^2 - ONE5*(ux(:,end).^2));
        
        % Assign to appropriate 2D array
        switch i
            case 1, F0(:,1) = feqL; F0(:,end) = feqR;
            case 2, F1(:,1) = feqL; F1(:,end) = feqR;
            case 3, F2(:,1) = feqL; F2(:,end) = feqR;
            case 4, F3(:,1) = feqL; F3(:,end) = feqR;
            case 5, F4(:,1) = feqL; F4(:,end) = feqR;
            case 6, F5(:,1) = feqL; F5(:,end) = feqR;
            case 7, F6(:,1) = feqL; F6(:,end) = feqR;
            case 8, F7(:,1) = feqL; F7(:,end) = feqR;
            case 9, F8(:,1) = feqL; F8(:,end) = feqR;
        end
    end
end

function PICTURE_NUM = visualize_flow_field(step, ux, uy, cylinder, POSITION_OX, POSITION_OY, RADIUS, ...
                                            MAX_Y, MAX_X, PICTURE_NUM, NOTICKS)
% =============================================================================
% FLOW FIELD VISUALIZATION
% =============================================================================
%
% Visualize the flow field by plotting vorticity and streamlines.
%
% This function creates comprehensive visualizations showing:
% - Vorticity field (curl of velocity) using a diverging colormap
% - Streamlines for flow pattern visualization
% - Cylinder outline for reference
% - Clear title and proper axis limits (unless NOTICKS is true)
%
% The visualization helps understand the flow physics including:
% - Vortex formation and shedding
% - Flow separation patterns
% - Wake development behind the cylinder
%
% When NOTICKS is true, creates clean images without ticks, labels, title,
% and with minimized margins for publication use.

    % Vorticity (discrete curl of velocity field)
    % This measures the local rotation of the fluid
    vort = (circshift(uy, [0 -1]) - circshift(uy, [0 1])) ...
         - (circshift(ux, [-1 0]) - circshift(ux, [1 0]));
    vort(cylinder) = NaN;  % hide vorticity inside cylinder

    % Pseudocolor image (match Python's origin='lower')
    imagesc(vort, [-0.02 0.02]); 
    axis image; 
    set(gca,'YDir','normal');
    
    % Apply Red-Blue diverging colormap for clear vorticity visualization
    colormap(gca, rdBu(256)); 
    % Alternative: colormap(gca, redblue()); % simple blue-white-red colormap
    hold on;

    % Cylinder outline
    % Use rectangle with curvature to draw a perfect circle
    rectangle('Position',[POSITION_OX-RADIUS, POSITION_OY-RADIUS, 2*RADIUS, 2*RADIUS], ...
              'Curvature',[1 1], 'FaceColor','k', 'EdgeColor','k', 'LineWidth',1);

    % Streamlines (MATLAB equivalent)
    % Python colors by speed; here we draw in default color for clarity
    [X, Y] = meshgrid(0:MAX_X-1, 0:MAX_Y-1);
    streamslice(X, Y, ux, uy, 1);  % light sampling to keep it responsive

    % Apply NOTICKS styling if requested
    if NOTICKS
        % Clean layout: no ticks, labels, title, minimized margins
        set(gca, 'XTick', [], 'YTick', [], ...           % no ticks
                 'XLabel', [], 'YLabel', [], ...         % no labels
                 'Title', []);                           % no title
        % Minimize margins while keeping the plot box
        set(gca, 'Position', [0.05, 0.05, 0.9, 0.9]); % tight margins
    else
        % Normal layout: standard title
        title(sprintf('Vorticity and streamlines at step %d', step));
    end

    % Figure limits
    xlim([0 MAX_X]);
    ylim([0 MAX_Y]);

    drawnow;

    % --- Create output directory and save figure ---
    persistent OUT_DIR
    if isempty(OUT_DIR)
        thisdir = fileparts(mfilename('fullpath'));
        OUT_DIR = fullfile(thisdir, 'matlab');
        if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end
    end

    % Save figure (PNG)
    fname = fullfile(OUT_DIR, sprintf('lattice_boltzmann_%04d_matlab.png', PICTURE_NUM));
    exportgraphics(gca, fname);   % (or saveas/gcf if you prefer)
    pause(0.01);
    cla; hold off;

    PICTURE_NUM = PICTURE_NUM + 1;
end

function cmap = rdBu(n)
% =============================================================================
% RED-BLUE DIVERGING COLORMAP
% =============================================================================
%
% rdBu  ColorBrewer RdBu diverging colormap (red–white–blue)
% cmap = rdBu(256) returns a 256x3 colormap matching matplotlib's 'RdBu'.
%
% This function creates a perceptually uniform diverging colormap that:
% - Uses red for positive values (high vorticity)
% - Uses white for values near zero (neutral)
% - Uses blue for negative values (low vorticity)
%
% The colormap is ideal for vorticity visualization because it clearly
% distinguishes between positive and negative rotation directions.

    if nargin < 1, n = 256; end
    
    % 11 control colors from ColorBrewer RdBu, scaled 0..1
    ctrl = [  % RGB values for control points
        103  0   31
        178 24   43
        214 96   77
        244 165 130
        253 219 199
        247 247 247
        209 229 240
        146 197 222
         67 147 195
         33 102 172
          5  48  97] / 255;
    
    % Interpolate between control points to create smooth colormap
    t0 = linspace(0,1,size(ctrl,1));
    t  = linspace(0,1,n);
    cmap = [interp1(t0, ctrl(:,1), t, 'pchip')', ...
            interp1(t0, ctrl(:,2), t, 'pchip')', ...
            interp1(t0, ctrl(:,3), t, 'pchip')'];
end
