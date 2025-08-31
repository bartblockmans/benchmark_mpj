function lbm_cylinder()
% =============================================================================
% MATLAB Program: Lattice Boltzmann Method (D2Q9) - Fluid Flow Past a Fixed Cylinder
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
% - Optimized simulation parameters for faster execution
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
% Author: BB
% Modified: Enhanced structure and documentation for educational use
% Language: MATLAB (with built-in plotting and export capabilities)
%
% =============================================================================
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

    % -- Initialize all parameters
    [MAX_X, MAX_Y, LATTICE_NUM, CX, CY, WEIGHTS, OPP, ...
     POSITION_OX, POSITION_OY, RADIUS, REYNOLDS, U_MAX, ...
     kinematic_viscosity, relaxation_time, MAX_STEP, OUTPUT_STEP, PICTURE_NUM, VISUALIZE, NOTICKS] = initialize_parameters();

    % -- Print simulation parameters for user reference
    fprintf('Reynolds number = %g\n', REYNOLDS);
    fprintf('Relaxation time = %g\n', relaxation_time);
    fprintf('Domain size: %d x %d\n', MAX_X, MAX_Y);
    fprintf('Maximum velocity: %g\n', U_MAX);
    fprintf('Kinematic viscosity: %g\n', kinematic_viscosity);
    fprintf('Visualization enabled: %s\n', string(VISUALIZE));
    if VISUALIZE
        fprintf('Output directory: matlab/\n');
        fprintf('Clean visualization (NOTICKS): %s\n', string(NOTICKS));
    end

    % -- Create cylinder mask (defines where the solid cylinder is located)
    cylinder = create_cylinder_mask(MAX_X, MAX_Y, POSITION_OX, POSITION_OY, RADIUS);

    % -- Initialize flow field with uniform density and inlet velocity
    [~, ~, ~, F] = initialize_flow_field(MAX_Y, MAX_X, LATTICE_NUM, U_MAX, CX, CY, WEIGHTS);

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
        F = apply_periodic_boundary_conditions(F);

        % 2. Streaming step: particles move along their velocity directions
        %    This is the advection part of the LBM
        F = streaming_step(F, LATTICE_NUM, CX, CY);

        % 3. Handle cylinder boundary (no-slip condition)
        %    Particles hitting the cylinder bounce back (bounce-back method)
        cylinderF = handle_cylinder_boundary(F, cylinder, LATTICE_NUM, CX, CY, OPP);

        % 4. Compute macroscopic variables (density, velocity)
        %    These are the physical quantities we actually care about
        [rho, ux, uy] = compute_macroscopic_variables(cylinderF, cylinder, CX, CY);
        F = cylinderF;

        % 5. Collision step: relaxation toward equilibrium
        %    This is the collision part of the LBM (BGK approximation)
        F = collision_step(F, rho, ux, uy, CX, CY, WEIGHTS, relaxation_time);

        % 6. Apply inlet/outflow boundary conditions
        %    Maintains constant velocity at domain boundaries
        [F, ux] = apply_inflow_outflow_boundary_conditions(F, rho, ux, LATTICE_NUM, CX, WEIGHTS, U_MAX);

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

function [MAX_X, MAX_Y, LATTICE_NUM, CX, CY, WEIGHTS, OPP, ...
          POSITION_OX, POSITION_OY, RADIUS, REYNOLDS, U_MAX, ...
          kinematic_viscosity, relaxation_time, MAX_STEP, OUTPUT_STEP, PICTURE_NUM, VISUALIZE, NOTICKS] = initialize_parameters()
% =============================================================================
% PARAMETER INITIALIZATION
% =============================================================================
%
% Initialize all simulation parameters including domain size, lattice properties,
% cylinder geometry, and fluid properties.
%
% This function sets up all the constants needed for the LBM simulation:
% - Computational domain dimensions
% - D2Q9 lattice model parameters (velocity directions, weights, opposite indices)
% - Cylinder geometry and position
% - Fluid properties and Reynolds number
% - Simulation control parameters
% - Visualization control flags

    % Fluid domain dimensions
    MAX_X = 400;
    MAX_Y = 100;

    % D2Q9 lattice model parameters
    % 9 velocity directions: center (1), cardinal directions (2-5), diagonal directions (6-9)
    LATTICE_NUM = 9;
    CX = [0  1  0 -1  0  1 -1 -1  1];      % x-velocity components
    CY = [0  0  1  0 -1  1  1 -1 -1];     % y-velocity components
    WEIGHTS = [4/9  1/9  1/9  1/9  1/9  1/36  1/36  1/36  1/36];  % lattice weights

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
    U_MAX = 0.1;         % maximum inlet velocity
    kinematic_viscosity = U_MAX * 2 * RADIUS / REYNOLDS;
    relaxation_time = 3.0 * kinematic_viscosity + 0.5;

    % Simulation control parameters
    MAX_STEP = 20001;    % total number of time steps
    OUTPUT_STEP = 2000;  % frequency of output visualization
    PICTURE_NUM = 1;     % counter for saved images
    
    % Visualization control flags
    % Set to false for performance benchmarking (no visualization or file I/O)
    % Set to true for normal operation with visualization and output
    VISUALIZE = true;
    
    % Clean visualization parameter: removes ticks, labels, title and minimizes margins
    % Set to true for clean, publication-ready images without visual clutter
    NOTICKS = false;
end

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

function [rho, ux, uy, F] = initialize_flow_field(MAX_Y, MAX_X, LATTICE_NUM, U_MAX, CX, CY, WEIGHTS)
% =============================================================================
% FLOW FIELD INITIALIZATION
% =============================================================================
%
% Initialize the flow field with uniform density and inlet velocity.
% Sets up the initial distribution functions F based on equilibrium conditions.
%
% This function creates the initial state of the simulation where:
% - Density is uniform (rho = 1) throughout the domain
% - Velocity is zero except at inlet/outlet boundaries
% - Distribution functions are set to equilibrium values
%
% The equilibrium distribution functions ensure that the simulation starts
% from a physically consistent state that satisfies the Chapman-Enskog expansion.

    % Initialize density and velocity fields
    rho = ones(MAX_Y, MAX_X);           % uniform density field
    ux  = zeros(MAX_Y, MAX_X);         % zero velocity initially
    uy  = zeros(MAX_Y, MAX_X);         % zero velocity initially
    
    % Set inlet and outlet velocities (left and right boundaries)
    ux(:,1)   = U_MAX;    % left boundary (inlet)
    ux(:,end) = U_MAX;    % right boundary (outlet)

    % Initialize distribution functions F using equilibrium distribution
    % This sets up the initial particle populations for each velocity direction
    F = zeros(MAX_Y, MAX_X, LATTICE_NUM);
    
    % Feq initialization using Chapman-Enskog expansion
    for i = 1:LATTICE_NUM
        cx = CX(i); cy = CY(i); w = WEIGHTS(i);
        cu = cx.*ux + cy.*uy;  % velocity component in this direction
        % Equilibrium distribution function (Chapman-Enskog expansion)
        F(:,:,i) = rho .* w .* (1 + 3*cu + 4.5*cu.^2 - 1.5*(ux.^2 + uy.^2));
    end
end

function F = apply_periodic_boundary_conditions(F)
% =============================================================================
% PERIODIC BOUNDARY CONDITIONS
% =============================================================================
%
% Apply periodic boundary conditions to the distribution functions.
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
    F(:,1,[2 6 9])   = F(:,end,[2 6 9]);
    %  right edge gets from left edge for dirs [3,6,7] -> MATLAB [4,7,8]
    F(:,end,[4 7 8]) = F(:,1,[4 7 8]);

    % Y-direction periodic boundaries (top/bottom edges)
    %  bottom row gets from top row for dirs [2,5,6] -> MATLAB [3,6,7]
    F(1,:,[3 6 7])   = F(end,:,[3 6 7]);
    %  top row gets from bottom row for dirs [4,7,8] -> MATLAB [5,8,9]
    F(end,:,[5 8 9]) = F(1,:,[5 8 9]);
end

function F = streaming_step(F, LATTICE_NUM, CX, CY)
% =============================================================================
% STREAMING STEP
% =============================================================================
%
% Perform the streaming step of the LBM algorithm.
%
% In this step, particles move along their velocity directions to neighboring 
% lattice sites. This is the advection part of the LBM that handles:
% - Particle movement across the lattice
% - Information propagation through the domain
% - Spatial discretization of the fluid flow
%
% The circshift function efficiently moves particles according to their
% velocity directions, implementing the streaming phase of the LBM algorithm.

    % In-place per-direction streaming (identical order to Python)
    for i = 1:LATTICE_NUM
        % Use circshift to move particles: [rows (y), cols (x)] directions
        F(:,:,i) = circshift(F(:,:,i), [CY(i), CX(i)]);
    end
end

function cylinderF = handle_cylinder_boundary(F, cylinder, LATTICE_NUM, CX, CY, OPP)
% =============================================================================
% CYLINDER BOUNDARY HANDLING
% =============================================================================
%
% Handle the no-slip boundary condition at the cylinder surface.
%
% Implements the bounce-back method where particles hitting the cylinder
% reverse their direction. This creates a solid wall effect by:
% - Identifying incoming particles (moving toward cylinder surface)
% - Reversing their velocity direction (bounce-back)
% - Maintaining mass conservation at the boundary
%
% The bounce-back method is a simple and effective way to implement
% no-slip boundary conditions in LBM simulations.

    cylinderF = F;
    for i = 2:LATTICE_NUM % skip rest particle (i=1)
        % Find incoming particles: inside cylinder but moving toward cylinder surface
        % incoming_particles = cylinder & ((roll_x & roll_y) == False)
        % i.e., NOT( roll_x & roll_y )
        roll_x = circshift(cylinder, [0, -CX(i)]);
        roll_y = circshift(cylinder, [-CY(i), 0]);
        incoming_particles = cylinder & ~(roll_x & roll_y);

        % Bounce back: f_i <- f_opp at boundary
        Fi   = cylinderF(:,:,i);
        Fopp = F(:,:,OPP(i));
        Fi(incoming_particles) = Fopp(incoming_particles);
        cylinderF(:,:,i) = Fi;
    end
end

function [rho, ux, uy] = compute_macroscopic_variables(F, cylinder, CX, CY)
% =============================================================================
% MACROSCOPIC VARIABLES COMPUTATION
% =============================================================================
%
% Compute macroscopic fluid variables (density and velocity) from the
% distribution functions using moment integrals.
%
% This function converts the microscopic particle distribution functions
% into macroscopic quantities that we can observe and analyze:
% - Density: sum of all distribution functions (mass conservation)
% - Velocity: momentum divided by density (momentum conservation)
%
% These macroscopic variables represent the actual fluid properties
% and are used for visualization and analysis.

    % Density: sum of all distribution functions (zeroth moment)
    rho = sum(F, 3);

    % Velocity: momentum divided by density (first moment)
    % Use implicit expansion via reshape for efficient computation
    ux = sum(F .* reshape(CX, 1,1,[]), 3) ./ rho;
    uy = sum(F .* reshape(CY, 1,1,[]), 3) ./ rho;

    % Set velocity to zero inside the cylinder (no-slip condition)
    ux(cylinder) = 0;
    uy(cylinder) = 0;
end

function F = collision_step(F, rho, ux, uy, CX, CY, WEIGHTS, relaxation_time)
% =============================================================================
% COLLISION STEP (BGK APPROXIMATION)
% =============================================================================
%
% Perform the collision step using the BGK (Bhatnagar-Gross-Krook) approximation.
%
% In this step, distribution functions relax toward their equilibrium values.
% The BGK approximation simplifies the collision operator by assuming:
% - Single relaxation time for all distribution functions
% - Linear relaxation toward equilibrium
% - Local equilibrium based on current macroscopic variables
%
% This is the collision part of the LBM that handles viscosity and
% drives the system toward thermodynamic equilibrium.

    LATTICE_NUM = numel(WEIGHTS);
    
    % Compute equilibrium distribution functions
    Feq = zeros(size(F));
    for i = 1:LATTICE_NUM
        cx = CX(i); cy = CY(i); w = WEIGHTS(i);
        cu = cx.*ux + cy.*uy;  % velocity component in this direction
        % Equilibrium distribution function (Chapman-Enskog expansion)
        Feq(:,:,i) = rho .* w .* (1 + 3*cu + 4.5*cu.^2 - 1.5*(ux.^2 + uy.^2));
    end
    
    % BGK collision: relax toward equilibrium
    F = F - (1/relaxation_time) * (F - Feq);
end

function [F, ux] = apply_inflow_outflow_boundary_conditions(F, rho, ux, LATTICE_NUM, CX, WEIGHTS, U_MAX)
% =============================================================================
% INFLOW/OUTFLOW BOUNDARY CONDITIONS
% =============================================================================
%
% Apply velocity boundary conditions at inlet and outlet.
%
% Maintains constant velocity at domain boundaries by:
% - Setting boundary velocities to U_MAX
% - Updating distribution functions using equilibrium distribution
% - Ensuring mass and momentum conservation at boundaries
%
% This creates a constant flow rate into and out of the domain,
% which is essential for maintaining the steady-state flow conditions.

    % Set boundary velocities
    ux(:,1)   = U_MAX;    % left boundary (inlet)
    ux(:,end) = U_MAX;    % right boundary (outlet)

    % Update F at left/right edges using equilibrium (uy=0 on edges as in Python)
    for i = 1:LATTICE_NUM
        cx = CX(i); w = WEIGHTS(i);
        
        % Inlet boundary (left column)
        cuL = cx .* ux(:,1);
        F(:,1,i) = rho(:,1) .* w .* (1 + 3*cuL + 4.5*cuL.^2 - 1.5*(ux(:,1).^2));
        
        % Outlet boundary (right column)
        cuR = cx .* ux(:,end);
        F(:,end,i) = rho(:,end) .* w .* (1 + 3*cuR + 4.5*cuR.^2 - 1.5*(ux(:,end).^2));
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
