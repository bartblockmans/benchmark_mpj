! LBM with half-way bounce-back (matching Python implementation exactly)
program lbm_halfway_bounceback
  implicit none
  
  ! Parameters
  integer, parameter :: NX = 400, NY = 100, Q = 9
  integer, parameter :: CX0 = 70, CY0 = 50, R = 20
  real, parameter :: UMAX = 0.1, REYNOLDS = 200.0
  real :: TAU  ! Relaxation time (calculated dynamically)
  integer :: NSTEPS, OUTPUT_STEP
  logical :: GENERATE_IMAGES
  
  ! Arrays
  real :: F(NY,NX,Q), F2(NY,NX,Q)
  real :: RHO(NY,NX), UX(NY,NX), UY(NY,NX)
  logical :: CYL(NY,NX)
  ! INCOMING_MASKS array removed - using original bounce-back approach
  real :: TMP2D(NY,NX)  ! 2D scratch buffer for streaming
  
  ! Lattice vectors
  integer :: CX(Q) = [0, 1, 0, -1, 0, 1, -1, -1, 1]
  integer :: CY(Q) = [0, 0, 1, 0, -1, 1, 1, -1, -1]
  integer :: OPP(Q) = [1, 4, 5, 2, 3, 8, 9, 6, 7]  ! Opposite directions (1-based)
  real :: W(Q) = [4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, &
                  1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0]
  
  integer :: i, j, k, step, pic
  real :: feq, vort(NY,NX)
  real :: cu(NY,NX), usq(NY,NX)
  real :: usqL(NY), usqR(NY)
  
  ! Parse command line arguments
  call parse_command_line_arguments()
  
  ! Calculate relaxation time dynamically (matching Python)
  TAU = 3.0 * (UMAX * 2.0 * real(R) / REYNOLDS) + 0.5
  
  write(*,*) 'LBM with Half-way Bounce-back (Python-matching)'
  write(*,*) 'Reynolds number =', REYNOLDS
  write(*,*) 'Relaxation time =', TAU
  write(*,*) 'Domain size =', NX, 'x', NY
  write(*,*) 'Umax =', UMAX
  write(*,*) 'NSTEPS =', NSTEPS
  write(*,*) 'OUTPUT_STEP =', OUTPUT_STEP
  write(*,*) 'Generate images =', GENERATE_IMAGES
  
  ! Initialize cylinder
  do j = 1, NX
    do i = 1, NY
      CYL(i,j) = ((j-CX0)**2 + (i-CY0)**2 <= R*R)
    end do
  end do
  write(*,*) 'Cylinder center: (', CX0, ',', CY0, '), radius:', R
  write(*,*) 'Cylinder mask:', count(CYL), 'solid cells out of', NY*NX
  
  ! No need for precomputed masks with original bounce-back approach
  
  ! Initialize flow
  RHO = 1.0
  UX = 0.0
  UY = 0.0
  UX(:,1) = UMAX
  UX(:,NX) = UMAX
  
  ! Initialize distribution functions
  do k = 1, Q
    do j = 1, NX
      do i = 1, NY
        cu(i,j) = real(CX(k))*UX(i,j) + real(CY(k))*UY(i,j)
        usq(i,j) = UX(i,j)**2 + UY(i,j)**2
        feq = RHO(i,j) * W(k) * (1.0 + 3.0*cu(i,j) + 4.5*cu(i,j)*cu(i,j) - 1.5*usq(i,j))
        F(i,j,k) = feq
      end do
    end do
  end do
  
  write(*,*) 'Starting simulation for', NSTEPS, 'steps...'
  pic = 1
  
  ! Main loop - matching Python algorithm exactly
  do step = 0, NSTEPS-1
    if (mod(step, 1000) == 0) then
      write(*,*) 'Step', step, '/', NSTEPS
      write(*,*) '  UX range:', minval(UX), 'to', maxval(UX)
      write(*,*) '  UY range:', minval(UY), 'to', maxval(UY)
      write(*,*) '  RHO range:', minval(RHO), 'to', maxval(RHO)
    end if
    
    ! Step 1: Apply periodic boundary conditions
    call apply_periodic_boundary_conditions()
    
    ! Step 2: Streaming step: particles move along their velocity directions
    call streaming_step()
    
    ! Step 3: Handle cylinder boundary (no-slip condition)
    call handle_cylinder_boundary()
    
    ! Step 4: Compute macroscopic variables
    call compute_macroscopic_variables()
    
    ! Step 5: Collision step: relaxation toward equilibrium
    call collision_step()
    
    ! Step 6: Apply inlet/outlet boundary conditions
    call apply_inflow_outflow_boundary_conditions()
    
    ! Step 7: Visualization
    if (GENERATE_IMAGES .and. mod(step, OUTPUT_STEP) == 0) then
      call plot_vorticity_png(step, pic)
      pic = pic + 1
    end if
  end do
  
  write(*,*) 'Simulation completed successfully!'
  if (GENERATE_IMAGES) then
    write(*,*) 'Images saved in ./images'
  else
    write(*,*) 'No images generated (GENERATE_IMAGES = .false.)'
  end if
  
contains

  subroutine parse_command_line_arguments()
    ! Parse command line arguments for NSTEPS, OUTPUT_STEP, and GENERATE_IMAGES
    integer :: argc, i, img_flag
    character(len=256) :: arg
    
    ! Set default values
    NSTEPS = 20001
    OUTPUT_STEP = 2000
    GENERATE_IMAGES = .true.
    
    ! Get number of command line arguments
    argc = command_argument_count()
    
    ! Check for help
    if (argc > 0) then
      call get_command_argument(1, arg)
      if (arg(1:2) == '-h' .or. arg(1:2) == '--') then
        write(*,*) 'Usage: ./lbm_cylinder_opt [NSTEPS] [OUTPUT_STEP] [GENERATE_IMAGES]'
        write(*,*) '  NSTEPS: Number of simulation steps (default: 20001)'
        write(*,*) '  OUTPUT_STEP: Steps between image outputs (default: 2000)'
        write(*,*) '  GENERATE_IMAGES: 1 to generate images, 0 to skip (default: 1)'
        write(*,*) ''
        write(*,*) 'Examples:'
        write(*,*) '  ./lbm_cylinder_opt                    ! Use defaults'
        write(*,*) '  ./lbm_cylinder_opt 10000 1000 1      ! 10k steps, output every 1k steps'
        write(*,*) '  ./lbm_cylinder_opt 5000 500 0        ! 5k steps, no images'
        stop
      end if
    end if
    
    ! Parse arguments
    do i = 1, argc
      call get_command_argument(i, arg)
      
      if (i == 1) then
        ! First argument: NSTEPS
        read(arg, *) NSTEPS
      else if (i == 2) then
        ! Second argument: OUTPUT_STEP
        read(arg, *) OUTPUT_STEP
      else if (i == 3) then
        ! Third argument: GENERATE_IMAGES (0 or 1)
        read(arg, *) img_flag
        GENERATE_IMAGES = (img_flag == 1)
      end if
    end do
    
    ! Validate arguments
    if (NSTEPS < 1) then
      write(*,*) 'Error: NSTEPS must be >= 1'
      stop
    end if
    
    if (OUTPUT_STEP < 1) then
      write(*,*) 'Error: OUTPUT_STEP must be >= 1'
      stop
    end if
    
    if (OUTPUT_STEP > NSTEPS) then
      write(*,*) 'Warning: OUTPUT_STEP > NSTEPS, no images will be generated'
    end if
  end subroutine parse_command_line_arguments

  ! Precomputed masks function removed - using original bounce-back approach
  
  ! roll_logical function removed - no longer needed with original bounce-back approach
  
  subroutine apply_periodic_boundary_conditions()
    ! Apply periodic boundary conditions (matching Python)
    do i = 1, NY
      F(i,1,2) = F(i,NX,2)    ! east at left edge
      F(i,1,6) = F(i,NX,6)    ! ne at left edge
      F(i,1,9) = F(i,NX,9)    ! se at left edge
      F(i,NX,4) = F(i,1,4)    ! west at right edge
      F(i,NX,7) = F(i,1,7)    ! nw at right edge
      F(i,NX,8) = F(i,1,8)    ! sw at right edge
    end do
    
    do j = 1, NX
      F(1,j,3) = F(NY,j,3)    ! north at bottom edge
      F(1,j,6) = F(NY,j,6)    ! ne at bottom edge
      F(1,j,7) = F(NY,j,7)    ! nw at bottom edge
      F(NY,j,5) = F(1,j,5)    ! south at top edge
      F(NY,j,8) = F(1,j,8)    ! sw at top edge
      F(NY,j,9) = F(1,j,9)    ! se at top edge
    end do
  end subroutine apply_periodic_boundary_conditions
  
  subroutine streaming_step()
    ! Original streaming step: particles move along their velocity directions
    integer :: k, dx, dy
    
    do k = 1, Q
      dx = CX(k)
      dy = CY(k)
      
      ! x shift
      if (dx == 0) then
        TMP2D = F(:,:,k)
      else if (dx == 1) then
        TMP2D(:,2:) = F(:,1:NX-1,k)
        TMP2D(:,1) = F(:,NX,k)
      else if (dx == -1) then
        TMP2D(:,1:NX-1) = F(:,2:,k)
        TMP2D(:,NX) = F(:,1,k)
      end if
      
      ! y shift
      if (dy == 0) then
        F(:,:,k) = TMP2D
      else if (dy == 1) then
        F(2:,:,k) = TMP2D(1:NY-1,:)
        F(1,:,k) = TMP2D(NY,:)
      else if (dy == -1) then
        F(1:NY-1,:,k) = TMP2D(2:,:)
        F(NY,:,k) = TMP2D(1,:)
      end if
    end do
  end subroutine streaming_step
  
  subroutine handle_cylinder_boundary()
    ! Original bounce-back method where particles hitting the cylinder reverse their direction
    integer :: k, i, j
    logical :: incoming_particles(NY,NX)
    
    F2 = F  ! copy distribution functions
    
    do k = 2, Q  ! skip rest particle (k=1)
      ! Find incoming particles: inside cylinder but moving toward cylinder surface
      do j = 1, NX
        do i = 1, NY
          if (CYL(i,j)) then
            ! Check if neighbor in direction -CX(k), -CY(k) is not solid
            incoming_particles(i,j) = .not. CYL(mod(i-CY(k)-1+NY,NY)+1, mod(j-CX(k)-1+NX,NX)+1)
          else
            incoming_particles(i,j) = .false.
          end if
        end do
      end do
      
      ! Bounce back: reverse particle direction
      where (incoming_particles)
        F2(:,:,k) = F(:,:,OPP(k))
      end where
    end do
    
    F = F2  ! update F with bounced-back values
  end subroutine handle_cylinder_boundary
  
  subroutine compute_macroscopic_variables()
    ! Compute macroscopic variables (matching Python)
    RHO = 0.0
    UX = 0.0
    UY = 0.0
    
    do k = 1, Q
      RHO = RHO + F(:,:,k)
      UX = UX + F(:,:,k) * real(CX(k))
      UY = UY + F(:,:,k) * real(CY(k))
    end do
    
    UX = UX / RHO
    UY = UY / RHO
    
    ! Set velocity to zero in cylinder
    where (CYL)
      UX = 0.0
      UY = 0.0
    end where
  end subroutine compute_macroscopic_variables
  
  subroutine collision_step()
    ! Original collision step using the BGK approximation
    real :: Feq(NY,NX,Q)
    integer :: k
    
    ! Compute equilibrium distribution functions
    do k = 1, Q
      cu = real(CX(k))*UX + real(CY(k))*UY
      usq = UX*UX + UY*UY
      Feq(:,:,k) = RHO * W(k) * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*usq)
    end do
    
    ! BGK collision: relax toward equilibrium
    F = F - (1.0/TAU) * (F - Feq)
  end subroutine collision_step
  
  subroutine apply_inflow_outflow_boundary_conditions()
    ! Apply inlet/outlet boundary conditions (matching Python)
    integer :: i, k
    real :: cu_col(NY)
    
    UX(:,1) = UMAX
    UX(:,NX) = UMAX
    UY(:,1) = 0.0
    UY(:,NX) = 0.0
    
    ! Left edge
    usqL = UX(:,1)*UX(:,1)
    do k = 1, Q
      cu_col = real(CX(k)) * UX(:,1)
      TMP2D(:,1) = 1.0 + 3.0*cu_col + 4.5*cu_col*cu_col - 1.5*usqL
      F(:,1,k) = RHO(:,1) * W(k) * TMP2D(:,1)
    end do
    
    ! Right edge
    usqR = UX(:,NX)*UX(:,NX)
    do k = 1, Q
      cu_col = real(CX(k)) * UX(:,NX)
      TMP2D(:,NX) = 1.0 + 3.0*cu_col + 4.5*cu_col*cu_col - 1.5*usqR
      F(:,NX,k) = RHO(:,NX) * W(k) * TMP2D(:,NX)
    end do
  end subroutine apply_inflow_outflow_boundary_conditions
  
  subroutine plot_vorticity_png(step, picnum)
    integer, intent(in) :: step, picnum
    
    character(len=256) :: fname_ppm
    integer, allocatable :: img(:,:,:)
    integer :: i, j, u
    real :: vort(NY,NX)
    logical :: ex
    
    ! Compute vorticity
    do j = 1, NX
      do i = 1, NY
        vort(i,j) = (UY(i,1 + modulo(j,NX)) - UY(i,1 + modulo(j-2,NX))) - &
                    (UX(1 + modulo(i,NY),j) - UX(1 + modulo(i-2,NY),j))
      end do
    end do
    
    if (picnum == 1) then
      write(*,*) 'vorticity min/max:', minval(vort), maxval(vort)
      write(*,*) 'UX min/max:', minval(UX), maxval(UX)
      write(*,*) 'UY min/max:', minval(UY), maxval(UY)
      write(*,*) 'RHO min/max:', minval(RHO), maxval(RHO)
    end if
    
    ! Allocate image array
    allocate(img(NY,NX,3))
    
    ! Map vorticity to colors (RdBu colormap)
    call map_to_rdBu(vort, -0.02, 0.02, img)
    
    ! Paint cylinder interior black
    do j = 1, NX
      do i = 1, NY
        if (CYL(i,j)) then
          img(i,j,1) = 0
          img(i,j,2) = 0
          img(i,j,3) = 0
        end if
      end do
    end do
    
    ! Draw cylinder outline
    call draw_circle_outline(img, CX0, CY0, R)
    
    ! Ensure output directory exists
    inquire(file='images', exist=ex)
    if (.not. ex) call execute_command_line('mkdir images', wait=.true.)
    
    ! Write PPM file
    write(fname_ppm,'(A,I4.4,A)') 'images/lattice_boltzmann_', picnum, '_fortran.ppm'
    open(newunit=u, file=fname_ppm, status='replace', action='write', form='formatted')
    write(u,'(A)') 'P3'
    write(u,'(I0,1X,I0)') NX, NY
    write(u,'(I0)') 255
    
    ! Write image data (top to bottom for matplotlib compatibility)
    do i = NY, 1, -1
      do j = 1, NX
        write(u,'(I0,1X,I0,1X,I0)') img(i,j,1), img(i,j,2), img(i,j,3)
      end do
    end do
    close(u)
    
    deallocate(img)
  end subroutine plot_vorticity_png
  
  subroutine map_to_rdBu(z, zmin, zmax, img)
    real, intent(in) :: z(:,:), zmin, zmax
    integer, intent(out) :: img(size(z,1), size(z,2), 3)
    integer :: i, j, r, g, b
    real :: t, val
    
    do j = 1, size(z,2)
      do i = 1, size(z,1)
        val = z(i,j)
        t = (val - zmin) / max(1.0e-20, (zmax - zmin))
        t = max(0.0, min(1.0, t))
        call rdBu_color(t, r, g, b)
        img(i,j,1) = r
        img(i,j,2) = g
        img(i,j,3) = b
      end do
    end do
  end subroutine map_to_rdBu
  
  subroutine rdBu_color(t, r, g, b)
    real, intent(in) :: t
    integer, intent(out) :: r, g, b
    real :: r1, g1, b1, r2, g2, b2, u
    
    if (t <= 0.5) then
      u = t / 0.5
      r1 = 49.0;  g1 = 54.0;  b1 = 149.0
      r2 = 255.0; g2 = 255.0; b2 = 255.0
    else
      u = (t - 0.5) / 0.5
      r1 = 255.0; g1 = 255.0; b1 = 255.0
      r2 = 165.0; g2 = 0.0;   b2 = 38.0
    end if
    
    r = int(nint((1.0 - u) * r1 + u * r2))
    g = int(nint((1.0 - u) * g1 + u * g2))
    b = int(nint((1.0 - u) * b1 + u * b2))
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
  end subroutine rdBu_color
  
  subroutine draw_circle_outline(img, xc, yc, radius)
    integer, intent(inout) :: img(:,:,:)
    integer, intent(in) :: xc, yc, radius
    integer, parameter :: N = 1024
    integer :: k, xi, yi
    real :: th
    
    do k = 0, N-1
      th = 6.283185307179586 * real(k) / real(N)
      xi = nint(real(xc) + real(radius) * cos(th))
      yi = nint(real(yc) + real(radius) * sin(th))
      if (xi >= 1 .and. xi <= size(img,2) .and. yi >= 1 .and. yi <= size(img,1)) then
        img(yi,xi,1) = 0
        img(yi,xi,2) = 0
        img(yi,xi,3) = 0
      end if
    end do
  end subroutine draw_circle_outline

end program lbm_halfway_bounceback