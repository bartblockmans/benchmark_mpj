! JSON Reader Module for N-Body Initial Conditions
! ======================================================
!
! Author: Bart Blockmans
! Date: August 2025

module json_reader
  implicit none
  
  private
  public :: import_initial_conditions
  
contains

  subroutine import_initial_conditions(filename, x, y, z, u, v, w, m)
    character(len=*), intent(in) :: filename
    real, intent(out) :: x(:), y(:), z(:), u(:), v(:), w(:), m(:)
    integer :: u_file, ios, i, n, line_count
    character(len=1000) :: line
    real :: G_actual, softening_actual
    integer :: N_actual, seed_actual
    character(len=50) :: scenario_actual
    logical :: in_particles, in_positions, in_velocities, in_masses
    logical :: in_x, in_y, in_z, in_u, in_v, in_w, in_m
    integer :: coord_count, particle_count
    real, allocatable :: temp_data(:)
    
    n = size(x)
    allocate(temp_data(n))
    
    ! Open file
    open(newunit=u_file, file=filename, status='old', action='read', iostat=ios)
    if (ios /= 0) then
      write(*,*) 'Error: Could not open file: ', trim(filename)
      stop
    end if
    
    ! Initialize
    in_particles = .false.
    in_positions = .false.
    in_velocities = .false.
    in_masses = .false.
    in_x = .false.
    in_y = .false.
    in_z = .false.
    in_u = .false.
    in_v = .false.
    in_w = .false.
    in_m = .false.
    coord_count = 0
    particle_count = 0
    G_actual = 1.0
    softening_actual = 0.015
    N_actual = 0
    seed_actual = 17
    scenario_actual = "galaxy_spiral"
    
    line_count = 0
    
    ! Read file line by line
    do
      read(u_file, '(A)', iostat=ios) line
      if (ios /= 0) exit
      line_count = line_count + 1
      
      ! Parse metadata
      if (index(line, '"N"') > 0) then
        call extract_integer(line, N_actual)
      else if (index(line, '"Gconst"') > 0) then
        call extract_number(line, G_actual)
      else if (index(line, '"softening"') > 0) then
        call extract_number(line, softening_actual)
      else if (index(line, '"seed"') > 0) then
        call extract_integer(line, seed_actual)
      end if
      
      ! Check for particles section
      if (index(line, '"particles"') > 0) then
        in_particles = .true.
        cycle
      end if
      
      if (in_particles) then
        ! Check for positions section
        if (index(line, '"positions"') > 0) then
          in_positions = .true.
          in_velocities = .false.
          in_masses = .false.
          cycle
        end if
        
        ! Check for velocities section
        if (index(line, '"velocities"') > 0) then
          in_positions = .false.
          in_velocities = .true.
          in_masses = .false.
          cycle
        end if
        
        ! Check for masses section
        if (index(line, '"masses"') > 0) then
          in_positions = .false.
          in_velocities = .false.
          in_masses = .true.
          in_m = .true.
          coord_count = 0
          cycle
        end if
        
        ! Check for coordinate arrays
        if (in_positions) then
          if (index(line, '"x"') > 0) then
            in_x = .true.
            in_y = .false.
            in_z = .false.
            coord_count = 0
            cycle
          else if (index(line, '"y"') > 0) then
            in_x = .false.
            in_y = .true.
            in_z = .false.
            coord_count = 0
            cycle
          else if (index(line, '"z"') > 0) then
            in_x = .false.
            in_y = .false.
            in_z = .true.
            coord_count = 0
            cycle
          end if
        else if (in_velocities) then
          if (index(line, '"u"') > 0) then
            in_u = .true.
            in_v = .false.
            in_w = .false.
            coord_count = 0
            cycle
          else if (index(line, '"v"') > 0) then
            in_u = .false.
            in_v = .true.
            in_w = .false.
            coord_count = 0
            cycle
          else if (index(line, '"w"') > 0) then
            in_u = .false.
            in_v = .false.
            in_w = .true.
            coord_count = 0
            cycle
          end if
        end if
        
        ! Extract coordinate data
        if (in_x .or. in_y .or. in_z .or. in_u .or. in_v .or. in_w .or. in_m) then
          call extract_coordinate_data_simple(line, temp_data, coord_count)
          
          ! Store data when we have enough coordinates
          if (coord_count >= n) then
            if (in_x) then
              x = temp_data
              in_x = .false.
            else if (in_y) then
              y = temp_data
              in_y = .false.
            else if (in_z) then
              z = temp_data
              in_z = .false.
            else if (in_u) then
              u = temp_data
              in_u = .false.
            else if (in_v) then
              v = temp_data
              in_v = .false.
            else if (in_w) then
              w = temp_data
              in_w = .false.
            else if (in_m) then
              m = temp_data
              in_m = .false.
            end if
            coord_count = 0
          end if
        end if
        
        ! Special handling for masses array (it's a flat array, not nested)
        if (in_masses .and. .not. in_m) then
          call extract_coordinate_data_simple(line, temp_data, coord_count)
          
          ! Store data when we have enough coordinates
          if (coord_count >= n) then
            m = temp_data
            in_masses = .false.
            coord_count = 0
          end if
        end if
        
        ! Check for end of particles section
        if (index(line, '}') > 0 .and. .not. in_positions .and. .not. in_velocities .and. .not. in_masses) then
          in_particles = .false.
          cycle
        end if
      end if
    end do
    
    close(u_file)
    deallocate(temp_data)
    
    ! Debug output removed for cleaner output
    
    ! Print import information
    write(*,*) 'Imported initial conditions from: ', trim(filename)
    write(*,*) '  Scenario: ', trim(scenario_actual)
    write(*,*) '  Particles: ', N_actual
    write(*,*) '  G: ', G_actual
    write(*,*) '  Softening: ', softening_actual
    write(*,*) '  Seed: ', seed_actual
  end subroutine import_initial_conditions
  
  subroutine extract_number(line, number)
    character(len=*), intent(in) :: line
    real, intent(out) :: number
    integer :: i, j, start_pos, end_pos
    character(len=100) :: number_str
    
    ! Find the first number in the line
    i = 1
    do while (i <= len(line))
      if (line(i:i) == '-' .or. (line(i:i) >= '0' .and. line(i:i) <= '9')) then
        ! Found start of a number
        start_pos = i
        j = i
        do while (j <= len(line) .and. &
                 (line(j:j) == '-' .or. line(j:j) == '.' .or. &
                  (line(j:j) >= '0' .and. line(j:j) <= '9') .or. &
                  line(j:j) == 'e' .or. line(j:j) == 'E' .or. line(j:j) == '+' .or. line(j:j) == '-'))
          j = j + 1
        end do
        end_pos = j - 1
        
        if (end_pos > start_pos) then
          number_str = line(start_pos:end_pos)
          read(number_str, *) number
          return
        end if
        
        i = j
      else
        i = i + 1
      end if
    end do
    
    ! If no number found, set to 0
    number = 0.0
  end subroutine extract_number
  
  subroutine extract_integer(line, number)
    character(len=*), intent(in) :: line
    integer, intent(out) :: number
    integer :: i, j, start_pos, end_pos
    character(len=100) :: number_str
    
    ! Find the first number in the line
    i = 1
    do while (i <= len(line))
      if (line(i:i) == '-' .or. (line(i:i) >= '0' .and. line(i:i) <= '9')) then
        ! Found start of a number
        start_pos = i
        j = i
        do while (j <= len(line) .and. &
                 (line(j:j) == '-' .or. (line(j:j) >= '0' .and. line(j:j) <= '9')))
          j = j + 1
        end do
        end_pos = j - 1
        
        if (end_pos > start_pos) then
          number_str = line(start_pos:end_pos)
          read(number_str, *) number
          return
        end if
        
        i = j
      else
        i = i + 1
      end if
    end do
    
    ! If no number found, set to 0
    number = 0
  end subroutine extract_integer
  
  subroutine extract_coordinate_data_simple(line, data, coord_count)
    character(len=*), intent(in) :: line
    real, intent(out) :: data(:)
    integer, intent(inout) :: coord_count
    integer :: i, j, start_pos, end_pos
    character(len=100) :: number_str
    real :: number
    
    ! Look for numbers in the line
    i = 1
    do while (i <= len(line))
      if (line(i:i) == '-' .or. (line(i:i) >= '0' .and. line(i:i) <= '9')) then
        ! Found start of a number
        start_pos = i
        j = i
        do while (j <= len(line) .and. &
                 (line(j:j) == '-' .or. line(j:j) == '.' .or. &
                  (line(j:j) >= '0' .and. line(j:j) <= '9') .or. &
                  line(j:j) == 'e' .or. line(j:j) == 'E' .or. line(j:j) == '+' .or. line(j:j) == '-'))
          j = j + 1
        end do
        end_pos = j - 1
        
        if (end_pos > start_pos) then
          number_str = line(start_pos:end_pos)
          read(number_str, *) number
          coord_count = coord_count + 1
          if (coord_count <= size(data)) then
            data(coord_count) = number
          end if
        end if
        
        i = j
      else
        i = i + 1
      end if
    end do
  end subroutine extract_coordinate_data_simple

end module json_reader