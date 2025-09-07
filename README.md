# Multi-Language Performance Benchmarking Suite

This repository contains performance benchmarking implementations of two computationally intensive simulations across three programming languages: **Python**, **Julia**, and **MATLAB**. The benchmarks compare both baseline and optimized versions of each simulation to evaluate language-specific performance characteristics.

## 🎯 Simulations Overview

### 1. Lattice Boltzmann Method (LBM) - Fluid Flow Past Cylinder
**Memory-bound simulation** that models 2D fluid dynamics using the Lattice Boltzmann Method with a D2Q9 lattice model.

- **Physics**: Simulates incompressible fluid flow around a circular cylinder
- **Method**: D2Q9 lattice model with collision and streaming steps
- **Characteristics**: Memory-intensive with regular memory access patterns
- **Phenomena**: Boundary layer separation, vortex shedding, wake formation
- **Files**: `lbm_cylinder.{py,jl,m}` (baseline), `lbm_cylinder_opt.{py,jl,m}` (optimized)

### 2. N-Body Simulation - Galaxy Dynamics
**Compute-bound simulation** that models gravitational interactions between particles in galactic systems.

- **Physics**: All-pairs gravitational force calculation with Plummer softening
- **Method**: Leapfrog integration scheme for orbital dynamics
- **Characteristics**: Compute-intensive with O(N²) complexity
- **Scenarios**: Spiral galaxy collision, simple galaxy, Plummer sphere, random distribution
- **Files**: `nbody.{py,jl,m}` (baseline), `nbody_opt.{py,jl,m}` (optimized)

## 🚀 Getting Started

### Julia Setup in VS Code

1. **Download Julia**:
   - Visit [julialang.org/downloads](https://julialang.org/downloads/)
   - Download Julia 1.9+ for your operating system
   - Add Julia to your system PATH during installation

2. **VS Code Extension**:
   - Install the "Julia" extension by Julia-VSCode
   - Extension ID: `julialang.language-julia`

3. **Configure Julia in VS Code**:
   - Open VS Code settings (Ctrl/Cmd + ,)
   - Search for "julia executable path"
   - Set the path to your Julia installation (if not auto-detected)

4. **Install Required Packages**:
   ```julia
   # Open Julia REPL in VS Code (Ctrl/Cmd + Shift + P → "Julia: Start REPL")
   using Pkg
   Pkg.add(["CairoMakie", "JSON3", "Distributions", "Random", "Statistics"])
   ```

### MATLAB Setup in VS Code

1. **Download MATLAB**:
   - Install MATLAB R2019b or later from [MathWorks](https://www.mathworks.com/products/matlab.html)
   - Ensure MATLAB is added to your system PATH during installation
   - Verify installation: open Command Prompt/Terminal and type `matlab -batch "version"`

2. **VS Code Extension**:
   - Install the "MATLAB" extension by MathWorks
   - Extension ID: `MathWorks.language-matlab`
   - This provides syntax highlighting, code navigation, and basic IntelliSense

3. **Configure MATLAB in VS Code**:
   - Open VS Code settings (Ctrl/Cmd + ,)
   - Search for "matlab executable path"
   - Set the path to your MATLAB installation (usually auto-detected)
   - Example path: `C:\Program Files\MATLAB\R2024b\bin\matlab.exe`

4. **Running MATLAB Code from VS Code**:
   ```bash
   # Run MATLAB scripts from VS Code terminal:
   matlab -batch "script_name"          # Run script (no .m extension)
   matlab -batch "function_name()"      # Run function
   matlab -singleCompThread -batch "script_name"  # Single-threaded (for benchmarking)
   ```

5. **VS Code Tips for MATLAB**:
   - Use `Ctrl/Cmd + Shift + P → "MATLAB: Open Command Window"` to open MATLAB terminal
   - Right-click in editor → "Run Current Section" to execute code blocks
   - Use `%%` to create code sections for interactive execution

### C and Fortran Setup

The repository includes C and Fortran implementations for additional performance comparison. These require compilation before execution.

#### Prerequisites

1. **C Compiler**:
   - **Windows**: Install [MinGW-w64](https://www.mingw-w64.org/) or [Microsoft Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
   - **Linux/macOS**: Install GCC (`sudo apt install gcc` on Ubuntu, `brew install gcc` on macOS)

2. **Fortran Compiler**:
   - **Windows**: Install [MinGW-w64 with Fortran](https://www.mingw-w64.org/) or [Intel Fortran Compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/fortran-compiler.html)
   - **Linux**: Install GFortran (`sudo apt install gfortran` on Ubuntu)
   - **macOS**: Install GFortran (`brew install gfortran`)

3. **Build Tools**:
   - **CMake**: Download from [cmake.org](https://cmake.org/download/) (version 3.10+)
   - **Ninja** (optional): Download from [ninja-build.org](https://ninja-build.org/) for faster builds

#### Compilation and Execution

**C Implementations**:
```bash
# Navigate to C code directory
cd C_lbm/          # or C_nbody/

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -G "MinGW Makefiles"    # Windows with MinGW
# or
cmake .. -G "Unix Makefiles"     # Linux/macOS

# Compile
cmake --build .

# Run the executable
./lbm_cylinder_c.exe             # Windows
./lbm_cylinder_c                 # Linux/macOS
```

**Fortran Implementations**:
```bash
# Navigate to Fortran code directory  
cd fortran_lbm/    # or fortran_nbody/

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -G "MinGW Makefiles"    # Windows with MinGW
# or  
cmake .. -G "Unix Makefiles"     # Linux/macOS

# Compile
cmake --build .

# Run the executable
./lbm_cylinder_fortran.exe       # Windows
./lbm_cylinder_fortran           # Linux/macOS
```

**Note**: The C and Fortran implementations generate visualization images in their respective `images/` folders. These images are included in the repository for reference but are not synced in future commits.

## 📋 Requirements

### Python Requirements
- **Python Version**: 3.8+
- **Required Libraries**:
  ```bash
  pip install numpy matplotlib numba
  ```
- **Optional for benchmarking**: `json` (built-in), `pathlib` (built-in)

### Julia Requirements
- **Julia Version**: 1.9+
- **Required Packages**:
  ```julia
  using Pkg
  Pkg.add([
      "CairoMakie",      # Plotting and visualization
      "JSON3",           # JSON file parsing
      "Distributions",   # Statistical distributions
      "Random",          # Random number generation
      "Statistics",      # Statistical functions
      "GeometryBasics"   # Geometric primitives (for LBM)
  ])
  ```

### MATLAB Requirements
- **MATLAB Version**: R2019b or later (requires support for `arguments` blocks)
- **Required Toolboxes**: None (uses built-in functions only)

## 🔧 Using the Benchmark Script

The `benchmark.py` script provides automated performance comparison across all three languages with statistical analysis and outlier detection.

### Basic Usage

1. **Run the benchmark script**:
   ```bash
   python benchmark.py
   ```

2. **Interactive Configuration**:
   - **Script Selection**: Choose which simulations to benchmark
     - Single: `lbm_cylinder` or `nbody`
     - Multiple: `lbm_cylinder,nbody_opt`
     - All: `lbm_cylinder,lbm_cylinder_opt,nbody,nbody_opt`
   
   - **Language Selection**: Choose which languages to compare
     - All languages: `p,j,m` (Python, Julia, MATLAB)
     - Subset: `p,j` (Python + Julia only)
     - Single: `m` (MATLAB only)

3. **Benchmark Modes**:
   - **No outlier detection**: Keeps all runs (useful for debugging)
   - **Outlier detection** (recommended): Finds consistent performance measurements
   - **Best of N**: Selects fastest runs from multiple attempts

### Example Benchmark Sessions

```bash
# Compare all implementations of LBM cylinder simulation
Script name(s): lbm_cylinder,lbm_cylinder_opt
Language codes: p,j,m

# Compare only optimized versions across Python and Julia
Script name(s): nbody_opt,lbm_cylinder_opt  
Language codes: p,j

# Full benchmark of all simulations and languages
Script name(s): lbm_cylinder,lbm_cylinder_opt,nbody,nbody_opt
Language codes: p,j,m
```

### Pre-Benchmark Setup

**Important**: Before running benchmarks, ensure visualization is disabled in all scripts:

- **Python**: Set `VISUALIZE = False`
- **Julia**: Set `VISUALIZE = false`  
- **MATLAB**: Set `VISUALIZE = false;`

This prevents matplotlib/plotting windows from appearing and ensures fair timing measurements.

### Benchmark Output

The script generates:
- **Performance rankings** with execution times and speedup ratios
- **Statistical analysis** including standard deviation and outlier detection
- **Visualization plots** comparing performance across languages
- **Detailed reports** saved to `benchmark_results_*` directories

### Advanced Configuration

Edit the configuration section in `benchmark.py`:

```python
N_WARMUPS = 2      # Warmup runs (for JIT compilation)
N_RUNS = 5         # Number of benchmark runs
N_MAX = 10         # Maximum attempts per language
THRESHOLD = 0.20   # Outlier detection threshold (20%)
SCRIPT_TIMEOUT = 900  # Timeout in seconds (15 minutes)
```

## 📊 Initial Conditions

The `nbody_ic.py` script generates reproducible initial conditions for N-body simulations:

```bash
python nbody_ic.py
```

This creates JSON files (e.g., `nbody_ic_galaxy_spiral_N4000.json`) that ensure identical starting conditions across all language implementations for fair performance comparison.

## 🏃‍♂️ Running Individual Simulations

### Python
```bash
python lbm_cylinder.py      # LBM baseline
python lbm_cylinder_opt.py  # LBM optimized
python nbody.py             # N-body baseline  
python nbody_opt.py         # N-body optimized
```

### Julia
```bash
julia lbm_cylinder.jl       # LBM baseline
julia lbm_cylinder_opt.jl   # LBM optimized
julia nbody.jl              # N-body baseline
julia nbody_opt.jl          # N-body optimized
```

### MATLAB
```matlab
lbm_cylinder()              % LBM baseline
lbm_cylinder_opt()          % LBM optimized  
nbody()                     % N-body baseline
nbody_opt()                 % N-body optimized
```

For MATLAB benchmarking, run single-threaded:
```bash
matlab -singleCompThread -batch "nbody()"
```

## 📁 Repository Structure

```
├── README.md                    # This file
├── benchmark.py                 # Main benchmarking script
├── nbody_ic.py                 # Initial conditions generator
│
├── lbm_cylinder.py             # LBM Python baseline
├── lbm_cylinder_opt.py         # LBM Python optimized
├── lbm_cylinder.jl             # LBM Julia baseline  
├── lbm_cylinder_opt.jl         # LBM Julia optimized
├── lbm_cylinder.m              # LBM MATLAB baseline
├── lbm_cylinder_opt.m          # LBM MATLAB optimized
│
├── nbody.py                    # N-body Python baseline
├── nbody_opt.py                # N-body Python optimized
├── nbody.jl                    # N-body Julia baseline
├── nbody_opt.jl                # N-body Julia optimized
├── nbody.m                     # N-body MATLAB baseline
└── nbody_opt.m                 # N-body MATLAB optimized
```

## 🎯 Performance Characteristics

### Expected Performance Patterns

- **Memory-bound (LBM)**: Performance often limited by memory bandwidth and cache efficiency
- **Compute-bound (N-body)**: Performance scales with computational throughput and vectorization

### Language-Specific Optimizations

- **Python**: NumPy vectorization, Numba JIT compilation
- **Julia**: Native performance, multiple dispatch, SIMD operations  
- **MATLAB**: Vectorized operations, built-in optimizations

## 📈 Benchmark Analysis

The benchmark suite provides insights into:
- **Absolute performance** across languages and optimization levels
- **Scalability** with problem size and complexity
- **Memory vs. compute** bound performance characteristics
- **JIT compilation** effects (Julia, Python/Numba)
- **Optimization effectiveness** across different algorithmic approaches

## 📄 License

This benchmarking suite is provided for educational and research purposes. Individual simulation implementations may have their own licensing terms.