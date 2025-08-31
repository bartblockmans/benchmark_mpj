#!/usr/bin/env python3
"""
General-Purpose Performance Benchmarking Script
==============================================

This script performs fair performance benchmarking of multi-language script implementations.
It supports Python (.py), Julia (.jl), and MATLAB (.m) scripts with flexible naming
and language selection.

Features:
- Dynamic script naming (user specifies base name)
- Flexible language selection (p=Python, j=Julia, m=MATLAB)
- Outlier detection and replacement
- Fair timing comparisons (single-threaded)
- Comprehensive reporting and visualization

Usage Examples:
- Benchmark scripts: script_name="lbm_cylinder", languages="p,m,j"
- Compare Python and Julia only: script_name="my_algorithm", languages="p,j"
- MATLAB-only benchmark: script_name="simulation", languages="m"

Author: Bart Blockmans
"""

import subprocess
import time
import platform
import sys
import os
import json

# Set matplotlib to non-interactive backend BEFORE importing pyplot
# This prevents any figures from popping up during benchmarking
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend that doesn't display figures

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')  # Suppress matplotlib warnings

# =============================================================================
# BENCHMARK CONFIGURATION PARAMETERS
# =============================================================================
# Modify these values to adjust benchmark behavior
N_WARMUPS = 1      # Number of warmup runs to handle compilation effects
N_RUNS = 3         # Number of good runs to collect for analysis
N_MAX = 5          # Maximum number of runs to attempt
THRESHOLD = 0.20   # Percentage threshold for outlier detection (20% = 0.20)
SCRIPT_TIMEOUT = 900  # Script timeout in seconds (15 minutes for long-running simulations)
# Note: Increase SCRIPT_TIMEOUT for simulations that legitimately take longer to complete
# =============================================================================

class Benchmarker:
    """
    General-purpose benchmarking class for multi-language script implementations.
    
    This class handles:
    - Running scripts in different programming languages
    - Fair timing measurement (accounting for compilation)
    - Result collection and analysis
    - Visualization and reporting
    - Flexible language selection (Python, Julia, MATLAB)
    - Dynamic script naming
    """
    
    def __init__(self, script_name='lbm_cylinder', languages='p,m,j', num_runs=N_RUNS, warmup_runs=N_WARMUPS, benchmark_mode='outlier_detection'):
        """
        Initialize the benchmarker.
        
        Parameters:
        -----------
        script_name : str
            Base name of the scripts to benchmark (without extension)
        languages : str
            Comma-separated language codes: 'p'=Python, 'j'=Julia, 'm'=MATLAB
            Examples: 'p,m,j', 'p,j', 'm,j', 'p'
        num_runs : int
            Number of benchmark runs to perform (excluding warmup)
        warmup_runs : int
            Number of warmup runs to handle compilation effects
        benchmark_mode : str
            Benchmark mode: 'no_outliers', 'outlier_detection', or 'best_of_10'
        """
        self.script_name = script_name
        self.languages = languages
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs
        self.benchmark_mode = benchmark_mode
        self.results = {}
        self.system_info = self._get_system_info()
        
        # Parse language selection and build script paths
        self.scripts = self._build_script_paths()
        self.benchmark_name = f"Script: {script_name} | Languages: {languages.replace(',', '_')}"
        
        # Output directories
        self.output_dir = f'benchmark_results_{script_name}_{languages.replace(",", "_")}'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _build_script_paths(self):
        """
        Build script paths based on user's language selection.
        
        Returns:
        --------
        dict : Dictionary mapping language names to script file paths
        """
        script_paths = {}
        language_codes = [code.strip().lower() for code in self.languages.split(',')]
        
        # Language code mapping
        language_mapping = {
            'p': ('Python', f'{self.script_name}.py'),
            'j': ('Julia', f'{self.script_name}.jl'),
            'm': ('MATLAB', f'{self.script_name}.m')
        }
        
        # Build scripts dictionary for selected languages
        for code in language_codes:
            if code in language_mapping:
                lang_name, script_file = language_mapping[code]
                script_paths[lang_name] = script_file
        
        return script_paths
    
    def _get_system_info(self):
        """Collect system information for the benchmark report."""
        info = {
            'platform': platform.platform(),
            'python_version': sys.version,
            'machine': platform.machine(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Try to get CPU information based on the operating system
        try:
            if platform.system() == 'Windows':
                # Windows: try multiple methods to get CPU info
                cpu_found = False
                
                # Method 1: Try wmic cpu get name
                try:
                    cpu_info = subprocess.check_output(['wmic', 'cpu', 'get', 'name'], 
                                                    text=True, stderr=subprocess.DEVNULL, timeout=10)
                    lines = cpu_info.strip().split('\n')
                    if len(lines) >= 2 and lines[1].strip():
                        info['cpu_model'] = lines[1].strip()
                        cpu_found = True
                except:
                    pass
                
                # Method 2: Try wmic cpu get caption (alternative)
                if not cpu_found:
                    try:
                        cpu_info = subprocess.check_output(['wmic', 'cpu', 'get', 'caption'], 
                                                        text=True, stderr=subprocess.DEVNULL, timeout=10)
                        lines = cpu_info.strip().split('\n')
                        if len(lines) >= 2 and lines[1].strip():
                            info['cpu_model'] = lines[1].strip()
                            cpu_found = True
                    except:
                        pass
                
                # Method 3: Try using platform.processor() as fallback
                if not cpu_found and platform.processor():
                    info['cpu_model'] = platform.processor()
                    cpu_found = True
                
                # Method 4: Use registry query (Windows-specific)
                if not cpu_found:
                    try:
                        cpu_info = subprocess.check_output(['reg', 'query', 'HKEY_LOCAL_MACHINE\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0', '/v', 'ProcessorNameString'], 
                                                        text=True, stderr=subprocess.DEVNULL, timeout=10)
                        for line in cpu_info.split('\n'):
                            if 'ProcessorNameString' in line:
                                cpu_name = line.split('REG_SZ')[1].strip()
                                if cpu_name:
                                    info['cpu_model'] = cpu_name
                                    cpu_found = True
                                    break
                    except:
                        pass
                
                if not cpu_found:
                    # Manual override: If you know your CPU model, you can uncomment and modify this line
                    # info['cpu_model'] = 'Intel(R) Core(TM) i7-1065G7 CPU @ 1.30GHz'
                    info['cpu_model'] = 'Windows CPU (Intel/AMD processor)'
                
                # Try to get more user-friendly CPU info if we have technical details
                if 'cpu_model' in info and info['cpu_model'] != 'Windows CPU (Intel/AMD processor)':
                    try:
                        # Try to extract more readable CPU info
                        cpu_raw = info['cpu_model']
                        if 'Intel64 Family 6' in cpu_raw:
                            # This is likely an Intel processor, try to get more details
                            try:
                                # Try to get additional CPU info
                                cpu_freq = subprocess.check_output(['wmic', 'cpu', 'get', 'MaxClockSpeed'], 
                                                                text=True, stderr=subprocess.DEVNULL, timeout=10)
                                lines = cpu_freq.strip().split('\n')
                                if len(lines) >= 2 and lines[1].strip().isdigit():
                                    freq_mhz = int(lines[1].strip())
                                    freq_ghz = freq_mhz / 1000.0
                                    info['cpu_model'] = f"Intel Processor @ {freq_ghz:.1f} GHz"
                                else:
                                    info['cpu_model'] = "Intel Processor"
                            except:
                                info['cpu_model'] = "Intel Processor"
                    except:
                        pass
                    
            elif platform.system() in ['Linux', 'Darwin']:
                # Linux/Unix: use lscpu
                try:
                    cpu_info = subprocess.check_output(['lscpu'], text=True, stderr=subprocess.DEVNULL, timeout=10)
                    for line in cpu_info.split('\n'):
                        if 'Model name' in line:
                            info['cpu_model'] = line.split(':')[1].strip()
                            break
                    else:
                        info['cpu_model'] = 'Unix CPU (details unavailable)'
                except:
                    info['cpu_model'] = 'Unix CPU (details unavailable)'
            else:
                info['cpu_model'] = 'Unknown OS CPU'
                
        except Exception:
            info['cpu_model'] = 'CPU info unavailable'
        
        # Ensure cpu_model key always exists as a fallback
        if 'cpu_model' not in info:
            info['cpu_model'] = 'CPU info unavailable'
            
        return info
    
    def _get_language_versions(self):
        """Get version information for each programming language."""
        versions = {}
        
        # Python version
        versions['Python'] = f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # Julia version
        try:
            result = subprocess.run(['julia', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                versions['Julia'] = result.stdout.strip()
            else:
                versions['Julia'] = 'Julia (version unknown)'
        except:
            versions['Julia'] = 'Julia (not found)'
        
        # MATLAB version
        try:
            # Try to get MATLAB version through command line
            result = subprocess.run(['matlab', '-batch', 'version'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                # Parse MATLAB version output
                for line in result.stdout.split('\n'):
                    if 'MATLAB Version' in line:
                        versions['MATLAB'] = line.strip()
                        break
                else:
                    # If no version found in output, try alternative approach
                    try:
                        # Try to get MATLAB version from executable path
                        result_path = subprocess.run(['where', 'matlab'], 
                                                  capture_output=True, text=True, timeout=10)
                        if result_path.returncode == 0:
                            matlab_path = result_path.stdout.strip().split('\n')[0]
                            if 'R2024b' in matlab_path or '2024b' in matlab_path:
                                versions['MATLAB'] = 'MATLAB R2024b'
                            elif 'R2024a' in matlab_path or '2024a' in matlab_path:
                                versions['MATLAB'] = 'MATLAB R2024a'
                            elif 'R2023b' in matlab_path or '2023b' in matlab_path:
                                versions['MATLAB'] = 'MATLAB R2023b'
                            else:
                                versions['MATLAB'] = 'MATLAB (version detected from path)'
                        else:
                            versions['MATLAB'] = 'MATLAB (version unknown)'
                    except:
                        versions['MATLAB'] = 'MATLAB (version unknown)'
            else:
                # If MATLAB command fails, try to detect from common paths
                try:
                    common_paths = [
                        r'C:\Program Files\MATLAB\R2024b\bin\matlab.exe',
                        r'C:\Program Files\MATLAB\R2024a\bin\matlab.exe',
                        r'C:\Program Files\MATLAB\R2023b\bin\matlab.exe',
                        r'C:\Program Files\MATLAB\R2023a\bin\matlab.exe'
                    ]
                    
                    for path in common_paths:
                        if os.path.exists(path):
                            if 'R2024b' in path:
                                versions['MATLAB'] = 'MATLAB R2024b'
                                break
                            elif 'R2024a' in path:
                                versions['MATLAB'] = 'MATLAB R2024a'
                                break
                            elif 'R2023b' in path:
                                versions['MATLAB'] = 'MATLAB R2023b'
                                break
                            elif 'R2023a' in path:
                                versions['MATLAB'] = 'MATLAB R2023a'
                                break
                    else:
                        versions['MATLAB'] = 'MATLAB (version unknown)'
                except:
                    versions['MATLAB'] = 'MATLAB (version unknown)'
        except:
            # Final fallback: try to detect from environment or common paths
            try:
                if 'MATLAB' in os.environ.get('PATH', ''):
                    versions['MATLAB'] = 'MATLAB (detected in PATH)'
                else:
                    versions['MATLAB'] = 'MATLAB (not found)'
            except:
                versions['MATLAB'] = 'MATLAB (not found)'
        
        # Manual override for known MATLAB version (you can modify this)
        if 'MATLAB' in versions and 'R2024b' not in versions['MATLAB']:
            # If we couldn't detect R2024b but you know it's that version
            try:
                # Check if MATLAB executable exists and try to infer version
                result_where = subprocess.run(['where', 'matlab'], 
                                            capture_output=True, text=True, timeout=10)
                if result_where.returncode == 0:
                    matlab_path = result_where.stdout.strip().split('\n')[0]
                    if 'MATLAB' in matlab_path:
                        versions['MATLAB'] = 'MATLAB R2024b'
            except:
                pass
        
        return versions
    
    def _get_benchmark_mode_description(self):
        """Get a human-readable description of the current benchmark mode."""
        mode_descriptions = {
            'no_outliers': 'No outlier detection - keeps all runs',
            'outlier_detection': f'Outlier detection - finds consistent subset or most consistent subset (threshold: {THRESHOLD*100:.0f}%)',
            'best_of_10': f'Best {N_RUNS} out of {N_MAX} - selects {N_RUNS} fastest runs from {N_MAX} total runs'
        }
        return mode_descriptions.get(self.benchmark_mode, 'Unknown mode')
    
    def _run_python_script(self):
        """Run the Python script and measure execution time."""
        start_time = time.time()
        
        try:
            result = subprocess.run([sys.executable, self.scripts['Python']], 
                                  capture_output=True, text=True, timeout=SCRIPT_TIMEOUT)
            end_time = time.time()
            
            if result.returncode == 0:
                # Check if there are any error messages in stderr
                if result.stderr and result.stderr.strip():
                    return end_time - start_time, False, f"Script completed but with errors: {result.stderr}"
                else:
                    return end_time - start_time, True, result.stdout
            else:
                return end_time - start_time, False, result.stderr
        except subprocess.TimeoutExpired:
            return SCRIPT_TIMEOUT, False, f"Script timed out after {SCRIPT_TIMEOUT//60} minutes"
        except Exception as e:
            return 0, False, str(e)
    
    def _run_julia_script(self):
        """Run the Julia script and measure execution time."""
        start_time = time.time()
        
        try:
            # Use single-threaded Julia for fair comparison
            result = subprocess.run(['julia', '--threads', '1', self.scripts['Julia']], 
                                  capture_output=True, text=True, timeout=SCRIPT_TIMEOUT)
            end_time = time.time()
            
            if result.returncode == 0:
                return end_time - start_time, True, result.stdout
            else:
                return end_time - start_time, False, result.stderr
        except subprocess.TimeoutExpired:
            return SCRIPT_TIMEOUT, False, f"Script timed out after {SCRIPT_TIMEOUT//60} minutes"
        except Exception as e:
            return 0, False, str(e)
    
    def _run_matlab_script(self):
        """Run the MATLAB script and measure execution time."""
        start_time = time.time()
        
        try:
            # Use single-threaded MATLAB for fair comparison
            # Extract script name without extension for MATLAB function call
            script_name = os.path.splitext(self.scripts['MATLAB'])[0]
            result = subprocess.run(['matlab', '-singleCompThread', '-batch', script_name], 
                                  capture_output=True, text=True, timeout=SCRIPT_TIMEOUT)
            end_time = time.time()
            
            if result.returncode == 0:
                return end_time - start_time, True, result.stdout
            else:
                return end_time - start_time, False, result.stderr
        except subprocess.TimeoutExpired:
            return SCRIPT_TIMEOUT, False, f"Script timed out after {SCRIPT_TIMEOUT//60} minutes"
        except Exception as e:
            return 0, False, str(e)
    
    def _detect_outliers(self, times):
        """
        Detect outliers using a simple percentage threshold relative to the mean.
        This is appropriate for performance benchmarking where we want consistent results.
        
        Parameters:
        -----------
        times : list
            List of execution times
            
        Returns:
        --------
        list : Boolean list indicating which times are outliers
        """
        if len(times) < 3:
            return [False] * len(times)  # Need at least 3 points
        
        times_array = np.array(times)
        
        # Use a percentage threshold relative to the mean
        # For performance benchmarking, THRESHOLD deviation is reasonable
        # This means times outside [(1-THRESHOLD)*mean, (1+THRESHOLD)*mean] are considered outliers
        mean_time = np.mean(times_array)
        
        lower_bound = mean_time * (1 - THRESHOLD)
        upper_bound = mean_time * (1 + THRESHOLD)
        
        # Identify outliers
        outliers = (times_array < lower_bound) | (times_array > upper_bound)
        
        # Additional safety check: if more than 40% of points are flagged as outliers,
        # the threshold is too strict, so don't flag any as outliers
        if np.sum(outliers) > len(times) * 0.4:
            return [False] * len(times)
        
        return outliers.tolist()
    
    def _check_consistency(self, times):
        """
        Check if the latest run is consistent with the existing pattern.
        This is used to provide real-time feedback during data collection.
        
        Strategy:
        - For the first 3 runs: check if all are consistent with each other
        - For subsequent runs: check if the new run is consistent with the 
          median of the existing runs (ignoring obvious outliers)
        
        Parameters:
        -----------
        times : list
            List of execution times
            
        Returns:
        --------
        bool : True if the pattern looks consistent, False otherwise
        """
        if len(times) < 3:
            return True  # Need at least 3 points to assess consistency
        
        times_array = np.array(times)
        
        if len(times) == 3:
            # For first 3 runs, check if they're all consistent with each other
            median_time = np.median(times_array)
            lower_bound = median_time * (1 - THRESHOLD)
            upper_bound = median_time * (1 + THRESHOLD)
            return np.all((times_array >= lower_bound) & (times_array <= upper_bound))
        else:
            # For subsequent runs, check if the new run is consistent with 
            # the median of what looks like the "good" runs so far
            
            # Strategy: Find the largest subset that's internally consistent
            # and check if the new run fits with that subset
            latest_run = times_array[-1]
            previous_runs = times_array[:-1]
            
            # Find a reference point from the previous runs
            # Use the median of the previous runs as a robust estimate
            reference_time = np.median(previous_runs)
            
            # Check if the latest run is consistent with this reference
            lower_bound = reference_time * (1 - THRESHOLD)
            upper_bound = reference_time * (1 + THRESHOLD)
            
            latest_is_consistent = lower_bound <= latest_run <= upper_bound
            
            if latest_is_consistent:
                # If the latest run seems consistent, check how many of the 
                # previous runs are also consistent with this pattern
                consistent_count = np.sum((previous_runs >= lower_bound) & (previous_runs <= upper_bound))
                # If at least 60% of previous runs are consistent with this pattern, 
                # consider the latest run as consistent too
                return consistent_count >= len(previous_runs) * 0.6
            else:
                # Latest run is clearly an outlier
                return False
    
    def _find_consistent_subset(self, times, target_size=5):
        """
        Find a subset of 'target_size' runs that are consistent with each other.
        This is the key fix: we look for ANY subset of 5 runs that are consistent,
        not just the entire dataset.
        
        If we find a consistent subset of 4 runs, we'll try to pad it to 5 by adding
        the most compatible additional run.
        
        Parameters:
        -----------
        times : list
            List of execution times
        target_size : int
            Number of runs needed for consistency (default: 5)
            
        Returns:
        --------
        dict or None: Dictionary with 'indices' and 'times' of consistent subset, or None if not found
        """
        if len(times) < target_size:
            return None
        
        times_array = np.array(times)
        from itertools import combinations
        
        # First try to find a subset of the exact target size
        for subset_indices in combinations(range(len(times)), target_size):
            subset_times = times_array[list(subset_indices)]
            subset_mean = np.mean(subset_times)
            
            # Check if this subset is consistent (all within THRESHOLD of subset mean)
            lower_bound = subset_mean * (1 - THRESHOLD)
            upper_bound = subset_mean * (1 + THRESHOLD)
            
            if np.all((subset_times >= lower_bound) & (subset_times <= upper_bound)):
                return {
                    'indices': list(subset_indices),
                    'times': subset_times.tolist(),
                    'mean': subset_mean
                }
        
        # If no exact subset found, try to find a consistent subset of 4 and pad it to {N_RUNS}
        if target_size == 5 and len(times) >= 5:
            for subset_indices in combinations(range(len(times)), 4):
                subset_times = times_array[list(subset_indices)]
                subset_mean = np.mean(subset_times)
                
                # Check if this 4-run subset is consistent
                lower_bound = subset_mean * (1 - THRESHOLD)
                upper_bound = subset_mean * (1 + THRESHOLD)
                
                if np.all((subset_times >= lower_bound) & (subset_times <= upper_bound)):
                    # Found a consistent 4-run subset, now try to add a {N_RUNS}th run
                    remaining_indices = [i for i in range(len(times)) if i not in subset_indices]
                    
                    # Try to find the best {N_RUNS}th run (closest to the subset mean)
                    best_additional_index = None
                    best_distance = float('inf')
                    
                    for candidate_idx in remaining_indices:
                        candidate_time = times_array[candidate_idx]
                        distance = abs(candidate_time - subset_mean)
                        
                        # Check if adding this run keeps the subset consistent
                        extended_times = np.append(subset_times, candidate_time)
                        extended_mean = np.mean(extended_times)
                        extended_lower = extended_mean * (1 - THRESHOLD)
                        extended_upper = extended_mean * (1 + THRESHOLD)
                        
                        if (np.all((extended_times >= extended_lower) & (extended_times <= extended_upper)) and 
                            distance < best_distance):
                            best_additional_index = candidate_idx
                            best_distance = distance
                    
                    if best_additional_index is not None:
                        final_indices = list(subset_indices) + [best_additional_index]
                        final_times = times_array[final_indices]
                        return {
                            'indices': final_indices,
                            'times': final_times.tolist(),
                            'mean': np.mean(final_times)
                        }
        
        return None
    
    def _select_most_consistent_subset(self, times_array, target_size):
        """
        Selects the most consistent subset of runs from a larger set.
        This is useful when outlier detection fails to achieve consistency.
        
        Parameters:
        -----------
        times_array : np.array
            Array of execution times.
        target_size : int
            The desired size of the subset.
            
        Returns:
        --------
        dict or None: A dictionary containing 'indices' of the selected subset
                      or None if no subset can be found.
        """
        if len(times_array) < target_size:
            return None

        # Calculate pairwise distances
        distances = np.zeros((len(times_array), len(times_array)))
        for i in range(len(times_array)):
            for j in range(i + 1, len(times_array)):
                distances[i, j] = np.abs(times_array[i] - times_array[j])
                distances[j, i] = distances[i, j]

        # Find the subset that minimizes the maximum distance
        best_subset_indices = []
        min_max_distance = float('inf')

        for i in range(len(times_array) - target_size + 1):
            current_subset_indices = list(range(i, i + target_size))
            current_max_distance = 0
            for j in range(target_size):
                for k in range(j + 1, target_size):
                    current_max_distance = max(current_max_distance, distances[current_subset_indices[j], current_subset_indices[k]])
            
            if current_max_distance < min_max_distance:
                min_max_distance = current_max_distance
                best_subset_indices = current_subset_indices

                return {'indices': best_subset_indices}
    
    def _select_best_runs(self, times_array, target_size):
        """
        Selects the fastest 'target_size' runs from a larger set.
        This is used for the 'best_of_10' mode.
        
        Parameters:
        -----------
        times_array : np.array
            Array of execution times.
        target_size : int
            The desired size of the subset.
            
        Returns:
        --------
        dict or None: A dictionary containing 'indices' of the selected subset
                      or None if no subset can be found.
        """
        if len(times_array) < target_size:
            return None
        
        # Get indices of the fastest runs
        fastest_indices = np.argsort(times_array)[:target_size]
        
        return {'indices': fastest_indices.tolist()}
    
    def _run_implementation(self, language):
        """Run benchmark for a specific language implementation."""
        print(f"Benchmarking {language} Implementation ({self.script_name})")
        print("=" * 60)
        
        # Warmup runs
        print(f"Performing {self.warmup_runs} warmup runs...")
        for i in range(self.warmup_runs):
            print(f"  Warmup run {i+1}/{self.warmup_runs}...")
            if language == 'Python':
                exec_time, success, output = self._run_python_script()
            elif language == 'Julia':
                exec_time, success, output = self._run_julia_script()
            elif language == 'MATLAB':
                exec_time, success, output = self._run_matlab_script()
            
            if not success:
                print(f"    Warmup failed: {output}")
                return None
        
        # Actual benchmark runs - collect ALL runs without removing anything
        print(f"Performing benchmark runs with mode: {self._get_benchmark_mode_description()}")
        
        times = []
        outputs = []
        total_runs = 0
        
        # Determine max attempts based on benchmark mode
        if self.benchmark_mode == 'best_of_10':
            max_attempts = N_MAX
        elif self.benchmark_mode == 'outlier_detection':
            max_attempts = N_MAX
        else:  # no_outliers
            max_attempts = self.num_runs
        
        # SIMPLIFIED APPROACH: Collect ALL runs first, then analyze
        while len(times) < max_attempts and total_runs < max_attempts:
            total_runs += 1
            good_runs_needed = max(0, self.num_runs - len(times))
            print(f"  Benchmark run {total_runs} (collecting data, need {N_RUNS} good runs total)...")
            
            if language == 'Python':
                exec_time, success, output = self._run_python_script()
            elif language == 'Julia':
                exec_time, success, output = self._run_julia_script()
            elif language == 'MATLAB':
                exec_time, success, output = self._run_matlab_script()
            
            if not success:
                print(f"    Failed: {output}")
                continue
            
            # Add this run to our collection (NEVER remove during collection)
            times.append(exec_time)
            outputs.append(output)
            
            # Show current progress immediately after successful run
            if len(times) < 3:
                print(f"    ✓ Run {total_runs}: {exec_time:.2f}s (building up sample)")
            else:
                # Check if current set is consistent (for informational purposes only)
                if self._check_consistency(times):
                    print(f"    ✓ Run {total_runs}: {exec_time:.2f}s (consistent)")
                else:
                    print(f"    ⚠️  Run {total_runs}: {exec_time:.2f}s (inconsistent, continuing...)")
            
            # Quick check: if we already have enough consistent runs, we can stop early
            if len(times) >= self.num_runs:
                if self.benchmark_mode == 'outlier_detection':
                    consistent_subset = self._find_consistent_subset(times, self.num_runs)
                    if consistent_subset:
                        print(f"    ✅ Found consistent subset of {len(consistent_subset['times'])} runs!")
                        print(f"    Stopping early - no need for additional runs.")
                        break
                elif self.benchmark_mode == 'no_outliers':
                    # For no_outliers mode, we can stop when we have exactly 5 runs
                    if len(times) == self.num_runs:
                        print(f"    ✅ Collected {len(times)} runs as requested")
                        break
        
        # Check if we hit the maximum attempts limit
        if total_runs >= max_attempts and self.benchmark_mode == 'outlier_detection':
            print(f"  ⚠️  Warning: Maximum attempts ({max_attempts}) reached. Using available runs.")
            print(f"     This may indicate that outlier detection is too strict for your data.")
        
        # POST-COLLECTION ANALYSIS: Now analyze the collected runs
        outliers_detected = []
        
        if self.benchmark_mode == 'outlier_detection':
            # First try to find a consistent subset of {N_RUNS} runs
            consistent_subset = self._find_consistent_subset(times, self.num_runs)
            if consistent_subset:
                print(f"    ✅ Using consistent subset of {len(consistent_subset['times'])} runs")
                
                # Update times and outputs to use the consistent subset
                best_indices = consistent_subset['indices']
                selected_times = [times[i] for i in best_indices]
                selected_outputs = [outputs[i] for i in best_indices]
                
                # Mark the removed runs as outliers for reporting
                outliers_detected = [times[i] for i in range(len(times)) if i not in best_indices]
                
                times = selected_times
                outputs = selected_outputs
                
            else:
                print(f"    ⚠️  No consistent subset found, selecting {self.num_runs} most consistent runs...")
                
                # Find the most consistent subset of {N_RUNS} runs
                times_array = np.array(times)
                best_subset = self._select_most_consistent_subset(times_array, self.num_runs)
                
                if best_subset is not None:
                    # Update times and outputs to use the most consistent subset
                    best_indices = best_subset['indices']
                    selected_times = [times[i] for i in best_indices]
                    selected_outputs = [outputs[i] for i in best_indices]
                    
                    # Mark the removed runs as outliers for reporting
                    outliers_detected = [times[i] for i in range(len(times)) if i not in best_indices]
                    
                    times = selected_times
                    outputs = selected_outputs
                else:
                    print(f"      ⚠️  Could not find consistent subset. Using all available runs.")
                    
        elif self.benchmark_mode == 'best_of_10':
            print(f"    ✅ Selecting {N_RUNS} fastest runs from {len(times)} total runs...")
            
            # Select the {N_RUNS} fastest runs
            times_array = np.array(times)
            best_subset = self._select_best_runs(times_array, self.num_runs)
            
            if best_subset is not None:
                # Update times and outputs to use the fastest subset
                best_indices = best_subset['indices']
                selected_times = [times[i] for i in best_indices]
                selected_outputs = [outputs[i] for i in best_indices]
                
                # Mark the removed runs as outliers for reporting
                outliers_detected = [times[i] for i in range(len(times)) if i not in best_indices]
                
                times = selected_times
                outputs = selected_outputs
            else:
                print(f"      ⚠️  Could not select best runs. Using all available runs.")
                
        else:  # no_outliers
            print(f"    Using all {len(times)} collected runs")
        
        # Calculate statistics
        times = np.array(times)
        stats = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
            'times': times.tolist(),
            'outputs': outputs,
            'total_runs': total_runs,
            'outliers_detected': outliers_detected,
            'outlier_count': len(outliers_detected)
        }
        
        print(f"  Results: {stats['mean']:.2f} ± {stats['std']:.2f} seconds")
        if outliers_detected:
            print(f"  Outliers removed: {len(outliers_detected)} runs")
        print(f"  Total runs performed: {total_runs}")
        
        return stats
    
    def run_benchmarks(self):
        """Run benchmarks for all three implementations."""
        print()
        print("Computational Performance Benchmarking")
        print("=" * 60)
        print(f"Script Type: {self.benchmark_name}")
        print(f"System: {self.system_info['platform']}")
        print(f"CPU: {self.system_info['cpu_model']}")
        print(f"Timestamp: {self.system_info['timestamp']}")
        print(f"Configuration: {self.num_runs} runs, {self.warmup_runs} warmup runs")
        print(f"Threading: Single-threaded for fair comparison")
        mode_description = self._get_benchmark_mode_description()
        print(f"Benchmark Mode: {mode_description}")
        
        # Get language versions
        versions = self._get_language_versions()
        print(f"\nLanguage Versions:")
        for lang, ver in versions.items():
            print(f"  {lang}: {ver}")
        
        # Run benchmarks for each implementation
        for language in self.scripts.keys():
            print(f"\n{'='*60}")
            print(f"Starting {language} benchmark...")
            
            result = self._run_implementation(language)
            if result:
                self.results[language] = result
                print(f"✓ {language} benchmark completed successfully")
            else:
                print(f"✗ {language} benchmark failed")
                self.results[language] = None
        
        # Generate report
        self._generate_report(versions)
        self._generate_plot(versions)
        
        print(f"\n{'='*60}")
        print("Benchmarking completed!")
        print(f"Results saved to: {self.output_dir}/")
        print(f"{'='*60}")
    
    def _generate_report(self, versions):
        """Generate a detailed benchmark report."""
        report = {
            'script_name': self.script_name,
            'languages': self.languages,
            'benchmark_name': self.benchmark_name,
            'system_info': self.system_info,
            'language_versions': versions,
            'benchmark_config': {
                'num_runs': self.num_runs,
                'warmup_runs': self.warmup_runs,
                'threading': 'single-threaded',
                'benchmark_mode': self.benchmark_mode
            },
            'results': self.results
        }
        
        # Save JSON report
        report_file = os.path.join(self.output_dir, 'benchmark_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save text report
        text_file = os.path.join(self.output_dir, 'benchmark_report.txt')
        with open(text_file, 'w') as f:
            f.write("Performance Benchmark Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Script Name: {self.script_name}\n")
            f.write(f"Languages: {self.languages}\n")
            f.write(f"Threading: Single-threaded for fair comparison\n\n")
            
            f.write(f"System Information:\n")
            f.write(f"  Platform: {self.system_info['platform']}\n")
            f.write(f"  Processor: {self.system_info['cpu_model']}\n")
            f.write(f"  Timestamp: {self.system_info['timestamp']}\n\n")
            
            f.write(f"Language Versions:\n")
            for lang, ver in versions.items():
                f.write(f"  {lang}: {ver}\n")
            f.write("\n")
            
            f.write(f"Benchmark Configuration:\n")
            f.write(f"  Number of runs: {self.num_runs}\n")
            f.write(f"  Warmup runs: {self.warmup_runs}\n")
            f.write(f"  Threading: Single-threaded\n")
            f.write(f"  Benchmark mode: {self._get_benchmark_mode_description()}\n\n")
            
            f.write(f"Results Summary:\n")
            f.write("-" * 30 + "\n")
            
            # Sort results by mean execution time
            sorted_results = sorted(
                [(lang, data) for lang, data in self.results.items() if data is not None],
                key=lambda x: x[1]['mean']
            )
            
            for i, (language, data) in enumerate(sorted_results):
                f.write(f"{i+1}. {language}:\n")
                f.write(f"   Mean: {data['mean']:.3f} ± {data['std']:.3f} seconds\n")
                f.write(f"   Range: {data['min']:.3f} - {data['max']:.3f} seconds\n")
                f.write(f"   Median: {data['median']:.3f} seconds\n")
                f.write(f"   Total runs performed: {data['total_runs']}\n")
                if data['outlier_count'] > 0:
                    f.write(f"   Outliers removed: {data['outlier_count']} (times: {[f'{t:.1f}s' for t in data['outliers_detected']]})\n")
                if i == 0:
                    f.write(f"   Performance: Best (baseline)\n")
                else:
                    speedup = sorted_results[0][1]['mean'] / data['mean']
                    f.write(f"   Performance: {speedup:.2f}x slower than best\n")
                f.write("\n")
        
        print()
        print(f"Report saved to: {report_file}")
        print(f"Text report saved to: {text_file}")
    
    def _generate_plot(self, versions):
        """Generate a bar plot comparing execution times."""
        # Filter out failed implementations
        valid_results = {lang: data for lang, data in self.results.items() if data is not None}
        
        if not valid_results:
            print("No valid results to plot")
            return
        
        # Prepare data for plotting
        languages = list(valid_results.keys())
        means = [valid_results[lang]['mean'] for lang in languages]
        stds = [valid_results[lang]['std'] for lang in languages]
        
        # Create labels with full language versions
        labels = [versions[lang] for lang in languages]
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Bar plot with error bars
        bars = plt.bar(range(len(languages)), means, yerr=stds, 
                      capsize=5, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Color the bars
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Customize the plot
        plt.xlabel('Implementation', fontsize=14, fontweight='bold')
        plt.ylabel('Execution Time (seconds)', fontsize=14, fontweight='bold')
        
        # Create informative title and subtitle
        title = f'Performance Benchmark Results - {self.script_name}'
        subtitle = f"Languages: {self.languages} | Single-threaded execution | {self.num_runs} runs, {self.warmup_runs} warmup runs"
        subtitle += f" | {self._get_benchmark_mode_description()}"
        if 'cpu_model' in self.system_info and self.system_info['cpu_model']:
            subtitle += f" | {self.system_info['cpu_model']}"
        
        # Add language versions to subtitle
        lang_versions = []
        for lang in languages:
            if lang in versions:
                lang_versions.append(f"{lang}: {versions[lang]}")
        if lang_versions:
            subtitle += f"\n{', '.join(lang_versions)}"
        
        plt.title(title + '\n' + subtitle, 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Set x-axis labels
        plt.xticks(range(len(languages)), labels, fontsize=12, rotation=0)
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            plt.text(i, mean + std + max(means) * 0.01, 
                    f'{mean:.2f}s\n±{std:.2f}s', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add grid for better readability
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(self.output_dir, 'performance_comparison.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance plot saved to: {plot_file}")
    
    def print_summary(self):
        """Print a summary of the benchmark results."""
        if not self.results:
            print("No benchmark results available")
            return
        
        print(f"\n{'='*60}")
        print(f"BENCHMARK SUMMARY - {self.script_name}")
        print(f"Languages: {self.languages}")
        print(f"Benchmark Mode: {self._get_benchmark_mode_description()}")
        print("="*60)
        
        # Sort results by mean execution time
        sorted_results = sorted(
            [(lang, data) for lang, data in self.results.items() if data is not None],
            key=lambda x: x[1]['mean']
        )
        
        print(f"{'Rank':<6} {'Language':<10} {'Mean (s)':<12} {'Std (s)':<10} {'Speedup':<10} {'Runs':<6} {'Outliers':<10}")
        print("-" * 70)
        
        baseline_time = sorted_results[0][1]['mean']
        for i, (language, data) in enumerate(sorted_results):
            speedup = baseline_time / data['mean']
            # Use benchmark_mode instead of enable_outlier_detection
            if self.benchmark_mode != 'no_outliers':
                outlier_info = f"{data['outlier_count']}" if data['outlier_count'] > 0 else "0"
            else:
                outlier_info = "N/A"
            print(f"{i+1:<6} {language:<10} {data['mean']:<12.3f} {data['std']:<10.3f} {speedup:<10.2f}x {data['total_runs']:<6} {outlier_info:<10}")
        
        print("\n" + "="*60)
        print("OUTLIER DETECTION SUMMARY:")
        print("="*60)
        
        if self.benchmark_mode == 'no_outliers':
            print("No outlier detection was used for this benchmark run.")
            print("All runs were kept for analysis.")
        else:
            for language, data in sorted_results:
                if data['outlier_count'] > 0:
                    print(f"{language}: {data['outlier_count']} outlier(s) removed")
                    print(f"  Outlier times: {[f'{t:.1f}s' for t in data['outliers_detected']]}")
                    print(f"  Total runs performed: {data['total_runs']}")
                    print()
                else:
                    print(f"{language}: No outliers detected ({data['total_runs']} runs)")
        
        print("="*60)


def main():
    """Main function to run the benchmarking."""
    print("General Performance Benchmarking Script")
    print("=" * 50)
    
    # Ask user for script names (can be multiple, comma-separated)
    print("\nWhat is the name of the scripts you want to benchmark?")
    print("(Enter the base name(s) without extension, e.g., 'lbm_cylinder' or 'nbody,nbody_opt')")
    print("Multiple scripts can be specified by separating them with commas.")
    
    while True:
        try:
            script_names_input = input("\nScript name(s): ").strip()
            if script_names_input:
                # Parse comma-separated script names
                script_names = [name.strip() for name in script_names_input.split(',') if name.strip()]
                if script_names:
                    break
                else:
                    print("Please enter at least one valid script name.")
            else:
                print("Please enter at least one script name.")
        except KeyboardInterrupt:
            print("\nBenchmarking cancelled.")
            return
    
    # Ask user for language selection
    print(f"\nWhich programming languages do you want to benchmark?")
    print("Use comma-separated codes:")
    print("  'p' = Python (.py)")
    print("  'j' = Julia (.jl)")
    print("  'm' = MATLAB (.m)")
    print("Examples: 'p,m,j' (all three), 'p,j' (Python + Julia), 'm' (MATLAB only)")
    
    while True:
        try:
            languages = input("\nLanguage codes: ").strip().lower()
            if languages:
                # Validate language codes
                valid_codes = ['p', 'j', 'm']
                input_codes = [code.strip() for code in languages.split(',')]
                
                if all(code in valid_codes for code in input_codes):
                    break
                else:
                    print("Invalid language codes. Please use only 'p', 'j', and/or 'm'.")
            else:
                print("Please enter at least one language code.")
        except KeyboardInterrupt:
            print("\nBenchmarking cancelled.")
            return
    
    # Determine which scripts to check based on user choice
    required_scripts = []
    language_mapping = {'p': '.py', 'j': '.jl', 'm': '.m'}
    
    # Check each script name for the selected languages
    for script_name in script_names:
        for code in languages.split(','):
            code = code.strip()
            if code in language_mapping:
                required_scripts.append(f"{script_name}{language_mapping[code]}")
    
    # Check if required scripts exist
    missing_scripts = [script for script in required_scripts if not os.path.exists(script)]
    
    if missing_scripts:
        print(f"\nError: Missing required scripts:")
        for script in missing_scripts:
            print(f"  - {script}")
        print(f"\nPlease ensure all required scripts are in the current directory.")
        return
    
    # Check if scripts are in performance mode
    print(f"\n⚠️  IMPORTANT: Before running benchmarks, ensure VISUALIZE = False in all scripts!")
    print("   This prevents matplotlib figures from appearing and ensures fair timing.")
    print("   - Python: VISUALIZE = False")
    print("   - Julia:  VISUALIZE = false") 
    print("   - MATLAB: VISUALIZE = false;")
    print()
    
    # Ask for confirmation
    response = input("Have you set VISUALIZE = False in all scripts? (y/n): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Please set VISUALIZE = False in all scripts and run again.")
        return
    
    # Ask about benchmark mode
    print(f"\nBenchmark Mode Selection:")
    print("   Choose one of the following approaches:")
    print(f"   1. No outlier detection - keeps all {N_RUNS} runs")
    print(f"   2. Outlier detection - finds consistent subset of {N_RUNS} runs")
    print(f"   3. Best {N_RUNS} out of {N_MAX} - selects {N_RUNS} fastest runs from {N_MAX} total runs")
    
    while True:
        try:
            mode_response = input("Select benchmark mode (1/2/3): ").strip()
            
            if mode_response == '1':
                benchmark_mode = 'no_outliers'
                break
            elif mode_response == '2':
                benchmark_mode = 'outlier_detection'
                break
            elif mode_response == '3':
                benchmark_mode = 'best_of_10'
                break
            else:
                print("Please enter '1', '2', or '3'.")
        except KeyboardInterrupt:
            print("\nBenchmarking cancelled.")
            return
    
    # Run benchmarks for each script
    for i, script_name in enumerate(script_names):
        print(f"\n{'='*80}")
        print(f"BENCHMARKING SCRIPT {i+1}/{len(script_names)}: {script_name}")
        print(f"{'='*80}")
        
        # Create and run benchmarker for this script
        benchmarker = Benchmarker(script_name=script_name, languages=languages, num_runs=N_RUNS, warmup_runs=N_WARMUPS, benchmark_mode=benchmark_mode)
        
        try:
            benchmarker.run_benchmarks()
            benchmarker.print_summary()
            
            # Add separator between scripts (except after the last one)
            if i < len(script_names) - 1:
                print(f"\n{'='*80}")
                print(f"Completed benchmark for '{script_name}'. Moving to next script...")
                print(f"{'='*80}")
                
        except KeyboardInterrupt:
            print(f"\n\nBenchmarking interrupted by user during '{script_name}'")
            break
        except Exception as e:
            print(f"\n\nError during benchmarking of '{script_name}': {e}")
            import traceback
            traceback.print_exc()
            
            # Continue with remaining scripts automatically (no user interaction)
            if i < len(script_names) - 1:
                print(f"Continuing with remaining scripts automatically...")
            else:
                print("This was the last script to benchmark.")
    
    # Final summary
    print(f"\n{'='*80}")
    print("ALL BENCHMARKING COMPLETED!")
    print(f"Successfully benchmarked {len(script_names)} script(s): {', '.join(script_names)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main() 