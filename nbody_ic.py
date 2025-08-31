#!/usr/bin/env python3
"""
N-Body Initial Conditions Generator
==================================

This script generates initial conditions for N-body simulations and exports them
in a portable format (JSON) that can be read by Python, MATLAB, and Julia.

The exported file contains all particle positions, velocities, masses, and
simulation parameters needed to ensure identical starting conditions across
different language implementations.

Usage:
    python nbody_ic.py

This will create a file named 'nbody_ic_<SCENARIO>_N<N>.json' containing
the initial conditions for the specified scenario.
"""

import numpy as np
import json
import os
from pathlib import Path

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# Initial condition scenario
SCENARIO = "galaxy_spiral"    # "galaxy_spiral" | "galaxy" | "plummer" | "random"

# Simulation parameters
N = 4000                      # Number of particles
Gconst = 1.0                  # Gravitational constant (normalized units)
softening = 1.5e-2            # Plummer softening parameter
SEED = 17                     # Random seed for reproducible results

# Output settings
OUTPUT_DIR = "."              # Directory to save the initial conditions file
FORMAT_VERSION = "1.0"        # Format version for compatibility checking

# =============================================================================
# INITIAL CONDITION GENERATION FUNCTIONS
# =============================================================================

def generate_disk(N, mass_total, Rmax=1.2, z_thick=0.1, v_rot=0.9, jitter=0.05):
    """Generate initial conditions for a rotating stellar disk."""
    # Generate radial positions (more particles near center)
    R = Rmax * np.sqrt(np.random.rand(N))
    
    # Generate random azimuthal angles
    theta = 2 * np.pi * np.random.rand(N)
    
    # Convert to Cartesian coordinates
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    z = (z_thick * 0.5) * np.random.randn(N)
    
    # Circular velocity for orbital stability
    vtan = v_rot * R / (0.3 + R)
    u = -vtan * np.sin(theta)
    v = vtan * np.cos(theta)
    w = np.zeros(N)
    
    # Add random velocity perturbations
    u = u + jitter * np.random.randn(N)
    v = v + jitter * np.random.randn(N)
    w = w + 0.5 * jitter * np.random.randn(N)
    
    # Equal mass particles
    m = (mass_total / N) * np.ones(N)
    
    return x, y, z, u, v, w, m

def generate_spiral_disk(N, mass_total, Rd=0.6, Rmax=1.8, m_arms=2, pitch_deg=18,
                        arm_amp=0.65, z_thick=0.08, v0=1.0, v_rise=0.35,
                        nudge_r=0.05, nudge_t=0.02, jitter=0.03, phi0=None):
    """Generate initial conditions for a spiral galaxy disk."""
    if phi0 is None:
        phi0 = 2 * np.pi * np.random.rand()
    
    # Generate radial positions using gamma distribution
    R = Rd * (-np.log(np.random.rand(N) * np.random.rand(N)))
    
    # Ensure all particles are within Rmax
    while np.any(R > Rmax):
        mask = R > Rmax
        R[mask] = Rd * (-np.log(np.random.rand(np.sum(mask)) * np.random.rand(np.sum(mask))))
    
    # Generate azimuthal angles with spiral arm overdensity
    k = 1 / np.tan(np.deg2rad(pitch_deg))
    theta = np.zeros(N)
    filled = 0
    
    # Use rejection sampling to create spiral arm structure
    while filled < N:
        need = N - filled
        th_try = (2 * np.pi) * np.random.rand(2 * need)
        Rrep = np.tile(R[filled:filled + need], 2)
        
        # Probability density for spiral arms
        pacc = 1 + arm_amp * np.cos(m_arms * (th_try - k * np.log(Rrep + 1e-6) - phi0))
        uacc = (1 + arm_amp) * np.random.rand(2 * need)
        
        # Accept particles based on spiral arm probability
        keep = uacc < pacc
        nkeep = min(need, np.sum(keep))
        theta[filled:filled + nkeep] = th_try[np.where(keep)[0][:nkeep]]
        filled = filled + nkeep
    
    # Convert to Cartesian coordinates
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    z = (z_thick * 0.5) * np.random.randn(N)
    
    # Velocity field with spiral structure
    v_circ = v0 * np.tanh(R / v_rise)
    phase = m_arms * (theta - k * np.log(R + 1e-6) - phi0)
    
    # Velocity perturbations aligned with spiral arms
    v_r = nudge_r * v_circ * np.cos(phase)
    v_t = v_circ * (1 + nudge_t * np.sin(phase))
    
    # Convert to Cartesian velocity components
    u = -v_t * np.sin(theta) + v_r * np.cos(theta)
    v = v_t * np.cos(theta) + v_r * np.sin(theta)
    w = 0.5 * jitter * np.random.randn(N)
    
    # Add random perturbations
    u = u + jitter * np.random.randn(N)
    v = v + jitter * np.random.randn(N)
    
    # Equal mass particles
    m = (mass_total / N) * np.ones(N)
    
    return x, y, z, u, v, w, m

def generate_plummer(N, mass_total, a=0.5):
    """Generate initial conditions for a Plummer sphere."""
    # Generate positions using Plummer distribution
    U = np.random.rand(N)
    r = a / np.sqrt(U ** (-2/3) - 1 + 1e-6)
    
    # Random spherical angles
    phi = 2 * np.pi * np.random.rand(N)
    cos_t = 2 * np.random.rand(N) - 1
    sin_t = np.sqrt(np.maximum(0, 1 - cos_t**2))
    
    # Convert to Cartesian coordinates
    x = r * sin_t * np.cos(phi)
    y = r * sin_t * np.sin(phi)
    z = r * cos_t
    
    # Small random velocities
    u = 0.02 * np.random.randn(N)
    v = 0.02 * np.random.randn(N)
    w = 0.02 * np.random.randn(N)
    m = (20.0 / N) * np.ones(N)
    
    return x, y, z, u, v, w, m

def generate_initial_conditions(N, seed, scenario):
    """Generate initial conditions for the specified scenario."""
    np.random.seed(seed)
    
    if scenario == "galaxy_spiral":
        # Two spiral galaxies on collision course
        N1 = N // 2
        N2 = N - N1
        
        # Different initial phases for the two galaxies
        phi1 = 2 * np.pi * np.random.rand()
        phi2 = phi1 + np.pi / 3
        
        # Generate first spiral galaxy
        x1, y1, z1, u1, v1, w1, m1 = generate_spiral_disk(
            N1, 10.0, Rd=0.55, Rmax=1.7, m_arms=2, pitch_deg=18, arm_amp=0.70,
            z_thick=0.07, v0=1.05, v_rise=0.32, nudge_r=0.06, nudge_t=0.03,
            jitter=0.025, phi0=phi1
        )
        
        # Generate second spiral galaxy
        x2, y2, z2, u2, v2, w2, m2 = generate_spiral_disk(
            N2, 10.0, Rd=0.55, Rmax=1.7, m_arms=2, pitch_deg=18, arm_amp=0.70,
            z_thick=0.07, v0=1.05, v_rise=0.32, nudge_r=0.06, nudge_t=0.03,
            jitter=0.025, phi0=phi2
        )
        
        # Reverse velocity of second galaxy and offset positions
        u2 = -u2
        v2 = -v2  # Counter-rotating
        d = 2.1
        vcm = 0.45  # Initial separation and center-of-mass velocity
        
        # Position and velocity offsets for collision
        x1 = x1 - d
        v1 = v1 + vcm
        x2 = x2 + d
        v2 = v2 - vcm
        
        # Combine the two galaxies
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])
        z = np.concatenate([z1, z2])
        u = np.concatenate([u1, u2])
        v = np.concatenate([v1, v2])
        w = np.concatenate([w1, w2])
        m = np.concatenate([m1, m2])
        
    elif scenario == "galaxy":
        # Two simple disk galaxies on collision course
        N1 = N // 2
        N2 = N - N1
        
        # Generate two identical disk galaxies
        x1, y1, z1, u1, v1, w1, m1 = generate_disk(
            N1, 10.0, Rmax=1.1, z_thick=0.08, v_rot=1.0, jitter=0.05
        )
        x2, y2, z2, u2, v2, w2, m2 = generate_disk(
            N2, 10.0, Rmax=1.1, z_thick=0.08, v_rot=1.0, jitter=0.05
        )
        
        # Counter-rotating galaxies
        u2 = -u2
        v2 = -v2
        d = 2.0
        vcm = 0.5  # Initial separation and center-of-mass velocity
        
        # Position and velocity offsets
        x1 = x1 - d
        v1 = v1 + vcm
        x2 = x2 + d
        v2 = v2 - vcm
        
        # Combine the galaxies
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])
        z = np.concatenate([z1, z2])
        u = np.concatenate([u1, u2])
        v = np.concatenate([v1, v2])
        w = np.concatenate([w1, w2])
        m = np.concatenate([m1, m2])
        
    elif scenario == "plummer":
        # Plummer sphere: a realistic model for globular clusters
        x, y, z, u, v, w, m = generate_plummer(N, 20.0, a=0.5)
        
    else:  # "random"
        # Random particle distribution (useful for testing)
        x = np.random.randn(N)
        y = np.random.randn(N)
        z = np.random.randn(N)
        u = np.random.randn(N)
        v = np.random.randn(N)
        w = np.random.randn(N)
        m = (20.0 / N) * np.ones(N)
    
    # Transform to center-of-mass frame for numerical stability
    mu = np.mean(m * u)
    mv = np.mean(m * v)
    mw = np.mean(m * w)
    mbar = np.mean(m)
    
    u = u - mu / mbar
    v = v - mv / mbar
    w = w - mw / mbar
    
    return x, y, z, u, v, w, m

def export_initial_conditions(x, y, z, u, v, w, m, scenario, N, Gconst, softening, seed, output_dir):
    """Export initial conditions to a portable JSON file."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    filename = f"nbody_ic_{scenario}_N{N}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Prepare data for export
    # Convert numpy arrays to lists for JSON compatibility
    export_data = {
        "format_version": FORMAT_VERSION,
        "metadata": {
            "scenario": scenario,
            "N": int(N),
            "Gconst": float(Gconst),
            "softening": float(softening),
            "seed": int(seed),
            "generated_by": "nbody_ic.py",
            "timestamp": str(np.datetime64('now'))
        },
        "particles": {
            "positions": {
                "x": x.tolist(),
                "y": y.tolist(),
                "z": z.tolist()
            },
            "velocities": {
                "u": u.tolist(),
                "v": v.tolist(),
                "w": w.tolist()
            },
            "masses": m.tolist()
        }
    }
    
    # Write to JSON file
    with open(filepath, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Initial conditions exported to: {filepath}")
    print(f"File size: {os.path.getsize(filepath) / 1024:.1f} KB")
    
    return filepath

def main():
    """Main function to generate and export initial conditions."""
    print("N-Body Initial Conditions Generator")
    print("=" * 50)
    print(f"Scenario: {SCENARIO}")
    print(f"Number of particles: {N}")
    print(f"Gravitational constant: {Gconst}")
    print(f"Softening parameter: {softening}")
    print(f"Random seed: {SEED}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Generate initial conditions
    print("Generating initial conditions...")
    x, y, z, u, v, w, m = generate_initial_conditions(N, SEED, SCENARIO)
    
    # Verify the results
    print(f"Generated {len(x)} particles")
    print(f"Position range: x=[{x.min():.3f}, {x.max():.3f}], y=[{y.min():.3f}, {y.max():.3f}], z=[{z.min():.3f}, {z.max():.3f}]")
    print(f"Velocity range: u=[{u.min():.3f}, {u.max():.3f}], v=[{v.min():.3f}, {v.max():.3f}], w=[{w.min():.3f}, {w.max():.3f}]")
    print(f"Mass range: [{m.min():.3f}, {m.max():.3f}]")
    
    # Check center-of-mass frame
    total_momentum_x = np.sum(m * u)
    total_momentum_y = np.sum(m * v)
    total_momentum_z = np.sum(m * w)
    print(f"Total momentum: ({total_momentum_x:.2e}, {total_momentum_y:.2e}, {total_momentum_z:.2e})")
    
    if abs(total_momentum_x) < 1e-10 and abs(total_momentum_y) < 1e-10 and abs(total_momentum_z) < 1e-10:
        print("✓ Center-of-mass frame transformation successful")
    else:
        print("✗ Center-of-mass frame transformation failed")
    
    # Export to file
    print("\nExporting initial conditions...")
    filepath = export_initial_conditions(x, y, z, u, v, w, m, SCENARIO, N, Gconst, softening, SEED, OUTPUT_DIR)
    
    print("\nInitial conditions generation complete!")
    print(f"File: {filepath}")
    print("\nThis file can now be imported by the N-body simulation scripts")
    print("in Python, MATLAB, and Julia to ensure identical starting conditions.")

if __name__ == "__main__":
    main() 