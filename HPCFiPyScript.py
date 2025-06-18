#!/usr/bin/env python3
import os
import h5py
from fipy import (CellVariable, Grid3D, dump, TransientTerm, DiffusionTerm,
                   LinearGMRESSolver, ImplicitSourceTerm)
from mpi4py import MPI
import time
import numpy as np
from fipy import dump
def get_current_time(ramdisk_path):
    """
    Get the current simulation time in minutes.

    Parameters:
    -----------
    ramdisk_path : str
        Path to the ramdisk directory

    Returns:
    --------
    time : float
        Current simulation time in minutes
    """
    # Try to read time from simulation parameters
    params_file = os.path.join(ramdisk_path, "simulation_params.h5")
    if os.path.exists(params_file):
        try:
            with h5py.File(params_file, 'r') as h5f:
                # Check for explicit time value first
                if 'current_time' in h5f:
                    return h5f['current_time'][()]
                # Otherwise calculate from step and dt
                elif 'current_step' in h5f and 'dt' in h5f:
                    return h5f['current_step'][()] * h5f['dt'][()]
        except Exception as e:
            print(f"Error reading time from params file: {e}")

    # Try to read step from step file and calculate time using dt
    step_file = os.path.join(ramdisk_path, "current_step.txt")
    if os.path.exists(step_file):
        try:
            with open(step_file, 'r') as f:
                step = int(f.read().strip())
                # Try to get dt value
                if os.path.exists(params_file):
                    try:
                        with h5py.File(params_file, 'r') as h5f:
                            if 'dt' in h5f:
                                return step * h5f['dt'][()]
                    except:
                        pass
                # Default dt of 1.0 if not available
                return float(step)
        except Exception as e:
            print(f"Error reading step file: {e}")

    # Default to 0 if time cannot be determined
    return 0.0

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

t_start_total = time.time()

# Define the ramdisk path – this must match what the main steppable uses.
ramdisk_path = "/home/anton/miniconda3/envs/cc3d/lib/python3.10/site-packages/cancer-modeling/ramdisk"

params_filepath = os.path.join(ramdisk_path, "simulation_params.h5")
with h5py.File(params_filepath, 'r') as h5f:
    sim_params = {key: h5f[key][()] for key in h5f.keys()}
    print(sim_params)

# Read in the mesh.
mesh_filepath = os.path.join(ramdisk_path, "mesh.dat")
mesh = dump.read(mesh_filepath)
nx = len(np.unique(mesh.cellCenters[0].value))
ny = len(np.unique(mesh.cellCenters[1].value))
nz = len(np.unique(mesh.cellCenters[2].value))
num_voxels = nx * ny * nz
print(f"Number of voxels in the PDE domain: {num_voxels}")


# Read in the PDE fields.
oxygen = dump.read(os.path.join(ramdisk_path, "oxygen.dat"))
nutrient = dump.read(os.path.join(ramdisk_path, "nutrient.dat"))
IL8 = dump.read(os.path.join(ramdisk_path, "cytokine_il8.dat"))
IL6 = dump.read(os.path.join(ramdisk_path, "cytokine_il6.dat"))
VEGF = dump.read(os.path.join(ramdisk_path, "vegf.dat"))

# Read in the cell presence fields.
cellpresent_e = dump.read(os.path.join(ramdisk_path, "cellpresent_e.dat"))
cellpresent_s = dump.read(os.path.join(ramdisk_path, "cellpresent_s.dat"))

# Read in the pleural mask.
pleural_mask = dump.read(os.path.join(ramdisk_path, "pleural_mask.dat"))

# Wrap fields as CellVariables if necessary.
oxygen_var   = oxygen if not hasattr(oxygen, 'flatten') else CellVariable(mesh=mesh, value=oxygen.flatten(), name="Oxygen")
nutrient_var = nutrient if not hasattr(nutrient, 'flatten') else CellVariable(mesh=mesh, value=nutrient.flatten(), name="Nutrient")
IL8_var      = IL8 if not hasattr(IL8, 'flatten') else CellVariable(mesh=mesh, value=IL8.flatten(), name="IL8")
IL6_var      = IL6 if not hasattr(IL6, 'flatten') else CellVariable(mesh=mesh, value=IL6.flatten(), name="IL6")
VEGF_var = VEGF if not hasattr(VEGF, 'flatten') else  CellVariable(mesh=mesh, value=VEGF.flatten(), name="VEGF")

cellpresent_e_var = cellpresent_e if not hasattr(cellpresent_e, 'flatten') else CellVariable(mesh=mesh, value=cellpresent_e.flatten(), name="CellPresentE")
cellpresent_s_var = cellpresent_s if not hasattr(cellpresent_s, 'flatten') else CellVariable(mesh=mesh, value=cellpresent_s.flatten(), name="CellPresentS")
pleural_mask_var  = pleural_mask

# Set up the solver.
mysolver = LinearGMRESSolver()
dt = sim_params["dt"]  # Now dt is 1 minute

# Wrap constant parameters as CellVariables.
mu_oxygen_var = CellVariable(mesh=mesh, value=sim_params["mu_oxygen"], name="mu_oxygen")
k_oxygen_epi_var = CellVariable(mesh=mesh, value=sim_params["k_oxygen_epi"], name="k_oxygen_epi")
k_oxygen_sarc_var = CellVariable(mesh=mesh, value=sim_params["k_oxygen_sarc"], name="k_oxygen_sarc")

mu_nutrient_var = CellVariable(mesh=mesh, value=sim_params["mu_nutrient"], name="mu_nutrient")
k_nutrient_epi_var = CellVariable(mesh=mesh, value=sim_params["k_nutrient_epi"], name="k_nutrient_epi")
k_nutrient_sarc_var = CellVariable(mesh=mesh, value=sim_params["k_nutrient_sarc"], name="k_nutrient_sarc")

# Create effective diffusion coefficients that apply only in the pleural space.
D_oxygen_eff = sim_params["D_oxygen"] * pleural_mask_var
D_nutrient_eff = sim_params["D_nutrient"] * pleural_mask_var

# Build source terms that are masked by the pleural space.
source_oxygen = (mu_oxygen_var * pleural_mask_var) + \
                ((k_oxygen_epi_var * cellpresent_e_var) * pleural_mask_var) + \
                ((k_oxygen_sarc_var * cellpresent_s_var) * pleural_mask_var)

source_nutrient = (mu_nutrient_var * pleural_mask_var) + \
                  ((k_nutrient_epi_var * cellpresent_e_var) * pleural_mask_var) + \
                  ((k_nutrient_sarc_var * cellpresent_s_var) * pleural_mask_var)

# Build the oxygen and nutrient equations.
eq_oxygen = (TransientTerm() == DiffusionTerm(coeff=D_oxygen_eff) - ImplicitSourceTerm(source_oxygen))
eq_nutrient = (TransientTerm() == DiffusionTerm(coeff=D_nutrient_eff) - ImplicitSourceTerm(source_nutrient))
eq_VEGF = TransientTerm() == DiffusionTerm(coeff=sim_params["D_VEGF"]) - ImplicitSourceTerm(sim_params["mu_VEGF"])

# Cytokine equations.
eq_IL8 = TransientTerm() == DiffusionTerm(coeff=sim_params["D_IL8"]) - ImplicitSourceTerm(sim_params["mu_IL8"])
eq_IL6 = TransientTerm() == DiffusionTerm(coeff=sim_params["D_IL6"]) - ImplicitSourceTerm(sim_params["mu_IL6"])

comm.Barrier()
local_comp_start = time.time()
# Solve the equations for one time step.
eq_oxygen.solve(var=oxygen_var, dt=dt, solver=mysolver)
eq_nutrient.solve(var=nutrient_var, dt=dt, solver=mysolver)
eq_IL8.solve(var=IL8_var, dt=dt, solver=mysolver)
eq_IL6.solve(var=IL6_var, dt=dt, solver=mysolver)
eq_VEGF.solve(var=VEGF_var, dt=dt, solver=mysolver)

local_comp_end = time.time()
local_comp_time = local_comp_end - local_comp_start
sim_time = local_comp_time
comm.Barrier()

# Write the updated PDE fields back to disk.
dump.write(oxygen_var, os.path.join(ramdisk_path, "oxygen.dat"))
dump.write(nutrient_var, os.path.join(ramdisk_path, "nutrient.dat"))
dump.write(IL8_var, os.path.join(ramdisk_path, "cytokine_il8.dat"))
dump.write(IL6_var, os.path.join(ramdisk_path, "cytokine_il6.dat"))
dump.write(VEGF_var, os.path.join(ramdisk_path, "vegf.dat"))

t_end_total = time.time()
total_time = t_end_total - t_start_total

max_sim_time = comm.reduce(sim_time, op=MPI.MAX, root=0)
max_total_time = comm.reduce(total_time, op=MPI.MAX, root=0)

all_comp_times = comm.gather(local_comp_time, root=0)

if rank == 0:
    for i, ctime in enumerate(all_comp_times):
        print(f"LOCAL_COMP_TIME[{i}]: {ctime}", flush=True)
    print(f"SIM_TIME: {max_sim_time:.6f}", flush=True)
    print(f"TOTAL_TIME: {max_total_time:.6f}", flush=True)
    current_time = get_current_time(ramdisk_path)
    try:
        # Get current time
        current_time = get_current_time(ramdisk_path)
        print(f"Current simulation time: {current_time} minutes")

    except Exception as e:
        print(f"Error in visualization script: {e}")
        import traceback

        traceback.print_exc()

print("HPC FiPy script completed successfully.")

