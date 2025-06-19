#!/usr/bin/env python3
import os
import sys
import h5py
import numpy as np
from fipy import (CellVariable, Grid3D, dump, TransientTerm, DiffusionTerm,
                   LinearGMRESSolver, ImplicitSourceTerm)
from mpi4py import MPI
from CANCER_3DSteppables import SystemMonitor
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from fipy import dump
from fipy.tools import serialComm



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


# --- MPI setup and 3D Cartesian communicator ---
def exchange_ghost_cells_3d(field_with_ghost, cart_comm, local_shape):
    """
    Exchange ghost cells in all three directions (x, y, z) using direct neighbor communication.
    Handles non-uniform domain decomposition by matching buffer sizes to neighbor subdomain shapes.
    """
    ghost = 1
    local_nx, local_ny, local_nz = local_shape
    rank = cart_comm.Get_rank()
    dims = cart_comm.Get_topo()[1]
    coords = cart_comm.Get_coords(rank)

    # --- X-direction exchange ---
    left, right = cart_comm.Shift(0, 1)
    if left != MPI.PROC_NULL and 0 <= coords[0] - 1 < dims[0]:
        left_coords = [coords[0]-1, coords[1], coords[2]]
        left_nx, _ = decompose(nx, dims[0], left_coords[0])
        send_buf = np.ascontiguousarray(field_with_ghost[ghost, :, :])
        recv_buf = np.empty((left_nx, local_ny, local_nz), dtype=np.float64)
        cart_comm.Sendrecv(sendbuf=send_buf, dest=left, sendtag=0,
                           recvbuf=recv_buf, source=left, recvtag=1)
        field_with_ghost[0, :left_nx, :] = recv_buf

    if right != MPI.PROC_NULL and 0 <= coords[0] + 1 < dims[0]:
        right_coords = [coords[0]+1, coords[1], coords[2]]
        right_nx, _ = decompose(nx, dims[0], right_coords[0])
        send_buf = np.ascontiguousarray(field_with_ghost[-ghost-1, :, :])
        recv_buf = np.empty((right_nx, local_ny, local_nz), dtype=np.float64)
        cart_comm.Sendrecv(sendbuf=send_buf, dest=right, sendtag=1,
                           recvbuf=recv_buf, source=right, recvtag=0)
        field_with_ghost[-1, :right_nx, :] = recv_buf

    # --- Y-direction exchange ---
    up, down = cart_comm.Shift(1, 1)
    if up != MPI.PROC_NULL and 0 <= coords[0] + 1 < dims[0]:
        up_coords = [coords[0], coords[1]-1, coords[2]]
        up_ny, _ = decompose(ny, dims[1], up_coords[1])
        send_buf = np.ascontiguousarray(field_with_ghost[:, ghost, :])
        recv_buf = np.empty((local_nx, up_ny, local_nz), dtype=np.float64)
        cart_comm.Sendrecv(sendbuf=send_buf, dest=up, sendtag=2,
                           recvbuf=recv_buf, source=up, recvtag=3)
        field_with_ghost[:, 0, :up_ny] = recv_buf

    if down != MPI.PROC_NULL and 0 <= coords[1] + 1 < dims[1]:
        down_coords = [coords[0], coords[1]+1, coords[2]]
        down_ny, _ = decompose(ny, dims[1], down_coords[1])
        send_buf = np.ascontiguousarray(field_with_ghost[:, -ghost-1, :])
        recv_buf = np.empty((local_nx, down_ny, local_nz), dtype=np.float64)
        cart_comm.Sendrecv(sendbuf=send_buf, dest=down, sendtag=3,
                           recvbuf=recv_buf, source=down, recvtag=2)
        field_with_ghost[:, -1, :down_ny] = recv_buf

    # --- Z-direction exchange ---
    front, back = cart_comm.Shift(2, 1)
    if front != MPI.PROC_NULL and 0 <= coords[2] - 1 < dims[2]:
        front_coords = [coords[0], coords[1], coords[2]-1]
        front_nz, _ = decompose(nz, dims[2], front_coords[2])
        send_buf = np.ascontiguousarray(field_with_ghost[:, :, ghost])
        recv_buf = np.empty((local_nx, local_ny, front_nz), dtype=np.float64)
        cart_comm.Sendrecv(sendbuf=send_buf, dest=front, sendtag=4,
                           recvbuf=recv_buf, source=front, recvtag=5)
        field_with_ghost[:, :, 0][:, :, :front_nz] = recv_buf

    if back != MPI.PROC_NULL and 0 <= coords[2] + 1 < dims[2]:
        back_coords = [coords[0], coords[1], coords[2]+1]
        back_nz, _ = decompose(nz, dims[2], back_coords[2])
        send_buf = np.ascontiguousarray(field_with_ghost[:, :, -ghost-1])
        recv_buf = np.empty((local_nx, local_ny, back_nz), dtype=np.float64)
        cart_comm.Sendrecv(sendbuf=send_buf, dest=back, sendtag=5,
                           recvbuf=recv_buf, source=back, recvtag=4)
        field_with_ghost[:, :, -1][:, :, :back_nz] = recv_buf

    return field_with_ghost




def collect_and_save_hpc_metrics(comm, rank, size, local_comp_time, total_time,
                                 mesh_size=None, output_dir="./",
                                 csv_filename="hpc_metrics.csv", is_baseline=False):
    """
    Collect HPC performance metrics from all ranks and save to CSV file.
    """
    import csv
    import os

    # Gather all local computation times to rank 0
    all_comp_times = comm.gather(local_comp_time, root=0)
    all_total_times = comm.gather(total_time, root=0)

    # Get maximum times across all ranks
    max_comp_time = comm.reduce(local_comp_time, op=MPI.MAX, root=0)
    max_total_time = comm.reduce(total_time, op=MPI.MAX, root=0)

    # Calculate metrics only on rank 0
    if rank == 0:
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, csv_filename)
            baseline_path = os.path.join(output_dir, "baseline_times.txt")

            # Handle baseline case (n=1)
            if is_baseline or size == 1:
                baseline_time = max_total_time
                with open(baseline_path, 'w') as f:
                    f.write(f"{baseline_time}\n")
                    f.write(f"{mesh_size}\n")
                print(f"Baseline time T_{{i,s}} = {baseline_time:.6f} saved for mesh size {mesh_size}")

                metrics = {
                    'mesh_size': mesh_size if mesh_size is not None else 'unknown',
                    'processors': size,
                    'max_total_time': max_total_time,
                    'max_comp_time': max_comp_time,
                    'avg_comp_time': sum(all_comp_times) / len(all_comp_times),
                    'sum_comp_times': sum(all_comp_times),
                    'T_comm': 0.0,
                    'load_imbalance': 0.0,
                    'f_t_i_max': 0.0,
                    'f_t_i_std': 0.0,
                    'speedup': 1.0,
                    'efficiency': 1.0,
                    'baseline_time': baseline_time
                }
            else:
                # Load baseline time from previous run
                if not os.path.exists(baseline_path):
                    print(f"Warning: Baseline file not found at {baseline_path}")
                    return None

                with open(baseline_path, 'r') as f:
                    baseline_time = float(f.readline().strip())
                    saved_mesh_size = f.readline().strip()

                # Calculate HPC metrics
                T_comm_approx = max_total_time - max(all_comp_times)
                sum_Tm_i = sum(all_comp_times)
                avg_local_time = sum_Tm_i / len(all_comp_times)

                f_t_i_list = []
                ideal_time_per_rank = baseline_time / size
                for t_m in all_comp_times:
                    f_t = (t_m - ideal_time_per_rank) / avg_local_time
                    f_t_i_list.append(f_t)

                load_imbalance = max(all_comp_times) / avg_local_time
                speedup = baseline_time / max_total_time
                efficiency = speedup / size

                metrics = {
                    'mesh_size': mesh_size if mesh_size is not None else 'unknown',
                    'processors': size,
                    'max_total_time': max_total_time,
                    'max_comp_time': max_comp_time,
                    'avg_comp_time': avg_local_time,
                    'sum_comp_times': sum_Tm_i,
                    'T_comm': T_comm_approx,
                    'load_imbalance': load_imbalance,
                    'f_t_i_max': max(f_t_i_list),
                    'f_t_i_std': np.std(f_t_i_list),
                    'speedup': speedup,
                    'efficiency': efficiency,
                    'baseline_time': baseline_time
                }

            # Write to CSV file
            file_exists = os.path.exists(csv_path)
            with open(csv_path, 'a', newline='') as csvfile:
                fieldnames = [
                    'mesh_size', 'processors', 'max_total_time', 'max_comp_time',
                    'avg_comp_time', 'sum_comp_times', 'T_comm', 'load_imbalance',
                    'f_t_i_max', 'f_t_i_std', 'speedup', 'efficiency', 'baseline_time'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                writer.writerow(metrics)

            print(f"HPC metrics saved to {csv_path}")
            print(f"Processors: {size}, Speedup: {metrics['speedup']:.3f}, Efficiency: {metrics['efficiency']:.3f}")
            if not is_baseline and size > 1:
                print(f"Load Imbalance: {metrics['load_imbalance']:.3f}, T_comm: {metrics['T_comm']:.6f}")

            return metrics

        except Exception as e:
            print(f"Error saving HPC metrics: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        return None


def decompose(n, nparts, coord):
    base = n // nparts
    rem = n % nparts
    if coord < rem:
        local = base + 1
        start = coord * (base + 1)
    else:
        local = base
        start = rem * (base + 1) + (coord - rem) * base
    return local, start
# --- MPI setup and 3D Cartesian communicator ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Compute 3D process grid
dims = MPI.Compute_dims(size, 3)
cart_comm = comm.Create_cart(dims, periods=[False, False, False], reorder=True)
coords = cart_comm.Get_coords(rank)

# --- Read parameters and mesh info (on all ranks) ---
ramdisk_path = "/home/anton/miniconda3/envs/cc3d/lib/python3.10/site-packages/cancer-modeling/ramdisk"
params_filepath = os.path.join(ramdisk_path, "simulation_params.h5")

with h5py.File(params_filepath, 'r') as h5f:
    sim_params = {key: h5f[key][()] for key in h5f.keys()}

mesh_filepath = os.path.join(ramdisk_path, "mesh.dat")
mesh = dump.read(mesh_filepath)

x = mesh.cellCenters[0].value
y = mesh.cellCenters[1].value
z = mesh.cellCenters[2].value

x_unique = np.unique(x)
y_unique = np.unique(y)
z_unique = np.unique(z)

nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)

# --- Decompose the global mesh in 3D ---
local_nx, start_x = decompose(nx, dims[0], coords[0])
local_ny, start_y = decompose(ny, dims[1], coords[1])
local_nz, start_z = decompose(nz, dims[2], coords[2])

dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
origin = (x_unique[start_x], y_unique[start_y], z_unique[start_z])

local_mesh = Grid3D(nx=local_nx, ny=local_ny, nz=local_nz, dx=dx, dy=dy, dz=dz, communicator=serialComm)


def get_local_block(arr, nx, ny, nz, start_x, local_nx, start_y, local_ny, start_z, local_nz):
    arr3d = arr.reshape((nx, ny, nz))
    return arr3d[start_x:start_x + local_nx, start_y:start_y + local_ny, start_z:start_z + local_nz].flatten()


# --- Load local fields ---
def load_field(field_name):
    arr = dump.read(os.path.join(ramdisk_path, f"{field_name}.dat"))
    arr_val = arr.value if hasattr(arr, 'value') else arr
    return get_local_block(arr_val, nx, ny, nz, start_x, local_nx, start_y, local_ny, start_z, local_nz)


oxygen_local = load_field("oxygen")
nutrient_local = load_field("nutrient")
IL8_local = load_field("cytokine_il8")
IL6_local = load_field("cytokine_il6")
VEGF_local = load_field("vegf")
cellpresent_e_local = load_field("cellpresent_e")
cellpresent_s_local = load_field("cellpresent_s")
pleural_mask_local = load_field("pleural_mask")

# --- Create fields with ghost layers ---
ghost = 1

# Allocate arrays with ghost layers
oxygen_with_ghost = np.zeros((local_nx + 2 * ghost, local_ny + 2 * ghost, local_nz + 2 * ghost))
nutrient_with_ghost = np.zeros((local_nx + 2 * ghost, local_ny + 2 * ghost, local_nz + 2 * ghost))
IL8_with_ghost = np.zeros((local_nx + 2 * ghost, local_ny + 2 * ghost, local_nz + 2 * ghost))
IL6_with_ghost = np.zeros((local_nx + 2 * ghost, local_ny + 2 * ghost, local_nz + 2 * ghost))
VEGF_with_ghost = np.zeros((local_nx + 2 * ghost, local_ny + 2 * ghost, local_nz + 2 * ghost))

# Copy local data into interior of ghosted arrays
oxygen_with_ghost[ghost:-ghost, ghost:-ghost, ghost:-ghost] = oxygen_local.reshape(local_nx, local_ny, local_nz)
nutrient_with_ghost[ghost:-ghost, ghost:-ghost, ghost:-ghost] = nutrient_local.reshape(local_nx, local_ny, local_nz)
IL8_with_ghost[ghost:-ghost, ghost:-ghost, ghost:-ghost] = IL8_local.reshape(local_nx, local_ny, local_nz)
IL6_with_ghost[ghost:-ghost, ghost:-ghost, ghost:-ghost] = IL6_local.reshape(local_nx, local_ny, local_nz)
VEGF_with_ghost[ghost:-ghost, ghost:-ghost, ghost:-ghost] = VEGF_local.reshape(local_nx, local_ny, local_nz)

# --- Wrap as CellVariables ---
oxygen_var = CellVariable(mesh=local_mesh, value=oxygen_local, name="Oxygen")
nutrient_var = CellVariable(mesh=local_mesh, value=nutrient_local, name="Nutrient")
IL8_var = CellVariable(mesh=local_mesh, value=IL8_local, name="IL8")
IL6_var = CellVariable(mesh=local_mesh, value=IL6_local, name="IL6")
VEGF_var = CellVariable(mesh=local_mesh, value=VEGF_local, name="VEGF")
cellpresent_e_var = CellVariable(mesh=local_mesh, value=cellpresent_e_local, name="CellPresentE")
cellpresent_s_var = CellVariable(mesh=local_mesh, value=cellpresent_s_local, name="CellPresentS")
pleural_mask_var = CellVariable(mesh=local_mesh, value=pleural_mask_local, name="PleuralMask")

# --- Set up parameter fields ---
mu_oxygen_var = CellVariable(mesh=local_mesh, value=sim_params["mu_oxygen"], name="mu_oxygen")
k_oxygen_epi_var = CellVariable(mesh=local_mesh, value=sim_params["k_oxygen_epi"], name="k_oxygen_epi")
k_oxygen_sarc_var = CellVariable(mesh=local_mesh, value=sim_params["k_oxygen_sarc"], name="k_oxygen_sarc")
mu_nutrient_var = CellVariable(mesh=local_mesh, value=sim_params["mu_nutrient"], name="mu_nutrient")
k_nutrient_epi_var = CellVariable(mesh=local_mesh, value=sim_params["k_nutrient_epi"], name="k_nutrient_epi")
k_nutrient_sarc_var = CellVariable(mesh=local_mesh, value=sim_params["k_nutrient_sarc"], name="k_nutrient_sarc")

D_oxygen_eff = sim_params["D_oxygen"] * pleural_mask_var
D_nutrient_eff = sim_params["D_nutrient"] * pleural_mask_var

source_oxygen = (mu_oxygen_var * pleural_mask_var) + \
                ((k_oxygen_epi_var * cellpresent_e_var) * pleural_mask_var) + \
                ((k_oxygen_sarc_var * cellpresent_s_var) * pleural_mask_var)

source_nutrient = (mu_nutrient_var * pleural_mask_var) + \
                  ((k_nutrient_epi_var * cellpresent_e_var) * pleural_mask_var) + \
                  ((k_nutrient_sarc_var * cellpresent_s_var) * pleural_mask_var)

# --- Build equations ---
eq_oxygen = (TransientTerm() == DiffusionTerm(coeff=D_oxygen_eff) - ImplicitSourceTerm(source_oxygen))
eq_nutrient = (TransientTerm() == DiffusionTerm(coeff=D_nutrient_eff) - ImplicitSourceTerm(source_nutrient))
eq_VEGF = TransientTerm() == DiffusionTerm(coeff=sim_params["D_VEGF"]) - ImplicitSourceTerm(sim_params["mu_VEGF"])
eq_IL8 = TransientTerm() == DiffusionTerm(coeff=sim_params["D_IL8"]) - ImplicitSourceTerm(sim_params["mu_IL8"])
eq_IL6 = TransientTerm() == DiffusionTerm(coeff=sim_params["D_IL6"]) - ImplicitSourceTerm(sim_params["mu_IL6"])

mysolver = LinearGMRESSolver()
dt = sim_params["dt"]

comm.Barrier()
local_comp_start = time.time()

# --- Main simulation loop with ghost cell exchange ---
num_steps = 1  # You can adjust this or read from parameters

for step in range(num_steps):
    # Exchange ghost cells for all fields before solving
    oxygen_with_ghost = exchange_ghost_cells_3d(oxygen_with_ghost, cart_comm, (local_nx, local_ny, local_nz))
    nutrient_with_ghost = exchange_ghost_cells_3d(nutrient_with_ghost, cart_comm, (local_nx, local_ny, local_nz))
    IL8_with_ghost = exchange_ghost_cells_3d(IL8_with_ghost, cart_comm, (local_nx, local_ny, local_nz))
    IL6_with_ghost = exchange_ghost_cells_3d(IL6_with_ghost, cart_comm, (local_nx, local_ny, local_nz))
    VEGF_with_ghost = exchange_ghost_cells_3d(VEGF_with_ghost, cart_comm, (local_nx, local_ny, local_nz))

    # Update FiPy variables with interior data (excluding ghost cells)
    oxygen_var.setValue(oxygen_with_ghost[ghost:-ghost, ghost:-ghost, ghost:-ghost].flatten())
    nutrient_var.setValue(nutrient_with_ghost[ghost:-ghost, ghost:-ghost, ghost:-ghost].flatten())
    IL8_var.setValue(IL8_with_ghost[ghost:-ghost, ghost:-ghost, ghost:-ghost].flatten())
    IL6_var.setValue(IL6_with_ghost[ghost:-ghost, ghost:-ghost, ghost:-ghost].flatten())
    VEGF_var.setValue(VEGF_with_ghost[ghost:-ghost, ghost:-ghost, ghost:-ghost].flatten())

    # Solve equations for one time step
    eq_oxygen.solve(var=oxygen_var, dt=dt, solver=mysolver)
    eq_nutrient.solve(var=nutrient_var, dt=dt, solver=mysolver)
    eq_IL8.solve(var=IL8_var, dt=dt, solver=mysolver)
    eq_IL6.solve(var=IL6_var, dt=dt, solver=mysolver)
    eq_VEGF.solve(var=VEGF_var, dt=dt, solver=mysolver)

    # Update ghost arrays with new solution
    oxygen_with_ghost[ghost:-ghost, ghost:-ghost, ghost:-ghost] = oxygen_var.value.reshape(local_nx, local_ny, local_nz)
    nutrient_with_ghost[ghost:-ghost, ghost:-ghost, ghost:-ghost] = nutrient_var.value.reshape(local_nx, local_ny,
                                                                                               local_nz)
    IL8_with_ghost[ghost:-ghost, ghost:-ghost, ghost:-ghost] = IL8_var.value.reshape(local_nx, local_ny, local_nz)
    IL6_with_ghost[ghost:-ghost, ghost:-ghost, ghost:-ghost] = IL6_var.value.reshape(local_nx, local_ny, local_nz)
    VEGF_with_ghost[ghost:-ghost, ghost:-ghost, ghost:-ghost] = VEGF_var.value.reshape(local_nx, local_ny, local_nz)

local_comp_end = time.time()
local_comp_time = local_comp_end - local_comp_start
sim_time = local_comp_time

comm.Barrier()


# --- Gather and save results (simplified version) ---
def simple_gather_and_save(var, var_name):
    """Simplified version for saving results"""
    if rank == 0:
        try:
            # Just save the local data from rank 0 for now
            filename = f"{var_name}_rank0_simple.npy"
            np.save(filename, var.value.reshape(local_nx, local_ny, local_nz))
            print(f"Saved {var_name} from rank 0 to {filename}")
        except Exception as e:
            print(f"Error saving {var_name}: {e}")


def robust_gather_and_write(var, var_name):
    """
    Robust version that handles dimensional inconsistencies and empty ranks
    """
    try:
        # Get actual coordinates from the local mesh
        local_x = var.mesh.cellCenters[0].value
        local_y = var.mesh.cellCenters[1].value
        local_z = var.mesh.cellCenters[2].value

        # Check for empty mesh on this rank
        if len(local_x) == 0 or len(local_y) == 0 or len(local_z) == 0:
            print(f"Rank {rank}: Empty mesh detected for {var_name}, using placeholder data")
            # Send placeholder data to indicate empty rank
            local_data_info = {
                'has_data': False,
                'shape': (0, 0, 0),
                'start': (0, 0, 0),
                'data': np.array([])
            }
        else:
            # Get actual data
            local_data = var.value

            # Determine actual bounds
            actual_x_min, actual_x_max = local_x.min(), local_x.max()
            actual_y_min, actual_y_max = local_y.min(), local_y.max()
            actual_z_min, actual_z_max = local_z.min(), local_z.max()

            # Gather all coordinate information to rank 0
            all_coords = comm.gather({
                'x_min': actual_x_min, 'x_max': actual_x_max,
                'y_min': actual_y_min, 'y_max': actual_y_max,
                'z_min': actual_z_min, 'z_max': actual_z_max,
                'x_coords': local_x,
                'y_coords': local_y,
                'z_coords': local_z,
                'data_shape': local_data.shape
            }, root=0)

            # Create global coordinate system on rank 0
            if rank == 0:
                # Collect all unique coordinates
                all_x = np.concatenate([coords['x_coords'] for coords in all_coords if len(coords['x_coords']) > 0])
                all_y = np.concatenate([coords['y_coords'] for coords in all_coords if len(coords['y_coords']) > 0])
                all_z = np.concatenate([coords['z_coords'] for coords in all_coords if len(coords['z_coords']) > 0])

                x_unique = np.unique(all_x)
                y_unique = np.unique(all_y)
                z_unique = np.unique(all_z)

                global_shape = (len(x_unique), len(y_unique), len(z_unique))
                # print(f"Global coordinate system shape: {global_shape}")
            else:
                x_unique = None
                y_unique = None
                z_unique = None
                global_shape = None

            # Broadcast global coordinates to all ranks
            x_unique = comm.bcast(x_unique, root=0)
            y_unique = comm.bcast(y_unique, root=0)
            z_unique = comm.bcast(z_unique, root=0)
            global_shape = comm.bcast(global_shape, root=0)

            # Find indices in global coordinate system with tolerance
            tolerance = 1e-10

            # Use broadcasting for efficient coordinate matching
            x_mask = np.abs(x_unique[:, np.newaxis] - local_x).min(axis=1) < tolerance
            y_mask = np.abs(y_unique[:, np.newaxis] - local_y).min(axis=1) < tolerance
            z_mask = np.abs(z_unique[:, np.newaxis] - local_z).min(axis=1) < tolerance

            x_indices = np.where(x_mask)[0]
            y_indices = np.where(y_mask)[0]
            z_indices = np.where(z_mask)[0]

            # Validate indices exist
            if len(x_indices) == 0 or len(y_indices) == 0 or len(z_indices) == 0:
                # print(f"Rank {rank}: No coordinate overlap found for {var_name}, using placeholder")
                local_data_info = {
                    'has_data': False,
                    'shape': (0, 0, 0),
                    'start': (0, 0, 0),
                    'data': np.array([])
                }
            else:
                # Calculate actual dimensions and starting position
                actual_start = (x_indices[0], y_indices[0], z_indices[0])
                actual_shape = (len(x_indices), len(y_indices), len(z_indices))

                # Validate shape consistency
                local_data_flat = local_data.flatten()
                expected_size = np.prod(actual_shape)

                if len(local_data_flat) != expected_size:
                    print(
                        f"Rank {rank}: Shape mismatch for {var_name} - data size {len(local_data_flat)} vs expected {expected_size}")

                    # Try to use actual data shape if it makes sense
                    if len(local_data.shape) == 3:
                        actual_shape = local_data.shape
                        print(f"Rank {rank}: Using actual data shape {actual_shape} for {var_name}")

                        # Recalculate indices based on actual shape
                        if actual_shape[0] <= len(x_indices) and actual_shape[1] <= len(y_indices) and actual_shape[
                            2] <= len(z_indices):
                            x_indices = x_indices[:actual_shape[0]]
                            y_indices = y_indices[:actual_shape[1]]
                            z_indices = z_indices[:actual_shape[2]]
                            actual_start = (x_indices[0], y_indices[0], z_indices[0])
                        else:
                            print(f"Rank {rank}: Cannot reconcile shapes for {var_name}, using placeholder")
                            local_data_info = {
                                'has_data': False,
                                'shape': (0, 0, 0),
                                'start': (0, 0, 0),
                                'data': np.array([])
                            }

                if 'local_data_info' not in locals():
                    # Successfully validated data
                    try:
                        local_shaped = local_data_flat.reshape(actual_shape)
                        local_data_info = {
                            'has_data': True,
                            'shape': actual_shape,
                            'start': actual_start,
                            'data': local_shaped
                        }
                        # print(f"Rank {rank}: Successfully prepared {var_name} data with shape {actual_shape}")
                    except ValueError as e:
                        print(f"Rank {rank}: Reshape failed for {var_name}: {e}")
                        local_data_info = {
                            'has_data': False,
                            'shape': (0, 0, 0),
                            'start': (0, 0, 0),
                            'data': np.array([])
                        }

        # Gather all data information to rank 0
        all_data_info = comm.gather(local_data_info, root=0)

        # Reconstruct global array on rank 0
        if rank == 0:
            try:
                global_array = np.zeros(global_shape)
                successful_ranks = 0

                for i, data_info in enumerate(all_data_info):
                    if data_info['has_data']:
                        start = data_info['start']
                        shape = data_info['shape']
                        data = data_info['data']

                        # Place data in global array
                        end = (start[0] + shape[0], start[1] + shape[1], start[2] + shape[2])

                        # Validate bounds
                        if (end[0] <= global_shape[0] and end[1] <= global_shape[1] and end[2] <= global_shape[2]):
                            global_array[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = data
                            successful_ranks += 1
                        else:
                            print(f"Warning: Rank {i} data bounds {start} to {end} exceed global shape {global_shape}")
                    else:
                        print(f"Rank {i}: No valid data for {var_name}")

                # print(f"Successfully gathered {var_name} from {successful_ranks}/{size} ranks")

                # Write to file
                filename = f"{var_name}_global_rank0.npy"
                np.save(filename, global_array)
                # print(f"Saved {var_name} to {filename} with shape {global_array.shape}")

                # Also save coordinate information
                coord_filename = f"{var_name}_coordinates.npz"
                np.savez(coord_filename,
                         x=x_unique, y=y_unique, z=z_unique,
                         shape=global_shape)
                # print(f"Saved coordinates to {coord_filename}")

            except Exception as e:
                print(f"Error during global array reconstruction for {var_name}: {e}")
                print("Falling back to individual rank data saving...")

                # Fallback: save individual rank data
                for i, data_info in enumerate(all_data_info):
                    if data_info['has_data']:
                        fallback_filename = f"{var_name}_rank{i}_fallback.npy"
                        np.save(fallback_filename, data_info['data'])
                        print(f"Saved rank {i} data to {fallback_filename}")

    except Exception as e:
        print(f"Rank {rank}: Critical error in robust_gather_and_write for {var_name}: {e}")
        import traceback
        traceback.print_exc()

        # Emergency fallback: save local data only
        try:
            emergency_filename = f"{var_name}_rank{rank}_emergency.npy"
            np.save(emergency_filename, var.value)
            print(f"Rank {rank}: Emergency save to {emergency_filename}")
        except:
            print(f"Rank {rank}: Even emergency save failed for {var_name}")


robust_gather_and_write(oxygen_var, "oxygen")
robust_gather_and_write(nutrient_var, "nutrient")
robust_gather_and_write(IL8_var, "cytokine_il8")
robust_gather_and_write(IL6_var, "cytokine_il6")
robust_gather_and_write(VEGF_var, "vegf")

# --- Timing and diagnostics ---
t_end_total = time.time()
total_time = t_end_total - local_comp_start

max_sim_time = comm.reduce(sim_time, op=MPI.MAX, root=0)
max_total_time = comm.reduce(total_time, op=MPI.MAX, root=0)
all_comp_times = comm.gather(local_comp_time, root=0)

if rank == 0:
    for i, ctime in enumerate(all_comp_times):
        print(f"LOCAL_COMP_TIME[{i}]: {ctime}", flush=True)
    print(f"SIM_TIME: {max_sim_time:.6f}", flush=True)
    print(f"TOTAL_TIME: {max_total_time:.6f}", flush=True)

    # # Collect and save HPC metrics
    # is_baseline_run = (size == 1)
    # mesh_size = nx * ny * nz
    #
    # metrics = collect_and_save_hpc_metrics(
    #     comm=comm,
    #     rank=rank,
    #     size=size,
    #     local_comp_time=local_comp_time,
    #     total_time=total_time,
    #     mesh_size=mesh_size,
    #     output_dir=ramdisk_path,
    #     csv_filename="hpc_performance_metrics.csv",
    #     is_baseline=is_baseline_run
    # )

# print(f"Rank {rank} completed with ghost cell communication.")