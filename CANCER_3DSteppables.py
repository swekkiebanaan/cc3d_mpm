import csv
import gc
import os
import random
import subprocess
import threading
import time
import shutil
import traceback
from collections import deque
from datetime import datetime
import nibabel as nib
import SimpleITK as sitk
import h5py
import matplotlib.pyplot as plt
import numpy as np
import petsc4py
import psutil
from cc3d.core.PySteppables import *
from cc3d.core.PySteppables import SteppableBasePy
from fipy import (CellVariable, Grid3D, dump)
from mpi4py import MPI
from scipy import ndimage
from scipy.ndimage import binary_fill_holes
from skimage.morphology import binary_dilation
from skimage.transform import resize
import cancer3d_variables as cv

petsc4py.init()
random.seed(42)
parallelComm = MPI.COMM_WORLD


class SystemMonitor:
    def __init__(self, max_points=1000):
        self.max_points = max_points
        self.timestamps = deque(maxlen=max_points)
        self.memory_usage = deque(maxlen=max_points)
        self.cpu_percent = deque(maxlen=max_points)
        self.cpu_per_core = deque(maxlen=max_points)
        self.load_avg = deque(maxlen=max_points)
        self.monitoring = False
        self.output_dir = "monitoring_output"
        os.makedirs(self.output_dir, exist_ok=True)

    def start_monitoring(self, interval=1.0):
        """Start monitoring system resources."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources, args=(interval,))
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop monitoring and save final plots."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        self.save_plots()

    def _get_total_process_tree_usage(self, pid=None):
        """
        Returns (total_memory_gb, total_cpu_percent) for the process with given pid
        and all its children (e.g., MPI processes spawned via mpiexec).
        """
        if pid is None:
            pid = os.getpid()
        parent_proc = psutil.Process(pid)
        # Get parent and all its children recursively
        all_procs = [parent_proc] + parent_proc.children(recursive=True)

        # Initialize CPU percent measurements for all processes.
        for p in all_procs:
            try:
                p.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Wait a short interval to let CPU usage accumulate.
        time.sleep(0.1)

        total_cpu = 0.0
        total_rss = 0
        for p in all_procs:
            try:
                total_cpu += p.cpu_percent(interval=0.0)
                total_rss += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        total_memory_gb = total_rss / (1024 ** 3)
        return total_memory_gb, total_cpu

    def _monitor_resources(self, interval):
        """Monitor system resources at specified interval."""
        while self.monitoring:
            current_time = datetime.now()

            # Get total memory and CPU usage for the current process and its children
            memory, total_cpu = self._get_total_process_tree_usage()

            # For per-core usage (system-wide)
            cpu_per_core = psutil.cpu_percent(interval=0.0, percpu=True)

            # System load averages (1, 5, 15 minutes)
            load = psutil.getloadavg()

            # Store the metrics
            self.timestamps.append(current_time)
            self.memory_usage.append(memory)
            self.cpu_percent.append(total_cpu)
            self.cpu_per_core.append(cpu_per_core)
            self.load_avg.append(load)

            # Generate plots periodically
            if len(self.timestamps) % 10 == 0:
                self.save_plots()

            time.sleep(interval)

    def save_plots(self):
        """Generate and save monitoring plots."""
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

        # Convert timestamps to relative seconds
        start_time = min(self.timestamps)
        rel_times = [(t - start_time).total_seconds() for t in self.timestamps]

        # Memory usage plot
        ax1.plot(rel_times, self.memory_usage, 'b-', label='Total Memory Usage')
        ax1.set_ylabel('Memory Usage (GB)')
        ax1.set_title('Total Process Memory Usage Over Time')
        ax1.grid(True)

        # CPU usage plot (process tree and per-core usage)
        ax2.plot(rel_times, self.cpu_percent, 'r-', label='Total Process CPU')
        if self.cpu_per_core:
            core_data = np.array(self.cpu_per_core)
            for i in range(core_data.shape[1]):
                ax2.plot(rel_times, core_data[:, i], '--', alpha=0.5, label=f'Core {i}')
        ax2.set_ylabel('CPU Usage (%)')
        ax2.set_title('CPU Usage Over Time')
        ax2.legend()
        ax2.grid(True)

        # Load average plot
        load_data = np.array(self.load_avg)
        ax3.plot(rel_times, load_data[:, 0], 'g-', label='1 min')
        ax3.plot(rel_times, load_data[:, 1], 'y-', label='5 min')
        ax3.plot(rel_times, load_data[:, 2], 'r-', label='15 min')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Load Average')
        ax3.set_title('System Load Average Over Time')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'system_metrics.png'))
        plt.close()

        # Save raw data for further analysis
        np.save(os.path.join(self.output_dir, 'monitoring_data.npy'), {
            'timestamps': np.array(rel_times),
            'memory': np.array(self.memory_usage),
            'cpu': np.array(self.cpu_percent),
            'cpu_per_core': np.array(self.cpu_per_core),
            'load': np.array(self.load_avg)
        })


class FiPySolverSteppable(SteppableBasePy):
    def start(self):
        """
        Called once at simulation start. Set up simulation parameters,
        load shared variables, and initialize the PDE bounding box.
        """
        self.lattice_unit_size_mm = cv.LATTICE_UNIT_SIZE_MM
        self.dx = cv.DX
        self.dt = cv.DT

        # PDE coefficients.
        self.D_oxygen = cv.D_OXYGEN
        self.D_nutrient = cv.D_NUTRIENT
        self.D_IL8 = cv.D_IL8
        self.D_IL6 = cv.D_IL6

        # Define tumor cell types.
        self.EPITHELIOIDTUMOR = cv.EPITHELIOIDTUMOR
        self.SARCOMATOIDTUMOR = cv.SARCOMATOIDTUMOR

        # Retrieve the pleural mask (global boolean NumPy array).
        self.space_between_pleuras = self.shared_steppable_vars.get('space_between_pleuras', None)
        if self.space_between_pleuras is None:
            print("Error: 'space_between_pleuras' not found in shared_steppable_vars.")
            return

            # Initialize bounding box variables.
        self.nx = None
        self.ny = None
        self.nz = None
        self.origin_x = None
        self.origin_y = None
        self.origin_z = None

        # PDE arrays.
        self.oxygen_array = None
        self.nutrient_array = None
        self.IL8_array = None
        self.IL6_array = None
        # We will also store the current FiPy objects for later dumping.
        self.mesh = None
        self.oxygen_var = None
        self.nutrient_var = None
        self.IL8_var = None
        self.IL6_var = None

        self.VEGF_array = None
        self.VEGF_var = None

        self.monitor = SystemMonitor()
        # self.monitor.start_monitoring(interval=1.0)

        # Initialize bounding box.
        self.update_bounding_box()

        self.shared_steppable_vars['fipy_solver_steppable'] = self

    def step(self, mcs):

        try:
            # Update bounding box every 50 steps.
            if mcs % 50 == 0:
                self.update_bounding_box()
            if self.nx is None:
                return

            # Gather consumption data for tumor cells.
            oxy_cons = np.zeros((self.nz, self.ny, self.nx), dtype=float)
            nut_cons = np.zeros((self.nz, self.ny, self.nx), dtype=float)
            for cell in self.cell_list:
                x = int((cell.xCOM * self.lattice_unit_size_mm - self.origin_x) / self.dx)
                y = int((cell.yCOM * self.lattice_unit_size_mm - self.origin_y) / self.dx)
                z = int((cell.zCOM * self.lattice_unit_size_mm - self.origin_z) / self.dx)
                if 0 <= x < self.nx and 0 <= y < self.ny and 0 <= z < self.nz:
                    if cell.type in (self.EPITHELIOIDTUMOR, self.SARCOMATOIDTUMOR):
                        rate = 0.04 if cell.type == self.EPITHELIOIDTUMOR else 0.06
                        oxy_cons[z, y, x] += rate
                        nut_cons[z, y, x] += rate

            # Create the FiPy 3D mesh.
            self.mesh = Grid3D(dx=self.dx, dy=self.dx, dz=self.dx,
                               nx=self.nx, ny=self.ny, nz=self.nz)

            # Compute local pleural mask for the current bounding box.
            start_x = int(self.origin_x / self.dx)
            start_y = int(self.origin_y / self.dx)
            start_z = int(self.origin_z / self.dx)
            pleural_mask_in_bbox = np.array(self.space_between_pleuras, dtype=bool)[
                                   start_z:start_z + self.nz,
                                   start_y:start_y + self.ny,
                                   start_x:start_x + self.nx
                                   ]

            # Initialize PDE arrays for the first run.
            if self.oxygen_array is None:
                self.oxygen_array = np.ones((self.nz, self.ny, self.nx), dtype=float)
                self.nutrient_array = np.ones((self.nz, self.ny, self.nx), dtype=float)
                self.IL8_array = np.zeros((self.nz, self.ny, self.nx), dtype=float)
                self.IL6_array = np.zeros((self.nz, self.ny, self.nx), dtype=float)
                self.VEGF_array = np.zeros((self.nz, self.ny, self.nx), dtype=float)

            # Create FiPy CellVariables.
            self.oxygen_var = CellVariable(mesh=self.mesh, value=self.oxygen_array.flatten(), name="Oxygen")
            self.nutrient_var = CellVariable(mesh=self.mesh, value=self.nutrient_array.flatten(), name="Nutrient")
            self.IL8_var = CellVariable(mesh=self.mesh, value=self.IL8_array.flatten(), name="IL8")
            self.IL6_var = CellVariable(mesh=self.mesh, value=self.IL6_array.flatten(), name="IL6")
            self.VEGF_var = CellVariable(mesh=self.mesh, value=self.VEGF_array.flatten(), name="VEGF")

            # Prepare the pleural mask as a CellVariable.
            pleural_mask_numeric = pleural_mask_in_bbox.astype(float)
            pleural_mask_var = CellVariable(mesh=self.mesh,
                                            value=pleural_mask_numeric.flatten(),
                                            name="PleuralMask")
            # Save the pleural mask as an attribute so that it can be dumped.
            self.pleural_mask_var = pleural_mask_var

            # Dump all simulation data (mesh, PDE fields, simulation parameters, and cell presence data).
            self.dump_simulation_data()


            n = cv.CORES
            if mcs == 0:
                            cmd = (f"mpiexec -n {n} /home/anton/miniconda3/envs/cc3d/bin/python "
                                   r"/home/anton/miniconda3/envs/cc3d/lib/python3.10/site-packages/cancer-modeling/Simulation/HPCFiPy_june1.py --petsc --first")
            else:
                            cmd = (f"mpiexec -n {n} /home/anton/miniconda3/envs/cc3d/bin/python "
                                   r"/home/anton/miniconda3/envs/cc3d/lib/python3.10/site-packages/cancer-modeling/Simulation/HPCFiPy_june1.py --petsc")
                            print(f"[FiPySolverSteppable] MCS={mcs} calling: {cmd}")

            subprocess.call(cmd, shell=True)

            # Read back updated PDE fields from the ramdisk.
            ramdisk_path = "/home/anton/miniconda3/envs/cc3d/lib/python3.10/site-packages/cancer-modeling/ramdisk"
            max_attempts = 10
            attempt = 0
            read_success = False
            oxygen_filepath = os.path.join(ramdisk_path, "oxygen.dat")
            nutrient_filepath = os.path.join(ramdisk_path, "nutrient.dat")
            IL8_filepath = os.path.join(ramdisk_path, "cytokine_il8.dat")
            IL6_filepath = os.path.join(ramdisk_path, "cytokine_il6.dat")
            vegf_filepath = os.path.join(ramdisk_path, "vegf.dat")
            while attempt < max_attempts and not read_success:
                try:
                    files_ready = (
                            os.path.exists(oxygen_filepath) and os.path.getsize(oxygen_filepath) > 0 and
                            os.path.exists(nutrient_filepath) and os.path.getsize(nutrient_filepath) > 0 and
                            os.path.exists(IL8_filepath) and os.path.getsize(IL8_filepath) > 0 and
                            os.path.exists(IL6_filepath) and os.path.getsize(IL6_filepath) > 0 and
                            os.path.exists(vegf_filepath) and os.path.getsize(vegf_filepath) > 0
                    )
                    if files_ready:
                        print("All PDE files ready; reading them...", flush=True)
                        oxy_var_updated = dump.read(oxygen_filepath)
                        nut_var_updated = dump.read(nutrient_filepath)
                        IL8_var_updated = dump.read(IL8_filepath)
                        IL6_var_updated = dump.read(IL6_filepath)
                        vegf_var_updated = dump.read(vegf_filepath)
                        print('vegf_var_updated',vegf_var_updated)

                        self.oxygen_array = np.array(oxy_var_updated).reshape((self.nz, self.ny, self.nx))
                        self.nutrient_array = np.array(nut_var_updated).reshape((self.nz, self.ny, self.nx))
                        self.IL8_array = np.array(IL8_var_updated).reshape((self.nz, self.ny, self.nx))
                        self.IL6_array = np.array(IL6_var_updated).reshape((self.nz, self.ny, self.nx))
                        self.VEGF_array = np.array(vegf_var_updated).reshape((self.nz, self.ny, self.nx))
                        print('0')

                        read_success = True
                        break
                    else:
                        print("One or more PDE files missing or empty; waiting...", flush=True)
                except Exception as e:
                    print(f"Attempt {attempt}: Exception reading PDE files: {e}", flush=True)
                    traceback.print_exc()
                attempt += 1
                time.sleep(1.5 * attempt)


            # Clamp negative values.
            self.oxygen_array = np.clip(self.oxygen_array, 0.0, None)
            self.nutrient_array = np.clip(self.nutrient_array, 0.0, None)

            # Update shared variables.
            self.shared_steppable_vars["oxygen_array"] = self.oxygen_array
            self.shared_steppable_vars["nutrient_array"] = self.nutrient_array
            self.shared_steppable_vars["IL8_array"] = self.IL8_array
            self.shared_steppable_vars["IL6_array"] = self.IL6_array
            self.shared_steppable_vars["VEGF_array"] = self.VEGF_array


            self.output_folder = "/home/anton/miniconda3/envs/cc3d/lib/python3.10/site-packages/cancer-modeling/ramdisk/visualizations"
            os.makedirs(self.output_folder, exist_ok=True)
            if mcs % 50 == 0:
                try:
                    self.create_2d_contour_plots(mcs, pleural_mask_in_bbox)
                    self.visualize_mesh_and_tumour_slice(self.mesh, self.oxygen_array, mcs, self.output_folder)
                    #
                    # self.visualize_full_mesh_3d(self.oxygen_array, mcs, self.output_folder,
                    #                         title="Oxygen Concentration")
                    # self.visualize_bounded_mesh_3d(self.oxygen_array, pleural_mask_in_bbox, mcs,
                    #                             self.output_folder, title="Oxygen in Pleural Cavity")

                except Exception as e:
                    print(f"Error creating plots at MCS {mcs}: {e}")
                    traceback.print_exc()
            #     gc.collect()
            #     if mcs == 100:
            #         self.monitor.stop_monitoring()

            gc.collect()

        except Exception as e:
            print(f"Exception in FiPySolverSteppable at step {mcs}: {e}")
            traceback.print_exc()

    def _factor_cores(self, cores):
        """Determine optimal 3D block distribution for given core count."""
        from itertools import product

        best_factors = (cores, 1, 1)
        min_surface = float('inf')

        # Generate all possible 3D factor combinations
        for factors in product(*([range(1, cores + 1)] * 3)):
            if np.prod(factors) != cores:
                continue
            # Calculate surface area (communication cost heuristic)
            surface = factors[0] * factors[1] + factors[1] * factors[2] + factors[0] * factors[2]
            if surface < min_surface:
                min_surface = surface
                best_factors = sorted(factors)  # Maintain sorted order for dimension matching

        return best_factors  # Returns (bx, by, bz) for x,y,z dimensions
    def visualize_mesh_and_tumour_slice(self, mesh, concentration_field, step, output_folder):
        """
        Visualize the FiPy mesh (as seen in CC3D) and overlay a concentration contour of a tumor slice.

        Parameters:
            mesh: FiPy mesh (e.g., a Grid3D object)
            concentration_field: 3D numpy array of the concentration (shape: [nz, ny, nx])
            step: int, simulation step number (used in filename)
            output_folder: str, path to folder where image will be saved
        """
        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Get grid dimensions from the concentration field shape
        nz, ny, nx = concentration_field.shape

        # Extract cell center coordinates from the FiPy mesh.
        # In FiPy, cellCenters is typically a tuple/list with three arrays (for x, y, z)
        cell_centers = mesh.cellCenters  # shape: (3, numberOfCells)
        # Reshape cell centers to 3D arrays matching the grid dimensions
        x_centers = np.array(cell_centers[0]).reshape((nz, ny, nx))
        y_centers = np.array(cell_centers[1]).reshape((nz, ny, nx))
        z_centers = np.array(cell_centers[2]).reshape((nz, ny, nx))

        # Choose a slice along the z-direction (for example, the middle slice)
        mid_index = nz // 2

        # Create a new figure
        plt.figure(figsize=(8, 6))

        # Plot the mesh cell centers as a scatter plot (projected on the XY plane)
        plt.scatter(x_centers[mid_index, :, :], y_centers[mid_index, :, :],
                    s=1, color='gray', label='Mesh cells')

        # Overlay the concentration contour for the same XY slice
        # Here we assume that the x and y positions are given by the cell centers in the slice.
        X = x_centers[mid_index, :, :]
        Y = y_centers[mid_index, :, :]
        conc_slice = concentration_field[mid_index, :, :]

        # Draw a filled contour plot (adjust levels and colormap as needed)
        contour = plt.contourf(X, Y, conc_slice, levels=20, cmap='jet', alpha=0.6)
        plt.colorbar(contour, label='Concentration')

        plt.title(f'Mesh and Tumor Slice Contour at Step {step}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()

        # Save the figure to a file with the step number in its name
        filename = os.path.join(output_folder, f'visualization_step_{step}.png')
        plt.savefig(filename)
        plt.close()
        print(f"Visualization saved to {filename}")

    def dump_simulation_data(self):
        """
        Dump simulation parameters, mesh, PDE fields, and tumor cell presence arrays
        to disk. All file writing is done here to avoid duplicate writes.
        """
        ramdisk_path = "/home/anton/miniconda3/envs/cc3d/lib/python3.10/site-packages/cancer-modeling/ramdisk"
        os.makedirs(ramdisk_path, exist_ok=True)

        # Dump simulation parameters.
        sim_params = cv.SIM_PARAMS
        params_filepath = os.path.join(ramdisk_path, "simulation_params.h5")
        with h5py.File(params_filepath, 'w') as h5f:
            for key, value in sim_params.items():
                h5f.create_dataset(key, data=value)

        # Dump the mesh.
        dump.write(self.mesh, os.path.join(ramdisk_path, "mesh.dat"))

        # Dump the PDE fields.
        dump.write(self.IL8_var, os.path.join(ramdisk_path, "cytokine_il8.dat"))
        dump.write(self.IL6_var, os.path.join(ramdisk_path, "cytokine_il6.dat"))
        dump.write(self.nutrient_var, os.path.join(ramdisk_path, "nutrient.dat"))
        dump.write(self.oxygen_var, os.path.join(ramdisk_path, "oxygen.dat"))
        dump.write(self.VEGF_var, os.path.join(ramdisk_path, "vegf.dat"))

        # Dump the pleural mask.
        dump.write(self.pleural_mask_var, os.path.join(ramdisk_path, "pleural_mask.dat"))

        # --- Dump cell presence data for tumor cells ---
        # Create arrays (shape: [nz, ny, nx]) for epithelioid and sarcomatoid cell presence.
        cellpresent_e_array = np.zeros((self.nz, self.ny, self.nx), dtype=float)
        cellpresent_s_array = np.zeros((self.nz, self.ny, self.nx), dtype=float)

        # Compute global starting indices using the bounding box origin.
        start_x = int(self.origin_x / self.dx)
        start_y = int(self.origin_y / self.dx)
        start_z = int(self.origin_z / self.dx)

        # Loop over the local PDE grid (z, y, x order).
        for x_local in range(self.nx):
            for y_local in range(self.ny):
                for z_local in range(self.nz):
                    # Map to global coordinates.
                    global_x = start_x + x_local
                    global_y = start_y + y_local
                    global_z = start_z + z_local
                    cell = self.cell_field[global_x, global_y, global_z]
                    if cell is not None:
                        if cell.type == self.EPITHELIOIDTUMOR:
                            cellpresent_e_array[z_local, y_local, x_local] = 1.0
                        elif cell.type == self.SARCOMATOIDTUMOR:
                            cellpresent_s_array[z_local, y_local, x_local] = 1.0

        # Wrap these arrays as FiPy CellVariables.
        cellpresent_e_var = CellVariable(mesh=self.mesh,
                                         value=cellpresent_e_array.flatten(),
                                         name="CellPresentE")
        cellpresent_s_var = CellVariable(mesh=self.mesh,
                                         value=cellpresent_s_array.flatten(),
                                         name="CellPresentS")

        # Dump the cell presence variables.
        dump.write(cellpresent_e_var, os.path.join(ramdisk_path, "cellpresent_e.dat"))
        dump.write(cellpresent_s_var, os.path.join(ramdisk_path, "cellpresent_s.dat"))

        print("Simulation parameters, PDE fields, pleural mask, and cell presence data dumped to:", ramdisk_path)

    def update_bounding_box(self):
        """
        Recompute the bounding box based on tumor cell positions and update PDE arrays.
        """
        tumor_cells = [c for c in self.cell_list_by_type(self.EPITHELIOIDTUMOR, self.SARCOMATOIDTUMOR)]
        if not tumor_cells:
            return

        x_coords = [cell.xCOM for cell in tumor_cells]
        y_coords = [cell.yCOM for cell in tumor_cells]
        z_coords = [cell.zCOM for cell in tumor_cells]

        margin = 75
        min_x = max(int(min(x_coords) - margin), 0)
        max_x = min(int(max(x_coords) + margin), self.dim.x - 1)
        min_y = max(int(min(y_coords) - margin), 0)
        max_y = min(int(max(y_coords) + margin), self.dim.y - 1)
        min_z = max(int(min(z_coords) - margin), 0)
        max_z = min(int(max(z_coords) + margin), self.dim.z - 1)

        space_indices = np.argwhere(self.space_between_pleuras)
        if space_indices.size == 0:
            print("No space between pleuras detected.")
            return

        space_in_bbox = space_indices[
            (space_indices[:, 2] >= min_x) & (space_indices[:, 2] <= max_x) &
            (space_indices[:, 1] >= min_y) & (space_indices[:, 1] <= max_y) &
            (space_indices[:, 0] >= min_z) & (space_indices[:, 0] <= max_z)
            ]
        if space_in_bbox.size == 0:
            return

        min_z_sp, min_y_sp, min_x_sp = space_in_bbox.min(axis=0)
        max_z_sp, max_y_sp, max_x_sp = space_in_bbox.max(axis=0)

        new_bbox_width = (max_x_sp - min_x_sp + 1) * self.lattice_unit_size_mm
        new_bbox_height = (max_y_sp - min_y_sp + 1) * self.lattice_unit_size_mm
        new_bbox_depth = (max_z_sp - min_z_sp + 1) * self.lattice_unit_size_mm

        new_nx = max(int(new_bbox_width / self.dx), 1)
        new_ny = max(int(new_bbox_height / self.dx), 1)
        new_nz = max(int(new_bbox_depth / self.dx), 1)

        cores = cv.CORES
        bx, by, bz = self._factor_cores(cores)

        # Adjust dimensions to be multiples of block factors
        new_nx = ((new_nx + bx - 1) // bx) * bx
        new_ny = ((new_ny + by - 1) // by) * by
        new_nz = ((new_nz + bz - 1) // bz) * bz

        new_origin_x = min_x_sp * self.lattice_unit_size_mm
        new_origin_y = min_y_sp * self.lattice_unit_size_mm
        new_origin_z = min_z_sp * self.lattice_unit_size_mm

        changed = (
                new_nx != self.nx or new_ny != self.ny or new_nz != self.nz or
                new_origin_x != self.origin_x or new_origin_y != self.origin_y or new_origin_z != self.origin_z
        )

        print("[update_bounding_box] New: nx={}, ny={}, nz={}".format(new_nx, new_ny, new_nz))
        print("[update_bounding_box] Old: nx={}, ny={}, nz={}".format(self.nx, self.ny, self.nz))
        print("[update_bounding_box] New origin: ({}, {}, {})".format(new_origin_x, new_origin_y, new_origin_z))

        if not changed:
            return

        print("[update_bounding_box] Updating bounding box.")
        old_nx, old_ny, old_nz = self.nx, self.ny, self.nz
        old_ox, old_oy, old_oz = self.origin_x, self.origin_y, self.origin_z

        self.nx = new_nx
        self.ny = new_ny
        self.nz = new_nz
        self.origin_x = new_origin_x
        self.origin_y = new_origin_y
        self.origin_z = new_origin_z

        print('self.oxygen_array', self.oxygen_array)
        if self.oxygen_array is not None:
            new_oxygen = np.ones((self.nz, self.ny, self.nx), dtype=float)
            new_nutrient = np.ones((self.nz, self.ny, self.nx), dtype=float)
            new_IL8 = np.zeros((self.nz, self.ny, self.nx), dtype=float)
            new_IL6 = np.zeros((self.nz, self.ny, self.nx), dtype=float)
            new_vegf = np.zeros((self.nz, self.ny, self.nx), dtype=float)

            self._copy_old_values_to_new_np_array(
                old_nx, old_ny, old_nz,
                old_ox, old_oy, old_oz,
                self.oxygen_array, self.nutrient_array, self.IL8_array, self.IL6_array,
                new_oxygen, new_nutrient, new_IL8, new_IL6, new_vegf
            )
            self.oxygen_array = new_oxygen
            self.nutrient_array = new_nutrient
            self.IL8_array = new_IL8
            self.IL6_array = new_IL6
            self.VEGF_array=new_vegf



    def _copy_old_values_to_new_np_array(self, old_nx, old_ny, old_nz,
                                         old_origin_x, old_origin_y, old_origin_z,
                                         old_oxygen, old_nutrient, old_IL8, old_IL6,
                                         new_oxygen, new_nutrient, new_IL8, new_IL6,new_vegf):
        if old_oxygen is None:
            return

        old_min_x = int(old_origin_x / self.dx)
        old_min_y = int(old_origin_y / self.dx)
        old_min_z = int(old_origin_z / self.dx)
        old_max_x = old_min_x + old_nx - 1
        old_max_y = old_min_y + old_ny - 1
        old_max_z = old_min_z + old_nz - 1

        new_min_x = int(self.origin_x / self.dx)
        new_min_y = int(self.origin_y / self.dx)
        new_min_z = int(self.origin_z / self.dx)
        new_max_x = new_min_x + self.nx - 1
        new_max_y = new_min_y + self.ny - 1
        new_max_z = new_min_z + self.nz - 1

        overlap_xmin = max(old_min_x, new_min_x)
        overlap_xmax = min(old_max_x, new_max_x)
        overlap_ymin = max(old_min_y, new_min_y)
        overlap_ymax = min(old_max_y, new_max_y)
        overlap_zmin = max(old_min_z, new_min_z)
        overlap_zmax = min(old_max_z, new_max_z)

        if (overlap_xmin > overlap_xmax or overlap_ymin > overlap_ymax or overlap_zmin > overlap_zmax):
            print('No overlap between old and new bounding boxes.')
            return

        for gz in range(overlap_zmin, overlap_zmax + 1):
            for gy in range(overlap_ymin, overlap_ymax + 1):
                for gx in range(overlap_xmin, overlap_xmax + 1):
                    old_ix = gx - old_min_x
                    old_iy = gy - old_min_y
                    old_iz = gz - old_min_z
                    new_ix = gx - new_min_x
                    new_iy = gy - new_min_y
                    new_iz = gz - new_min_z

                    new_oxygen[new_iz, new_iy, new_ix] = old_oxygen[old_iz, old_iy, old_ix]
                    new_nutrient[new_iz, new_iy, new_ix] = old_nutrient[old_iz, old_iy, old_ix]
                    new_IL8[new_iz, new_iy, new_ix] = old_IL8[old_iz, old_iy, old_ix]
                    new_IL6[new_iz, new_iy, new_ix] = old_IL6[old_iz, old_iy, old_ix]
                    new_vegf[new_iz, new_iy, new_ix] = new_vegf[old_iz, old_iy, old_ix]



        default_oxygen_count = np.count_nonzero(new_oxygen == 0.5)
        print(f"[_copy_old_values_to_new_np_array] Oxygen values at default (0.5): {default_oxygen_count}")

    def create_2d_contour_plots(self, mcs, pleural_mask):
        mid_z = self.nz // 2
        mid_y = self.ny // 2
        mid_x = self.nx // 2

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # XY plane (slice at mid_z)
        xy_slice = self.oxygen_array[mid_z, :, :]
        im1 = ax1.contourf(xy_slice, levels=20, cmap='viridis')
        ax1.set_title('XY Plane (mid Z)')
        plt.colorbar(im1, ax=ax1)
        if pleural_mask is not None:
            mask_xy = pleural_mask[mid_z, :, :]
            ax1.contour(mask_xy, levels=[0.5], colors='red', linewidths=2)

        # XZ plane (slice at mid_y)
        xz_slice = self.oxygen_array[:, mid_y, :]
        im2 = ax2.contourf(xz_slice, levels=20, cmap='viridis')
        ax2.set_title('XZ Plane (mid Y)')
        plt.colorbar(im2, ax=ax2)
        if pleural_mask is not None:
            mask_xz = pleural_mask[:, mid_y, :]
            ax2.contour(mask_xz, levels=[0.5], colors='red', linewidths=2)

        # YZ plane (slice at mid_x)
        yz_slice = self.oxygen_array[:, :, mid_x]
        im3 = ax3.contourf(yz_slice, levels=20, cmap='viridis')
        ax3.set_title('YZ Plane (mid X)')
        plt.colorbar(im3, ax=ax3)
        if pleural_mask is not None:
            mask_yz = pleural_mask[:, :, mid_x]
            ax3.contour(mask_yz, levels=[0.5], colors='red', linewidths=2)

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f'contour_2d_mcs_{mcs}.png')
        plt.savefig(plot_path)
        print(f'2D contour plot saved to {plot_path}')
        plt.close()

    def create_3d_cell_gradient_plot(self, mcs, space_mask):
        try:
            import pyvista as pv
            grad_z, grad_y, grad_x = np.gradient(self.oxygen_array, self.dx)
            grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)

            if space_mask is None:
                print("No pleural mask available for 3D visualization.")
                return

            indices = np.argwhere(space_mask)
            if indices.size == 0:
                print("Pleural mask is empty; nothing to visualize.")
                return

            points = np.zeros((indices.shape[0], 3))
            scalar_values = np.zeros(indices.shape[0])
            for i, (z, y, x) in enumerate(indices):
                points[i, 0] = self.origin_x + x * self.dx
                points[i, 1] = self.origin_y + y * self.dx
                points[i, 2] = self.origin_z + z * self.dx
                scalar_values[i] = grad_magnitude[z, y, x]

            pd = pv.PolyData(points)
            pd["gradient"] = scalar_values

            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(pd, scalars="gradient", cmap="viridis",
                             render_points_as_spheres=True, point_size=8)
            plotter.add_scalar_bar(title="Oxygen Gradient", vertical=True)
            plotter.camera_position = 'iso'
            plotter.camera.zoom(1.3)
            plotter.add_text(f"Oxygen Gradient at MCS {mcs}", font_size=14, color="black", position="upper_left")
            output_path = os.path.join(self.output_dir, f'cell_gradient_mcs_{mcs}.png')
            plotter.screenshot(output_path)
            plotter.close()
            print(f"3D cell gradient visualization saved to: {output_path}")
        except Exception as e:
            print(f"Error in create_3d_cell_gradient_plot: {e}")
            traceback.print_exc()

    def visualize_full_mesh_3d(self, field_data, mcs, output_folder, colormap='viridis',
                               opacity=0.7, title="Full 3D Mesh Visualization"):
        """
        Visualize the full 3D mesh with the provided field data.

        Args:
            field_data: 3D numpy array containing the field values (e.g., oxygen_array)
            mcs: Monte Carlo step for labeling the output file
            output_folder: Directory to save the visualization
            colormap: Matplotlib colormap to use for visualization
            opacity: Opacity value for the 3D volume rendering (0-1)
            title: Title for the visualization
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.cm as cm

            # Create figure and 3D axis
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Get dimensions
            nz, ny, nx = field_data.shape

            # Create meshgrid for visualization
            z, y, x = np.mgrid[0:nz, 0:ny, 0:nx]

            # Normalize the data for coloring
            normalized_data = (field_data - field_data.min()) / (field_data.max() - field_data.min() + 1e-10)

            # Choose a threshold for visualization (adjust as needed)
            threshold = 0.1
            mask = normalized_data > threshold

            # Apply mask
            x_masked = x[mask]
            y_masked = y[mask]
            z_masked = z[mask]
            values_masked = normalized_data[mask]

            # Create scatter plot with colors based on field values
            scatter = ax.scatter(
                x_masked, y_masked, z_masked,
                c=values_masked,
                cmap=colormap,
                alpha=opacity,
                s=5  # Point size
            )

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label(f'Normalized {title} Values')

            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{title} at MCS {mcs}')

            # Save the visualization
            filename = os.path.join(output_folder, f'full_mesh_3d_mcs_{mcs}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)

            print(f"Saved 3D full mesh visualization to {filename}")

        except Exception as e:
            print(f"Error in visualize_full_mesh_3d: {e}")
            traceback.print_exc()

    def visualize_bounded_mesh_3d(self, field_data, pleural_mask, mcs, output_folder,
                                  colormap='viridis', opacity=0.7, title="Bounded 3D Mesh Visualization"):
        """
        Visualize the 3D mesh within the bounding box, showing only the parts inside the pleural cavity.

        Args:
            field_data: 3D numpy array containing the field values (e.g., oxygen_array)
            pleural_mask: Boolean mask indicating the pleural cavity region
            mcs: Monte Carlo step for labeling the output file
            output_folder: Directory to save the visualization
            colormap: Matplotlib colormap to use for visualization
            opacity: Opacity value for the 3D volume rendering (0-1)
            title: Title for the visualization
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.cm as cm

            # Create figure and 3D axis
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Get dimensions
            nz, ny, nx = field_data.shape

            # Create meshgrid for visualization
            z, y, x = np.mgrid[0:nz, 0:ny, 0:nx]

            # Normalize the data for coloring
            normalized_data = (field_data - field_data.min()) / (field_data.max() - field_data.min() + 1e-10)

            # Apply the pleural mask to show only data inside the pleural cavity
            combined_mask = pleural_mask & (normalized_data > 0.1)  # Adjust threshold as needed

            # Apply mask

            x_masked = x[combined_mask]
            y_masked = y[combined_mask]
            z_masked = z[combined_mask]
            values_masked = normalized_data[combined_mask]

            # Create scatter plot with colors based on field values
            scatter = ax.scatter(
                x_masked, y_masked, z_masked,
                c=values_masked,
                cmap=colormap,
                alpha=opacity,
                s=5  # Point size
            )

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label(f'Normalized {title} Values in Pleural Cavity')

            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{title} (Bounded by Pleural Cavity) at MCS {mcs}')

            # Save the visualization
            filename = os.path.join(output_folder, f'bounded_mesh_3d_mcs_{mcs}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)

            print(f"Saved 3D bounded mesh visualization to {filename}")

            # Create an additional visualization showing the bounding box outline
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Plot the data points
            scatter = ax.scatter(
                x_masked, y_masked, z_masked,
                c=values_masked,
                cmap=colormap,
                alpha=opacity,
                s=3  # Smaller point size
            )

            # Draw the bounding box edges
            bbox_x = [0, nx - 1, nx - 1, 0, 0, nx - 1, nx - 1, 0]
            bbox_y = [0, 0, ny - 1, ny - 1, 0, 0, ny - 1, ny - 1]
            bbox_z = [0, 0, 0, 0, nz - 1, nz - 1, nz - 1, nz - 1]

            # Define the edges of the box
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
                (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
                (0, 4), (1, 5), (2, 6), (3, 7)  # Connecting edges
            ]

            # Plot the edges
            for start, end in edges:
                ax.plot(
                    [bbox_x[start], bbox_x[end]],
                    [bbox_y[start], bbox_y[end]],
                    [bbox_z[start], bbox_z[end]],
                    'r-', linewidth=2
                )

            # Add colorbar and labels
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label(f'Normalized {title} Values in Pleural Cavity')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{title} with Bounding Box at MCS {mcs}')

            # Save the bounding box visualization
            bbox_filename = os.path.join(output_folder, f'bounded_mesh_with_box_3d_mcs_{mcs}.png')
            plt.savefig(bbox_filename, dpi=300, bbox_inches='tight')
            plt.close(fig)

            print(f"Saved 3D bounded mesh with box visualization to {bbox_filename}")

        except Exception as e:
            print(f"Error in visualize_bounded_mesh_3d: {e}")
            traceback.print_exc()

class CellInitializerSteppable(SteppableBasePy):
    def start(self):
        try:
            # --- Define Cell Type IDs (make sure cv has a definition for BONE) ---
            # --- Define Cell Type IDs (ensure cv has these attributes) ---
            self.MEDIUM = cv.MEDIUM
            self.EPITHELIOIDTUMOR = cv.EPITHELIOIDTUMOR
            self.SARCOMATOIDTUMOR = cv.SARCOMATOIDTUMOR
            self.VISCERALPLEURA = cv.VISCERALPLEURA
            self.PARIETALPLEURA = cv.PARIETALPLEURA
            self.M2MACROPHAGE = cv.M2MACROPHAGE
            self.CAF = cv.CAF
            self.FIBROBLAST = cv.FIBROBLAST
            self.BONE = cv.BONE

            # --- Create Scalar Fields for Visualization ---
            self.create_scalar_field_py("LungMaskField")
            self.create_scalar_field_py("BoneMaskField")
            self.create_scalar_field_py("TumorMaskField")
            self.create_scalar_field_py("SpaceBetweenPleuras")
            self.create_scalar_field_py("VEGF")

            # --- Load masks from compressed file ---
            mask_file = "/home/anton/miniconda3/envs/cc3d/lib/python3.10/site-packages/cancer-modeling/Simulation/lung_mask_MPM_0782-20161209_0000.npz"
            with np.load(mask_file) as data:
                lung_mask = data['lung_mask']
                bone_mask = data['bone_mask']
                tumor_mask = data['tumor_mask']
                spacing = data['spacing']

            # mask.shape is (Z, Y, X)
            mask_shape = lung_mask.shape
            physical_size_mm = (mask_shape[2] * spacing[0],  # X
                                mask_shape[1] * spacing[1],  # Y
                                mask_shape[0] * spacing[2])  # Z

            sim_shape = (self.dim.x, self.dim.y, self.dim.z)
            lattice_unit_size_mm = (physical_size_mm[0] / sim_shape[0],
                                    physical_size_mm[1] / sim_shape[1],
                                    physical_size_mm[2] / sim_shape[2])

            lattice_unit_size_mm3 = lattice_unit_size_mm[0] * lattice_unit_size_mm[1] * lattice_unit_size_mm[2]

            # Resample masks (order=0 for label masks)
            lung_mask_resized = resize(lung_mask, (sim_shape[2], sim_shape[1], sim_shape[0]), order=0,
                                       preserve_range=True, anti_aliasing=False).astype(bool)
            # bone_mask_resized = resize(bone_mask, (sim_shape[2], sim_shape[1], sim_shape[0]), order=0,
            #                            preserve_range=True, anti_aliasing=False).astype(bool)
            tumor_mask_resized = resize(tumor_mask, (sim_shape[2], sim_shape[1], sim_shape[0]), order=0,
                                        preserve_range=True, anti_aliasing=False).astype(bool)


            lung_mask_xyz = lung_mask_resized
            # bone_mask_xyz = bone_mask_resized
            tumor_mask_xyz = tumor_mask_resized

            # --- Set Up Scalar Fields for Visualization ---
            self.field.LungMaskField[:, :, :] = np.transpose(lung_mask_resized, (2, 1, 0)).astype(float)
            # self.field.BoneMaskField[:, :, :] = np.transpose(bone_mask_resized, (2, 1, 0)).astype(float)
            self.field.TumorMaskField[:, :, :] = np.transpose(tumor_mask_resized, (2, 1, 0)).astype(float)


            lung_minus_tumor = lung_mask_xyz & (~tumor_mask_xyz)
            dilated = ndimage.binary_dilation(lung_minus_tumor, iterations=2)
            visceral_region = dilated & (~lung_minus_tumor)
            core_mask = lung_mask_xyz | tumor_mask_xyz
            core_dilated_4 = ndimage.binary_dilation(core_mask, iterations=4)
            core_dilated_3 = ndimage.binary_dilation(core_mask, iterations=3)
            parietal_region = core_dilated_4 & (~core_dilated_3)
            background = ~core_mask
            # Compute the Euclidean distance from each background voxel to the nearest core voxel
            distance_from_core = ndimage.distance_transform_edt(background)

            # 3. Define the parietal pleura as a shell at a fixed distance from the core
            # For example, a shell between 4 and 6 voxels from the core surface
            pleura_inner = 4  # inner boundary in voxels
            pleura_outer = 6  # outer boundary in voxels

            parietal_region = (distance_from_core >= pleura_inner) & (distance_from_core < pleura_outer)
            # 5. Pleural space: the space between visceral and parietal pleura
            core_dilated_1 = ndimage.binary_dilation(core_mask, iterations=1)
            tumor_mask_dilated = ndimage.binary_dilation(tumor_mask_xyz, iterations=1)
            space_between_pleuras = ((core_dilated_3 & (~core_dilated_1)) | tumor_mask_dilated) & (~parietal_region)


            # --- Store for other steppables ---
            self.shared_steppable_vars['lattice_unit_size_mm'] = cv.LATTICE_UNIT_SIZE_MM
            # self.shared_steppable_vars['space_between_pleuras'] = np.transpose(space_between_pleuras, (2, 1, 0)).astype(float)
            # self.shared_steppable_vars['visceral_pleura'] = np.transpose(visceral_region, (2, 1, 0))
            # self.shared_steppable_vars['parietal_pleura'] = np.transpose(parietal_region, (2, 1, 0))
            self.shared_steppable_vars['space_between_pleuras'] = space_between_pleuras
            self.shared_steppable_vars['visceral_pleura'] = visceral_region
            self.shared_steppable_vars['parietal_pleura'] = parietal_region
            self.shared_steppable_vars['lung_mask'] = lung_mask_resized
            # self.shared_steppable_vars['bone_mask'] = bone_mask_resized
            self.shared_steppable_vars['tumor_mask'] = tumor_mask_resized

            # --- Assign scalar field for visualization ---
            self.field.SpaceBetweenPleuras[:, :, :] = np.transpose(space_between_pleuras, (2, 1, 0)).astype(float)

            sim_voxel_volume = (cv.LATTICE_UNIT_SIZE_MM) ** 3
            print('sim voxel volume', sim_voxel_volume)  # e.g., 0.4³ = 0.064 mm³
            target_volume_mm3 = 451207  # 10826.88  # 676 #2680.32 #451
            target_voxels = int(target_volume_mm3 / sim_voxel_volume)
            total_space_voxels = np.count_nonzero(space_between_pleuras)
            total_pleural_space_volume_mm3 = total_space_voxels * sim_voxel_volume
            total_tumor_voxels = np.count_nonzero(tumor_mask_xyz)
            total_tumor_voxels_volume_mm3 = total_tumor_voxels * sim_voxel_volume
            print(
                f"Total pleural space voxels={total_space_voxels}, Total pleural space volume={total_pleural_space_volume_mm3:.2f} mm³")

            print(
                f"Total tumor voxels = {total_tumor_voxels}, total tumour volume = {total_tumor_voxels_volume_mm3}"
            )
            # 4. Get the coordinates
            solid_tumor_mask = binary_fill_holes(tumor_mask_xyz)
            candidate_indices = np.argwhere(solid_tumor_mask)

            if candidate_indices.shape[0] == 0:
                print("No intersection between visceral pleura and tumor mask found.")
                return

            # If more candidates than needed, subsample randomly
            if candidate_indices.shape[0] > target_voxels:
                selected_indices = candidate_indices[
                    np.random.choice(candidate_indices.shape[0], target_voxels, replace=False)]
            else:
                selected_indices = candidate_indices

            tumor_indices = selected_indices
            # Place tumor cells
            placed = 0
            init_vol = 0
            cell_vol = 0
            for (z, y, x) in selected_indices:
                x, y, z = int(x), int(y), int(z)
                # if self.cell_field[x, y, z] is None or self.cell_field[x, y, z].type == self.MEDIUM:
                #     if not (visceral_region[z, y, x] or parietal_region[z, y, x]):
                cell = self.new_cell(self.EPITHELIOIDTUMOR)
                self.cell_field[x, y, z] = cell
                cell.targetVolume = 1
                cell.lambdaVolume = 100
                cell.dict['division_volume'] = cell.targetVolume * 1.5
                cell.dict['mass'] = cv.MASS_EPITHELIOID_CELL_NG
                placed += 1
                init_vol += cell.targetVolume * sim_voxel_volume
                cell_vol = cell.targetVolume * sim_voxel_volume

            actual_volume_mm3 = placed * sim_voxel_volume
            volume_error_pct = abs(actual_volume_mm3 - target_volume_mm3) / target_volume_mm3 * 100
            print(f"Initialized {placed} tumor cells along visceral pleura (target: {target_voxels})")
            print(f"""
                        Initialized tumor cluster:
                        - Cells placed: {placed}/{target_voxels} ({placed / target_voxels * 100:.1f}% of target cells)
                        - Actual volume: {init_vol:.2f} mm³
                        - Target volume: {target_volume_mm3:.2f} mm³
                        - Volume error: {volume_error_pct:.2f}%
                        - Voxel size: {sim_voxel_volume:.3f} mm³
                        - Single cell target volume: {cell_vol} mm³
                        """)

            # --- Initialize Visceral Pleura Cells ---
            for (z, y, x) in np.argwhere(visceral_region):
                x, y, z = int(x), int(y), int(z)
                # if not self.cell_field[x, y, z]:
                c = self.new_cell(self.VISCERALPLEURA)
                self.cell_field[x, y, z] = c
                c.targetVolume = 1
                c.lambdaVolume = 500
                # c.targetSurface = 4
                # c.lambdaSurface = 10
                c.fluctAmpl = 1
                c.dict['division_volume'] = 1e9
                c.connectivityOn = True

            # --- Initialize Parietal Pleura Cells ---
            for (z, y, x) in np.argwhere(parietal_region):
                x, y, z = int(x), int(y), int(z)
                # if not self.cell_field[x, y, z]:
                c = self.new_cell(self.PARIETALPLEURA)
                self.cell_field[x, y, z] = c
                c.targetVolume = 1
                c.lambdaVolume = 500
                # c.targetSurface = 4
                # c.lambdaSurface = 10
                c.fluctAmpl = 1
                c.dict['division_volume'] = 1e9
                c.connectivityOn = True

            #self.pre_equilibrate_fields_with_hpc(iterations=25)
            ramdisk_path = "/home/anton/miniconda3/envs/cc3d/lib/python3.10/site-packages/cancer-modeling/ramdisk"
            steady_oxygen_path = os.path.join(ramdisk_path, "steady_oxygen.dat")
            steady_nutrient_path = os.path.join(ramdisk_path, "steady_nutrient.dat")

            if os.path.exists(steady_oxygen_path) and os.path.exists(steady_nutrient_path):
                print("Loading steady-state fields in CellInitializerSteppable.")
                oxygen_var = dump.read(steady_oxygen_path)
                nutrient_var = dump.read(steady_nutrient_path)
                self.shared_steppable_vars["oxygen_array"] = np.array(oxygen_var)
                self.shared_steppable_vars["nutrient_array"] = np.array(nutrient_var)
            bone_cells = 0
            pleura_dilation_distance = 4



            # Use bone_mask_shifted for cell placement:
            for (z, y, x) in np.argwhere(bone_mask):
                x, y, z = int(x), int(y), int(z)
                if not self.cell_field[x, y, z]:
                    c = self.new_cell(self.BONE)
                    self.cell_field[x, y, z] = c
                    c.targetVolume = 2
                    c.lambdaVolume = 100
                    c.dict['division_volume'] = 1e9
                    bone_cells += 1
            # --- Initialize Bone Cells Based on the Bone Mask ---

            # --- Initialize Stromal Cells Near the Tumor ---
            # Use the pleural space computed from the lung mask
            between_indices = np.argwhere(space_between_pleuras)
            if not np.any(between_indices):
                print("No space detected between pleuras for stromal cell placement.")
            else:
                # Use the tumor cell bounding box to limit the region
                if tumor_indices.size > 0:
                    tumor_min_z = max(int(np.min(tumor_indices[:, 0])) - 20, 0)
                    tumor_max_z = min(int(np.max(tumor_indices[:, 0])) + 20, self.dim.z - 1)
                    tumor_min_y = max(int(np.min(tumor_indices[:, 1])) - 20, 0)
                    tumor_max_y = min(int(np.max(tumor_indices[:, 1])) + 20, self.dim.y - 1)
                    tumor_min_x = max(int(np.min(tumor_indices[:, 2])) - 20, 0)
                    tumor_max_x = min(int(np.max(tumor_indices[:, 2])) + 20, self.dim.x - 1)
                else:
                    tumor_min_z = tumor_max_z = tumor_min_y = tumor_max_y = tumor_min_x = tumor_max_x = 0

                valid_space_indices = [
                    (z, y, x)
                    for z, y, x in np.argwhere(space_between_pleuras)
                    if (tumor_min_z <= z <= tumor_max_z) and
                       (tumor_min_y <= y <= tumor_max_y) and
                       (tumor_min_x <= x <= tumor_max_x)
                ]
                if not valid_space_indices:
                    print("No valid space detected near tumor for stromal cell placement.")
                else:
                    max_attempts = 1000
                    num_m2 = cv.NUM_M2
                    num_caf = cv.NUM_CAF
                    fibroblast_count = cv.FIBROBLAST_COUNT

                    # Place M2 Macrophages
                    attempts = 0
                    placed_m2 = 0

                    while placed_m2 < num_m2 and attempts < max_attempts:
                        z, y, x = random.choice(valid_space_indices)
                        x, y, z = int(x), int(y), int(z)
                        if not self.cell_field[x, y, z]:
                            cell = self.new_cell(self.M2MACROPHAGE)
                            self.cell_field[x, y, z] = cell
                            cell.targetVolume = 2.0
                            cell.lambdaVolume = 10.0
                            placed_m2 += 1
                        attempts += 1
                    if placed_m2 < num_m2:
                        print(f"Could only place {placed_m2}/{num_m2} M2 Macrophages after {max_attempts} attempts.")

                    # Place CAFs
                    attempts = 0
                    placed_caf = 0
                    while placed_caf < num_caf and attempts < max_attempts:
                        z, y, x = random.choice(valid_space_indices)
                        x, y, z = int(x), int(y), int(z)
                        if not self.cell_field[x, y, z]:
                            cell = self.new_cell(self.CAF)
                            self.cell_field[x, y, z] = cell
                            cell.targetVolume = 2.0
                            cell.lambdaVolume = 10.0
                            cell.dict['division_volume'] = 100.0
                            placed_caf += 1
                        attempts += 1
                    if placed_caf < num_caf:
                        print(f"Could only place {placed_caf}/{num_caf} CAFs after {max_attempts} attempts.")

                    # Place Fibroblasts
                    attempts = 0
                    placed_fibroblasts = 0
                    while placed_fibroblasts < fibroblast_count and attempts < max_attempts:
                        z, y, x = random.choice(valid_space_indices)
                        x, y, z = int(x), int(y), int(z)
                        if not self.cell_field[x, y, z]:
                            cell = self.new_cell(self.FIBROBLAST)
                            self.cell_field[x, y, z] = cell
                            cell.targetVolume = 2.0
                            cell.lambdaVolume = 10.0
                            placed_fibroblasts += 1
                        attempts += 1
                    if placed_fibroblasts < fibroblast_count:
                        print(f"Could only place {placed_fibroblasts}/{fibroblast_count} Fibroblasts after {max_attempts} attempts.")

        except Exception as e:
            print(f"Exception in CellInitializerSteppable: {e}")
            traceback.print_exc()

    def pre_equilibrate_fields_with_hpc(self, iterations=25):
        """
        Calls the parallelized HPCFiPyScript N times to equilibrate oxygen/nutrient fields.
        """
        print(f"[Pre-equilibration] Running HPCFiPyScript for {iterations} iterations...")

        ramdisk_path = "/home/anton/miniconda3/envs/cc3d/lib/python3.10/site-packages/cancer-modeling/ramdisk"
        hpc_script = "/home/anton/miniconda3/envs/cc3d/lib/python3.10/site-packages/cancer-modeling/Simulation/HPCFiPy_june1.py"
        python_bin = "/home/anton/miniconda3/envs/cc3d/bin/python"
        n = cv.CORES  # or set to your desired number of MPI processes

        for i in range(iterations):
            cmd = (
                f"mpiexec -n {n} {python_bin} {hpc_script} --petsc"
            )
            print(f"[Pre-equilibration] Iteration {i + 1}/{iterations}: {cmd}")
            subprocess.call(cmd, shell=True)

            # Optionally, check that output files exist and are non-empty
            oxy_path = os.path.join(ramdisk_path, "oxygen.dat")
            nut_path = os.path.join(ramdisk_path, "nutrient.dat")
            vegf_path = os.path.join(ramdisk_path, "vegf.dat")
            il6_path = os.path.join(ramdisk_path, "cytokine_il6.dat")
            il8_path = os.path.join(ramdisk_path, "cytokine_il8.dat")
            max_attempts = 10
            for attempt in range(max_attempts):
                if (os.path.exists(oxy_path) and os.path.getsize(oxy_path) > 0 and
                        os.path.exists(nut_path) and os.path.getsize(nut_path) > 0 and
                os.path.exists(vegf_path) and os.path.getsize(vegf_path) > 0 and
                os.path.exists(il6_path) and os.path.getsize(il6_path) > 0 and
                os.path.exists(il8_path) and os.path.getsize(il8_path) > 0
                 ):
                    break
                print(f"  Waiting for PDE output files (attempt {attempt + 1})...")
                time.sleep(1.0)
            else:
                print(f"  Warning: PDE output files not found or empty after {max_attempts} attempts.")

        print("[Pre-equilibration] Complete. Fields are now at steady-state.")

        # Optionally, load the final arrays into shared_steppable_vars for immediate use
        oxy = dump.read(os.path.join(ramdisk_path, "oxygen.dat"))
        nut = dump.read(os.path.join(ramdisk_path, "nutrient.dat"))
        il6 = dump.read(os.path.join(ramdisk_path, "cytokine_il6.dat"))
        il8 = dump.read(os.path.join(ramdisk_path, "cytokine_il8.dat"))
        vegf = dump.read(os.path.join(ramdisk_path, "vegf.dat"))
        # You may need to reshape based on your mesh dimensions
        self.shared_steppable_vars["oxygen_array"] = np.array(oxy)
        self.shared_steppable_vars["nutrient_array"] = np.array(nut)
        oxygen_src = os.path.join(ramdisk_path, "oxygen.dat")
        nutrient_src = os.path.join(ramdisk_path, "nutrient.dat")
        oxygen_dst = os.path.join(ramdisk_path, "steady_oxygen.dat")
        nutrient_dst = os.path.join(ramdisk_path, "steady_nutrient.dat")
        shutil.copy2(oxygen_src, oxygen_dst)
        shutil.copy2(nutrient_src, nutrient_dst)
        print(f"[Pre-equilibration] Saved steady-state fields as steady_oxygen.dat and steady_nutrient.dat")


class CellGrowthSteppable(SteppableBasePy):

    def start(self):
        # Cell type IDs
        self.MEDIUM = cv.MEDIUM
        self.EPITHELIOIDTUMOR = cv.EPITHELIOIDTUMOR
        self.SARCOMATOIDTUMOR = cv.SARCOMATOIDTUMOR
        self.VISCERALPLEURA = cv.VISCERALPLEURA
        self.PARIETALPLEURA = cv.PARIETALPLEURA
        self.M2MACROPHAGE = cv.M2MACROPHAGE
        self.CAF = cv.CAF
        self.FIBROBLAST = cv.FIBROBLAST

        # Retrieve the 'FiPySolverSteppable' reference (not the old fipy_solver)
        # We'll call it solver_step. This is the steppable that handles bounding box + PDE arrays
        self.solver_step = self.shared_steppable_vars.get('fipy_solver_steppable', None)
        if self.solver_step is None:
            print("Error: 'fipy_solver_steppable' not found in shared_steppable_vars.")
            return

        # Access space between pleuras
        self.space_between_pleuras = self.shared_steppable_vars.get('space_between_pleuras', None)
        if self.space_between_pleuras is None:
            print("Error: 'space_between_pleuras' not found in shared_steppable_vars.")

        self.lattice_unit_size_mm = self.shared_steppable_vars.get('lattice_unit_size_mm')
        self.ideal_growth_increment = cv.IDEAL_GROWTH  # Increase per minute for ideal doubling in 30h
        # Create a plot window (unchanged)
        self.pW = self.addNewPlotWindow(
            _title='Number of Cells',
            _xAxisTitle='MonteCarlo Step (MCS)',
            _yAxisTitle='Quantity',
            _xScaleType='linear',
            _yScaleType='linear',
            _grid=True
        )

        self.pW.addPlot('Epithelioid', _style='Lines', _color='red', _size=5)
        self.pW.addPlot('Sarcomatoid', _style='Lines', _color='green', _size=5)
        self.pW.addPlot('M2Macrophage', _style='Lines', _color='blue', _size=5)
        self.pW.addPlot('CAF', _style='Lines', _color='orange', _size=5)
        self.pW.addPlot('Fibroblast', _style='Lines', _color='cyan', _size=5)

        self.pW_volume = self.addNewPlotWindow(
            _title='Tumor Volume Growth',
            _xAxisTitle='MonteCarlo Step (MCS)',
            _yAxisTitle='Total Tumor Volume',
            _xScaleType='linear',
            _yScaleType='linear',
            _grid=True
        )
        self.pW_volume.addPlot('TumorVolume', _style='Lines', _color='magenta', _size=5)

        self.pW_apoptosis = self.addNewPlotWindow(
            _title='Number of Apoptotic Tumor Cells',
            _xAxisTitle='MonteCarlo Step (MCS)',
            _yAxisTitle='Apoptotic Cells',
            _xScaleType='linear',
            _yScaleType='linear',
            _grid=True
        )
        self.pW_apoptosis.addPlot('Apoptotic', _style='Lines', _color='black', _size=5)
        self.tumor_volume_history = []

    def step(self, mcs):
        try:
            # Growth rates
            growth_rates = cv.GROWTH_RATES

            # Retrieve PDE arrays from shared_steppable_vars
            oxygen_field = self.shared_steppable_vars.get("oxygen_array", None)
            nutrient_field = self.shared_steppable_vars.get("nutrient_array", None)
            il6_field = self.shared_steppable_vars.get("IL6_array", None)
            il8_field = self.shared_steppable_vars.get("IL8_array", None)

            if oxygen_field is None or nutrient_field is None or il6_field is None or il8_field is None:
                # PDE might not have run yet or bounding box is 0 in size
                return

            # Also get bounding box info from our FiPySolverSteppable reference
            if (self.solver_step.nx is None or
                    self.solver_step.ny is None or
                    self.solver_step.nz is None):
                return  # no bounding box => skip

            nx = self.solver_step.nx
            ny = self.solver_step.ny
            nz = self.solver_step.nz

            origin_x = self.solver_step.origin_x
            origin_y = self.solver_step.origin_y
            origin_z = self.solver_step.origin_z
            dx = self.solver_step.dx

            # Now each PDE array is shape [nz, ny, nx], so we can do oxygen_field[z,y,x]
            # Loop over tumor cells
            for cell in self.cell_list_by_type(self.EPITHELIOIDTUMOR, self.SARCOMATOIDTUMOR):

                if cell.dict.get('is_apoptotic', False):
                    continue

                # global CC3D coords
                z_idx = int(round(cell.zCOM))
                y_idx = int(round(cell.yCOM))
                x_idx = int(round(cell.xCOM))

                # Clamp indices to valid range
                z_idx = min(max(z_idx, 0), self.dim.z - 1)
                y_idx = min(max(y_idx, 0), self.dim.y - 1)
                x_idx = min(max(x_idx, 0), self.dim.x - 1)

                # Check domain
                if (0 <= x_idx < self.dim.x and
                        0 <= y_idx < self.dim.y and
                        0 <= z_idx < self.dim.z):

                    # Are we in the pleural space
                    if self.space_between_pleuras[z_idx, y_idx, x_idx]:
                        # Convert to bounding box local PDE coords
                        x_pde = int((cell.xCOM * self.lattice_unit_size_mm - origin_x) / dx)
                        y_pde = int((cell.yCOM * self.lattice_unit_size_mm - origin_y) / dx)
                        z_pde = int((cell.zCOM * self.lattice_unit_size_mm - origin_z) / dx)

                        if (0 <= x_pde < nx and
                                0 <= y_pde < ny and
                                0 <= z_pde < nz):

                            oxygen_level = oxygen_field[z_pde, y_pde, x_pde]
                            nutrient_level = nutrient_field[z_pde, y_pde, x_pde]
                            il6_level = il6_field[z_pde, y_pde, x_pde]
                            il8_level = il8_field[z_pde, y_pde, x_pde]
                        else:
                            # If PDE coords out of bounding box => assume defaults
                            oxygen_level = 1.0
                            nutrient_level = 1.0
                            il6_level = 0.0
                            il8_level = 0.0

                        # Simple growth logic
                        min_oxygen = cv.MIN_OXYGEN
                        min_nutrient = cv.MIN_NUTRIENT
                        if oxygen_level > min_oxygen and nutrient_level > min_nutrient:
                            # Use cv.MIN_OXYGEN and cv.MIN_NUTRIENT as half-saturation constants.
                            K_oxygen = cv.MIN_OXYGEN
                            K_nutrient = cv.MIN_NUTRIENT
                            alpha = 0.1  # cytokine sensitivity for IL6
                            beta = 0.1  # cytokine sensitivity for IL8
                            dt = cv.DT

                            oxygen_effect = oxygen_level / (K_oxygen + oxygen_level)
                            nutrient_effect = nutrient_level / (K_nutrient + nutrient_level)
                            growth_factor = min(oxygen_effect, nutrient_effect)
                            cytokine_factor = 1.0 + alpha * il6_level + beta * il8_level

                            # print(growth_rates[cell.type],growth_factor,cytokine_factor)
                            # Dynamic growth increment based on local conditions.
                            deltaV = growth_rates[cell.type] * growth_factor * cytokine_factor * dt

                            cell.targetVolume += deltaV

                            # Update mass proportionally
                            volume_increase_ratio = cell.targetVolume / cell.volume

                            cell.dict['mass'] *= volume_increase_ratio
                            # print(cell.dict['mass'],volume_increase_ratio)
                            oxygen_field[z_pde, y_pde, x_pde] -= cv.MITOSIS_CONS_COST
                            nutrient_field[z_pde, y_pde, x_pde] -= cv.MITOSIS_CONS_COST

                        else:
                            # If starved
                            cell.targetVolume -= cv.STARVATION_TARGETV

            # --- Visualization / Plotting ---
            cell_counts = {}
            for cell_type in [self.EPITHELIOIDTUMOR, self.SARCOMATOIDTUMOR,
                              self.M2MACROPHAGE, self.CAF, self.FIBROBLAST]:
                cell_counts[cell_type] = len(self.cell_list_by_type(cell_type))

            self.pW.addDataPoint("Epithelioid", mcs, cell_counts[self.EPITHELIOIDTUMOR])
            self.pW.addDataPoint("Sarcomatoid", mcs, cell_counts[self.SARCOMATOIDTUMOR])
            self.pW.addDataPoint("M2Macrophage", mcs, cell_counts[self.M2MACROPHAGE])
            self.pW.addDataPoint("CAF", mcs, cell_counts[self.CAF])
            self.pW.addDataPoint("Fibroblast", mcs, cell_counts[self.FIBROBLAST])

            # Calculate total tumor volume in mm³
            DX = self.lattice_unit_size_mm  # should be 0.4 mm
            tumor_volume_voxels = sum(
                cell.volume for cell in self.cell_list_by_type(self.EPITHELIOIDTUMOR, self.SARCOMATOIDTUMOR))
            tumor_volume_mm3 = tumor_volume_voxels * (DX ** 3)
            self.pW_volume.addDataPoint('TumorVolume', mcs, tumor_volume_mm3)
            self.tumor_volume_history.append((mcs, tumor_volume_mm3))

            # Debug: occupancy
            space_array = self.space_between_pleuras
            total_space_voxels = np.count_nonzero(space_array)
            occupied_count = 0
            for z, y, x in np.ndindex(space_array.shape):
                if space_array[z, y, x]:
                    if self.cell_field[x, y, z] is not None:
                        occupied_count += 1
            free_voxels = total_space_voxels - occupied_count
            print(f"Total pleural space voxels={total_space_voxels}, Occupied={occupied_count}, Free={free_voxels}")

            # Count apoptotic tumor cells (both types)
            apoptotic_count = 0
            for cell in self.cell_list_by_type(self.EPITHELIOIDTUMOR, self.SARCOMATOIDTUMOR):
                if cell.dict.get('is_apoptotic', False):
                    apoptotic_count += 1
            print('apoptotic count:', apoptotic_count)
            self.pW_apoptosis.addDataPoint('Apoptotic', mcs, apoptotic_count)

            if mcs % 100:
                self.finish()

            # Calculate percentage of tumor cells outside the pleural space
            total_tumor_cells = 0
            outside_count = 0
            for cell in self.cell_list_by_type(self.EPITHELIOIDTUMOR, self.SARCOMATOIDTUMOR):
                z_idx = int(round(cell.zCOM))
                y_idx = int(round(cell.yCOM))
                x_idx = int(round(cell.xCOM))

                # Clamp indices to valid range
                z_idx = min(max(z_idx, 0), self.dim.z - 1)
                y_idx = min(max(y_idx, 0), self.dim.y - 1)
                x_idx = min(max(x_idx, 0), self.dim.x - 1)
                if (0 <= x_idx < self.dim.x and 0 <= y_idx < self.dim.y and 0 <= z_idx < self.dim.z):
                    total_tumor_cells += 1
                    if not self.space_between_pleuras[z_idx, y_idx, x_idx]:
                        outside_count += 1
            if total_tumor_cells > 0:
                percent_outside = 100.0 * outside_count / total_tumor_cells
                print(
                    f"[MCS {mcs}] Tumor cells outside pleural space: {outside_count} / {total_tumor_cells} ({percent_outside:.2f}%)")
            else:
                print(f"[MCS {mcs}] No tumor cells present.")


        except Exception as e:
            print(f"Exception in CellGrowthSteppable at step {mcs}: {e}")
            import traceback
            traceback.print_exc()

        # self.debug_tumor_distribution(mcs)

    def finish(self):
        with open('tumor_volume_growth.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['MCS', 'TumorVolume_mm3'])
            for mcs, volume in self.tumor_volume_history:
                writer.writerow([mcs, volume])

    def debug_tumor_distribution(self, mcs):
        """
        Debug function to capture graded responses and investigate the 'O' shape growth pattern.
        It prints:
          - Each tumor cell's id, type, position, actual volume, targetVolume, division threshold,
          - The cell's computed radial distance from the tumor's center,
          - The lambdaVec (motility/force vector) components.
        Also prints summary statistics (min, max, mean radial distance).
        """
        # Get list of tumor cells (both epithelioid and sarcomatoid)
        tumor_cells = self.cell_list_by_type(self.EPITHELIOIDTUMOR, self.SARCOMATOIDTUMOR)
        if not tumor_cells:
            print("No tumor cells present at MCS", mcs)
            return

        # Compute tumor center (center of mass of tumor cell positions)
        center_x = np.mean([cell.xCOM for cell in tumor_cells])
        center_y = np.mean([cell.yCOM for cell in tumor_cells])
        center_z = np.mean([cell.zCOM for cell in tumor_cells])

        radial_distances = []

        print(f"--- Debug at MCS {mcs} ---")
        print(f"Tumor Center: ({center_x:.2f}, {center_y:.2f}, {center_z:.2f})")

        for cell in tumor_cells:
            # Get basic info: position, actual volume, targetVolume, division threshold
            pos = (cell.xCOM, cell.yCOM, cell.zCOM)
            volume = cell.volume
            target = cell.targetVolume
            div_thresh = cell.dict.get('division_volume', 'N/A')
            # Calculate radial distance from tumor center:
            r = np.sqrt((cell.xCOM - center_x) ** 2 + (cell.yCOM - center_y) ** 2 + (cell.zCOM - center_z) ** 2)
            radial_distances.append(r)
            # Retrieve the mechanical force vector (if available)
            lambda_vec = (cell.lambdaVecX, cell.lambdaVecY, cell.lambdaVecZ)
            # print(f"Cell {cell.id}: type={cell.type}, pos={pos}, volume={volume:.3f}, targetVolume={target:.3f}, "
            #       f"div_threshold={div_thresh}, radial_distance={r:.3f}, lambdaVec=({lambda_vec[0]:.2e}, {lambda_vec[1]:.2e}, {lambda_vec[2]:.2e})")

        # Summary statistics of radial distances:
        print(f"Radial distance (tumor spread): min={np.min(radial_distances):.2f}, "
              f"max={np.max(radial_distances):.2f}, mean={np.mean(radial_distances):.2f}")
        print("------------------------------\n")


class TumorCellMitosisSteppable(MitosisSteppableBase):
    def start(self):
        self.MEDIUM = cv.MEDIUM
        self.EPITHELIOIDTUMOR = cv.EPITHELIOIDTUMOR
        self.SARCOMATOIDTUMOR = cv.SARCOMATOIDTUMOR
        self.VISCERALPLEURA = cv.VISCERALPLEURA
        self.PARIETALPLEURA = cv.PARIETALPLEURA
        self.M2MACROPHAGE = cv.M2MACROPHAGE
        self.CAF = cv.CAF
        self.FIBROBLAST = cv.FIBROBLAST

        self.fipy_solver = self.shared_steppable_vars.get('fipy_solver_steppable', None)
        self.space_between_pleuras = self.shared_steppable_vars.get('space_between_pleuras', None)
        self.lattice_unit_size_mm = self.shared_steppable_vars.get('lattice_unit_size_mm')

        self.mitosis_consump_cost = cv.MITOSIS_CONS_COST

    def step(self, mcs):
        try:
            fipy_solver = self.fipy_solver
            if fipy_solver is None:
                return

            oxygen_field = self.shared_steppable_vars.get("oxygen_array", None)
            nutrient_field = self.shared_steppable_vars.get("nutrient_array", None)

            for cell in self.cell_list_by_type(self.EPITHELIOIDTUMOR, self.SARCOMATOIDTUMOR):
                x_idx = int(round(cell.xCOM))
                y_idx = int(round(cell.yCOM))
                z_idx = int(round(cell.zCOM))

                if (0 <= x_idx < self.dim.x and 0 <= y_idx < self.dim.y and 0 <= z_idx < self.dim.z):
                    if self.space_between_pleuras[z_idx, y_idx, x_idx]:
                        x = int((cell.xCOM * self.lattice_unit_size_mm - fipy_solver.origin_x) / fipy_solver.dx)
                        y = int((cell.yCOM * self.lattice_unit_size_mm - fipy_solver.origin_y) / fipy_solver.dx)
                        z = int((cell.zCOM * self.lattice_unit_size_mm - fipy_solver.origin_z) / fipy_solver.dx)

                        if 0 <= x < fipy_solver.nx and 0 <= y < fipy_solver.ny and 0 <= z < fipy_solver.nz:
                            oxygen_level = oxygen_field[z, y, x]
                            nutrient_level = nutrient_field[z, y, x]
                        else:
                            oxygen_level = 0.0
                            nutrient_level = 0.0

                        # print(
                        #     f"Cell {cell.id}: volume={cell.volume}, targetVolume={cell.targetVolume}, division_volume={cell.dict['division_volume']}")
                        initial_mass = cv.MASS_EPITHELIOID_CELL_NG if cell.type == self.EPITHELIOIDTUMOR else cv.MASS_SARCOMATOID_CELL_NG
                        if oxygen_level > 0.1 and nutrient_level > 0.1 and cell.volume >= cell.dict['division_volume']:
                            preferred_direction = self.get_preferred_division_direction(cell)
                            if preferred_direction is not None:
                                print('mitosis 1')
                                self.divide_cell_random_orientation(cell)
                                #self.divide_cell_orientation_vector_based(cell, *preferred_direction)
                            else:
                                print('mitosis 2')
                                # Optional: Use random direction if no valid preferred direction
                                self.divide_cell_random_orientation(cell)
                else:
                    print(f"Cell ID {cell.id} is outside the simulation domain ({x_idx}, {y_idx}, {z_idx})")
        except Exception as e:
            print(f"Exception in TumorCellMitosisSteppable at step {mcs}: {e}")
            traceback.print_exc()

    def update_attributes(self):
        # Assume parent's targetVolume is near 2. At division, split equally:
        self.parent_cell.targetVolume /= 2.0  # Now parent's target becomes 1.0
        self.parent_cell.dict['division_volume'] = self.parent_cell.targetVolume * 2.0
        self.clone_parent_2_child()
        self.child_cell.targetVolume = self.parent_cell.targetVolume
        self.child_cell.dict['division_volume'] = self.parent_cell.dict['division_volume']
        # Divide parent's mass equally among the two daughter cells
        self.parent_cell.dict['mass'] /= 2
        self.parent_cell.connectivityOn = True
        self.clone_parent_2_child()
        self.child_cell.dict['mass'] = self.parent_cell.dict['mass']

    def get_preferred_division_direction(self, cell):
        # Find a neighbor position still within space_between_pleuras
        search_radius = 1
        x_cell = int(round(cell.xCOM))
        y_cell = int(round(cell.yCOM))
        z_cell = int(round(cell.zCOM))

        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                for dz in range(-search_radius, search_radius + 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    x_neighbor = x_cell + dx
                    y_neighbor = y_cell + dy
                    z_neighbor = z_cell + dz

                    if (0 <= x_neighbor < self.dim.x and 0 <= y_neighbor < self.dim.y and 0 <= z_neighbor < self.dim.z):
                        if self.space_between_pleuras[z_neighbor, y_neighbor, x_neighbor]:
                            xVec = dx
                            yVec = dy
                            zVec = dz
                            norm = (xVec ** 2 + yVec ** 2 + zVec ** 2) ** 0.5
                            if norm != 0:
                                return (xVec / norm, yVec / norm, zVec / norm)
        return None


class ApoptosisSteppable(SteppableBasePy):
    def start(self):
        self.MEDIUM = cv.MEDIUM
        self.EPITHELIOIDTUMOR = cv.EPITHELIOIDTUMOR
        self.SARCOMATOIDTUMOR = cv.SARCOMATOIDTUMOR
        self.VISCERALPLEURA = cv.VISCERALPLEURA
        self.PARIETALPLEURA = cv.PARIETALPLEURA
        self.M2MACROPHAGE = cv.M2MACROPHAGE
        self.CAF = cv.CAF
        self.FIBROBLAST = cv.FIBROBLAST

        self.fipy_solver = self.shared_steppable_vars.get('fipy_solver_steppable', None)
        self.lattice_unit_size_mm = self.shared_steppable_vars.get('lattice_unit_size_mm')

    def step(self, mcs):
        try:

            fipy_solver = self.fipy_solver
            oxygen_field = self.shared_steppable_vars.get("oxygen_array", None)
            nutrient_field = self.shared_steppable_vars.get("nutrient_array", None)

            critical_oxygen = cv.CRITICAL_OXYGEN  # 2% for mild Hypoxia,0.1–0.5% Anoxia
            critical_nutrient = cv.CRITICAL_NUTRIENT

            for cell in self.cell_list_by_type(self.EPITHELIOIDTUMOR, self.SARCOMATOIDTUMOR):

                if cell.dict.get('is_apoptotic', False):
                    cell.targetVolume -= 0.3
                    if cell.volume < 0.5:  # Threshold for removal (adjust as needed)
                        self.deleteCell(cell)
                    # print("cell is already apoptotic")
                    continue

                x = int((cell.xCOM * self.lattice_unit_size_mm - fipy_solver.origin_x) / fipy_solver.dx)
                y = int((cell.yCOM * self.lattice_unit_size_mm - fipy_solver.origin_y) / fipy_solver.dx)
                z = int((cell.zCOM * self.lattice_unit_size_mm - fipy_solver.origin_z) / fipy_solver.dx)
                if 0 <= x < fipy_solver.nx and 0 <= y < fipy_solver.ny and 0 <= z < fipy_solver.nz:
                    oxygen_level = oxygen_field[z, y, x]
                    nutrient_level = nutrient_field[z, y, x]
                else:
                    # print("outside of bounds apotosis")
                    oxygen_level = 0.0
                    nutrient_level = 0.0

                # print(f"Cell:{cell.id}, has oxygen level: {oxygen_level}, nutrient level: {nutrient_level}")
                if oxygen_level < critical_oxygen or nutrient_level < critical_nutrient:
                    print(f"Apoptosis at step {mcs}: Cell ID {cell.id} "
                          f"at ({x}, {y}, {z}) - "
                          f"Oxygen: {oxygen_level}, Nutrient: {nutrient_level}")
                    cell.dict['is_apoptotic'] = True  # Mark the cell as apoptotic
                    cell.targetVolume -= 0.5  # Gradual shrinkage
                    if cell.targetVolume <= 0.0:
                        cell.targetVolume = 0.0
                        cell.lambdaVolume = 100.0  # High lambda to prevent further changes

            center_x = np.mean(
                [cell.xCOM for cell in self.cell_list_by_type(self.EPITHELIOIDTUMOR, self.SARCOMATOIDTUMOR)])
            print(
                f"Tumor Center: {center_x}, Cell Count: {len(self.cell_list_by_type(self.EPITHELIOIDTUMOR, self.SARCOMATOIDTUMOR))}")


        except Exception as e:
            print(f"Exception in ApoptosisSteppable at step {mcs}: {e}")


class HypoxiaResponseSteppable(SteppableBasePy):
    def start(self):
        self.MEDIUM = cv.MEDIUM
        self.EPITHELIOIDTUMOR = cv.EPITHELIOIDTUMOR
        self.SARCOMATOIDTUMOR = cv.SARCOMATOIDTUMOR
        self.VISCERALPLEURA = cv.VISCERALPLEURA
        self.PARIETALPLEURA = cv.PARIETALPLEURA
        self.M2MACROPHAGE = cv.M2MACROPHAGE
        self.CAF = cv.CAF
        self.FIBROBLAST = cv.FIBROBLAST

        self.fipy_solver = self.shared_steppable_vars.get('fipy_solver_steppable', None)
        self.lattice_unit_size_mm = self.shared_steppable_vars.get('lattice_unit_size_mm', 1.88)
        self.EMT_odd = cv.HYPOXIA_TRANSFORMATION_PROB
        self.hypoxia_thresh = cv.HYPOXIA_OXYGEN_THRESHOLD
        self.field.VEGF[:, :, :] = 0.0

    def step(self, mcs):
        try:

            fipy_solver = self.fipy_solver


            oxygen_field = self.shared_steppable_vars.get("oxygen_array", None)

            for cell in self.cell_list_by_type(1):  # Epithelioid
                x = int((cell.xCOM * self.lattice_unit_size_mm - fipy_solver.origin_x) / fipy_solver.dx)
                y = int((cell.yCOM * self.lattice_unit_size_mm - fipy_solver.origin_y) / fipy_solver.dx)
                z = int((cell.zCOM * self.lattice_unit_size_mm - fipy_solver.origin_z) / fipy_solver.dx)
                if 0 <= x < fipy_solver.nx and 0 <= y < fipy_solver.ny and 0 <= z < fipy_solver.nz:
                    if oxygen_field[z, y, x] < self.hypoxia_thresh:
                        oxygen_level = oxygen_field[z, y, x]
                        if oxygen_level < cv.HYPOXIA_OXYGEN_THRESHOLD:
                            self.fipy_solver.VEGF_array[z, y, x] += cv.VEGF_SECRETION_RATE
                            self.field.VEGF[x,y,z] += cv.VEGF_SECRETION_RATE
                            print('VEGF happening')
                        # in future should be depending also on cytokines
                        if random.random() < self.EMT_odd:  # DOI: 10.1038/nrc2620
                            print(f"Hypoxia at step {mcs}: Cell ID {cell.id} "
                                  f"at ({x}, {y}, {z}) - "
                                  f"Oxygen: {oxygen_level}")
                            cell.type = self.SARCOMATOIDTUMOR  # Sarcomatoid
                            cell.dict['division_volume'] = 50.0
                            cell.targetVolume = cell.volume
                            cell.lambdaVolume = 2.0
                # print(f"motility directions {cell.lambdaVecX, cell.lambdaVecY, cell.lambdaVecZ}")


        except Exception as e:
            print(f"Exception in HypoxiaResponseSteppable at step {mcs}: {e}")


# Base,requires refinement and accuracy
class FibroblastToCAFTransformationSteppable(SteppableBasePy):
    def start(self):
        self.TGFbeta_threshold = cv.TGFbeta_THRESHOLD  # Threshold for transformation

        self.MEDIUM = cv.MEDIUM
        self.FIBROBLAST = cv.FIBROBLAST
        self.CAF = cv.CAF

        self.fipy_solver = self.shared_steppable_vars.get('fipy_solver_steppable', None)
        self.lattice_unit_size_mm = self.shared_steppable_vars.get('lattice_unit_size_mm', 1.88)

        if self.fipy_solver is None:
            print("Error: 'fipy_solver' not found in shared_steppable_vars.")

    def step(self, mcs):
        try:
            fipy_solver = self.fipy_solver
            if fipy_solver is None:
                return

            tgf_beta_field = self.shared_steppable_vars.get("tgf_beta_array", None)

            for cell in self.cell_list_by_type(self.FIBROBLAST):
                x_idx = int((cell.xCOM * self.lattice_unit_size_mm - fipy_solver.origin_x) / fipy_solver.dx)
                y_idx = int((cell.yCOM * self.lattice_unit_size_mm - fipy_solver.origin_y) / fipy_solver.dx)
                z_idx = int((cell.zCOM * self.lattice_unit_size_mm - fipy_solver.origin_z) / fipy_solver.dx)

                # Validate indices
                if 0 <= x_idx < fipy_solver.nx and 0 <= y_idx < fipy_solver.ny and 0 <= z_idx < fipy_solver.nz:

                    tgf_beta_level = tgf_beta_field[z_idx, y_idx, x_idx]
                    if tgf_beta_level > self.TGFbeta_threshold:
                        cell.type = self.CAF  # Transform fibroblast to CAF
                        print(f"Fibroblast transformed into CAF at ({x_idx}, {y_idx}, {z_idx})")
                else:
                    print('tgf_beta_level:', tgf_beta_level)


        except Exception as e:
            print(f"Exception in FibroblastToCAFTransformationSteppable at step {mcs}: {e}")


class FiPyToCC3DFieldSynchronizationSteppable(SteppableBasePy):
    def start(self):
        self.fipy_solver = self.shared_steppable_vars.get('fipy_solver_steppable', None)
        if self.fipy_solver is None:
            print("Error: fipy_solver not found in shared variables.")
            return

        # Access CC3D concentration fields
        self.il6_field = CompuCell.getConcentrationField(self.simulator, "IL6")
        if self.il6_field is None:
            print("Error: IL6 field not found in CC3D.")
            return
        self.il8_field = CompuCell.getConcentrationField(self.simulator, "IL8")
        if self.il8_field is None:
            print("Error: il8_field not found in CC3D.")
            return
        self.oxygen_field = CompuCell.getConcentrationField(self.simulator, "Oxygen")
        if self.oxygen_field is None:
            print("Error: oxygen_field not found in CC3D.")
            return
        self.nutrient_field = CompuCell.getConcentrationField(self.simulator, "Nutrient")
        if self.nutrient_field is None:
            print("Error: nutrient_field not found in CC3D.")
            return
        self.vegf_field = CompuCell.getConcentrationField(self.simulator, "VEGF")
        if self.vegf_field is None:
            print("Error: vegf_field not found in CC3D.")
            return
        # self.tgfbeta_field = CompuCell.getConcentrationField(self.simulator, "TGFbeta")

        self.lattice_unit_size_mm = self.shared_steppable_vars.get('lattice_unit_size_mm', 1.88)
        self.origin_x = self.fipy_solver.origin_x
        self.origin_y = self.fipy_solver.origin_y
        self.origin_z = self.fipy_solver.origin_z
        self.dx = self.fipy_solver.dx

    def step(self, mcs):
        print('00')

        if not self.fipy_solver:
            return

        # Extract FiPy arrays
        oxygen_array = self.shared_steppable_vars.get("oxygen_array", None)
        nutrient_array = self.shared_steppable_vars.get("nutrient_array", None)
        il6_array = self.shared_steppable_vars.get("IL6_array", None)
        il8_array = self.shared_steppable_vars.get("IL8_array", None)
        VEGF_array = self.shared_steppable_vars.get("VEGF_array", None)
        # tgf_beta_array = self.shared_steppable_vars.get("tgf_beta_array", None)

        # Compute integer offsets
        min_x_sp = int(self.fipy_solver.origin_x / self.fipy_solver.lattice_unit_size_mm)
        min_y_sp = int(self.fipy_solver.origin_y / self.fipy_solver.lattice_unit_size_mm)
        min_z_sp = int(self.fipy_solver.origin_z / self.fipy_solver.lattice_unit_size_mm)

        # Synchronize
        for iz in range(self.fipy_solver.nz):
            for iy in range(self.fipy_solver.ny):
                for ix in range(self.fipy_solver.nx):
                    global_x = min_x_sp + ix
                    global_y = min_y_sp + iy
                    global_z = min_z_sp + iz

                    if 0 <= global_x < self.dim.x and 0 <= global_y < self.dim.y and 0 <= global_z < self.dim.z:
                        self.il6_field[global_x, global_y, global_z] = il6_array[iz, iy, ix]
                        self.il8_field[global_x, global_y, global_z] = il8_array[iz, iy, ix]
                        self.oxygen_field[global_x, global_y, global_z] = oxygen_array[iz, iy, ix]
                        self.nutrient_field[global_x, global_y, global_z] = nutrient_array[iz, iy, ix]
                        self.vegf_field[global_x, global_y, global_z] = VEGF_array[iz, iy, ix]


# Base,requires refinement and accuracy
class ChemotaxisSteppable(SteppableBasePy):
    def start(self):
        try:
            # Define cell type IDs used for chemotaxis
            self.M2MACROPHAGE = cv.M2MACROPHAGE
            self.CAF = cv.CAF

            # Initialize chemotaxis for M2 Macrophages towards IL6 and IL8
            for cell in self.cell_list_by_type(self.M2MACROPHAGE):
                il6_chemotaxis_data = self.chemotaxisPlugin.addChemotaxisData(cell, "IL6")
                il6_chemotaxis_data.setLambda(cv.IL6_CHEMOTAXIS_LAMBDA)  # Attract towards IL6

                il8_chemotaxis_data = self.chemotaxisPlugin.addChemotaxisData(cell, "IL8")
                il8_chemotaxis_data.setLambda(cv.IL8_CHEMOTAXIS_LAMBDA)  # Attract towards IL8

            # Initialize chemotaxis for CAF cells towards TGFbeta
            # for cell in self.cell_list_by_type(self.CAF):
            # tgf_beta_chemotaxis_data = self.chemotaxisPlugin.addChemotaxisData(cell, "TGFbeta")
            # tgf_beta_chemotaxis_data.setLambda(cv.TGFB_CHEMOTAXIS_LAMBDA)  # Attract towards TGFbeta

        except Exception as e:
            print(f"Exception in ChemotaxisSteppable.start: {e}")
            traceback.print_exc()

    def step(self, mcs):
        try:
            # Decrease chemotaxis strength over time for M2 Macrophages
            for cell in self.cell_list_by_type(self.M2MACROPHAGE):
                il6_data = self.chemotaxisPlugin.getChemotaxisData(cell, "IL6")
                if il6_data:
                    lambda_val = il6_data.getLambda() - cv.IL6_CHEMOTAXIS_DECAY
                    il6_data.setLambda(max(0.0, lambda_val))
                else:
                    print("no il_6_data")

                il8_data = self.chemotaxisPlugin.getChemotaxisData(cell, "IL8")
                if il8_data:
                    lambda_val = il8_data.getLambda() - cv.IL8_CHEMOTAXIS_DECAY
                    il8_data.setLambda(max(0.0, lambda_val))

            # Decrease chemotaxis strength over time for CAF cells
            # for cell in self.cell_list_by_type(self.CAF):
            # tgf_data = self.chemotaxisPlugin.getChemotaxisData(cell, "TGFbeta")
            # if tgf_data:
            # lambda_val = tgf_data.getLambda() - cv.TGFB_CHEMOTAXIS_DECAY
            # tgf_data.setLambda(max(0.0, lambda_val))

        except Exception as e:
            print(f"Exception in ChemotaxisSteppable.step at MCS {mcs}: {e}")
            traceback.print_exc()


# Base,requires refinement and accuracy
class AngiogenesisSteppable(SteppableBasePy):
    def step(self, mcs):
        try:
            fipy_solver = self.shared_steppable_vars.get('fipy_solver_steppable', None)
            if fipy_solver is None:
                return

            vegf_field = fipy_solver.VEGF_array

            threshold_vegf = cv.VEGF_THRESHOLD

            for z in range(fipy_solver.nz):
                for y in range(fipy_solver.ny):
                    for x in range(fipy_solver.nx):
                        if vegf_field[z, y, x] > threshold_vegf:
                            print(f"increasing oxygen and nutrients at ({x}, {y}, {z})")
                            fipy_solver.oxygen.value[z, y, x] += cv.OXYGEN_BOOST  # Boost oxygen
                            fipy_solver.nutrient.value[z, y, x] += cv.NUTRIENT_BOOST  # Boost nutrients

        except Exception as e:
            print(f"Exception in AngiogenesisSteppable at step {mcs}: {e}")


# Incorrect ruling of LamdaVec
class PleuraBoundarySteppable(SteppableBasePy):
    def start(self):
        try:
            self.EPITHELIOIDTUMOR = cv.EPITHELIOIDTUMOR
            self.SARCOMATOIDTUMOR = cv.SARCOMATOIDTUMOR

            self.space_between_pleuras = self.shared_steppable_vars.get('space_between_pleuras', None)
            self.visceral_pleura = self.shared_steppable_vars.get('visceral_pleura', None)
            self.parietal_pleura = self.shared_steppable_vars.get('parietal_pleura', None)
            self.lattice_unit_size_mm = self.shared_steppable_vars.get('lattice_unit_size_mm', 1.88)

            # Distance map is now optional; for simplicity, we can still compute it as distance to any pleura
            # Combine pleura masks
            combined_pleura = self.visceral_pleura | self.parietal_pleura
            self.distance_map = ndimage.distance_transform_edt(~combined_pleura).astype(np.float32)

        except Exception as e:
            print(f"Exception in PleuraBoundarySteppable.start: {e}")
            traceback.print_exc()

    def step(self, mcs):
        try:
            for cell in self.cell_list_by_type(self.EPITHELIOIDTUMOR, self.SARCOMATOIDTUMOR):
                x_idx = int(round(cell.xCOM))
                y_idx = int(round(cell.yCOM))
                z_idx = int(round(cell.zCOM))

                # Clamp indices to valid range
                z_idx = min(max(z_idx, 0), self.dim.z - 1)
                y_idx = min(max(y_idx, 0), self.dim.y - 1)
                x_idx = min(max(x_idx, 0), self.dim.x - 1)

                if 0 <= x_idx < self.dim.x and 0 <= y_idx < self.dim.y and 0 <= z_idx < self.dim.z:
                    # If cell is not in the space between pleuras (i.e., in pleura), push it back
                    # print('z,y,x:',z_idx,y_idx,x_idx,'shape_vis and shape_par:',self.visceral_pleura.shape,self.parietal_pleura.shape)
                    if self.visceral_pleura[z_idx, y_idx, x_idx] or self.parietal_pleura[z_idx, y_idx, x_idx]:
                        fx = self.compute_gradient_component(self.distance_map, x_idx, y_idx, z_idx, axis='x')
                        fy = self.compute_gradient_component(self.distance_map, x_idx, y_idx, z_idx, axis='y')
                        fz = self.compute_gradient_component(self.distance_map, x_idx, y_idx, z_idx, axis='z')

                        # Invert gradient to push cell away from the pleura (into the space between)
                        fx = -fx
                        fy = -fy
                        fz = -fz

                        magnitude = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2) + 1e-6
                        fx /= magnitude
                        fy /= magnitude
                        fz /= magnitude

                        force_magnitude = 1e6
                        cell.lambdaVecX = fx * force_magnitude
                        cell.lambdaVecY = fy * force_magnitude
                        cell.lambdaVecZ = fz * force_magnitude

                    else:
                        cell.lambdaVecX = 0.0
                        cell.lambdaVecY = 0.0
                        cell.lambdaVecZ = 0.0
                else:
                    cell.lambdaVecX = 0.0
                    cell.lambdaVecY = 0.0
                    cell.lambdaVecZ = 0.0

                # print(f"motility directions {cell.lambdaVecX, cell.lambdaVecY, cell.lambdaVecZ}")

        except Exception as e:
            print(f"Exception in PleuraBoundarySteppable at step {mcs}: {e}")
            traceback.print_exc()

    def compute_gradient_component(self, distance_map, x, y, z, axis):
        try:
            if axis == 'x':
                x_minus = max(x - 1, 0)
                x_plus = min(x + 1, self.dim.x - 1)
                gradient = (distance_map[z, y, x_plus] - distance_map[z, y, x_minus]) / 2.0
            elif axis == 'y':
                y_minus = max(y - 1, 0)
                y_plus = min(y + 1, self.dim.y - 1)
                gradient = (distance_map[z, y_plus, x] - distance_map[z, y_minus, x]) / 2.0
            elif axis == 'z':
                z_minus = max(z - 1, 0)
                z_plus = min(z + 1, self.dim.z - 1)
                gradient = (distance_map[z_plus, y, x] - distance_map[z_minus, y, x]) / 2.0
            else:
                gradient = 0.0
            return gradient
        except IndexError:
            return 0.0


class FPPPleuraSetupSteppable(SteppableBasePy):
    def start(self):
        # We will link adjacent pleura cells to form a mesh-like membrane.

        # Retrieve references to pleura arrays
        visceral_pleura = self.shared_steppable_vars.get('visceral_pleura', None)
        parietal_pleura = self.shared_steppable_vars.get('parietal_pleura', None)

        if visceral_pleura is None or parietal_pleura is None:
            print("FPPPleuraSetupSteppable: Pleura arrays not found.")
            return

        # Link visceral pleura cells among themselves
        self.link_pleura_sheet(visceral_pleura, cell_type_name="VisceralPleura")

        # Link parietal pleura cells among themselves
        self.link_pleura_sheet(parietal_pleura, cell_type_name="ParietalPleura")

    def link_pleura_sheet(self, pleura_mask, cell_type_name):
        """
        Scans through the given pleura mask (3D boolean array) and for each pleura cell:
        - Identify neighbors (e.g., in x, y, z directions)
        - If neighbor is also pleura cell, create a FocalPointPlasticity (FPP) link
        """
        # Directions to check for neighbors (6-connected)
        directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

        # We assume pleura_mask is shape [z,y,x]
        dim_x = self.dim.x
        dim_y = self.dim.y
        dim_z = self.dim.z

        # We will store cell references in a dictionary keyed by coordinates for quick access
        pleura_cells = {}

        # First pass: store references
        for z in range(dim_z):
            for y in range(dim_y):
                for x in range(dim_x):
                    if pleura_mask[z, y, x]:
                        cell = self.cell_field[x, y, z]
                        if cell and cell.type == self.get_cell_type_id(cell_type_name):
                            pleura_cells[(x, y, z)] = cell

        # Second pass: create links
        for (x, y, z), cell in pleura_cells.items():
            for dx, dy, dz in directions:
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < dim_x and 0 <= ny < dim_y and 0 <= nz < dim_z:
                    if pleura_mask[nz, ny, nx]:
                        neighbor_cell = pleura_cells.get((nx, ny, nz), None)
                        if neighbor_cell and neighbor_cell.id != cell.id:
                            # Create a FPP link only if not already linked (to avoid duplicates)
                            # CompuCell3D ensures no duplicate links, but you can also check
                            self.add_fpp_link(cell, neighbor_cell)

    def add_fpp_link(self, cell1, cell2):

        lambda_fpp = 1000  # Elastic strength
        target_distance = 1  # Equilibrium distance
        max_number_of_junctions = 16  # Ensure not to exceed the max allowed junctions

        # Ensure the number of current junctions does not exceed the maximum
        if cell1.dict.get("num_junctions", 0) < max_number_of_junctions and \
                cell2.dict.get("num_junctions", 0) < max_number_of_junctions:
            # Create the FPP link with explicit parameters
            self.focalPointPlasticityPlugin.createFocalPointPlasticityLink(
                cell1, cell2, lambda_fpp, target_distance
            )

            # Update the number of junctions for each cell
            cell1.dict["num_junctions"] = cell1.dict.get("num_junctions", 0) + 1
            cell2.dict["num_junctions"] = cell2.dict.get("num_junctions", 0) + 1

    def get_cell_type_id(self, cell_type_name):
        """
        Utility to convert cell type name string to a type ID.
        Adjust or use a lookup if necessary.
        """
        type_map = {
            "Medium": 0,
            "EpithelioidTumor": 1,
            "SarcomatoidTumor": 2,
            "VisceralPleura": 3,
            "ParietalPleura": 4,
            "M2Macrophage": 5,
            "CAF": 6,
            "Fibroblast": 7
        }
        return type_map.get(cell_type_name, None)


class TumorCellMotilitySteppable(SteppableBasePy):
    def start(self):
        # Define cell types
        self.EPITHELIOIDTUMOR = cv.EPITHELIOIDTUMOR
        self.SARCOMATOIDTUMOR = cv.SARCOMATOIDTUMOR

        # Retrieve the FiPy solver
        self.fipy_solver = self.shared_steppable_vars.get('fipy_solver_steppable', None)
        if self.fipy_solver is None:
            print("Error: 'fipy_solver' not found in shared_steppable_vars.")
            return

        # Retrieve lattice unit size
        self.lattice_unit_size_mm = self.shared_steppable_vars.get('lattice_unit_size_mm', 1.88)

        # Retrieve boundary information
        self.space_between_pleuras = self.shared_steppable_vars.get('space_between_pleuras', None)
        if self.space_between_pleuras is None:
            print("Error: 'space_between_pleuras' not found in shared_steppable_vars.")
            return

    def step(self, mcs):
        try:
            # Check if FiPy solver is initialized
            if self.fipy_solver is None:
                return

            # Extract oxygen and nutrient fields from FiPy

            oxygen_field = self.shared_steppable_vars.get("oxygen_array", None)
            nutrient_field = self.shared_steppable_vars.get("nutrient_array", None)

            # Motility strength for tumor cells (adjustable parameters)
            motility_strength_epithelioid = cv.MOTILITY_STRENGTH_EPI
            motility_strength_sarcomatoid = cv.MOTILITY_STRENGTH_SARC

            # Iterate over all tumor cells
            for cell in self.cell_list_by_type(self.EPITHELIOIDTUMOR, self.SARCOMATOIDTUMOR):
                # Calculate the grid indices of the cell in the FiPy field
                x = int((cell.xCOM * self.lattice_unit_size_mm - self.fipy_solver.origin_x) / self.fipy_solver.dx)
                y = int((cell.yCOM * self.lattice_unit_size_mm - self.fipy_solver.origin_y) / self.fipy_solver.dx)
                z = int((cell.zCOM * self.lattice_unit_size_mm - self.fipy_solver.origin_z) / self.fipy_solver.dx)

                # Skip cells outside the valid simulation grid
                if not (0 <= x < self.fipy_solver.nx and 0 <= y < self.fipy_solver.ny and 0 <= z < self.fipy_solver.nz):
                    continue

                # Compute the gradients of oxygen and nutrient at the cell's location
                grad_oxygen_x, grad_oxygen_y, grad_oxygen_z = self.compute_gradient(oxygen_field, x, y, z)
                grad_nutrient_x, grad_nutrient_y, grad_nutrient_z = self.compute_gradient(nutrient_field, x, y, z)

                # Calculate the combined chemotactic gradient (weighted sum of oxygen and nutrient gradients)
                grad_x = grad_oxygen_x + grad_nutrient_x
                grad_y = grad_oxygen_y + grad_nutrient_y
                grad_z = grad_oxygen_z + grad_nutrient_z
                # print(f"gradients are {grad_x}, {grad_y}, {grad_z}")

                # Normalize the gradient vector
                magnitude = (grad_x ** 2 + grad_y ** 2 + grad_z ** 2) ** 0.5 + 1e-6
                grad_x /= magnitude
                grad_y /= magnitude
                grad_z /= magnitude

                # Apply motility based on cell type
                if cell.type == self.EPITHELIOIDTUMOR:
                    motility_strength = motility_strength_epithelioid
                elif cell.type == self.SARCOMATOIDTUMOR:
                    motility_strength = motility_strength_sarcomatoid

                # Update motility direction
                # cell.lambdaVecX = grad_x * motility_strength
                # cell.lambdaVecY = grad_y * motility_strength
                # cell.lambdaVecZ = grad_z * motility_strength

                # Add random component
                random_component = 0.4  # Strength of random motility
                random_x = random.uniform(-1, 1)
                random_y = random.uniform(-1, 1)
                random_z = random.uniform(-1, 1)

                # Normalize random vector
                random_magnitude = (random_x ** 2 + random_y ** 2 + random_z ** 2) ** 0.5 + 1e-6
                random_x /= random_magnitude
                random_y /= random_magnitude
                random_z /= random_magnitude

                # Combine gradient and random components
                cell.lambdaVecX = (1 - random_component) * grad_x * motility_strength + \
                                  random_component * random_x * motility_strength
                cell.lambdaVecY = (1 - random_component) * grad_y * motility_strength + \
                                  random_component * random_y * motility_strength
                cell.lambdaVecZ = (1 - random_component) * grad_z * motility_strength + \
                                  random_component * random_z * motility_strength

                # Apply boundary constraints
                if not self.is_within_boundaries(cell):
                    # Reverse motility direction to push cell back into valid space
                    cell.lambdaVecX *= -1
                    cell.lambdaVecY *= -1
                    cell.lambdaVecZ *= -1


        except Exception as e:
            print(f"Exception in TumorCellMotilitySteppable at step {mcs}: {e}")
            traceback.print_exc()

    def compute_gradient(self, field, x, y, z):
        """Compute the gradient of a field at a given point."""
        try:
            dx = self.fipy_solver.dx
            nx, ny, nz = self.fipy_solver.nx, self.fipy_solver.ny, self.fipy_solver.nz

            # Partial derivatives using central differences
            grad_x = (field[z, y, min(x + 1, nx - 1)] - field[z, y, max(x - 1, 0)]) / (2 * dx)
            grad_y = (field[z, min(y + 1, ny - 1), x] - field[z, max(y - 1, 0), x]) / (2 * dx)
            grad_z = (field[min(z + 1, nz - 1), y, x] - field[max(z - 1, 0), y, x]) / (2 * dx)

            return grad_x, grad_y, grad_z

        except IndexError:
            # If indices are out of bounds, return zero gradient
            return 0.0, 0.0, 0.0

    def is_within_boundaries(self, cell):
        """Check if the cell is within the valid boundaries (space between pleuras)."""
        x_idx = int(round(cell.xCOM))
        y_idx = int(round(cell.yCOM))
        z_idx = int(round(cell.zCOM))

        # Ensure indices are within bounds
        if not (0 <= x_idx < self.dim.x and 0 <= y_idx < self.dim.y and 0 <= z_idx < self.dim.z):
            return False

        # Check if the cell is within the space between pleuras
        return self.space_between_pleuras[z_idx, y_idx, x_idx]


class FieldVisualizationSteppable(SteppableBasePy):
    """
    Creates new CC3D fields to visualize the oxygen/nutrient levels
    even when they're ~0.9999 to 1.0. One field is absolute, one is scaled.
    """

    def start(self):
        # Make new CC3D scalar fields
        self.create_scalar_field_py("Oxygen_Absolute")
        self.create_scalar_field_py("Oxygen_Scaled")
        self.create_scalar_field_py("Cell_Oxygen_Level")
        self.create_scalar_field_py("Nutrient_Absolute")
        self.create_scalar_field_py("Nutrient_Scaled")
        self.create_scalar_field_py("Cell_Nutrient_Level")

        # Grab the FiPy solver reference
        self.fipy_solver = self.shared_steppable_vars.get('fipy_solver_steppable', None)
        if self.fipy_solver is None:
            print("FieldVisualizationSteppable: No FiPy solver found. Visualization will be inactive.")

    def step(self, mcs):
        if not self.fipy_solver:
            return

        # Extract the current FiPy arrays
        nx = self.fipy_solver.nx
        ny = self.fipy_solver.ny
        nz = self.fipy_solver.nz
        oxygen_array = self.fipy_solver.oxygen_array.reshape((nz, ny, nx))
        nutrient_array = self.fipy_solver.nutrient_array.reshape((nz, ny, nx))

        # Compute min & max to do scaling
        o2_min, o2_max = oxygen_array.min(), oxygen_array.max()
        nut_min, nut_max = nutrient_array.min(), nutrient_array.max()

        # Avoid division by zero if fields are uniform
        eps = 1e-12
        o2_range = max(o2_max - o2_min, eps)
        nut_range = max(nut_max - nut_min, eps)

        # Convert FiPy (ix, iy, iz) to CC3D global (x, y, z)
        origin_x = int(self.fipy_solver.origin_x / self.fipy_solver.lattice_unit_size_mm)
        origin_y = int(self.fipy_solver.origin_y / self.fipy_solver.lattice_unit_size_mm)
        origin_z = int(self.fipy_solver.origin_z / self.fipy_solver.lattice_unit_size_mm)

        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    global_x = origin_x + ix
                    global_y = origin_y + iy
                    global_z = origin_z + iz

                    if (0 <= global_x < self.dim.x and
                            0 <= global_y < self.dim.y and
                            0 <= global_z < self.dim.z):
                        # Raw values
                        o2_val = oxygen_array[iz, iy, ix]
                        nut_val = nutrient_array[iz, iy, ix]

                        # Scaled to 0..1
                        o2_scaled = (o2_val - o2_min) / o2_range
                        nut_scaled = (nut_val - nut_min) / nut_range

                        # Store in the CC3D fields
                        self.field.Oxygen_Absolute[global_x, global_y, global_z] = o2_val
                        self.field.Nutrient_Absolute[global_x, global_y, global_z] = nut_val
                        self.field.Oxygen_Scaled[global_x, global_y, global_z] = o2_scaled
                        self.field.Nutrient_Scaled[global_x, global_y, global_z] = nut_scaled

            # Map oxygen levels to each cell
            for cell in self.cell_list:
                # Calculate cell's position in the FiPy field
                x = int((
                                cell.xCOM * self.fipy_solver.lattice_unit_size_mm - self.fipy_solver.origin_x) / self.fipy_solver.dx)
                y = int((
                                cell.yCOM * self.fipy_solver.lattice_unit_size_mm - self.fipy_solver.origin_y) / self.fipy_solver.dx)
                z = int((
                                cell.zCOM * self.fipy_solver.lattice_unit_size_mm - self.fipy_solver.origin_z) / self.fipy_solver.dx)

                # Ensure indices are within bounds
                if 0 <= x < nx and 0 <= y < ny and 0 <= z < nz:
                    # Get the oxygen value at the cell's position
                    cell_oxygen = oxygen_array[z, y, x]
                    cell_nutrient = nutrient_array[z, y, x]

                    # Map the oxygen value to the scalar field at the cell's position
                    self.field.Cell_Oxygen_Level[
                        int(round(cell.xCOM)), int(round(cell.yCOM)), int(round(cell.zCOM))] = cell_oxygen
                    self.field.Cell_Nutrient_Level[
                        int(round(cell.xCOM)), int(round(cell.yCOM)), int(round(cell.zCOM))] = cell_nutrient

        # Debug print every so often
        if mcs % 50 == 0:
            print(f"[FieldVisualizationSteppable] MCS={mcs} O2 range=({o2_min:.6f},{o2_max:.6f}), "
                  f"Nutrient range=({nut_min:.6f},{nut_max:.6f})")



class TumorConnectivitySteppable(SteppableBasePy):
    def start(self):
        # Initialize connectivity for existing tumor cells
        for cell in self.cell_list_by_type(self.EPITHELIOIDTUMOR, self.SARCOMATOIDTUMOR):
            cell.connectivityOn = True

class PleuralPressureVisualizationSteppable(SteppableBasePy):
    """
    Creates a CC3D scalar field visualizing the mechanical pressure on pleural cells.
    Pressure is defined as lambdaVolume * (actual volume - targetVolume).
    """

    def start(self):
        # Create a new scalar field for visualization
        self.create_scalar_field_py("Pleural_Pressure")

    def step(self, mcs):
        # Zero the field at each step
        self.field.Pleural_Pressure[:, :, :] = 0.0

        # Track max pressure for normalization (optional)
        max_abs_pressure = 1e-8

        # Loop over all pleural cells (visceral and parietal)
        for cell in self.cell_list_by_type(cv.VISCERALPLEURA, cv.PARIETALPLEURA):
            # Mechanical pressure: lambdaVolume * (actual volume - targetVolume)
            pressure = cell.lambdaVolume * (cell.volume - cell.targetVolume)
            max_abs_pressure = max(max_abs_pressure, abs(pressure))

            # Assign this pressure to all voxels occupied by the cell
            for ptd in self.get_cell_pixel_list(cell):
                x = ptd.pixel.x
                y = ptd.pixel.y
                z = ptd.pixel.z
                self.field.Pleural_Pressure[x, y, z] = pressure

        # Optional: Print min/max for debug
        if mcs % 50 == 0:
            min_p = np.min(self.field.Pleural_Pressure)
            max_p = np.max(self.field.Pleural_Pressure)
            print(f"[PleuralPressureVisualizationSteppable] MCS={mcs} Pleural_Pressure range: {min_p:} to {max_p:}")