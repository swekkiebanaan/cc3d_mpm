# CANCER_3D: 3D Multiscale Simulation of Pleural Tumor Growth

## **Overview**

CANCER_3D is a high-performance, 3D multiscale simulation framework for modeling tumor growth and microenvironmental interactions in the pleural cavity. Built on [CompuCell3D (CC3D)](https://compucell3d.org/) and [FiPy](https://www.ctcms.nist.gov/fipy/), it integrates agent-based cell modeling with parallelized PDE solvers to capture the spatial and biochemical complexity of cancer progression, including cell behavior, nutrient/oxygen diffusion, cytokine signaling, angiogenesis, and stromal transformation.

---

## **Key Features**

- **3D Agent-Based Model:** Simulates multiple cell types (tumor, pleura, immune, stromal, bone) in a realistic pleural geometry.
- **Parallel PDE Solvers:** Uses FiPy (with MPI) for scalable, high-resolution diffusion-reaction modeling of oxygen, nutrients, cytokines, and growth factors.
- **Dynamic Tumor Microenvironment:** Models hypoxia, apoptosis, chemotaxis, angiogenesis, fibroblast activation, and cell motility.
- **Data-Driven Initialization:** Imports patient-derived masks (lung, tumor, bone) for anatomically accurate simulations.
- **System Monitoring & Visualization:** Tracks resource usage and provides 2D/3D visualizations of fields and cell distributions.

---

## **Project Structure**

| File/Folder                | Purpose                                                                                  |
|---------------------------|------------------------------------------------------------------------------------------|
| `CANCER_3D.py`            | Main simulation entry point (registers and runs all steppables)                         |
| `CANCER_3DSteppables.py`  | All custom steppables (cell behaviors, PDE coupling, visualization, etc.)               |
| `cancer3d_variables.py`   | Centralized simulation parameters and constants                                         |
| `HPCFiPyScript.py`        | Parallel FiPy PDE solver script (invoked via MPI)                                       |
| `Simulation/*.npz`        | Mask files for lung, tumor, bone, etc.                                                  |
| `ramdisk/`                | Temporary storage for PDE fields and mesh exchange                                      |
| `visualizations/`         | Output folder for generated plots and images                                            |
| `*.xml`                   | CC3D XML configuration (cell types, plugins, field definitions)                         |

---

## **Installation & Requirements**

- Python 3.10+
- [CompuCell3D 4.6+](https://compucell3d.org/)
- [FiPy](https://www.ctcms.nist.gov/fipy/)
- [mpi4py](https://mpi4py.readthedocs.io/)
- [petsc4py](https://petsc4py.readthedocs.io/)
- numpy, scipy, matplotlib, h5py, nibabel, SimpleITK, skimage, psutil, pyvista (optional for 3D plots)

**Note:** The simulation is designed for Linux environments with MPI support and sufficient RAM/CPU resources for large 3D domains.

---

## **How to Run**

1. **Edit Parameters:** Adjust simulation parameters, cell counts, diffusion rates, etc., in `cancer3d_variables.py` as needed[1].
2. **Prepare Mask Files:** Ensure required `.npz` mask files (lung, tumor, bone) are present in the `Simulation/` directory.
3. **Launch Simulation:**
   ```bash
   runScript CANCER_3D.py
   ```
   or within the CC3D GUI, open the project and run.

4. **Parallel PDE Solving:** The simulation will automatically invoke `HPCFiPyScript.py` via `mpiexec` for parallel PDE solving[2][3].
5. **Outputs:** Visualization images, monitoring data, and simulation logs will be saved in the specified output folders.

---

## **Simulation Workflow**

### **1. Initialization**

- **Cell Placement:** The `CellInitializerSteppable` loads and resamples anatomical masks, then populates the domain with tumor, pleura, bone, immune, and stromal cells according to mask geometry and parameterized counts[2].
- **Field Initialization:** Scalar fields for lung, tumor, bone, and pleural space are created for visualization and analysis.
- **Pre-equilibration:** Optionally equilibrates oxygen/nutrient fields using the HPC FiPy script before cell dynamics begin.

### **2. Main Simulation Loop**

#### **Cellular Dynamics (CC3D Steppables)**

- **Growth & Division:** Tumor cells grow based on local oxygen/nutrient/cytokine availability and divide when reaching a threshold volume (`CellGrowthSteppable`, `TumorCellMitosisSteppable`)[2][1].
- **Apoptosis:** Cells undergo apoptosis if starved of critical resources (`ApoptosisSteppable`)[2][1].
- **Hypoxia Response:** Hypoxic tumor cells may upregulate VEGF and switch phenotype (epithelioid → sarcomatoid) with a defined probability (`HypoxiaResponseSteppable`)[2][1].
- **Motility & Chemotaxis:** Tumor and stromal cells move in response to gradients of oxygen, nutrients, and cytokines (`TumorCellMotilitySteppable`, `ChemotaxisSteppable`)[2][1].
- **Fibroblast Activation:** Fibroblasts transform into CAFs in response to TGF-beta (`FibroblastToCAFTransformationSteppable`)[2][1].
- **Angiogenesis:** VEGF secretion triggers local increases in oxygen/nutrient fields, modeling vessel recruitment (`AngiogenesisSteppable`)[2][1].
- **Pleura Mechanics:** Specialized steppables maintain pleural membrane integrity using Focal Point Plasticity links (`FPPPleuraSetupSteppable`)[2].

#### **PDE Coupling (FiPy)**

- **Field Exchange:** At defined intervals, the `FiPySolverSteppable`:
  - Gathers cell consumption/secretion data.
  - Exports mesh and fields to disk.
  - Calls `HPCFiPyScript.py` via MPI to solve 3D PDEs for oxygen, nutrient, IL-6, IL-8, VEGF, etc., using up-to-date cell presence and pleural masks[2][3].
  - Imports updated fields back into CC3D and synchronizes them for use by all steppables[2].

#### **Visualization & Monitoring**

- **2D/3D Plots:** Generates contour and volume renderings of fields and cell distributions at user-defined intervals[2].
- **Resource Monitoring:** Tracks CPU, memory, and system load, saving plots for performance analysis (`SystemMonitor`)[2].
- **Data Logging:** Tumor volume, cell counts, and apoptosis statistics are logged for downstream analysis[2].

---

## **Parameterization**

All key simulation constants are defined in `cancer3d_variables.py` for easy tuning[1]. These include:

- **Cell type IDs and counts**
- **Diffusion and decay coefficients for all fields**
- **Cell growth rates, division thresholds, and apoptosis criteria**
- **Chemotaxis strengths and decay rates**
- **Angiogenesis and transformation thresholds**
- **Motility strengths**

---

## **Extending the Model**

- **Add New Cell Types:** Define new IDs and properties in `cancer3d_variables.py`, update XML, and implement behaviors in `CANCER_3DSteppables.py`.
- **Modify PDEs:** Adjust or add new fields in both the CC3D XML and FiPy solver scripts.
- **Custom Analysis:** Implement additional steppables for new biological processes or data collection.

---

## **Troubleshooting**

- **Missing Mask Files:** Ensure all required `.npz` mask files are present and paths are correct.
- **PDE Solver Errors:** Check that MPI and petsc4py are installed and the correct number of cores is set in `cancer3d_variables.py` and the CC3D XML.
- **Performance:** For large domains, run on a multi-core server with sufficient RAM. Adjust `CORES` and domain size as needed[1][4].
- **Visualization Issues:** Install `pyvista` for advanced 3D plotting; otherwise, only 2D plots will be generated.

---

## **References**

- [CompuCell3D Documentation](https://compucell3d.org/)
- [FiPy PDE Solver](https://www.ctcms.nist.gov/fipy/)
- [MPI for Python (mpi4py)](https://mpi4py.readthedocs.io/)
- [petsc4py Documentation](https://petsc4py.readthedocs.io/)

---
