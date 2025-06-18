# cancer3d_variables.py
# This file contains all the simulation parameters and constants used in the CANCER 3D simulation.
# Tweak these values as needed without modifying the main simulation code.

# Simulation geometry and time step
LATTICE_UNIT_SIZE_MM = 1.88
DX = LATTICE_UNIT_SIZE_MM  # grid spacing in mm
DT = 1.0                  # time step
CORES = 4

# Cell type identifiers
MEDIUM = 0
EPITHELIOIDTUMOR = 1
SARCOMATOIDTUMOR = 2
VISCERALPLEURA = 3
PARIETALPLEURA = 4
M2MACROPHAGE = 5
CAF = 6
FIBROBLAST = 7
BONE = 8

#cell initialization
FIBROBLAST_COUNT = 2
NUM_CAF = 1
NUM_M2 = 2
# Diffusion coefficients for PDEs
D_OXYGEN = 6.0
D_NUTRIENT = 17.0
D_IL8 = 100.0
D_IL6 = 100.0

# Mass constants (ng per cell)
MASS_EPITHELIOID_CELL_NG = 2.0   # Approximate average mass of epithelioid tumor cell
MASS_SARCOMATOID_CELL_NG = 1.5   # Approximate average mass of sarcomatoid tumor cell



SIM_PARAMS = {
    "D_IL8": 1e-5,         # Diffusion coefficient for IL8 [mm²/s] (range: 1e-6 to 1e-5)
    "D_IL6": 1e-5,         # Diffusion coefficient for IL6 [mm²/s] (range: 1e-6 to 1e-5)
    "D_nutrient": 1e-3,    # Diffusion coefficient for nutrient [mm²/s] (range: 0.5e-3 to 2e-3)
    "D_oxygen": 2e-3,      # Diffusion coefficient for oxygen [mm²/s] (range: 1e-3 to 3e-3)
    "D_VEGF": 1.2e-6,  # mm²/s (rounded from 1.16e-6 for simplicity)

    "mu_IL8": 0.01,        # Decay/uptake rate for IL8 [min⁻¹] (range: 0.005 to 0.01)
    "mu_IL6": 0.01,        # Decay/uptake rate for IL6 [min⁻¹] (range: 0.005 to 0.01)
    "mu_nutrient": 0.01,   # Consumption rate for nutrient [min⁻¹] (range: 0.005 to 0.02)
    "mu_oxygen": 0.01,     # Consumption rate for oxygen [min⁻¹] (range: 0.005 to 0.02)
    "mu_VEGF": 0.03,    # Decay/uptake rate for VEGF [min⁻¹]
    "boundaryat": 2.0,     # Boundary threshold (model-specific)
    "dx": 1.88,             # Spatial discretization [mm] (each lattice site is 1.88 mm)
    "dt": 1.0,             # Time step [min] (1 minute per timestep)
    "origin_x": 0.0,       # Domain origin
    "origin_y": 0.0,
    "origin_z": 0.0,
    "k_oxygen_epi": 0.01,  # Oxygen consumption rate for epithelioid tumor cells (range: 0.005 to 0.02)
    "k_oxygen_sarc": 0.02, # Oxygen consumption rate for sarcomatoid tumor cells (range: 0.01 to 0.03)
    "k_nutrient_epi": 0.01,# Nutrient consumption rate for epithelioid tumor cells (range: 0.005 to 0.02)
    "k_nutrient_sarc": 0.02# Nutrient consumption rate for sarcomatoid tumor cells (range: 0.01 to 0.03)
}

# Growth rates for tumor cells
GROWTH_RATES = {
    EPITHELIOIDTUMOR: 0.00288,
    SARCOMATOIDTUMOR: 0.0030,
}

#min growth requirement
MIN_OXYGEN = 0.2
MIN_NUTRIENT =0.2
# Apoptosis thresholds
CRITICAL_OXYGEN = 0.02  # critical oxygen level
CRITICAL_NUTRIENT = 0.02  # critical nutrient level

STARVATION_TARGETV = 0.002
IDEAL_GROWTH = 0.00056
#MITOSIS CONS COST
MITOSIS_CONS_COST = 0.10
# Hypoxia response parameters
HYPOXIA_OXYGEN_THRESHOLD = 0.01
HYPOXIA_TRANSFORMATION_PROB = 0.15  # probability of switching to sarcomatoid under hypoxia

# Transformation threshold for Fibroblast-to-CAF conversion
TGFbeta_THRESHOLD = 1.0

# Chemotaxis parameters
IL6_CHEMOTAXIS_LAMBDA = 5.0  # chemotaxis strength towards IL6
IL8_CHEMOTAXIS_LAMBDA = 3.0  # chemotaxis strength towards IL8
TGFB_CHEMOTAXIS_LAMBDA = 8.0 # chemotaxis strength towards TGF-beta

#Chemotaxis decay strength
IL6_CHEMOTAXIS_DECAY = 0.001
IL8_CHEMOTAXIS_DECAY = 0.005
TGFB_CHEMOTAXIS_DECAY = 0.002

# Angiogenesis parameters
VEGF_SECRETION_RATE = 0.20
OXY_SECRETION_RATE = 0.20

VEGF_THRESHOLD = 0.6       # threshold to boost oxygen/nutrient levels
OXYGEN_BOOST = 0.15        # oxygen boost per step when VEGF threshold exceeded
NUTRIENT_BOOST = 0.15      # nutrient boost per step when VEGF threshold exceeded

MOTILITY_STRENGTH_EPI = 3.0
MOTILITY_STRENGTH_SARC = 6.0
