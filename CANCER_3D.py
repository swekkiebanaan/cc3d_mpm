from cc3d import CompuCellSetup
from CANCER_3DSteppables import FiPySolverSteppable
from CANCER_3DSteppables import CellInitializerSteppable
from CANCER_3DSteppables import CellGrowthSteppable
from CANCER_3DSteppables import TumorCellMitosisSteppable
from CANCER_3DSteppables import ApoptosisSteppable
from CANCER_3DSteppables import HypoxiaResponseSteppable
from CANCER_3DSteppables import FibroblastToCAFTransformationSteppable
from CANCER_3DSteppables import FiPyToCC3DFieldSynchronizationSteppable
from CANCER_3DSteppables import ChemotaxisSteppable
from CANCER_3DSteppables import AngiogenesisSteppable
from CANCER_3DSteppables import PleuraBoundarySteppable
from CANCER_3DSteppables import FPPPleuraSetupSteppable
from CANCER_3DSteppables import TumorCellMotilitySteppable
from CANCER_3DSteppables import FieldVisualizationSteppable
from CANCER_3DSteppables import TumorConnectivitySteppable
from CANCER_3DSteppables import PleuralPressureVisualizationSteppable

CompuCellSetup.register_steppable(steppable=CellInitializerSteppable(frequency=1))
#CompuCellSetup.register_steppable(steppable=TumorConnectivitySteppable(frequency=6))
CompuCellSetup.register_steppable(steppable=FiPySolverSteppable(frequency=24))
CompuCellSetup.register_steppable(steppable=FPPPleuraSetupSteppable(frequency=1))
CompuCellSetup.register_steppable(steppable=FiPyToCC3DFieldSynchronizationSteppable(frequency=24))
CompuCellSetup.register_steppable(steppable=TumorCellMotilitySteppable(frequency=12))
CompuCellSetup.register_steppable(steppable=CellGrowthSteppable(frequency=48))
CompuCellSetup.register_steppable(steppable=TumorCellMitosisSteppable(frequency=48))
CompuCellSetup.register_steppable(steppable=ApoptosisSteppable(frequency=12))
CompuCellSetup.register_steppable(steppable=HypoxiaResponseSteppable(frequency=12))
CompuCellSetup.register_steppable(steppable=ChemotaxisSteppable(frequency=12))
CompuCellSetup.register_steppable(steppable =FibroblastToCAFTransformationSteppable(frequency=12))
CompuCellSetup.register_steppable(steppable =AngiogenesisSteppable(frequency=12))
#CompuCellSetup.register_steppable(steppable=PleuraBoundarySteppable(frequency=2))
CompuCellSetup.register_steppable(steppable=PleuralPressureVisualizationSteppable(frequency=6))
#CompuCellSetup.register_steppable(steppable=FieldVisualizationSteppable(frequency=50))

try:
    CompuCellSetup.run()
except Exception as e:
    print(e)
