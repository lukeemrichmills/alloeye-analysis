import enum

from src.d00_utils.EnumFuncWrapper import FuncWrapper
from src.d03_processing.fixations.GazeCollision import GazeCollision
from src.d03_processing.fixations.I_VDT import I_VDT
from src.d03_processing.fixations.VR_IDT import VR_IDT
from src.d03_processing.fixations.ClusterFix import ClusterFix
from src.d03_processing.fixations.I_HMM import I_HMM



class FixAlgo(enum.Enum):
    GazeCollision = FuncWrapper(GazeCollision)
    I_VDT = FuncWrapper(I_VDT)
    VR_IDT = FuncWrapper(VR_IDT)
    ClusterFix = FuncWrapper(ClusterFix)
    I_HMM = FuncWrapper(I_HMM)

def fix_algo_dict():
    return {i.name: i.value for i in FixAlgo}
