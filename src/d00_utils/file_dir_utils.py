import platform

from src.d01_data.database.Errors import UnknownComputer

icn_pc = "DESKTOP-IBAG161"
personal_laptop = "LAPTOP-OOBPQ1A8"


def get_data_dir(folder="", from_ehd=False):
    if folder != "":
        folder = folder + "\\"

    if platform.uname().node == icn_pc:
        return f"C:\\Users\\Luke Emrich-Mills\\Documents\\AlloEye\\MainDataOutput\\{folder}"
    elif platform.uname().node == personal_laptop:
        if from_ehd:
            return f"E:\\Luke\\AlloEye\\Data\\VR\\upload\\{folder}"
        else:
            return f"C:\\Users\\Luke\\Documents\\AlloEye\\data\\{folder}"

    else:
        raise UnknownComputer


def get_doc_dir(folder=""):
    if folder != "":
        folder = folder + "\\"

    if platform.uname().node == icn_pc:
        return f"C:\\Users\\Luke Emrich-Mills\\OneDrive\\Documents\\PhD\\AlloEye\\{folder}"
    elif platform.uname().node == personal_laptop:
        return f"C:\\Users\\Luke\\OneDrive\\Documents\\PhD\\AlloEye\\{folder}"
    else:
        raise UnknownComputer
