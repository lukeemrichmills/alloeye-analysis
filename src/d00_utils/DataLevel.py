import enum


class DataLevel(enum.Enum):
    timepoint = "timepoint"
    fix_sacc = "fix_sacc"
    viewing = "viewing"
    trial = "trial"
    block = "block"
    condition = "condition"
    participant = "participant"
    group = "group"
