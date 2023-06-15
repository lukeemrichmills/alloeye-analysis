from src.d00_utils.condition_builders import *


class Conditions:
    list = ['StayStill', 'WalkStill', 'TeleportStill',
                         'StayRotate', 'WalkRotate', 'TeleportRotate']
    all = construct_dict(['Stay', 'Walk', 'Teleport'],
                         ['false', 'true'])
    stay = MoveType('Stay')
    walk = MoveType('Walk')
    teleport = MoveType('Teleport')
    still = TableRotation('false')
    rotate = TableRotation('true')

    def __init__(self):
        pass
