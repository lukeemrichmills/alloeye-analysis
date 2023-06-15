def construct_dict(move_list, rotation_list):
    return {'table_rotates': rotation_list,
            'move_type': move_list}


class MoveType:
    def __init__(self, string):
        self.string = string
        self.all = self.construct_dict(['false', 'true'])
        self.still = self.construct_dict(['false'])
        self.rotate = self.construct_dict(['true'])

    def construct_dict(self, rotation_list):
        return construct_dict([self.string], rotation_list)


class TableRotation:
    def __init__(self, string):
        self.string = string
        self.all = self.construct_dict(['Stay', 'Walk', 'Teleport'])
        self.stay = self.construct_dict(['Stay'])
        self.walk = self.construct_dict(['Walk'])
        self.teleport = self.construct_dict(['Teleport'])

    def construct_dict(self, move_list):
        return construct_dict(move_list, [self.string])

