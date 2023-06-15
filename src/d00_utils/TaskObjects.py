class TaskObjects:
    min_aoi_radius = 0.08
    max_aoi_radius = 0.1
    tabletop = 0.7
    floor = 'FloorFocusArea'
    dome = 'OcclusionDome'
    instructions = ['FirstInstructionsCanvas', 'SecondInstructionsCanvas', 'WalkWarningCanvas']
    invisible_object = 'InvisibleObject'
    distant_objects = ['Well', 'RockLarge', 'House', 'Tree',
                       'RockZ lod_1', 'RockZ lod_0', 'RockZ lod_2', 'DeadTrees']
    ext_cues = ['RockClose lod_0', 'RockClose lod_1', 'RockClose lod_2',
                'Log_LOD0', 'Log_LOD1', 'Log_LOD2']
    array_objects = ['5Ball', 'AlarmClock', 'Apple', 'Box', 'Candle', 'Crown',
                     'Cup', 'Donut', 'Duck', 'Grapes', 'Helmet',
                     'Lemon', 'Pipe', 'Plane', 'Shoe', 'Spray',
                     'Stapler', 'Tape', 'Teapot', 'Tomato', 'Truck']
    off_table = [floor, dome, *instructions, *distant_objects, *ext_cues]
    array_and_invisible = [invisible_object, *array_objects]
    on_table = ['Table', *array_and_invisible]
    standard_aois = ['External', 'Table', 'Moved', 'Obj2', 'Obj3', 'Obj4', 'Previous']
    standard_aoi_chars = {
        'External': 'x',
        'Table': 't',
        'Moved': 'm',
        'Previous': 'p',
        'Obj2': 'b',
        'Obj3': 'c',
        'Obj4': 'd'
    }


    lossy_scale_dict = {
        'Table': (0.55, 0.06538646, 0.55),
        'ArrayObjectParent': (1, 0.05, 1)   # need to check this
    }

    local_scale_dict = {
        '5Ball': (2.109091, 17.74068, 2.109091),
        'AlarmClock': (1.463642, 12.31142, 1.463642),
        'Apple': (1.672727, 14.07019, 1.672727),
        'Box': (0.6516335, 9.037549, 0.6343784),
        'Candle': (1.500001, 10.06172, 1.500001),
        'Crown': (1.454546, 12.23495, 1.454546),
        'Cup': (1.454545, 12.23495, 1.454545),
        'Donut': (1.545455, 12.99963, 1.545455),
        'Duck': (1.272728, 12.23495, 2.181819),
        'Grapes': (1.2,	7.646843, 1.2),
        'Helmet': (1.254546, 10.55264, 1.254546),
        'Lemon': (1.672727, 14.0702, 1.672727),
        'Pipe': (1.989274, 15.29369, 2.25582),
        'Plane': (1.259716, 16.67012, 1.648418),
        'Shoe': (0.866422, 7.287933, 0.9671868),
        'Spray': (1.504544, 10.45705, 1.862661),
        'Stapler': (1.09091, 10.70558, 1.454546),
        'Tape': (1.740707, 14.64199, 3.474312),
        'Teapot': (1.185003, 9.967682, 1.185003),
        'Tomato': (2.090909, 17.58775, 2.090909),
        'Truck': (2.590172, 14.28824, 1.562945),
        'InvisibleObject': (0.2727273, 1.529369, 0.2727273)
    }

    collider_y_offset = {       # IMPORTANT: this is offset from tabletop - everything but 5Ball will have position at tabletop when there
        '5Ball': 0.03,                  # 5Ball y position is centre of ball, collider is the same, position is adjusted up
        'AlarmClock': 0.05979824,
        'Apple': 0.04134846,
        'Box': 0.03112698,
        'Candle': 0.03289413,
        'Crown': 0.03391361,
        'Cup': 0.02536201,
        'Donut': 0.01404858,
        'Duck': 0.04053783,
        'Grapes': 0.02910519,
        'Helmet':  0.04826736,
        'Lemon': 0.03064823,
        'Pipe': 0.0422287,
        'Plane': 0.03458786,
        'Shoe':  0.04234028,
        'Spray':  0.04102516,
        'Stapler': 0.02948856,
        'Tape': 0.03571987,
        'Teapot': 0.04101372,
        'Tomato':  0.03482437,
        'Truck': 0.03144741,
        'InvisibleObject': 0
    }

    viewpoint_from_table_centre = 1   # m
    table_border_width = 0.12   # m
    array_obj_max_distance_from_table = viewpoint_from_table_centre + \
        lossy_scale_dict['Table'][0] - table_border_width
