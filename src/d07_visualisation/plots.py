# plot functions
import itertools

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.d00_utils.TaskObjects import TaskObjects
from src.d03_processing.aoi import collision_sphere_radius


def on_off_plot(tps, cols, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    # multipliers = list(range(1, len(cols) + 1))
    adders = [float(i) * 1.5 for i in range(len(cols))]
    t = tps.eye_timestamp_ms.to_numpy()
    print(tps.viewing_id[0])
    t = t - t[0]
    colors = itertools.cycle(["r", "b", "g", "c", "m"])

    for i in range(len(cols)):
        y = np.array(tps[cols[i]].to_numpy(), dtype=float)
        y += float(adders[i])
        ax.plot(t, y, color=next(colors))

    ax.set_yticks(adders, cols)


def get_center(obj, tps, y_adjust=0):
    # display(tps.iloc[:2, :].style)
    row = tps[tps.gaze_object == obj].reset_index(drop=True).iloc[0, :]
    center = (row.object_position_x, row.object_position_z, row.object_position_y + y_adjust)
    # print(f"{obj}: {row.gaze_object} {center}")
    return np.array(center)


def viewing_plot(tps, array_objects, gp_lab='table', gp_line=False, ax=None, alpha=0.5, camera_pos=(0, 1, 1.65),
                 table_centre=(0, 0, 0.7)):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    camera_pos = np.array(camera_pos) if tps is None else np.array([np.nanmean(tps.camera_x), np.nanmean(tps.camera_z), np.nanmean(tps.camera_y)])
    print(camera_pos)
    # plot gaze points as points
    if tps is not None:
        x, y, z = tps.gaze_collision_x.to_numpy(), tps.gaze_collision_y.to_numpy(), tps.gaze_collision_z.to_numpy()
        lab = tps.gaze_object == 'Table' if gp_lab == 'table' else np.array(tps[gp_lab].to_numpy(), dtype=bool)
        no_lab = np.invert(lab)
        # print(lab)
        # print(no_lab)
        ax.scatter(x[lab], z[lab], y[lab], c='red', s=2, alpha=1)
        ax.scatter(x[no_lab], z[no_lab], y[no_lab], c='g', s=2, alpha=alpha)
        if gp_line:
            ax.plot(x, z, y, c='k', linewidth=0.4, alpha=alpha)

    # draw a cylinder to represent a table
    table_y_scale = 0.06538646
    table_center = np.array(table_centre) if tps is None else get_center('Table', tps)  # center of the cylinder
    print(table_center)
    camera_vec = table_center - camera_pos
    print(camera_vec)
    # print(f"table: {table_center}")
    radius = 0.55  # radius of the cylinder
    height = table_y_scale * 2  # height of the cylinder
    resolution = 20

    # Create the cylinder mesh
    z = np.linspace(table_center[2] - height / 2, table_center[2] + height / 2, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + table_center[0]
    y_grid = radius * np.sin(theta_grid) + table_center[1]

    # Create the Poly3DCollection
    verts = []
    for i in range(len(z) - 1):
        verts += [list(zip(x_grid[i], y_grid[i], z_grid[i])),
                  list(zip(x_grid[i + 1], y_grid[i + 1], z_grid[i + 1])),
                  list(zip(x_grid[i + 1], y_grid[i + 1], z_grid[i])),
                  list(zip(x_grid[i], y_grid[i], z_grid[i + 1]))]
    table = Poly3DCollection(verts, facecolors=[1, 1, 1, 0], edgecolors=[0, 0, 0, 0.2], alpha=0.0)
    table.set_zsort('min')
    ax.add_collection3d(table)

    if tps is not None:
        for obj in array_objects:
            sphere_center = get_center(obj, tps)
            sphere_radius = 0.12 if obj == 'InvisibleObject' else collision_sphere_radius(sphere_center, camera_pos)
            sphere_center = get_center(obj, tps, sphere_radius)
            # print(f"{obj}: {sphere_center}")
            # sphere_center = get_center(obj, tps)
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            x = sphere_radius * np.outer(np.cos(u), np.sin(v)) + sphere_center[0]
            y = sphere_radius * np.outer(np.sin(u), np.sin(v)) + sphere_center[1]
            z = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + sphere_center[2]
            c = [0, 0.8, 0.8, 0.5] if obj == 'InvisibleObject' else [0, 0, 1, 0.5]
            # print(c)
            sphere = ax.plot_surface(x, y, z, color=c, alpha=0.05)
            sphere.set_edgecolor(c)

    # set axis limits and labels
    ax_len = 0.55
    ax.set_xlim([table_center[0] - ax_len, table_center[0] + ax_len])
    ax.set_ylim([table_center[1] - ax_len, table_center[1] + ax_len])
    ax.set_zlim([table_center[2] - ax_len, table_center[2] + ax_len])
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()

    azimuthal_angle = 180 + (np.arctan2(camera_vec[1], camera_vec[0]) * 180 / np.pi)
    print("azimuth: ", azimuthal_angle)
    polar_angle = 180 - (np.arccos(camera_vec[2] / np.linalg.norm(camera_vec)) * 180 / np.pi)
    print("polar: ", polar_angle)

    # Convert angles to radians
    azimuthal_angle_rad = np.deg2rad(azimuthal_angle)
    polar_angle_rad = np.deg2rad(polar_angle)
    # print(f'elev: {polar_angle}, azim: {azimuthal_angle}')
    ax.view_init(elev=polar_angle, azim=azimuthal_angle)


def fixation_plot(fix_df, tps=None, fix_range='default', ax=None, gp_lab='fixation', camera_pos=(0, 1, 1.65),
                 table_centre=(0, 0, 0.7)):
    full_objects = np.unique(fix_df.object) if tps is None else np.unique(tps.gaze_object)
    # array_objects = TaskObjects.array_and_invisible
    array_objects = ['InvisibleObject', *TaskObjects.array_objects]
    objects = full_objects[np.isin(full_objects, np.array(array_objects))]
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    viewing_plot(tps, objects, gp_lab=gp_lab, gp_line=False, ax=ax, alpha=0.2,
                 camera_pos=camera_pos, table_centre=table_centre)
    fixations = fix_df[fix_df.fixation_or_saccade == 'fixation'].reset_index(drop=True)
    size_scaler = 1
    arrow_len = 0.75
    start_fix = 0 if fix_range == 'default' else fix_range[0]
    end_fix = len(fixations) if fix_range == 'default' else fix_range[1]
    for i in range(start_fix, end_fix):
        s = size_scaler * fixations.duration_time[i]
        x, y, z = (fixations.centroid_x[i], fixations.centroid_z[i], fixations.centroid_y[i])
        ax.scatter(x, y, z, c='green', s=s, alpha=0.4)
        ax.text(x, y, z, i, color='green')
        if i > 0:
            prev_x, prev_y, prev_z = (
            fixations.centroid_x[i - 1], fixations.centroid_z[i - 1], fixations.centroid_y[i - 1])
            ax.quiver(prev_x, prev_y, prev_z,
                      arrow_len * (x - prev_x), arrow_len * (y - prev_y), arrow_len * (z - prev_z),
                      color='green', alpha=0.4, linewidth=1)
    ax.plot(fixations.centroid_x[start_fix:end_fix],
            fixations.centroid_z[start_fix:end_fix],
            fixations.centroid_y[start_fix:end_fix], 'g-', linewidth=1, alpha=0.4)