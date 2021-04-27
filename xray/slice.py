import math

import numpy as np


def manhattan_distance(p1, p2, d=2):
    assert (len(p1) == len(p2))
    all_distance = 0
    for i in range(d):
        all_distance += abs(p1[i] - p2[i])
    return all_distance


def to_intersecting_lines(mesh, height):
    # find relevant triangles
    mask = np.zeros(len(mesh))
    z_val = mesh[:, 2::3]
    above = z_val > height
    below = z_val < height
    same = z_val == height
    row_sum = same.sum(axis=1)
    mask[row_sum == 3] = 1
    mask[row_sum == 2] = 1
    mask[np.any(above, axis=1) & np.any(below, axis=1)] = 1
    # find intersecting triangles
    not_same_triangles = mesh[mask.astype(np.bool) & ~np.all(same, axis=1)].reshape(-1, 3, 3)
    # TODO: Make the following line faster
    lines = list(map(lambda tri: triangle_to_intersecting_lines(tri, height), not_same_triangles))
    return lines


def draw_line_on_pixels(p1, p2, pixels):
    line_steps = math.ceil(manhattan_distance(p1, p2))
    if line_steps == 0:
        pixels[int(p1[0]), int(p2[1])] = True
        return
    for j in range(line_steps + 1):
        point = linear_interpolation(p1, p2, j / line_steps)
        pixels[int(point[0]), int(point[1])] = True


def linear_interpolation(p1, p2, distance):
    '''
    :param p1: Point 1
    :param p2: Point 2
    :param distance: Between 0 and 1, Lower numbers return points closer to p1.
    :return: A point on the line between p1 and p2
    '''
    slopex = (p1[0] - p2[0])
    slopey = (p1[1] - p2[1])
    slopez = p1[2] - p2[2]
    return (
        p1[0] - distance * slopex,
        p1[1] - distance * slopey,
        p1[2] - distance * slopez
    )


def triangle_to_intersecting_lines(triangle, height):
    assert (len(triangle) == 3)
    above = triangle[triangle[:, 2] > height]
    below = triangle[triangle[:, 2] < height]
    same = triangle[triangle[:, 2] == height]
    assert len(same) != 3
    if len(same) == 2:
        return same[0], same[1]
    elif len(same) == 1:
        side1 = where_line_crosses_z(above[0], below[0], height)
        return side1, same[0]
    else:
        lines = []
        for a in above:
            for b in below:
                lines.append((b, a))
        side1 = where_line_crosses_z(lines[0][0], lines[0][1], height)
        side2 = where_line_crosses_z(lines[1][0], lines[1][1], height)
        return side1, side2


def where_line_crosses_z(p1, p2, z):
    if p1[2] > p2[2]:
        t = p1
        p1 = p2
        p2 = t
    # now p1 is below p2 in z
    if p2[2] == p1[2]:
        distance = 0
    else:
        distance = (z - p1[2]) / (p2[2] - p1[2])
    return linear_interpolation(p1, p2, distance)


# Depricated
def calculate_scale_shift(mesh, resolution):
    all_points = mesh.reshape(-1, 3)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    del all_points
    shift = -1 * mins
    xy_scale = float(resolution - 1) / (max(maxs[0] - mins[0], maxs[1] - mins[1]))
    # TODO: Change this to return one scale. If not, verify svx exporting still works.
    scale = [xy_scale, xy_scale, xy_scale]
    bounding_box = [resolution, resolution, math.ceil((maxs[2] - mins[2]) * xy_scale)]
    return scale, shift, bounding_box
