#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Leon Sütfeld
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon

def read_coordinates(file_path):
    """
    Reads a text file containing x, y, z coordinates separated by spaces and tabs.

    :param file_path: Path to the text file.
    :return: NumPy array of shape (N, 3) with float coordinates.
    """
    return np.loadtxt(file_path, delimiter=None)  # delimiter=None automatically handles spaces & tabs


def move_point(coords_in, direction, distance):
    """
    Moves a point in a given direction and distance.

    :param coords_in: Initial x- and  y-coordinates.
    :param direction: Direction in degrees (0 is up, 90 is right, 180 is down, 270 is left).
    :param distance: Distance to move.
    :return: New (x, y) coordinates.
    """

    x, y = coords_in
    rad = math.radians(direction)
    new_x = x + distance * -math.sin(rad)
    new_y = y + distance * -math.cos(rad)
    return new_x, new_y


def fixdir(deg):
    """
    Sanitizes any degree number into the corresponding degree number between 0 and 360.

    Args:
        deg: A degree number

    Returns:
        deg: A degree number between 0 and 360
    """

    if deg > 360:
        deg = deg % 360
    elif deg < 0:
        deg = deg % -360
        deg = deg + 360

    return deg


def create_wakefield(pt, direction, length, base_width, angle):
    """
    Creates a trapezoid extending from a point, in a defined direction.

    Args:
        pt: location of turbine
        direction: wind direction
        length: length of wake influence
        base_width: starting width for wake
        angle: wake angle

    Returns: list: pt, endpoint (central), base points, far points

    """

    endpt = move_point(pt, direction, length)

    # base points
    if base_width > 0:
        basept_1 = move_point(pt, fixdir(direction+90), base_width/2)
        basept_2 = move_point(pt, fixdir(direction-90), base_width/2)
    else:
        basept_1 = pt.copy()
        basept_2 = pt.copy()

    # far points
    angle = fixdir(angle)
    far_width = np.tan(np.deg2rad(angle))*length*2 + base_width
    farpt_1 = move_point(endpt, fixdir(direction+90), far_width/2)
    farpt_2 = move_point(endpt, fixdir(direction-90), far_width/2)

    return [basept_1, basept_2, farpt_2, farpt_1]


def get_wakefields(pts, direction, length, base_width, angle):
    """
    Creates wakefields for a number of points.

    Args:
        pts: list of turbine coordinates (x, y)
        direction: wind direction
        length: length of influence
        base_width: wake shape
        angle: wake shape

    Returns: list of wakefields

    """

    wakefields = []
    for pt in pts:
        wakefields.append(create_wakefield(pt, direction, length, base_width, angle))

    return wakefields


def is_point_in_area(pt, area_points):
    """
    Checks if a point (x, y) lies within an area defined by four other points.

    :param x: x-coordinate of the point.
    :param y: y-coordinate of the point.
    :param area_points: List or array of four (x, y) coordinate tuples defining the area.
    :return: True if the point is inside the area, False otherwise.
    """
    x, y = pt
    path = Path(area_points)
    return path.contains_point((x, y))


def get_wake_matrix(turbine_coords, wakefields):

    wake_matrix = np.zeros((len(turbine_coords), len(wakefields)))
    for i, turbine in enumerate(turbine_coords):
        for j, wakefield in enumerate(wakefields):
            if is_point_in_area(turbine, wakefield) and not i == j:
                wake_matrix[i, j] = 1

    return wake_matrix


def get_wake_impact_list(wake_matrix):
    wake_impact_list = []
    for r in range(wake_matrix.shape[0]):
        row = wake_matrix[r, :]
        indices = np.nonzero(row)[0]
        wake_impact_list.append(indices.tolist())

    return wake_impact_list


def plot_2d_coordinates(coords, title=None, xlabel=None, ylabel=None):
    """
    Plots 2D coordinates as dots on a canvas.

    :param coords: NumPy array of shape (N, 2) with x and y coordinates.
    """
    coords = np.array(coords)
    plt.scatter(coords[:, 0], coords[:, 1], marker='.', color='black')
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.show()


def plot_wakefields(coords, wakefields, wake_matrix, title=None, xlabel=None, ylabel=None, fsize=[8, 5]):
    """
    Plots 2D coordinates as dots on a canvas.

    :param coords: NumPy array of shape (N, 2) with x and y coordinates.
    """

    plt.figure(figsize=fsize)

    if wakefields:
        for i, field in enumerate(wakefields):
            polygon = Polygon(field, color='gray', alpha=0.3)
            plt.gca().add_patch(polygon)

    for i, (x, y) in enumerate(coords):
        offset = 30
        plt.text(x-offset, y-offset, str(i), fontsize=7, ha='right', va='top', color='blue')

    wake_affected_turbines = np.clip(np.sum(wake_matrix, 1), 0, 1)
    coords = np.array(coords)
    for i in range(coords.shape[0]):
        if wake_affected_turbines[i] == 0:
            plt.scatter(coords[i, 0], coords[i, 1], marker='.', color='black')
        else:
            plt.scatter(coords[i, 0], coords[i, 1], marker='.', color='red')

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)

    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()

def get_distance_matrix(coords):

    dist_matrix = np.zeros((len(coords), len(coords)))
    for i, coord_source in enumerate(coords):
        for j, coord_target in enumerate(coords):
            if not i == j:
                dist_matrix[i, j] = np.linalg.norm(coord_source - coord_target)

    return dist_matrix

def get_direction_matrix(coords):
    """
    Computes the direction angles between all coordinate pairs relative to 0° = up.

    :param coords: List of numpy arrays of shape (2,)
    :return: A matrix (numpy array) of angles in degrees
    """
    n = len(coords)
    angles = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                dx = coords[j][0] - coords[i][0]
                dy = coords[j][1] - coords[i][1]
                angle = np.degrees(np.arctan2(dx, dy))  # 0° = up
                angles[i, j] = angle

    return angles
