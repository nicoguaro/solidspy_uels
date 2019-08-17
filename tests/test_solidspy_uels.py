#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `solidspy_uels` package."""

import pytest
import numpy as np
import scipy.linalg as LA

# SolidsPy
from solidspy import gaussutil as gau
from solidspy.femutil import jacoper
from solidspy.assemutil import DME, dense_assem, loadasem
from solidspy.postprocesor import complete_disp

# SolidsPy UELs
from solidspy_uels.solidspy_uels import elast_brick8


def test_shape_brick8():
    pass


def test_interp_mat_3d():
    pass


def test_elast_brick8():

    # One element in uniaxial load
    coords = np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1]])
    params = [1, 1/4, 1]

    pts = np.column_stack((range(0, 8), coords))
    cons = np.array([
            [-1, -1, -1],
            [0, -1, -1],
            [0, 0, -1],
            [-1, 0, -1],
            [-1, -1, 0],
            [0, -1, 0],
            [0, 0, 0],
            [-1, 0, 0]])

    els = np.array([[0, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7]])
    mats = np.array([params])
    assem_op, bc_array, neq = DME(cons, els, ndof_node=3, 
                                  ndof_el=lambda iet: 24,
                                  ndof_el_max=24)
    stiff, _ = dense_assem(els, mats, pts, neq, assem_op, uel=elast_brick8)
    loads = np.array([
            [4, 0, 0, 1],
            [5, 0, 0, 1],
            [6, 0, 0, 1],
            [7, 0, 0, 1],])
    rhs = loadasem(loads, bc_array, neq, ndof_node=3)
    sol = LA.solve(stiff, rhs)
    disp = complete_disp(bc_array, pts, sol, ndof_node=3)
    disp_comp = np.array([
        [ 0.0,  0.0,  0.0],
        [-0.5,  0.0,  0.0],
        [-0.5, -0.5,  0.0],
        [ 0.0, -0.5,  0.0],
        [ 0.0,  0.0,  2.0],
        [-0.5,  0.0,  2.0],
        [-0.5, -0.5,  2.0],
        [ 0.0, -0.5,  2.0]])
    assert np.allclose(disp, disp_comp)
