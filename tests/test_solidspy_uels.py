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
from solidspy_uels.solidspy_uels import shape_brick8, shape_tri6, shape_quad9
from solidspy_uels.solidspy_uels import elast_brick8


#%% Test interpolators
def test_shape_tri6():
    # Interpolation condition check
    coords = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.0, 0.5]])
    N, _ = shape_tri6(coords[:, 0], coords[:, 1])
    assert np.allclose(N, np.eye(6))

    # Evaluation at (1/3, 1/3)
    N, dNdr = shape_tri6(1/3, 1/3)
    N_exp = np.array([-1., -1., -1., 4., 4., 4.])/9
    dNdr_exp = np.array([
                [-1.,  1.,  0.,  0.,  4., -4.],
                [-1.,  0.,  1., -4.,  4.,  0.]])/3
    assert np.allclose(N, N_exp)
    assert np.allclose(dNdr, dNdr_exp)


def test_shape_quad9():
    # Interpolation condition check
    coords = np.array([
            [-1.0, -1.0],
            [ 1.0, -1.0],
            [ 1.0,  1.0],
            [-1.0,  1.0],
            [ 0.0, -1.0],
            [ 1.0,  0.0],
            [ 0.0,  1.0],
            [-1.0,  0.0],
            [ 0.0,  0.0]])
    N, _ = shape_quad9(coords[:, 0], coords[:, 1])
    assert np.allclose(N, np.eye(9))

    # Evaluation at (1/4, 1/4)
    N, dNdr = shape_quad9(0.25, 0.25)
    N_exp = np.array([0.00878906, -0.01464844, 0.02441406, -0.01464844,
                   -0.08789062, 0.14648438, 0.14648438, -0.08789062,
                   0.87890625])

    dNdr_exp = np.array([
                [0.0234375, -0.0703125, 0.1171875, -0.0390625, 0.046875,
                 0.703125, -0.078125, -0.234375, -0.46875],
                [0.0234375, -0.0390625, 0.1171875, -0.0703125, -0.234375,
                 -0.078125, 0.703125, 0.046875, -0.46875]])
    assert np.allclose(N, N_exp)
    assert np.allclose(dNdr, dNdr_exp)


def test_shape_brick8():
    # Interpolation condition check
    coords = np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1]])
    N, _ = shape_brick8(coords[:, 0], coords[:, 1], coords[:, 2])
    assert np.allclose(N, np.eye(8))

    # Evaluation at (0, 0, 0)
    N, dNdr = shape_brick8(0, 0, 0)
    N_exp = 0.125*np.array([1., 1., 1., 1., 1., 1., 1., 1.])
    dNdr_exp = 0.125*np.array([
        [-1.,  1.,  1., -1., -1.,  1.,  1., -1.],
        [-1., -1.,  1.,  1., -1., -1.,  1.,  1.],
        [-1., -1., -1., -1.,  1.,  1.,  1.,  1.]])
    assert np.allclose(N, N_exp)
    assert np.allclose(dNdr, dNdr_exp)


#%% Test interpolation matrices
def test_interp_mat_3d():
    pass


#%% Test elements
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
