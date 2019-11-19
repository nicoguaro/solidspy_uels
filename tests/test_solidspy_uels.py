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
from solidspy_uels.solidspy_uels import elast_brick8, elast_tri6, elast_quad9


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
def test_elast_tri6():
    ## One element
    coords = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.0, 0.5]])
    params = [1, 1/4, 1]

    els = np.array([[0, 1, 0, 0, 1, 2, 3, 4, 5]])

    # Element without constraints
    cons = np.zeros((6, 2))
    pts = np.column_stack((range(0, 6), coords))
    mats = np.array([params])
    assem_op, bc_array, neq = DME(cons, els, ndof_node=2, 
                                  ndof_el=lambda iet: 12,
                                  ndof_el_max=12)
    assert np.allclose(assem_op, range(0, 12))

    stiff, mass = dense_assem(els, mats, pts, neq, assem_op, uel=elast_tri6)

    stiff_comp = 1/15 * np.array([
        [12, 6, 3, 1, 1, 1, -12, -4, 0, 0, -4, -4],
        [6, 12, 1, 1, 1, 3, -4, -4, 0, 0, -4, -12],
        [3, 1, 9, 0, 0, -1, -12, -4, 0, 4, 0, 0],
        [1, 1, 0, 3, -1, 0, -4, -4, 4, 0, 0, 0],
        [1, 1, 0, -1, 3, 0, 0, 0, 0, 4, -4, -4],
        [1, 3, -1, 0, 0, 9, 0, 0, 4, 0, -4, -12],
        [-12, -4, -12, -4, 0, 0, 32, 8, -8, -8, 0, 8],
        [-4, -4, -4, -4, 0, 0, 8, 32, -8, -24, 8, 0],
        [0, 0, 0, 4, 0, 4, -8, -8, 32, 8, -24, -8],
        [0, 0, 4, 0, 4, 0, -8, -24, 8, 32, -8, -8],
        [-4, -4, 0, 0, -4, -4, 0, 8, -24, -8, 32, 8],
        [-4, -12, 0, 0, -4, -12, 8, 0, -8, -8, 8, 32]])
    assert np.allclose(stiff, stiff_comp)

    mass_comp = 1/360 *np.array([
        [6, 0, -1, 0, -1, 0, 0, 0, -4, 0, 0, 0],
        [0, 6, 0, -1, 0, -1, 0, 0, 0, -4, 0, 0],
        [-1, 0, 6, 0, -1, 0, 0, 0, 0, 0, -4, 0],
        [0, -1, 0, 6, 0, -1, 0, 0, 0, 0, 0, -4],
        [-1, 0, -1, 0, 6, 0, -4, 0, 0, 0, 0, 0],
        [0, -1, 0, -1, 0, 6, 0, -4, 0, 0, 0, 0],
        [0, 0, 0, 0, -4, 0, 32, 0, 16, 0, 16, 0],
        [0, 0, 0, 0, 0, -4, 0, 32, 0, 16, 0, 16],
        [-4, 0, 0, 0, 0, 0, 16, 0, 32, 0, 16, 0],
        [0, -4, 0, 0, 0, 0, 0, 16, 0, 32, 0, 16],
        [0, 0, -4, 0, 0, 0, 16, 0, 16, 0, 32, 0],
        [0, 0, 0, -4, 0, 0, 0, 16, 0, 16, 0, 32]])
    assert np.allclose(mass, mass_comp)

    ## Uniaxial stress
    coords = np.array([
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 0.5],
            [0.5, 0.5],
            [1.0, 0.5],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 1.0]])
    params = [1, 1/4, 1]

    els = np.array([
            [0, 1, 0, 0, 2, 6, 1, 4, 3],
            [1, 1, 0, 8, 6, 2, 7, 4, 5]])
    cons = np.array([
            [-1,-1],
            [ 0,-1],
            [ 0,-1],
            [-1, 0],
            [ 0, 0],
            [ 0, 0],
            [-1, 0],
            [ 0, 0],
            [ 0, 0]])
    pts = np.column_stack((range(0, 9), coords))
    mats = np.array([params])
    assem_op, bc_array, neq = DME(cons, els, ndof_node=2, 
                                  ndof_el=lambda iet: 12,
                                  ndof_el_max=12)
    stiff, mass = dense_assem(els, mats, pts, neq, assem_op, uel=elast_tri6)
    loads = np.array([
        [6, 0, 1/3],
        [7, 0, 4/3],
        [8, 0, 1/3]])
    rhs = loadasem(loads, bc_array, neq, ndof_node=2)
    sol = LA.solve(stiff, rhs)
    disp = complete_disp(bc_array, pts, sol, ndof_node=2)
    disp_comp = np.array([
        [ 0.0,  0.0],
        [-0.3125, 0.0],
        [-0.625,  0.0],
        [ 0.0,  0.9375],
        [-0.3125, 0.9375],
        [-0.625,  0.9375],
        [ 0.0,  1.875],
        [-0.3125, 1.875],
        [-0.625,  1.875]])
    assert np.allclose(disp, disp_comp)


def test_elast_quad9():
    ## One element
    coords = np.array([
            [-1.0,-1.0],
            [ 1.0,-1.0],
            [ 1.0, 1.0],
            [-1.0, 1.0],
            [ 0.0,-1.0],
            [ 1.0, 0.0],
            [ 0.0, 1.0],
            [-1.0, 0.0],
            [ 0.0, 0.0]])
    params = [1, 1/4, 1]

    els = np.array([[0, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]])

    # Element without constraints
    cons = np.zeros((9, 2))
    pts = np.column_stack((range(0, 9), coords))
    mats = np.array([params])
    assem_op, bc_array, neq = DME(cons, els, ndof_node=2, 
                                  ndof_el=lambda iet: 18,
                                  ndof_el_max=18)
    assert np.allclose(assem_op, range(0, 18))

    stiff, mass = dense_assem(els, mats, pts, neq, assem_op, uel=elast_quad9)

    stiff_comp = 1/225 * np.array([
        [112, 45, 5, 0, -4, -5, -17, 0, -82, 0, 14, 20, 26, 20, 10, 0, -64, -80],
        [45, 112, 0, -17, -5, -4, 0, 5, 0, 10, 20, 26, 20, 14, 0, -82, -80, -64],
        [5, 0, 112, -45, -17, 0, -4, 5, -82, 0, 10, 0, 26, -20, 14, -20, -64, 80],
        [0, -17, -45, 112, 0, 5, 5, -4, 0, 10, 0, -82, -20, 14, -20, 26, 80, -64],
        [-4, -5, -17, 0, 112, 45, 5, 0, 26, 20, 10, 0, -82, 0, 14, 20, -64, -80],
        [-5, -4, 0, 5, 45, 112, 0, -17, 20, 14, 0, -82, 0, 10, 20, 26, -80, -64],
        [-17, 0, -4, 5, 5, 0, 112, -45, 26, -20, 14, -20, -82, 0, 10, 0, -64, 80],
        [0, 5, 5, -4, 0, -17, -45, 112, -20, 14, -20, 26, 0, 10, 0, -82, 80, -64],
        [-82, 0, -82, 0, 26, 20, 26, -20, 304, 0, -64, -80, -32, 0, -64, 80, -32, 0],
        [0, 10, 0, 10, 20, 14, -20, 14, 0, 400, -80, -64, 0, 32, 80, -64, 0, -352],
        [14, 20, 10, 0, 10, 0, 14, -20, -64, -80, 400, 0, -64, 80, 32, 0, -352, 0],
        [20, 26, 0, -82, 0, -82, -20, 26, -80, -64, 0, 304, 80, -64, 0, -32, 0, -32],
        [26, 20, 26, -20, -82, 0, -82, 0, -32, 0, -64, 80, 304, 0, -64, -80, -32, 0],
        [20, 14, -20, 14, 0, 10, 0, 10, 0, 32, 80, -64, 0, 400, -80, -64, 0, -352],
        [10, 0, 14, -20, 14, 20, 10, 0, -64, 80, 32, 0, -64, -80, 400, 0, -352, 0],
        [0, -82, -20, 26, 20, 26, 0, -82, 80, -64, 0, -32, -80, -64, 0, 304, 0, -32],
        [-64, -80, -64, 80, -64, -80, -64, 80, -32, 0, -352, 0, -32, 0, -352, 0, 1024, 0],
        [-80, -64, 80, -64, -80, -64, 80, -64, 0, -352, 0, -32, 0, -352, 0, -32, 0, 1024]])
    assert np.allclose(stiff, stiff_comp)

    mass_comp = 1/225 * np.array([
        [16, 0, -4, 0, 1, 0, -4, 0, 8, 0, -2, 0, -2, 0, 8, 0, 4, 0],
        [0, 16, 0, -4, 0, 1, 0, -4, 0, 8, 0, -2, 0, -2, 0, 8, 0, 4],
        [-4, 0, 16, 0, -4, 0, 1, 0, 8, 0, 8, 0, -2, 0, -2, 0, 4, 0],
        [0, -4, 0, 16, 0, -4, 0, 1, 0, 8, 0, 8, 0, -2, 0, -2, 0, 4],
        [1, 0, -4, 0, 16, 0, -4, 0, -2, 0, 8, 0, 8, 0, -2, 0, 4, 0],
        [0, 1, 0, -4, 0, 16, 0, -4, 0, -2, 0, 8, 0, 8, 0, -2, 0, 4],
        [-4, 0, 1, 0, -4, 0, 16, 0, -2, 0, -2, 0, 8, 0, 8, 0, 4, 0],
        [0, -4, 0, 1, 0, -4, 0, 16, 0, -2, 0, -2, 0, 8, 0, 8, 0, 4],
        [8, 0, 8, 0, -2, 0, -2, 0, 64, 0, 4, 0, -16, 0, 4, 0, 32, 0],
        [0, 8, 0, 8, 0, -2, 0, -2, 0, 64, 0, 4, 0, -16, 0, 4, 0, 32],
        [-2, 0, 8, 0, 8, 0, -2, 0, 4, 0, 64, 0, 4, 0, -16, 0, 32, 0],
        [0, -2, 0, 8, 0, 8, 0, -2, 0, 4, 0, 64, 0, 4, 0, -16, 0, 32],
        [-2, 0, -2, 0, 8, 0, 8, 0, -16, 0, 4, 0, 64, 0, 4, 0, 32, 0],
        [0, -2, 0, -2, 0, 8, 0, 8, 0, -16, 0, 4, 0, 64, 0, 4, 0, 32],
        [8, 0, -2, 0, -2, 0, 8, 0, 4, 0, -16, 0, 4, 0, 64, 0, 32, 0],
        [0, 8, 0, -2, 0, -2, 0, 8, 0, 4, 0, -16, 0, 4, 0, 64, 0, 32],
        [4, 0, 4, 0, 4, 0, 4, 0, 32, 0, 32, 0, 32, 0, 32, 0, 256, 0],
        [0, 4, 0, 4, 0, 4, 0, 4, 0, 32, 0, 32, 0, 32, 0, 32, 0, 256]])
    assert np.allclose(mass, mass_comp)

    ## Uniaxial stress
    coords = np.array([
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 0.5],
            [0.5, 0.5],
            [1.0, 0.5],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 1.0]])
    params = [1, 1/4, 1]

    els = np.array([[0, 1, 0, 0, 2, 8, 6, 1, 5, 7, 3, 4]])
    cons = np.array([
            [-1,-1],
            [ 0,-1],
            [ 0,-1],
            [-1, 0],
            [ 0, 0],
            [ 0, 0],
            [-1, 0],
            [ 0, 0],
            [ 0, 0]])
    pts = np.column_stack((range(0, 9), coords))
    mats = np.array([params])
    assem_op, bc_array, neq = DME(cons, els, ndof_node=2, 
                                  ndof_el=lambda iet: 18,
                                  ndof_el_max=18)
    stiff, mass = dense_assem(els, mats, pts, neq, assem_op, uel=elast_quad9)
    loads = np.array([
        [6, 0, 1/3],
        [7, 0, 4/3],
        [8, 0, 1/3]])
    rhs = loadasem(loads, bc_array, neq, ndof_node=2)
    sol = LA.solve(stiff, rhs)
    disp = complete_disp(bc_array, pts, sol, ndof_node=2)
    disp_comp = np.array([
        [ 0.0,  0.0],
        [-0.3125, 0.0],
        [-0.625,  0.0],
        [ 0.0,  0.9375],
        [-0.3125, 0.9375],
        [-0.625,  0.9375],
        [ 0.0,  1.875],
        [-0.3125, 1.875],
        [-0.625,  1.875]])
    assert np.allclose(disp, disp_comp)


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
