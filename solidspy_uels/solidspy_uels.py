# -*- coding: utf-8 -*-

"""Main module."""
import numpy as np
import scipy.linalg as LA
from solidspy import gaussutil as gau
from solidspy.femutil import jacoper
from solidspy.assemutil import DME, dense_assem, loadasem
from solidspy.postprocesor import complete_disp



def shape_brick8(r, s, t):
    """
    Shape functions and derivatives for a trilinear element
    
    Parameters
    ----------
    r : float
        Horizontal coordinate of the evaluation point.
    s : float
        Horizontal coordinate of the evaluation point.
    t : float
        Vertical coordinate of the evaluation point.

    Returns
    -------
    N : ndarray (float)
        Array with the shape functions evaluated at the point (r, s, t).
    dNdr : ndarray (float)
        Array with the derivative of the shape functions evaluated at
        the point (r, s, t).
    """
    N = np.array([
            (1 - r)*(1 - s)*(1 - t), (1 - s)*(1 - t)*(r + 1),
            (1 - t)*(r + 1)*(s + 1), (1 - r)*(1 - t)*(s + 1),
            (1 - r)*(1 - s)*(t + 1), (1 - s)*(r + 1)*(t + 1),
            (r + 1)*(s + 1)*(t + 1), (1 - r)*(s + 1)*(t + 1)])
    dNdr = np.array([
            [(1 - t)*(s - 1), (1 - s)*(1 - t),
             (1 - t)*(s + 1), (1 - t)*(-s - 1),
             (1 - s)*(-t - 1), (1 - s)*(t + 1),
             (s + 1)*(t + 1), -(s + 1)*(t + 1)],
            [(1 - t)*(r - 1), (1 - t)*(-r - 1),
             (1 - t)*(r + 1), (1 - r)*(1 - t),
             -(1 - r)*(t + 1), -(r + 1)*(t + 1),
             (r + 1)*(t + 1), (1 - r)*(t + 1)],
            [-(1 - r)*(1 - s), -(1 - s)*(r + 1),
             -(r + 1)*(s + 1), -(1 - r)*(s + 1),
             (1 - r)*(1 - s), (1 - s)*(r + 1),
             (r + 1)*(s + 1), (1 - r)*(s + 1)]])

    return 0.125*N, 0.125*dNdr


def interp_mat_3d(r, s, t, coord):
    """
    Shape functions and derivatives for a trilinear element
    
    Parameters
    ----------
    r : float
        Horizontal coordinate of the evaluation point.
    s : float
        Horizontal coordinate of the evaluation point.
    t : float
        Vertical coordinate of the evaluation point.
    coord : ndarray (float)
        Coordinates of the element.

    Returns
    -------
    H : ndarray (float)
        Array with the shape functions evaluated at the point (r, s, t)
        for each degree of freedom.
    B : ndarray (float)
        Array with the displacement to strain matrix evaluated
        at the point (r, s, t).
    det : float
        Determinant of the Jacobian.
    """
    N, dNdr = shape_brick8(r, s, t) # This line would be different for
                                    # other element types1
    det, jaco_inv = jacoper(dNdr, coord)
    dHdx = jaco_inv @ dNdr
    H = np.zeros((3, 24))
    B = np.zeros((6, 24))
    H[0, 0::3] = N
    H[1, 1::3] = N
    H[2, 2::3] = N
    B[0, 0::3] = dHdx[0, :]
    B[1, 1::3] = dHdx[1, :]
    B[2, 2::3] = dHdx[2, :]
    
    B[3, 1::3] = dHdx[2, :]
    B[3, 2::3] = dHdx[1, :]
    
    B[4, 0::3] = dHdx[2, :]
    B[4, 2::3] = dHdx[0, :]
    
    B[5, 0::3] = dHdx[1, :]
    B[5, 1::3] = dHdx[0, :]   
    return H, B, det


def elast_brick8(coord, params):
    """Brick element with 8 nodes for classic elasticity
    
    Parameters
    ----------
    coord : coord
        Coordinates of the element.
    params : list
        List with material parameters in the following order:
        [Young modulus, Poisson coefficient, density].

    Returns
    -------
    stiff_mat : ndarray (float)
        Local stifness matrix.
    mass_mat : ndarray (float)
        Local mass matrix.
    """
    E, nu, rho = params
    lamda = E*nu/(1 + nu)/(1 - 2*nu)
    mu = 0.5*E/(1 + nu)
    stiff_mat = np.zeros((24, 24))
    mass_mat = np.zeros((24, 24))
    C = np.array([
        [2*mu + lamda, lamda, lamda, 0, 0, 0],
        [lamda, 2*mu + lamda, lamda, 0, 0, 0],
        [lamda, lamda, 2*mu + lamda, 0, 0, 0],
        [0, 0, 0, mu, 0, 0],
        [0, 0, 0, 0, mu, 0],
        [0, 0, 0, 0, 0, mu]])
    gpts, gwts = gau.gauss_nd(2, ndim=3)
    for cont in range(gpts.shape[0]):
        r = gpts[cont, 0]
        s = gpts[cont, 1]
        t = gpts[cont, 2]
        H, B, det = interp_mat_3d(r, s, t, coord)
        factor = det * gwts[cont]
        stiff_mat  += factor * (B.T @ C @ B)
        mass_mat += rho*factor * (H.T @ H)
    return stiff_mat, mass_mat


if __name__ == "__main__":
    coords = np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1]])
    params = [1, 1/3, 1]

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
    stiff, mass = dense_assem(els, mats, pts, neq, assem_op, uel=elast_brick8)
    loads = np.array([
            [4, 0, 0, 1],
            [5, 0, 0, 1],
            [6, 0, 0, 1],
            [7, 0, 0, 1],])
    rhs = loadasem(loads, bc_array, neq, ndof_node=3)
    sol = LA.solve(stiff, rhs)
    disp = complete_disp(bc_array, pts, sol, ndof_node=3)
    print(disp)
