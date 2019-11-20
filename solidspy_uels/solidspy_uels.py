# -*- coding: utf-8 -*-

"""Main module."""
import numpy as np
from solidspy import gaussutil as gau
from solidspy.femutil import jacoper
from solidspy.assemutil import eqcounter


#%% Interpolators
def shape_tri6(r, s):
    """
    Shape functions and derivatives for a quadratic element

    Parameters
    ----------
    r : float
        Horizontal coordinate of the evaluation point.
    s : float
        Vertical coordinate of the evaluation point.

    Returns
    -------
    N : ndarray (float)
        Array with the shape functions evaluated at the point (r, s).
    dNdr : ndarray (float)
        Array with the derivative of the shape functions evaluated at
        the point (r, s).
    """
    N = np.array(
        [(1 - r - s) - 2*r*(1 - r - s) - 2*s*(1 - r - s),
         r - 2*r*(1 - r - s) - 2*r*s,
         s - 2*r*s - 2*s*(1-r-s),
         4*r*(1 - r - s),
         4*r*s,
         4*s*(1 - r - s)])
    dNdr = np.array([
        [4*r + 4*s - 3, 4*r - 1, 0, -8*r - 4*s + 4, 4*s, -4*s],
        [4*r + 4*s - 3, 0, 4*s - 1, -4*r, 4*r, -4*r - 8*s + 4]])
    return N, dNdr


def shape_quad9(r, s):
    """
    Shape functions and derivatives for a biquadratic element

    Parameters
    ----------
    r : float
        Horizontal coordinate of the evaluation point.
    s : float
        Vertical coordinate of the evaluation point.

    Returns
    -------
    N : ndarray (float)
        Array with the shape functions evaluated at the point (r, s).
    dNdr : ndarray (float)
        Array with the derivative of the shape functions evaluated at
        the point (r, s).
    """
    N = np.array(
        [0.25*r*s*(r - 1.0)*(s - 1.0),
         0.25*r*s*(r + 1.0)*(s - 1.0),
         0.25*r*s*(r + 1.0)*(s + 1.0),
         0.25*r*s*(r - 1.0)*(s + 1.0),
         0.5*s*(-r**2 + 1.0)*(s - 1.0),
         0.5*r*(r + 1.0)*(-s**2 + 1.0),
         0.5*s*(-r**2 + 1.0)*(s + 1.0),
         0.5*r*(r - 1.0)*(-s**2 + 1.0),
         (-r**2 + 1.0)*(-s**2 + 1.0)])
    dNdr = np.array([
        [0.25*s*(2.0*r - 1.0)*(s - 1.0),
         0.25*s*(2.0*r + 1.0)*(s - 1.0),
         0.25*s*(2.0*r + 1.0)*(s + 1.0),
         0.25*s*(2.0*r - 1.0)*(s + 1.0),
         r*s*(-s + 1.0),
         -0.5*(2.0*r + 1.0)*(s**2 - 1.0),
         -r*s*(s + 1.0),
         0.5*(-2.0*r + 1.0)*(s**2 - 1.0),
         2.0*r*(s**2 - 1.0)],
        [0.25*r*(r - 1.0)*(2.0*s - 1.0),
         0.25*r*(r + 1.0)*(2.0*s - 1.0),
         0.25*r*(r + 1.0)*(2.0*s + 1.0),
         0.25*r*(r - 1.0)*(2.0*s + 1.0),
         0.5*(r**2 - 1.0)*(-2.0*s + 1.0),
         -r*s*(r + 1.0),
         -0.5*(r**2 - 1.0)*(2.0*s + 1.0),
         r*s*(-r + 1.0),
         2.0*s*(r**2 - 1.0)]])
    return N, dNdr


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


#%% Interpolation matrices
def elast_mat_2d(r, s, coord, element):
    """
    Interpolation matrices for elements for plane elasticity

    Parameters
    ----------
    r : float
        Horizontal coordinate of the evaluation point.
    s : float
        Vertical coordinate of the evaluation point.
    coord : ndarray (float)
        Coordinates of the element.

    Returns
    -------
    H : ndarray (float)
        Array with the shape functions evaluated at the point (r, s)
        for each degree of freedom.
    B : ndarray (float)
        Array with the displacement to strain matrix evaluated
        at the point (r, s).
    det : float
        Determinant of the Jacobian.
    """
    N, dNdr = element(r, s)
    det, jaco_inv = jacoper(dNdr, coord)
    dNdx = jaco_inv @ dNdr
    H = np.zeros((2, 2*N.shape[0]))
    B = np.zeros((3, 2*N.shape[0]))
    H[0, 0::2] = N
    H[1, 1::2] = N
    B[0, 0::2] = dNdx[0, :]
    B[1, 1::2] = dNdx[1, :]
    B[2, 0::2] = dNdx[1, :]
    B[2, 1::2] = dNdx[0, :]
    return H, B, det


def elast_mat_3d(r, s, t, coord):
    """
    Interpolation matrices for a trilinear element for
    elasticity

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
                                    # other element types
    det, jaco_inv = jacoper(dNdr, coord)
    dNdx = jaco_inv @ dNdr
    H = np.zeros((3, 24))
    B = np.zeros((6, 24))
    H[0, 0::3] = N
    H[1, 1::3] = N
    H[2, 2::3] = N
    B[0, 0::3] = dNdx[0, :]
    B[1, 1::3] = dNdx[1, :]
    B[2, 2::3] = dNdx[2, :]
    B[3, 1::3] = dNdx[2, :]
    B[3, 2::3] = dNdx[1, :]
    B[4, 0::3] = dNdx[2, :]
    B[4, 2::3] = dNdx[0, :]
    B[5, 0::3] = dNdx[1, :]
    B[5, 1::3] = dNdx[0, :]
    return H, B, det


def micropolar_mat_2d(r, s, coord, element):
    """
    Interpolation matrices for a quadratic element for plane
    micropolar elasticity

    Parameters
    ----------
    r : float
        Horizontal coordinate of the evaluation point.
    s : float
        Vertical coordinate of the evaluation point.
    coord : ndarray (float)
        Coordinates of the element.

    Returns
    -------
    H : ndarray (float)
        Array with the shape functions evaluated at the point (r, s)
        for each degree of freedom.
    B : ndarray (float)
        Array with the displacement to strain matrix evaluated
        at the point (r, s).
    det : float
        Determinant of the Jacobian.

    References
    ----------
    .. [Guarin2020] Guarín-Zapata, N., Gomez, J., Valencia, C.,
       Dargush, G. F., & Hadjesfandiari, A. R. (2020).
       Finite element modeling of micropolar-based phononic crystals.
       Wave Motion, 92, 102406.
    """
    N, dNdr = element(r, s)
    det, jaco_inv = jacoper(dNdr, coord)
    dNdx = jaco_inv @ dNdr
    B = np.zeros((6, 3*N.shape[0]))
    H = np.zeros((3, 3*N.shape[0]))
    H[0, 0::3] = N
    H[1, 1::3] = N
    H[2, 2::3] = N
    B[0, 0::3] = dNdx[0, :]
    B[1, 1::3] = dNdx[1, :]
    B[2, 1::3] = dNdx[0, :]
    B[2, 2::3] = N
    B[3, 0::3] = dNdx[1, :]
    B[3, 2::3] = -N
    B[4, 2::3] = dNdx[0, :]
    B[5, 2::3] = dNdx[1, :]
    return H, B, det


def cst_mat_2d(r, s, coord, element):
    """
    Interpolation matrices for a quadratic element for plane
    c-cst elasticity

    Parameters
    ----------
    r : float
        Horizontal coordinate of the evaluation point.
    s : float
        Vertical coordinate of the evaluation point.
    coord : ndarray (float)
        Coordinates of the element.

    Returns
    -------
    H : ndarray (float)
        Array with the shape functions evaluated at the point (r, s)
        for each degree of freedom.
    B : ndarray (float)
        Array with the displacement to strain matrix evaluated
        at the point (r, s).
    det : float
        Determinant of the Jacobian.

    References
    ----------

    .. [CST] Ali R. Hadhesfandiari, Gary F. Dargush.
        Couple stress theory for solids. International Journal
        for Solids and Structures, 2011, 48, 2496-2510.
    """
    N, dNdr = element(r, s)
    det, jaco_inv = jacoper(dNdr, coord)
    dNdr = jaco_inv @ dNdr
    H = np.zeros((2, 2*N.shape[0]))
    B = np.zeros((3, 2*N.shape[0]))
    Bk = np.zeros((2, N.shape[0]))
    B_curl = np.zeros((2*N.shape[0],))
    H[0, 0::2] = N
    H[1, 1::2] = N
    B[0, 0::2] = dNdr[0, :]
    B[1, 1::2] = dNdr[1, :]
    B[2, 0::2] = dNdr[1, :]
    B[2, 1::2] = dNdr[0, :]
    Bk[0, :] = -dNdr[1, :]
    Bk[1, :] = dNdr[0, :]
    B_curl[0::2] = -dNdr[1, :]
    B_curl[1::2] = dNdr[0, :]
    return N, H, B, Bk, B_curl, det


#%% Elements
def elast_tri6(coord, params):
    """
    Triangular element with 6 nodes for classic elasticity
    under plane-strain

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
    stiff_mat = np.zeros((12, 12))
    mass_mat = np.zeros((12, 12))
    C = np.array([
        [2*mu + lamda, lamda, 0],
        [lamda, 2*mu + lamda, 0],
        [0, 0, mu]])
    gpts, gwts = gau.gauss_tri(order=3)
    for cont in range(gpts.shape[0]):
        r = gpts[cont, 0]
        s = gpts[cont, 1]
        H, B, det = elast_mat_2d(r, s, coord, shape_tri6)
        factor = 0.5 * det * gwts[cont]
        stiff_mat += factor * (B.T @ C @ B)
        mass_mat += rho*factor * (H.T @ H)
    return stiff_mat, mass_mat


def elast_quad9(coord, params):
    """
    Quadrilateral element with 9 nodes for classic elasticity
    under plane-strain

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
    stiff_mat = np.zeros((18, 18))
    mass_mat = np.zeros((18, 18))
    C = np.array([
        [2*mu + lamda, lamda, 0],
        [lamda, 2*mu + lamda, 0],
        [0, 0, mu,]])
    gpts, gwts = gau.gauss_nd(3, ndim=2)
    for cont in range(gpts.shape[0]):
        r = gpts[cont, 0]
        s = gpts[cont, 1]
        H, B, det = elast_mat_2d(r, s, coord, shape_quad9)
        factor = det * gwts[cont]
        stiff_mat += factor * (B.T @ C @ B)
        mass_mat += rho*factor * (H.T @ H)
    return stiff_mat, mass_mat


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
        H, B, det = elast_mat_3d(r, s, t, coord)
        factor = det * gwts[cont]
        stiff_mat += factor * (B.T @ C @ B)
        mass_mat += rho*factor * (H.T @ H)
    return stiff_mat, mass_mat


def micropolar_tri6(coord, params):
    """
    Triangular element with 6 nodes for micropolar elasticity
    under plane-strain as presented in [Guarin2020]_

    Parameters
    ----------
    coord : coord
        Coordinates of the element.
    params : list
        List with material parameters in the following order:
        [Young modulus, Poisson coefficient, couple modulus,
         bending modulus, mass density, inertia density].

    Returns
    -------
    stiff_mat : ndarray (float)
        Local stifness matrix.
    mass_mat : ndarray (float)
        Local mass matrix.

    References
    ----------
    .. [Guarin2020] Guarín-Zapata, N., Gomez, J., Valencia, C.,
       Dargush, G. F., & Hadjesfandiari, A. R. (2020).
       Finite element modeling of micropolar-based phononic crystals.
       Wave Motion, 92, 102406.
    """
    E, nu, alpha, xi, rho, J = params
    lam = E*nu/((1 + nu)*(1 - 2*nu))
    mu = 0.5*E/(1 + nu)
    stiff_mat = np.zeros((18, 18))
    mass_mat = np.zeros((18, 18))
    stiff = np.zeros((6, 6))
    stiff[0:2, 0:2] = [[lam + 2*mu, lam], [lam, lam + 2*mu]]
    stiff[2:4, 2:4] = [[mu + alpha, mu - alpha], [mu - alpha, mu + alpha]]
    stiff[4:6, 4:6] = [[xi, 0], [0, xi]]
    inertia = np.diag([rho, rho, J])
    gpts, gwts = gau.gauss_tri(order=3)
    for cont in range(gpts.shape[0]):
        r = gpts[cont, 0]
        s = gpts[cont, 1]
        H, B, det = micropolar_mat_2d(r, s, coord, shape_tri6)
        factor = 0.5 * det * gwts[cont]
        stiff_mat += factor * B.T @ stiff @ B
        mass_mat += factor* H.T @ inertia @ H

    return stiff_mat, mass_mat


def micropolar_quad9(coord, params):
    """
    Quadrilateral element with 9 nodes for micropolar elasticity
    under plane-strain as presented in [Guarin2020]_

    Parameters
    ----------
    coord : coord
        Coordinates of the element.
    params : list
        List with material parameters in the following order:
        [Young modulus, Poisson coefficient, couple modulus,
         bending modulus, mass density, inertia density].

    Returns
    -------
    stiff_mat : ndarray (float)
        Local stifness matrix.
    mass_mat : ndarray (float)
        Local mass matrix.

    References
    ----------
    .. [Guarin2020] Guarín-Zapata, N., Gomez, J., Valencia, C.,
       Dargush, G. F., & Hadjesfandiari, A. R. (2020).
       Finite element modeling of micropolar-based phononic crystals.
       Wave Motion, 92, 102406.
    """
    E, nu, alpha, xi, rho, J = params
    lam = E*nu/((1 + nu)*(1 - 2*nu))
    mu = 0.5*E/(1 + nu)
    stiff_mat = np.zeros((27, 27))
    mass_mat = np.zeros((27, 27))
    stiff = np.zeros((6, 6))
    stiff[0:2, 0:2] = [[lam + 2*mu, lam], [lam, lam + 2*mu]]
    stiff[2:4, 2:4] = [[mu + alpha, mu - alpha], [mu - alpha, mu + alpha]]
    stiff[4:6, 4:6] = [[xi, 0], [0, xi]]
    inertia = np.diag([rho, rho, J])
    npts = 3
    gpts, gwts = gau.gauss_nd(npts)
    for cont in range(0, npts**2):
        r = gpts[cont, 0]
        s = gpts[cont, 1]
        H, B, det = micropolar_mat_2d(r, s, coord, shape_quad9)
        factor = det * gwts[cont]
        stiff_mat += factor * B.T @ stiff @ B
        mass_mat += factor* H.T @ inertia @ H

    return stiff_mat, mass_mat


def cst_tri6(coord, params):
    """
    Triangular element with 6 nodes for Corrected Couple-Stress
    elasticity (C-CST) under plane-strain as presented in [CST]_

    Parameters
    ----------
    coord : coord
        Coordinates of the element.
    params : list
        List with material parameters in the following order:
        [Young modulus, Poisson coefficient, couple modulus,
         bending modulus, mass density, inertia density].

    Returns
    -------
    stiff_mat : ndarray (float)
        Local stifness matrix.
    mass_mat : ndarray (float)
        Local mass matrix.

    References
    ----------
    .. [CST] Ali R. Hadhesfandiari, Gary F. Dargush.
        Couple stress theory for solids. International Journal
        for Solids and Structures, 2011, 48, 2496-2510.
    """
    E, nu, eta, rho = params
    stiff_mat = np.zeros((19, 19))
    mass_mat = np.zeros((19, 19))
    c = E*(1 - nu)/((1 + nu)*(1 - 2*nu))*np.array([
        [1, nu/(1 - nu), 0],
        [nu/(1 - nu), 1, 0],
        [0, 0, (1 - 2*nu)/(2*(1 - nu))]])
    b = 4*eta*np.eye(2)
    gpts, gwts = gau.gauss_tri(order=3)
    for cont in range(gpts.shape[0]):
        r = gpts[cont, 0]
        s = gpts[cont, 1]
        N, H, B, Bk, B_curl, det = cst_mat_2d(r, s, coord, shape_tri6)
        Ku = B.T @ c @ B
        Kw = Bk.T @ b @ Bk
        K_w_s = -2*N
        factor = 0.5 * det * gwts[cont]
        stiff_mat[0:12, 0:12] += factor * Ku
        stiff_mat[12:18, 12:18] += factor * Kw
        stiff_mat[0:12, 18] += factor * B_curl.T
        stiff_mat[12:18, 18] += factor * K_w_s.T
        stiff_mat[18, 0:12] += factor * B_curl
        stiff_mat[18, 12:18] += factor * K_w_s
        mass_mat[0:12, 0:12] += rho*factor* (H.T @ H)
    order = [0, 1, 12, 2, 3, 13, 4, 5, 14, 6, 7, 15, 8, 9, 16, 10, 11, 17, 18]
    stiff_mat = stiff_mat[:, order]
    mass_mat = mass_mat[:, order]
    return stiff_mat[order, :], mass_mat[order, :]


def cst_quad9(coord, params):
    """
    Quadrilateral element with 9 nodes for Corrected Couple-Stress
    elasticity (C-CST) under plane-strain as presented in [CST]_

    Parameters
    ----------
    coord : coord
        Coordinates of the element.
    params : list
        List with material parameters in the following order:
        [Young modulus, Poisson coefficient, couple modulus,
         bending modulus, mass density, inertia density].

    Returns
    -------
    stiff_mat : ndarray (float)
        Local stifness matrix.
    mass_mat : ndarray (float)
        Local mass matrix.

    References
    ----------
    .. [CST] Ali R. Hadhesfandiari, Gary F. Dargush.
        Couple stress theory for solids. International Journal
        for Solids and Structures, 2011, 48, 2496-2510.
    """
    E, nu, eta, rho = params
    stiff_mat = np.zeros((28, 28))
    mass_mat = np.zeros((28, 28))
    c = E*(1 - nu)/((1 + nu)*(1 - 2*nu))*np.array([
        [1, nu/(1 - nu), 0],
        [nu/(1 - nu), 1, 0],
        [0, 0, (1 - 2*nu)/(2*(1 - nu))]])
    b = 4*eta*np.eye(2)
    npts = 3
    gpts, gwts = gau.gauss_nd(npts)
    for cont in range(0, npts**2):
        r = gpts[cont, 0]
        s = gpts[cont, 1]
        N, H, B, Bk, B_curl, det = cst_mat_2d(r, s, coord, shape_quad9)
        Ku = B.T @ c @ B
        Kw = Bk.T @ b @ Bk
        K_w_s = -2*N
        factor = det * gwts[cont]
        stiff_mat[0:18, 0:18] += factor * Ku
        stiff_mat[18:27, 18:27] += factor * Kw
        stiff_mat[0:18, 27] += factor * B_curl.T
        stiff_mat[18:27, 27] += factor * K_w_s.T
        stiff_mat[27, 0:18] += factor * B_curl
        stiff_mat[27, 18:27] += factor * K_w_s
        mass_mat[0:18, 0:18] += rho*factor* (H.T @ H)
    order = [0, 1, 18, 2, 3, 19, 4, 5, 20, 6, 7, 21, 8, 9, 22, 10, 11, 23,
             12, 13, 24, 14, 15, 25, 16, 17, 26, 27]
    stiff_mat = stiff_mat[:, order]
    mass_mat = mass_mat[:, order]
    return stiff_mat[order, :], mass_mat[order, :]


#%% Assembly
def assem_op_cst(cons, elements):
    """Create assembly array operator

    Count active equations, create boundary conditions array ``bc_array``
    and the assembly operator DME.

    Parameters
    ----------
    cons : ndarray.
      Array with constraints for each degree of freedom in each node.
    elements : ndarray
      Array with the number for the nodes in each element.

    Returns
    -------
    assem_op : ndarray (int)
      Assembly operator.
    bc_array : ndarray (int)
      Boundary conditions array.
    neq : int
      Number of active equations in the system.

    """
    nels = elements.shape[0]
    assem_op = np.zeros([nels, 28], dtype=np.integer)
    neq, bc_array = eqcounter(cons, ndof_node=3)
    for ele in range(nels):
        assem_op[ele, :27] = bc_array[elements[ele, 3:]].flatten()
        assem_op[ele, 27] = neq + ele
    return assem_op, bc_array, neq + nels


def assem_op_cst6(cons, elements):
    """Create assembly array operator

    Count active equations, create boundary conditions array ``bc_array``
    and the assembly operator DME.

    Parameters
    ----------
    cons : ndarray.
      Array with constraints for each degree of freedom in each node.
    elements : ndarray
      Array with the number for the nodes in each element.

    Returns
    -------
    assem_op : ndarray (int)
      Assembly operator.
    bc_array : ndarray (int)
      Boundary conditions array.
    neq : int
      Number of active equations in the system.

    """
    nels = elements.shape[0]
    assem_op = np.zeros([nels, 19], dtype=np.integer)
    neq, bc_array = eqcounter(cons, ndof_node=3)
    for ele in range(nels):
        assem_op[ele, :18] = bc_array[elements[ele, 3:]].flatten()
        assem_op[ele, 18] = neq + ele
    return assem_op, bc_array, neq + nels


#%% Checks
if __name__ == "__main__":
    import doctest
    doctest.testmod()
