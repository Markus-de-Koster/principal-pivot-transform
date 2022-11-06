import numpy as np

import ppt.matrix as mat
from ppt import helper


def principal_pivot_transform(m, alpha: list, generalized=False):
    """
    principal pivot transform according to
    M. J. Tsatsomeros, “Principal pivot transforms: properties and applications,” Linear Algebra and its Applications, vol. 307, no. 1–3, pp. 151–165, Mar. 2000, doi: 10.1016/S0024-3795(99)00281-5.
    :param m: matrix
    :param alpha: exchange indices. Index list of rows and columns of the principal sub-matrix from 0 to n-1; i.e. the rows that need to be exchanged - indexing starts with row 0
    :return: Matrix relative to alpha
    """
    if not mat.is_square(m):
        raise ValueError("Matrix is not a square matrix")

    if len(alpha) == 0:  # by convention
        return m

    n = m.shape[0]
    a, b, c, d = get_submatrices_for_indices(m, alpha)  # get sub-matrices
    if not generalized:
        a_inv = np.linalg.inv(a)
    else:
        if not check_gppt(a, b, c):
            raise ValueError("GPPT does not hold")
        a_inv = np.linalg.pinv(a)

    # Schur complement
    a_schur = np.subtract(d, (np.matmul(np.matmul(c, a_inv), b)))  # D - C*A^(-1)*B

    """
    return the reordered matrix 
    [[A⁻¹,-A⁻¹*B],[C*A⁻¹, D-C*A⁻¹*B]]
    NOTE the returned matrix is relative to alpha. That means it should be re-ordered to fit the original
    """
    a_repl = a_inv
    b_repl = - np.matmul(a_inv, b)
    c_repl = np.matmul(c, a_inv)
    d_repl = a_schur
    return combine_submatrices(a_repl, b_repl, c_repl, d_repl, n, alpha)

def check_gppt(a, b, c):
    """
    confirm if the domain range exchange property holds.
    According to:
    M. Rajesh Kannan and R. B. Bapat, “Generalized principal pivot transform,” Linear Algebra and its Applications, vol. 454, pp. 49–56, Aug. 2014, doi: 10.1016/j.laa.2014.04.015.
    Theorem 3.1

    The matrix has to be partitioned in the following way
    N (a) ⊆ N (c) and N (a*) ⊆ N (b*).
    Where N(a) is the nullspace of a and b* is the complex conjugate (or adjoint) of b
    :param a: sub-matrix a
    :param b: sub-matrix b
    :param c: sub-matrix c
    :return: true if the gppt holds true
    """
    nsa = mat.nullspace(a)
    nsc = mat.nullspace(c)
    if not mat.is_linear_subspace(nsa, nsc):
        return False
    nsac = mat.nullspace(np.conj(a))
    nsbc = mat.nullspace(np.conj(b))
    if not mat.is_linear_subspace(nsac, nsbc):
        return False
    return True

def get_submatrices_for_indices(m: np.array, s: list):
    """
    Get submatrices for given rows and column indices.
    The Matrix will be split into 4 matrices (a,b,c,d) according to the given row / column index list.
    E.g. if the set (1,3) is given on a 4x4 matrix the submatrix 'a' (also known as principal minor)
    will have those values of row 1,3 and column 1,3
    whereas submatrix 'b' will consist of the values in rows 1,3 and columns 2,4
    :param m: block matrix as input
    :param s: list of rows / columns, indexing starts at 0
    :return: 4 submatrices where a is the principal minor in relation to indices given in s
    """
    if not mat.is_square(m):
        raise ValueError("Matrix must be a square matrix")
    n = m.shape[0]
    # calculate the complement of s
    s_comp = list(set(range(0, n)) - set(s))

    a = m[np.ix_(s, s)]
    b = m[np.ix_(s, s_comp)]
    c = m[np.ix_(s_comp, s)]
    d = m[np.ix_(s_comp, s_comp)]
    return a, b, c, d


def combine_submatrices(a, b, c, d, n, s: list):
    """
    Combine sub-matrices applying the same scheme as in get_submatrices_for_indices
    :param a: sub-matrix with rows and columns contained in s
    :param b: sub-matrix with rows contained in s and columns contained in the complement of s
    :param c: sub-matrix with rows contained in the complement of s and columns contained in s
    :param d: sub-matrix with rows contained in the complement of s and columns contained in the complement of s
    :param n: number of rows / columns
    :param s: list of row / column indices used for determining combination
    :return: combined matrix
    """
    m = np.empty((n, n), dtype=np.complex_)  # empty array
    # complement of s
    s_comp = list(set(range(0, n)) - set(s))

    m[np.ix_(s, s)] = a
    m[np.ix_(s, s_comp)] = b
    m[np.ix_(s_comp, s)] = c
    m[np.ix_(s_comp, s_comp)] = d
    return m

def is_p_matrix(m: np.array):
    """
    Is the matrix a P-Matrix?
    A Matrix is a P-Matrix if all of its principal minors are positive.
    A principal minor is the determinant of a sub-matrix
    :param m: matrix
    :return: true if P-Matrix, else false
    """
    # the matrix needs to be square
    if not mat.is_square(m):
        raise ValueError("Matrix is not square")
    n = m.shape[0]
    s = set(range(0, n))
    subsets = list(helper.powerset(s))  # get all subsets
    for subset in subsets:
        # get all principal minors (a)
        a, _, _, _ = get_submatrices_for_indices(m, list(subset))
        if np.linalg.det(a) < 0:  # check if the principal minor's determinant is positive
            return False  # if this is not the case for any subset, the matrix is not a P-Matrix
    return True