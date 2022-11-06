import numpy as np
from sympy import Matrix
from ppt.helper import powerset


def is_symmetric(a, tol=1e-8):
    """
    check symmetry of a matrix by subtracting transpose of matrix
    :param a: matrix
    :param tol: tolerance required due to floating point imprecision
    :return:
    """
    return np.all(np.abs(a - a.T) < tol)


def is_square(a):
    """
    check if the matrix is a square matrix, i.e. number of rows equals number of columns
    :param a: matrix
    :return:
    """
    return a.shape[0] == a.shape[1]


def nullspace(a):
    """
    Null space is often described as Kernel.
    It is given by the vector space that multiplied with the given vector space is mapped to the zero vector
    For square matrices the null space only exists if the determinant is 0, otherwise the subspace only contains the zero vector.
    The zero vector is always part of the null space and thus trivial (will not be returned)
    :param a: matrix
    :return: spanning nullspace vector(s)
    """
    m = Matrix(a)
    ns = m.nullspace()
    ns_arr = np.array(ns).astype(complex)
    return ns_arr


def is_range_hermitian(a):
    """
    Is the Matrix range-hermitian (range symmetric)?
    rank(a) = rank(conj.(a))*
    R(A) = R(A∗)(R(A) = R(At)).
    null_space(A) = null_space(conj.(a))
    then also: group inverse = moore penrose inverse
    :param a: matrix
    :return: true if a is range-hermitian, else false
    """
    ra = np.linalg.matrix_rank(a)
    rac = np.linalg.matrix_rank(np.conj(a))
    if ra != rac:
        return False
    na = nullspace(a)
    nac = nullspace(np.conj(a))
    raise NotImplemented # TODO check everything below for accuracy
    if not is_linear_subspace(na, nac):
        return False
    A = Matrix(a)
    rs = A.columnspace()
    rsc = A.columnspace(Matrix(np.conj(a)))
    if not is_linear_subspace(rs, rsc):
        return False
    return True


def is_linear_subspace(vs1, vs2):
    """
    Checks whether vs1 is a linear subspace (i.e. a subset) of vs2.
    If a vector space is "empty" it still contains the zero vector which is a part of all vector spaces.
    (See Axiom for 'Identity Element'): https://en.wikipedia.org/wiki/Vector_space )
    Thus, if vs1 is empty, True will be returned.
    Methodology:
    - prove that zero vector is a part of the subspace (given see above)
    - prove each vector of vs1 multiplied by a scalar is a part of vs2 (closure over scalar multiplication)
        - this is proven through checking whether the vector is linearly dependent, then scaling doesn't matter
    - prove each vector of vs1 added to each other vector of vs1 is a part of vs2 (closure over vector addition)
        - iterate over all vectors, then again prove linear dependence
    :param vs1: vector space 1
    :param vs2: vector space 2
    :return: true if vs1 is a subset of vs2
    """
    for v in vs1:
        if not is_vector_linearly_dependent_vector_space(v, vs2):
            return False
        for v1 in vs1:
            if not is_vector_linearly_dependent_vector_space(v + v1, vs2):
                return False
    return True


def vector_in_vector_space(vs, v):
    """
    Checks whether the given vector space contains the given vector
    also accounts for floating point imprecision
    Does not account for scalar multiplication or vector addition within vector spaces
    :param vs: vector space
    :param v: vector
    :return: true if v is contained in vs within tolerance
    """
    for v1 in vs:
        if np.allclose(v, v1, rtol=1e-9):
            return True
    return False


def is_vector_linearly_dependent_vector_space(v, vs):
    """
    Is vector v linearly dependent on any of the vectors in the given vector space.
    This method uses row echolon reduction and checks whether the given row is all zeros
    Then it is lineraly dependent on other rows.
    :param v: vector
    :param vs: vector space
    :return: true if vector is linearly dependent on vector space
    """
    # confirm shape is equal
    n = v.shape[0]
    m = list(vs)[0].shape[0]
    if not n == m:
        raise ValueError("Wrong shape of vector")
    # vector space to matrix as row vectors
    a = np.empty([len(vs) + 1, n])
    i = 0
    for v1 in vs:
        vt = v1.transpose()
        a[i] = vt
        i += 1
    a[i] = v.transpose()
    # add vector to matrix
    # reduce matrix / calculate eigenvalues
    A = Matrix(a)
    A_rref, pivot_indices = A.rref()
    a_rref = np.array(A_rref).astype(complex)
    zero_vec = np.zeros(n, dtype=complex)
    if np.allclose(a_rref[i], zero_vec, rtol=1e-9):
        return True
    return False
