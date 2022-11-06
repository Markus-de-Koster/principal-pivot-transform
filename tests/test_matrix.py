import unittest
import numpy as np

import ppt.matrix as mat


class TestSymmetry(unittest.TestCase):
    """
    Unit Tests for symmetry checks
    """

    def test_symmetric(self):
        a = np.array([[1, 2, 1], [1, 1, 0], [2, 8, 1]])
        self.assertFalse(mat.is_symmetric(a))

    def test_asymmetric(self):
        a = np.array([[1, 2, 1], [2, 1, 0], [1, 0, 1]])
        self.assertTrue(mat.is_symmetric(a))


class TestSquare(unittest.TestCase):
    """
    Unit Tests for square checks
    """

    def test_square(self):
        a = np.array([[1, 2, 1], [1, 1, 0], [2, 8, 1]])
        self.assertTrue(mat.is_square(a))

    def test_not_square(self):
        a = np.array([[1, 2, 1], [1, 1, 0], [2, 8, 1], [1, 2, 3]])
        self.assertFalse(mat.is_square(a))


class TestPMatrix(unittest.TestCase):
    def test_is_p_matrix_00(self):
        a_mat = np.array([[1, 2, 1], [1, 1, 0], [2, 8, 1]])
        self.assertFalse(mat.is_p_matrix(a_mat))

    def test_is_p_matrix_01(self):
        a_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertTrue(mat.is_p_matrix(a_mat))



class TestNullSpace(unittest.TestCase):

    def test_null_space_02(self):
        from sympy import Matrix
        a = [[2, 3, 5], [-4, 2, 3], [0, 0, 0]]
        A = Matrix(a)
        zero_vec = np.array(A * A.nullspace()[0]).astype(float)
        ns = mat.nullspace(a)
        arr = np.array(ns).astype(float)
        self.assertTrue(np.allclose([[0, 0, 0]], zero_vec, rtol=1e-9))
        self.assertTrue(np.allclose([[[-1 / 16], [-13 / 8], [1]]], arr, rtol=1e-9))

    def test_null_space_03(self):
        """
        More than one result vector
        """
        a_mat = [[1, 2, 3, 0, 0], [4, 10, 0, 0, 1]]
        nsa = mat.nullspace(a_mat)
        exp_0 = np.array([[-15, 6, 1, 0, 0]]).transpose()
        exp_1 = np.array([[0, 0, 0, 1, 0]]).transpose()
        exp_2 = np.array([[1, -1 / 2, 0, 0, 1]]).transpose()
        self.assertTrue(np.allclose(exp_0, nsa[0], rtol=1e-9))
        self.assertTrue(np.allclose(exp_1, nsa[1], rtol=1e-9))
        self.assertTrue(np.allclose(exp_2, nsa[2], rtol=1e-9))


class TestRank(unittest.TestCase):
    """
    rank is the maximum number of linearly independent columns
    """

    def test_rank_00(self):
        a_mat = np.array([[3, 1, -1], [2, -2, 0], [1, 2, -1]])
        rank = np.linalg.matrix_rank(a_mat)
        self.assertEqual(3, rank)

    def test_rank_01(self):
        a_mat = np.array([[3, 1, -1], [6, 2, -2], [1, 2, -1]])  # linearly dependent
        rank = np.linalg.matrix_rank(a_mat)
        self.assertEqual(2, rank)


class TestHermitianTranspose(unittest.TestCase):
    def test_hermitian_transpose(self):
        a_mat = np.array([[3, 1, -1], [2, -2, 0], [1, 2, -1]])
        b_mat = np.matrix.getH(a_mat)
        c_mat = np.conj(a_mat).T  # should be the same
        self.assertTrue(np.allclose(b_mat, c_mat, rtol=1e-9))


class TestRangeHermitian(unittest.TestCase):
    def test_range_hermitian_00(self):
        # TODO: define test case
        pass


class TestLinearSubspace(unittest.TestCase):
    def test_vector_in_vector_space(self):
        v1 = np.array([[-15, 6, 1, 0, 0]]).transpose()
        v2 = np.array([[1, -1 / 2, 0, 0, 1]]).transpose()
        vs = [v1, v2]
        self.assertTrue(mat.vector_in_vector_space(vs, v1))

    def test_vector_in_vector_space(self):
        v1 = np.array([[-15, 6, 1, 0, 0]]).transpose()
        v2 = np.array([[1, -1 / 2, 0, 0, 1]]).transpose()
        v2m = np.array([[1, -1, 0, 0, 1]]).transpose()
        vs = [v1, v2]
        self.assertFalse(mat.vector_in_vector_space(vs, v2m))

    def test_is_linear_subspace_00(self):
        v1 = np.array([[-15, 6, 1, 0, 0]]).transpose()
        v2 = np.array([[0, 0, 0, 1, 0]]).transpose()
        v3 = np.array([[1, -1 / 2, 0, 0, 1]]).transpose()
        vs1 = [v1, v3]
        vs2 = [v1, v2, v3]
        self.assertTrue(mat.is_linear_subspace(vs1, vs2))

    def test_is_linear_subspace_01(self):
        v1 = np.array([[-15, 6, 1, 0, 0]]).transpose()
        v2 = np.array([[0, 0, 0, 1, 0]]).transpose()
        v3 = np.array([[1, -1 / 2, 0, 0, 1]]).transpose()
        vs1 = [v1, v3]
        vs2 = [v1, v2, v3]
        self.assertFalse(mat.is_linear_subspace(vs1=vs2, vs2=vs1))
        # wrong order of parameters, larger one will never be subset

    def test_is_vector_linearly_dependent_vector_space_00(self):
        """
        divide vector by 3, obviously dependent
        """
        v1 = np.array([[-15, 6, 1, 0, 0]]).transpose()
        v2 = np.array([[0, 0, 0, 1, 0]]).transpose()
        v3 = np.array([[1, -1 / 2, 0, 0, 1]]).transpose()
        vs1 = [v1, v2, v3]
        self.assertTrue(mat.is_vector_linearly_dependent_vector_space(v1 / 3, vs1))

    def test_is_vector_linearly_dependent_vector_space_01(self):
        """
        add vectors together, obviously dependent
        """
        v1 = np.array([[-15, 6, 1, 0, 0]]).transpose()
        v2 = np.array([[0, 0, 0, 1, 0]]).transpose()
        v3 = np.array([[1, -1 / 2, 0, 0, 1]]).transpose()
        vs1 = [v1, v2, v3]
        self.assertTrue(mat.is_vector_linearly_dependent_vector_space(v1 + v2, vs1))

    def test_is_vector_linearly_dependent_vector_space_02(self):
        """
        vs1 spans a plane, v3 goes into 3rd dimension, obviously not dependent
        """
        v1 = np.array([[0, 0, 1]]).transpose()
        v2 = np.array([[0, 1, 0]]).transpose()
        v3 = np.array([[5, 1, 0]]).transpose()
        vs1 = [v1, v2]
        self.assertFalse(mat.is_vector_linearly_dependent_vector_space(v3, vs1))


if __name__ == "__main__":
    unittest.main()  # only works from command line
