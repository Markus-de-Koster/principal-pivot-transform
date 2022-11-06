import unittest

import numpy as np

import ppt.ppt as ppt


class TestGetSubmatrices(unittest.TestCase):
    def test_get_submatrices_for_indices(self):
        m = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12],
                      [13, 14, 15, 16]])
        a_exp = [[6, 7], [10, 11]]
        b_exp = [[5, 8], [9, 12]]
        c_exp = [[2, 3], [14, 15]]
        d_exp = [[1, 4], [13, 16]]
        a, b, c, d = ppt.get_submatrices_for_indices(m, [1, 2])
        self.assertTrue(np.allclose(a_exp, a, rtol=1e-9))
        self.assertTrue(np.allclose(b_exp, b, rtol=1e-9))
        self.assertTrue(np.allclose(c_exp, c, rtol=1e-9))
        self.assertTrue(np.allclose(d_exp, d, rtol=1e-9))


class TestPrincipalPivotTransform(unittest.TestCase):

    def test_principal_pivot_transform_00(self):
        """
        Test with solution from paper
        Example 3.7
        """
        a_mat = np.array([[1, 2, 1], [1, 1, 0], [2, 8, 1]])
        alpha = [0, 2]
        b_mat = ppt.principal_pivot_transform(a_mat, alpha)
        b_mat_exp = np.array([[-1, -6, 1], [-1, -5, 1], [2, 4, -1]])

        self.assertTrue(np.allclose(b_mat_exp, b_mat, rtol=1e-9))

        # Test the partial exchange of values at row indices 1,3
        sol1 = np.dot(a_mat, [1, 1, 1])
        sol2 = np.dot(b_mat, [4, 1, 11])
        sol1_expected = [4, 2, 11]
        sol2_expected = [1, 2, 1]
        self.assertTrue(np.allclose(sol1_expected, sol1, rtol=1e-9))
        self.assertTrue(np.allclose(sol2_expected, sol2, rtol=1e-9))

    def test_principal_pivot_transform_01(self):
        """
        second example from paper
        """
        a_mat = np.array([[1, 2, 1], [1, 1, 0], [2, 8, 1]])
        b_mat_exp = np.array([[-1, -6, 1], [-1, -5, 1], [2, 4, -1]])
        a_inv = np.linalg.inv(a_mat)
        beta = [1]
        c_mat = ppt.principal_pivot_transform(b_mat_exp, beta)
        self.assertTrue(np.allclose(a_inv, c_mat, rtol=1e-9))

    def test_principal_pivot_transform_02(self):
        """
        Test complex numbers
        """
        a_mat = np.array([[(1 + 1j), (2 + 2j), (1 - 1j)], [(1 + 1j), (1 + 1j), 0j], [(2 - 0j), 8, (1 + 1j)]])
        b = ppt.principal_pivot_transform(a_mat, [0, 2])
        b_expected = np.array([[(0.1 - 0.3j), (-3.2 - 0.4j), (0.3 + 0.1j)],
                               [(0.4 - 0.2j), (-1.8 - 2.6j), (0.2 + 0.4j)],
                               [(0.2 + 0.4j), (-0.4 + 1.2j), (0.1 - 0.3j)]])
        self.assertTrue(np.allclose(b_expected, b, rtol=1e-9))

    def test_principal_pivot_transform_03(self):
        """
        setting alpha to <n> should give the inverse of the original matrix
        """
        a_mat = np.array([[1, 2, 1], [1, 1, 0], [2, 8, 1]])
        b_mat_exp = np.linalg.inv(a_mat)
        alpha = [0, 1, 2]
        b_mat = ppt.principal_pivot_transform(a_mat, alpha)
        self.assertTrue(np.allclose(b_mat_exp, b_mat, rtol=1e-9))

    def test_principal_pivot_transform_04(self):
        """
        linearly dependant matrix
        """
        a_mat = np.array([[1, 1], [2, 2]])
        b_mat_exp = np.array([[1, -1], [2, 0]])
        alpha = [0]
        b_mat = ppt.principal_pivot_transform(a_mat, alpha)
        from numpy.linalg import LinAlgError
        with self.assertRaises(LinAlgError):
            np.linalg.inv(a_mat)  # numpy.linalg.LinAlgError: Singular matrix
        self.assertTrue(np.allclose(b_mat_exp, b_mat, rtol=1e-9))

    def test_principal_pivot_transform_05(self):
        """
        linearly dependant principal submatrix
        """
        a_mat = np.array([[1, 2, 1], [1, 1, 0], [1, 8, 1]])
        alpha = [0, 2]
        from numpy.linalg import LinAlgError
        with self.assertRaises(LinAlgError):
            b_mat = ppt.principal_pivot_transform(a_mat, alpha)

    def test_principal_pivot_transform_06(self):
        """
        GPPT with linearly dependant submatrix
        """
        a_mat = np.array([[1, 2, 1], [1, 1, 0], [1, 8, 1]])
        alpha = [0, 2]
        a, b, c, d = ppt.get_submatrices_for_indices(a_mat, alpha)
        with self.assertRaises(ValueError):  # GPPT does not hold
            b_mat = ppt.principal_pivot_transform(a_mat, alpha, generalized=True)

    def test_theorem_3_3(self):
        """
        Rajesh Kannan Bapat 2014 Theorem 3.3
        If N (a) ⊆ N (c) and N (a*) ⊆ N (b*)
        pinv(gppt(m, a)) = gppt(m,d)
        :param m:
        :param alpha:
        :return:
        """
        a_mat = np.array([[1, 2, 1], [1, 1, 0], [2, 8, 1]])
        alpha = [0, 2]
        gppt_a = ppt.principal_pivot_transform(a_mat, alpha, generalized=True)
        gppt_d = ppt.principal_pivot_transform(a_mat, [1], generalized=True)
        gppt_a_pinv = np.linalg.pinv(gppt_a)
        a, b, c, d = ppt.get_submatrices_for_indices(a_mat, alpha)  # get sub-matrices
        if not ppt.check_gppt(a, b, c):
            raise ValueError("GPPT does not hold")
        self.assertTrue(np.allclose(gppt_a_pinv, gppt_d, rtol=1e-7))


class TestGeneralizedPrincipalPivotTransform(unittest.TestCase):
    def test_check_gppt_00(self):
        a_mat = np.array([[1, 2, 1], [1, 1, 0], [2, 8, 1]])
        a, b, c, _ = ppt.get_submatrices_for_indices(a_mat, [0, 2])
        self.assertTrue(ppt.check_gppt(a, b, c))

    def test_check_gppt_01(self):
        a_mat = np.array([[2, 3, 5], [-4, 2, 3], [0, 0, 0]])
        a, b, c, _ = ppt.get_submatrices_for_indices(a_mat, [0, 2])
        self.assertFalse(ppt.check_gppt(a, b, c))

    def test_check_gppt_02(self):
        a_mat = np.array([[1, 3, 0],
                          [-2, -6, 0],
                          [3, 9, 6]])
        a, b, c, _ = ppt.get_submatrices_for_indices(a_mat, [0, 1])
        self.assertFalse(ppt.check_gppt(a, b, c))


if __name__ == "__main__":
    unittest.main()  # only works from command line
