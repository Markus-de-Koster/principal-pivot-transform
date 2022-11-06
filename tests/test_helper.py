import unittest

from ppt.helper import powerset


class TestPowerSet(unittest.TestCase):
    def test_powerset(self):
        s = set(range(0, 3))
        exp = {(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)}
        self.assertSetEqual(exp, set(list(powerset(s))))


if __name__ == "__main__":
    unittest.main()  # only works from command line
