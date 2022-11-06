from itertools import chain, combinations


def powerset(iterable):
    """
    Create all subsets, source: https://stackoverflow.com/a/1482316
    e.g. for iterable = {0,1,2} this will return (), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)
    :param iterable: to create all sub-sets from
    :return: all sub-sets
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
