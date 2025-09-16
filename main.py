import numpy as np
from birkhoff import birkhoff_von_neumann_decomposition

def as_list(iterable_of_arrays):
    """Converts an iterable of permutation matrices given as NumPy
    arrays into a list of lists.

    """
    return [array.tolist() for array in iterable_of_arrays]



def main():
    D = (1 / 6) * np.array([[1, 4, 0, 1],
                            [2, 1, 3, 0],
                            [2, 1, 1, 2],
                            [1, 0, 2, 3]])
    P1 = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    P2 = np.identity(4)
    P3 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    P4 = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    expected_coefficients = [1 / 6, 1 / 6, 1 / 3, 1 / 3]
    expected_permutations = [P1, P2, P3, P4]
    actual = birkhoff_von_neumann_decomposition(D)
    actual_coefficients, actual_permutations = zip(*actual)
    assert sorted(actual_coefficients) == sorted(expected_coefficients)
    # Convert the permutation matrices into a list of lists for easy sorting.
    expected_permutations = as_list(expected_permutations)
    actual_permutations = as_list(actual_permutations)
    assert sorted(actual_permutations) == sorted(expected_permutations)
    # Now that we know the coefficients and permutations are as we expected,
    # let's double check that the doubly stochastic matrix is actually the sum
    # of the scaled permutation matrices.
    assert np.all(D == sum(c * P for c, P in actual))



if __name__ == "__main__":
    main()
