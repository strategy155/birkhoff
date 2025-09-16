# test_birkhoff.py - unit tests for Birkhoff--von Neumann decomposition
#
# Copyright 2015 Jeffrey Finkelstein.
#
# This file is part of Birkhoff.
#
# Birkhoff is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Birkhoff is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# Birkhoff.  If not, see <http://www.gnu.org/licenses/>.
"""Unit tests for the :mod:`birkhoff` module.

"""
# Imports from built-in packages.
from __future__ import division

# Imports from third-party packages.
import numpy as np

# Imports from this package.
from birkhoff import birkhoff_von_neumann_decomposition


def as_list(iterable_of_arrays):
    """Converts an iterable of permutation matrices given as NumPy
    arrays into a list of lists.

    """
    return [array.tolist() for array in iterable_of_arrays]




def test_birkhoff_von_neumann_decomposition():
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


def test_scaled():
    """Tests for computing the Birkhoff decomposition of a scaled doubly
    stochastic matrix.

    """
    # If this matrix were multiplied by the scalar 1 / 6, the result
    # would be a true doubly stochastic matrix.
    D = np.array([[1, 4, 0, 1],
                  [2, 1, 3, 0],
                  [2, 1, 1, 2],
                  [1, 0, 2, 3]])
    P1 = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    P2 = np.identity(4)
    P3 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    P4 = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    expected_coefficients = [1, 1, 2, 2]
    expected_permutations = [P1, P2, P3, P4]
    actual = birkhoff_von_neumann_decomposition(D)
    actual_coefficients, actual_permutations = zip(*actual)
    assert sorted(actual_coefficients) == sorted(expected_coefficients)
    expected_permutations = as_list(expected_permutations)
    actual_permutations = as_list(actual_permutations)
    assert sorted(actual_permutations) == sorted(expected_permutations)
    # Now that we know the coefficients and permutations are as we expected,
    # let's double check that the doubly stochastic matrix is actually the sum
    # of the scaled permutation matrices.
    assert np.all(D == sum(c * P for c, P in actual))
