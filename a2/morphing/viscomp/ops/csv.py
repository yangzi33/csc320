# CSC320 Spring 2023
# Assignment 1
# (c) Kyros Kutulakos, Towaki Takikawa, Esther Lin
#
# UPLOADING THIS CODE TO GITHUB OR OTHER CODE-SHARING SITES IS
# STRICTLY FORBIDDEN.
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY FORBIDDEN. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY.
#
# THE ABOVE STATEMENTS MUST ACCOMPANY ALL VERSIONS OF THIS CODE,
# WHETHER ORIGINAL OR MODIFIED.

import os
import numpy as np
import pandas as pd
import sys

def write_csv(mat, path):
    """Writes a numpy matrix as a csv file.

    Args:
        mat (np.array): The matrix to write as a CSV file.
        path (str): Path to a CSV file.
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    arr = pd.DataFrame(mat).to_csv(path, index=False, header=False)

def load_csv(path):
    """Loads a path to a csv file with a numpy matrix.

    Args:
        path (str): Path to a CSV file.

    Returns:
        (np.array): The csv file as a numpy matrix.
    """
    if not os.path.exists(path):
        print(f"Path {path} does not exist!")
        sys.exit(0)
    arr = pd.read_csv(path, header=None).to_numpy()
    return arr
