import numpy as np


def chi_square_distance(query_vec, reference_db):
    """
    Standard Chi-Square Distance Strategy.
    Calculates similarity between the uploaded logo and the database.
    """
    # 1e-10 prevents 'Division by Zero' if a vector is empty
    return 0.5 * np.sum(((reference_db - query_vec)**2) /
                        (reference_db + query_vec + 1e-10), axis=1)
