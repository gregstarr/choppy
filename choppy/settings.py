"""Module for parameters that the user should not pass in"""
import numpy as np


def uniform_normals(n_t, n_p):
    """http://corysimon.github.io/articles/uniformdistn-on-sphere/
    """
    theta = np.arange(0, np.pi, np.pi / n_t)
    phi = np.arccos(np.arange(0, 1, 1 / n_p))
    theta, phi = np.meshgrid(theta, phi)
    theta = theta.ravel()
    phi = phi.ravel()
    return np.column_stack(
        (np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi))
    )

# planes
PLANE_SPACING = 10
N_THETA = 4
N_PHI = 3
NORMALS = uniform_normals(N_THETA, N_PHI)
ADD_MIDDLE_PLANE = True
DIFFERENT_ORIGIN_TH = 9  # should be less than PLANE_SPACING
DIFFERENT_ANGLE_TH = np.pi / 10
N_RANDOM_ROTATIONS = 400

# objective parameters
OBJECTIVE_WEIGHTS = {
    "part": 1,
    "utilization": .25,
    "connector": .1,
    "fragility": 1,
    "seam": 0,  # set to zero until implemented
    "symmetry": 0  # set to zero until implemented
}
FRAGILITY_OBJECTIVE_TH = .95
CONNECTOR_OBJECTIVE_TH = 10
OBB_UTILIZATION = False

# connector / cross-section
CONNECTOR_COLLISION_PENALTY = 10 ** 6
EMPTY_CC_PENALTY = 10**-5
SA_INITIALIZATION_ITERATIONS = 5_000
SA_ITERATIONS = 100_000
CONNECTOR_SIZES = [2, 6, 10]  # diameter (mm)
CONNECTOR_TOLERANCE = .2

# run options
BEAM_WIDTH = 3
MAX_FACES = 3000
