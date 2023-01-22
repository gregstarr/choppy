import tempfile
from pathlib import Path

import numpy as np
import pytest
import trimesh
from pychop3d import settings, utils, bsp_node
from pychop3d.main import run


with tempfile.TemporaryDirectory() as tmpdir:
    run(
        Path(__file__).parent / "test_meshes" / "Bunny-LowPoly.stl",
        np.array([100, 100, 100]),
        "test_chop",
        Path(tmpdir),
    )
