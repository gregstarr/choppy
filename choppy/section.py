"""Cross section and connected component"""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon
from trimesh import Trimesh

from choppy import bsp_node, settings, utils
from choppy.exceptions import CcTooSmallError, CrossSectionError
from choppy.logger import logger


class CrossSection:
    """cross section created by plane intersection with mesh, should contain at least
    one connected component"""

    plane: bsp_node.Plane
    cc_polygons: list[Polygon]
    xform: np.ndarray
    objective: float

    def __init__(self, mesh: Trimesh, plane: bsp_node.Plane):
        self.plane = plane
        self.cc_polygons = []
        path3d = mesh.section(plane_origin=plane.origin, plane_normal=plane.normal)
        if path3d is None:
            # 'Missed' the part
            logger.warning("plane missed part")
            logger.info(
                "%s %s %s",
                min(mesh.vertices @ plane.normal),
                max(mesh.vertices @ plane.normal),
                plane.origin @ plane.normal
            )
            raise CrossSectionError(1)

        # triangulate the cross section
        path2d, self.xform = path3d.to_planar()
        path2d.merge_vertices()
        self.objective = 0
        for polygon in path2d.polygons_full:
            if polygon.area < settings.MIN_CC_AREA:
                raise CcTooSmallError(polygon)
            self.cc_polygons.append(polygon)
            eroded_polygon = polygon.buffer(-1 * settings.CONNECTOR_BUFFER)
            if eroded_polygon.area == 0:
                raise CcTooSmallError(polygon)
            cc_obj = polygon.area / eroded_polygon.area
            self.objective = max(self.objective, cc_obj)

    def split(self, mesh: Trimesh, separate=True) -> tuple[Trimesh, Trimesh]:
        """splits mesh
        """
        positive = mesh.slice_plane(
            plane_origin=self.plane.origin,
            plane_normal=self.plane.normal,
            cap=True,
            use_pyembree=True,
        )
        utils.trimesh_repair(positive)

        negative = mesh.slice_plane(
            plane_origin=self.plane.origin,
            plane_normal=-1 * self.plane.normal,
            cap=True,
            use_pyembree=True,
        )
        utils.trimesh_repair(negative)

        # split parts and assign to connected components
        if positive.body_count > 1 and separate:
            positive_parts = []
            for p in positive.split(only_watertight=False):
                if not split_mesh_check(p):
                    continue
                utils.trimesh_repair(p)
                positive_parts.append(p)
        else:
            positive_parts = [positive]
        if negative.body_count > 1 and separate:
            negative_parts = []
            for p in negative.split(only_watertight=False):
                if not split_mesh_check(p):
                    continue
                utils.trimesh_repair(p)
                negative_parts.append(p)
        else:
            negative_parts = [negative]
        
        if len(positive_parts) == 0 or len(negative_parts) == 0:
            logger.warning("split lost a part")
            raise CrossSectionError(0)

        return positive_parts, negative_parts


def split_mesh_check(mesh):
    return (
        (mesh.vertices.shape[0] >= 5) & 
        (mesh.faces.shape[0] >= 4) & 
        (mesh.volume > 1)
    )
