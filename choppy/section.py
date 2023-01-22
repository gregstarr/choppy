"""Cross section and connected component"""
from __future__ import annotations
import numpy as np
from shapely.affinity import rotate
from shapely.geometry import MultiPoint, Point, Polygon
from trimesh import Trimesh, transform_points
from trimesh.creation import triangulate_polygon

from choppy import settings, utils, bsp_node, connector as _conn
from choppy.logger import logger


class ConnectedComponentError(Exception):
    ...


class CrossSectionError(Exception):
    ...


connector_spec = np.dtype(
    [
        ("center", np.float32, (3, )),
        ("radius", np.float32, (1, )),
        ("center_2d", np.float32, (2, )),
    ]
)


class ConnectedComponent:
    """a connected component of a mesh-plane intersection"""

    polygon: Polygon
    plane: bsp_node.Plane
    a_point: np.ndarray
    area: float
    positive: int
    negative: int
    objective: float
    connectors: np.ndarray[connector_spec]
    mesh: Trimesh

    def __init__(
        self, polygon: Polygon, xform: np.ndarray, plane: bsp_node.Plane, mesh: Trimesh
    ):
        self.polygon = polygon
        self.plane = plane
        self.area = self.polygon.area
        self.positive = None
        self.negative = None

        self.connectors = self._get_connectors(mesh, xform)
        self.objective = self._get_objective()

        verts, faces = triangulate_polygon(polygon, triangle_args="p")
        verts = np.column_stack((verts, np.zeros(len(verts))))
        verts = transform_points(verts, xform)
        faces = np.fliplr(faces)
        self.mesh = Trimesh(verts, faces)
        self.a_point = self.mesh.sample(1)[0]

    def _get_connectors(self, mesh, xform):
        connectors = []
        for diameter in settings.CONNECTOR_SIZES:
            radius = diameter / 2
            # potential connector sites
            plane_samples = grid_sites_polygon(self.polygon, radius)
            if len(plane_samples) == 0:
                continue
            mesh_samples = transform_points(
                np.column_stack((plane_samples, np.zeros(plane_samples.shape[0]))),
                xform,
            )
            # distance from connector sites to mesh
            dists = mesh.nearest.signed_distance(mesh_samples)
            # check that connector sites are far enough away from mesh (won't protrude)
            valid_mask = dists >= radius
            if valid_mask.sum() == 0:
                continue
            cdata = [
                (center, radius, center_2d) for center, center_2d in 
                zip(mesh_samples[valid_mask], plane_samples[valid_mask])
            ]
            connectors.append(np.array(cdata, dtype=connector_spec))
        if len(connectors) == 0:
            logger.info("no valid connector sites")
            raise ConnectedComponentError("no valid connector sites")
        connectors = np.concatenate(connectors)
        return connectors

    def _get_objective(self):
        """compute objective and valid connector sites for the interface

        Args:
            positive (Trimesh): mesh pre-split

        Returns:
            bool: success
        """
        plane_samples = self.connectors["center_2d"]
        chull_area = MultiPoint(plane_samples).convex_hull.area
        objective = max(self.area / chull_area - settings.CONNECTOR_OBJECTIVE_TH, 0)
        return objective


def grid_sites_polygon(polygon: Polygon, radius: float) -> np.ndarray:
    """Returns a grid of connector cantidate locations. The connected component
    (chop cross section) is rotated to align with the minimum rotated bounding box,
    then the resulting polygon is grid sampled

    Args:
        polygon (Polygon)
        diameter (float)

    Returns:
        np.ndarray: samples
    """
    spacing = 2 * radius

    # erode polygon by radius
    eroded_polygon = polygon.buffer(-1 * radius)
    # cancel if too small
    min_area = 1.5 * np.pi * radius ** 2
    if eroded_polygon.area < min_area:
        return np.array([])

    # points of minimum rotated rectangle
    mrr_points = np.column_stack(eroded_polygon.minimum_rotated_rectangle.boundary.xy)
    mrr_edges = np.diff(mrr_points, axis=0)
    # decide on spacing
    mrr_edge_lengths = np.hypot(mrr_edges[:, 0], mrr_edges[:, 1])
    if min(mrr_edge_lengths) < 10 * radius:
        spacing = radius
    else:
        spacing = 2 * radius
    
    # get angle of MRR
    angle = -1 * np.arctan2(mrr_edges[0, 1], mrr_edges[0, 0])
    # rotate eroded polygon so it has no rotation
    rotated_polygon = rotate(eroded_polygon, angle, use_radians=True, origin=(0, 0))
    min_x, min_y, max_x, max_y = rotated_polygon.bounds

    # sample rotated polygon, cancel if no samples
    xp = np.arange(min_x, max_x, spacing)
    if len(xp) == 0:
        return np.array([])
    xp += (min_x + max_x) / 2 - (xp.min() + xp.max()) / 2
    yp = np.arange(min_y, max_y, spacing)
    if len(yp) == 0:
        return np.array([])
    yp += (min_y + max_y) / 2 - (yp.min() + yp.max()) / 2
    X, Y = np.meshgrid(xp, yp)
    xy = np.stack((X.ravel(), Y.ravel()), axis=1)

    # unrotate points
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    xy = xy @ rotation

    # check which points are inside eroded polygon
    mask = np.zeros(xy.shape[0], dtype=bool)
    for i in range(xy.shape[0]):
        point = Point(xy[i])
        if point.within(eroded_polygon):
            mask[i] = True
    return xy[mask]


class CrossSection:
    """cross section created by plane intersection with mesh, should contain at least
    one connected component"""

    valid: bool
    plane: bsp_node.Plane
    connected_components: list[ConnectedComponent]
    xform: np.ndarray

    def __init__(self, mesh: Trimesh, plane: bsp_node.Plane):
        self.valid = False
        self.plane = plane
        self.connected_components = []
        path3d = mesh.section(plane_origin=plane.origin, plane_normal=plane.normal)
        if path3d is None:
            # 'Missed' the part
            logger.warning("plane missed part")
            CrossSectionError("plane missed part")

        # triangulate the cross section
        path2d, self.xform = path3d.to_planar()
        path2d.merge_vertices()
        for polygon in path2d.polygons_full:
            cc = ConnectedComponent(polygon, self.xform, self.plane, mesh)
            self.connected_components.append(cc)

    def split(self, mesh: Trimesh) -> tuple[Trimesh, Trimesh]:
        """splits mesh

        Args:
            mesh (Trimesh)

        Returns:
            tuple[Trimesh, Trimesh]: two meshes resulting from split
        """
        positive = mesh.slice_plane(
            plane_origin=self.plane.origin, plane_normal=self.plane.normal, cap=True
        )
        utils.trimesh_repair(positive)

        negative = mesh.slice_plane(
            plane_origin=self.plane.origin, plane_normal=-1 * self.plane.normal, cap=True
        )
        utils.trimesh_repair(negative)

        # split parts and assign to connected components
        positive_parts = positive.split(only_watertight=False)
        negative_parts = negative.split(only_watertight=False)
        if len(positive_parts) == 0 or len(negative_parts) == 0:
            logger.warning("split lost a part")
            raise CrossSectionError("split lost a part")

        cc_pts = np.array(
            [cc.a_point for cc in self.connected_components]
        )

        for i, part in enumerate(positive_parts):
            dist = abs(part.nearest.signed_distance(cc_pts))
            for idx in np.argwhere(dist < 1.0e-6)[:, 0]:
                self.connected_components[idx].positive = i

        for i, part in enumerate(negative_parts):
            dist = abs(part.nearest.signed_distance(cc_pts))
            for idx in np.argwhere(dist < 1.0e-6)[:, 0]:
                self.connected_components[idx].negative = i + len(positive_parts)
        
        cc_idx = [cc.positive for cc in self.connected_components]
        cc_idx += [cc.negative for cc in self.connected_components]
        if None in cc_idx:
            logger.warning("cc missing part")
            raise CrossSectionError("cc missing part")

        parts = np.concatenate((positive_parts, negative_parts))
        if not np.all(np.isin(np.arange(len(parts)), np.unique(cc_idx))):
            logger.warning("part missing cc")
            raise CrossSectionError("part missing cc")
        return parts
