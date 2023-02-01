from __future__ import annotations

import numpy as np
from numba import jit
from shapely import Polygon, contains_xy
from trimesh import Trimesh, transform_points
from trimesh.creation import triangulate_polygon
from trimesh.primitives import Sphere

from choppy import bsp_tree, settings, utils
from choppy.exceptions import (
    InvalidOperationError,
    OperationFailedError,
    CcTooSmallError
)
from choppy.logger import logger, progress


@jit(nopython=True)
def evaluate_connector_objective(
    connectors: np.ndarray[connector_t],
    collisions: np.array,
    cc_area: list[float],
    state: np.ndarray,
) -> float:
    """evaluates connector objective given a state vector"""
    objective = 0
    n_collisions = collisions[state, :][:, state].sum()
    objective += settings.CONNECTOR_COLLISION_PENALTY * n_collisions
    if n_collisions > 0:
        return objective

    for i, area in enumerate(cc_area):
        ci = 0
        cc_mask = connectors["cci"] == i
        cc_conn = connectors[cc_mask & state]
        if cc_conn.shape[0] > 0:
            rc = 12 * cc_conn["radius"]
            rc = np.minimum(rc, np.sqrt(area) / 2)
            ci = np.pi * np.sum(rc**2)
            arr1 = np.expand_dims(cc_conn["center"], 1)
            arr2 = np.expand_dims(cc_conn["center"], 0)
            distances = np.sqrt(np.sum((arr1 - arr2) ** 2, axis=2))
            mask = distances > 0
            rc_sq = np.expand_dims(rc, 1) + np.expand_dims(rc, 0)
            mask &= distances < rc_sq
            ci -= np.sum(
                np.pi * 
                (np.where(mask, rc_sq, 0) - np.where(mask, distances, 0) / 2) ** 2
            )
        objective += area / (settings.EMPTY_CC_PENALTY + max(0, ci))

    return objective


@jit(nopython=True)
def sa_iteration(
    connectors: np.ndarray[connector_t],
    collisions: np.array,
    cc_area: list[float],
    state: np.ndarray,
    objective: float,
    temp: float,
) -> tuple[np.ndarray, float]:
    """run a single simulated annealing iteration"""
    new_state = state.copy()
    if np.random.randint(0, 2) or not state.any() or state.all():
        idx = sample(np.ones(connectors.shape[0]).astype("bool"))
        new_state[idx] = 0 if state[idx] else 1
    else:
        add = sample(~state)
        remove = sample(state)
        new_state[add] = 1
        new_state[remove] = 0

    new_objective = evaluate_connector_objective(
        connectors, collisions, cc_area, new_state
    )
    if new_objective < objective:
        return new_state, new_objective
    elif temp > 0:
        if np.random.rand() < np.exp(-(new_objective - objective) / temp):
            return new_state, new_objective
    return state, objective


@jit(nopython=True)
def sample(mask):
    r = np.random.randint(0, mask.sum())
    return np.flatnonzero(mask)[r]


@jit(nopython=True)
def sa_connector_placement(
    state: np.ndarray,
    connectors: np.ndarray[connector_t],
    collisions: np.array,
    cc_area: list[float],
):
    objective = evaluate_connector_objective(connectors, collisions, cc_area, state)
    # initialization
    for _ in range(settings.SA_INITIALIZATION_ITERATIONS):
        state, objective = sa_iteration(
            connectors, collisions, cc_area, state, objective, 0
        )

    initial_temp = objective / 2
    for temp in np.linspace(initial_temp, 0, settings.SA_ITERATIONS):
        state, objective = sa_iteration(
            connectors, collisions, cc_area, state, objective, temp
        )

    return state


def sample_polygon(polygon: Polygon, radius: float) -> np.ndarray:
    """Returns a grid of connector cantidate locations. The connected component
    (chop cross section) is rotated to align with the minimum rotated bounding box,
    then the resulting polygon is grid sampled

    Args:
        polygon (Polygon)
        diameter (float)

    Returns:
        np.ndarray: samples
    """
    density = 1 / 16  # samples per sq mm
    n_samples = int(np.ceil(2 * density * polygon.area))
    # erode polygon by radius
    eroded_polygon = polygon.buffer(-1 * radius)
    minx, miny, maxx, maxy = polygon.bounds
    x_samples = minx + np.random.rand(n_samples) * (maxx - minx)
    y_samples = miny + np.random.rand(n_samples) * (maxy - miny)
    # check which points are inside eroded polygon
    mask = contains_xy(eroded_polygon, x_samples, y_samples)
    points = np.column_stack((x_samples, y_samples))[mask]
    if points.shape[0] > 200:
        points = np.concatenate((points[:200], points[200::4]))
    return points


connector_t = np.dtype(
    [
        ("center", np.float32, (3,)),
        ("radius", np.float32),
        ("cci", np.uint32)
    ]
)


def _get_connectors(polygon, mesh: Trimesh, xform: np.ndarray, cci):
    connectors = []
    for diameter in settings.CONNECTOR_SIZES:
        radius = diameter / 2
        # potential connector sites
        plane_samples = sample_polygon(polygon, radius)
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
        cdata = [(center, radius, cci) for center in mesh_samples[valid_mask]]
        connectors.append(np.array(cdata, dtype=connector_t))
    connectors = np.concatenate(connectors)
    if connectors.shape[0] == 0:
        raise NoConnectorSitesFoundError()
    return connectors


def get_cap(polygon, xform):
    verts, faces = triangulate_polygon(polygon, triangle_args="p")
    verts = np.column_stack((verts, np.zeros(len(verts))))
    verts = transform_points(verts, xform)
    faces = np.fliplr(faces)
    return Trimesh(verts, faces)


class ConnectorPlacerInputError(Exception):
    ...
class NoConnectorSitesFoundError(Exception):
    ...


class ConnectorPlacer:
    """Manages optimization and placement of connectors"""

    cc_area: list[float]
    cc_path: list[tuple[int]]
    connectors: np.ndarray[connector_t]
    n_connectors: int
    collisions: np.ndarray

    def __init__(self, tree: bsp_tree.BSPTree):
        self.cc_area = []
        connectors = []
        caps = []
        self.n_connectors = 0
        self.cc_path = []
        if len(tree.nodes) < 2:
            raise ConnectorPlacerInputError()

        # create connectors
        logger.info("Creating connectors...")
        for node in tree.nodes:
            if node.cross_section is None:
                continue
            for ccp in node.cross_section.cc_polygons:
                connectors.append(
                    _get_connectors(
                        ccp, node.part, node.cross_section.xform, len(caps)
                    )
                )
                self.cc_area.append(ccp.area)
                self.cc_path.append(node.path)
                caps.append(get_cap(ccp, node.cross_section.xform))
        self.connectors = np.concatenate(connectors)

        # get rid of bad connectors
        logger.info("determining connector-cut intersections")
        intersections = np.zeros(self.connectors.shape[0], dtype=int)
        for cap in caps:
            dist = cap.nearest.on_surface(self.connectors["center"])[1]
            mask = dist < self.connectors["radius"]
            intersections[mask] += 1
        good_connector_mask = intersections <= 1
        self.connectors = self.connectors[good_connector_mask]

        # connector stats
        self.n_connectors = self.connectors.shape[0]
        logger.info("Connection placer stats")
        logger.info("connectors %s", self.n_connectors)
        logger.info("connected components %s", len(self.cc_area))

        # collisions
        self.collisions = np.zeros((self.n_connectors, self.n_connectors), dtype=bool)
        logger.info("determining connector-connector intersections")
        distances = np.sqrt(
            np.sum(
                (
                    self.connectors["center"][:, None, :]
                    - self.connectors["center"][None, :, :]
                )
                ** 2,
                axis=2,
            )
        )
        mask = (distances > 0) & (
            distances
            < (self.connectors["radius"][:, None] + self.connectors["radius"][None, :])
        )
        self.collisions = np.logical_or(self.collisions, mask)
        logger.info("Done setting up connector placer")

    def get_initial_state(self) -> np.ndarray:
        """gets initial random state vector, enables one connector for each connected
        component
        """
        state = np.zeros(self.n_connectors, dtype=bool)
        for i in range(len(self.cc_area)):
            mask = self.connectors["cci"] == i
            idx = sample(mask)
            state[idx] = True
        return state

    def simulated_annealing_connector_placement(self) -> np.ndarray:
        """run simulated annealing to optimize placement of connectors"""
        initial_state = self.get_initial_state()
        initial_objective = evaluate_connector_objective(
            self.connectors,
            self.collisions,
            self.cc_area,
            initial_state
        )
        logger.info("initial connector objective %s", initial_objective)
        logger.info("initial connectors %s", initial_state.sum())
        logger.info(
            "initial collisions %s",
            np.argwhere(self.collisions[initial_state, :][:, initial_state])
        )
        state = sa_connector_placement(
            initial_state,
            self.connectors,
            self.collisions,
            self.cc_area,
        )
        final_objective = evaluate_connector_objective(
            self.connectors,
            self.collisions,
            self.cc_area,
            state
        )
        progress.update(connector_progress=0.3)
        logger.info(
            "finished with connector placement, final objective %s", final_objective
        )
        return state

    def insert_connectors(
        self, tree: bsp_tree.BSPTree, state: np.ndarray, printer_extents: np.ndarray
    ) -> bsp_tree.BSPTree:
        """Insert connectors into a tree according to a state vector

        Args:
            tree (BSPTree): tree
            state (np.ndarray): state vector
            printer_extents (np.ndarray): printer dims (mm)

        Returns:
            BSPTree: tree but all the part meshes have connectors
        """
        logger.info("inserting %s connectors", state.sum())
        if tree.nodes[0].plane is None:
            new_tree = bsp_tree.separate_starter(tree.nodes[0].part, printer_extents)
        else:
            new_tree = bsp_tree.BSPTree(tree.nodes[0].part, printer_extents)
        for ni, node in enumerate(tree.nodes):
            if node.plane is None:
                continue
            try:
                new_tree = bsp_tree.expand_node(
                    new_tree, node.path, node.plane, separate=False
                )
            except CcTooSmallError:
                logger.exception("why")
            new_node = new_tree.get_node(node.path)
            for child_node in new_node.children:
                if child_node.positive:
                    adj = 0
                    op = "union"
                else:
                    adj = settings.CONNECTOR_TOLERANCE / 2
                    op = "difference"
                c_dist = child_node.part.nearest.signed_distance(
                    self.connectors["center"]
                )
                mask = state & (abs(c_dist) < 1.0e-3)
                cc_conn = self.connectors[mask]
                conn = []
                for center, radius, *_ in cc_conn:
                    conn.append(Sphere(radius=radius + adj, center=center))
                conn = sum(conn)
                
                child_node.part = insert_connector(child_node.part, conn, op)
            progress.update(connector_progress=0.3 + 0.7 * (ni + 1) / len(tree.nodes))
        return new_tree


def insert_connector(part: Trimesh, connector: Trimesh, operation: str):
    """Adds a box / connector to a part using boolean union. Operating under the
    assumption that adding a connector MUST INCREASE the number of vertices of the
    resulting part.

    Args:
        part (Trimesh): part to add connector to
        connector (Trimesh): connector to add to part
        operation (str): 'union' or 'difference'

    Raises:
        Exception: retries exceeded
        ValueError: incorrect operation input

    Returns:
        Trimesh: part with connector inserted
    """
    if operation not in ["union", "difference"]:
        raise InvalidOperationError()
    utils.trimesh_repair(part)
    if operation == "union":
        new_part = part.union(connector)
        if new_part.volume > part.volume:
            return new_part
        check = new_part.intersection(connector)
        logger.warning(
            "union connector check failed, intersect volume diff %s",
            abs(check.volume - connector.volume),
        )
        if abs(check.volume - connector.volume) < 1:
            return new_part
    else:
        new_part = part.difference(connector)
        if new_part.volume < part.volume:
            return new_part
        check = new_part.intersection(connector)
        logger.warning(
            "difference connector check failed, intersect volume %s", check.volume
        )
        if check.volume < 1:
            return new_part
    raise OperationFailedError()
