from __future__ import annotations

import numpy as np
from numba import jit
from shapely import Polygon, contains_xy
from trimesh import Trimesh, transform_points
from trimesh.creation import triangulate_polygon
from trimesh.primitives import Box
from trimesh.collision import CollisionManager
from trimesh.transformations import rotation_matrix, translation_matrix

from choppy import bsp_tree, settings, utils
from choppy.logger import logger, progress
from choppy.exceptions import (
    CcTooSmallError, 
    ConnectorPlacerInputError,
    InvalidOperationError,
    NoConnectorSitesFoundError,
    OperationFailedError
)


@jit(nopython=True)
def evaluate_connector_objective(
    connector_locations: np.ndarray[float],
    connector_cci: np.ndarray[int],
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
        cc_mask = connector_cci == i
        locs = connector_locations[cc_mask & state]
        if locs.shape[0] > 0:
            connector_area = min(settings.CONNECTOR_AREA, area / 4)
            # covered area
            ci = connector_area * locs.shape[0]
            arr1 = np.expand_dims(locs, 1)
            arr2 = np.expand_dims(locs, 0)
            dist2 = np.sum((arr1 - arr2) ** 2, axis=2)
            mask = dist2 > 0
            mask &= dist2 < (4 * connector_area)
            ci -= np.sum(np.where(mask, connector_area - dist2 / 4, 0))
        objective += area / (settings.EMPTY_CC_PENALTY + max(0, ci))

    return objective


@jit(nopython=True)
def sa_iteration(
    connector_locations: np.ndarray[float],
    connector_cci: np.ndarray[int],
    collisions: np.array,
    cc_area: list[float],
    state: np.ndarray,
    objective: float,
    temp: float,
) -> tuple[np.ndarray, float]:
    """run a single simulated annealing iteration"""
    new_state = state.copy()
    if np.random.randint(0, 2) or not state.any() or state.all():
        idx = sample(np.ones(connector_cci.shape[0]).astype("bool"))
        new_state[idx] = 0 if state[idx] else 1
    else:
        add = sample(~state)
        remove = sample(state)
        new_state[add] = 1
        new_state[remove] = 0

    new_objective = evaluate_connector_objective(
        connector_locations, connector_cci, collisions, cc_area, new_state
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
    connector_locations: np.ndarray[float],
    connector_cci: np.ndarray[int],
    collisions: np.array,
    cc_area: list[float],
):
    objective = evaluate_connector_objective(
        connector_locations, connector_cci, collisions, cc_area, state
    )
    # initialization
    for _ in range(settings.SA_INITIALIZATION_ITERATIONS):
        state, objective = sa_iteration(
            connector_locations, connector_cci, collisions, cc_area, state, objective, 0
        )

    initial_temp = objective / 2
    for temp in np.linspace(initial_temp, 0, settings.SA_ITERATIONS):
        state, objective = sa_iteration(
            connector_locations, connector_cci, collisions, cc_area, state, objective, temp
        )

    return state


def sample_polygon(polygon: Polygon) -> np.ndarray:
    """Returns a grid of connector cantidate locations. The connected component
    (chop cross section) is rotated to align with the minimum rotated bounding box,
    then the resulting polygon is grid sampled

    Args:
        polygon (Polygon)
        diameter (float)

    Returns:
        np.ndarray: samples
    """
    density = 1 / 9  # samples per sq mm
    n_samples = int(np.ceil(density * polygon.area))
    # erode polygon by radius
    eroded_polygon = polygon.buffer(-1 * settings.CONNECTOR_BUFFER)
    minx, miny, maxx, maxy = polygon.bounds
    x_samples = minx + np.random.rand(4 * n_samples) * (maxx - minx)
    y_samples = miny + np.random.rand(4 * n_samples) * (maxy - miny)
    # check which points are inside eroded polygon
    mask = contains_xy(eroded_polygon, x_samples, y_samples)
    points = np.column_stack((x_samples, y_samples))[mask]
    return points[:n_samples]


def _get_connectors(polygon, mesh: Trimesh, xform: np.ndarray, cci):
    connectors = []
    corners = []
    # potential connector sites
    plane_samples = sample_polygon(polygon)
    angles = np.random.rand(plane_samples.shape[0]) * np.pi
    for sample, angle in zip(plane_samples, angles):
        c_xform = (
            xform @ 
            translation_matrix([*sample, 0]) @ 
            rotation_matrix(angle, [0, 0, 1])
        )
        b = Box(
            extents=settings.CONNECTOR_SIZE + settings.CONNECTOR_TOLERANCE,
            transform=c_xform
        )
        connectors.append(b)
        corners.append(b.vertices)
    corners = np.stack(corners, axis=0)
    connectors = np.array(connectors)
    # distance from connector sites to mesh
    dists = mesh.nearest.signed_distance(corners.reshape((-1, 3))).reshape((-1, 8))
    # check that connector sites are far enough away from mesh (won't protrude)
    valid_mask = dists.min(axis=1) >= settings.CONNECTOR_BUFFER
    if valid_mask.sum() == 0:
        raise NoConnectorSitesFoundError()
    return connectors[valid_mask]


def get_cap(polygon, xform):
    verts, faces = triangulate_polygon(polygon, triangle_args="p")
    verts = np.column_stack((verts, np.zeros(len(verts))))
    verts = transform_points(verts, xform)
    faces = np.fliplr(faces)
    return Trimesh(verts, faces)


class ConnectorPlacer:
    """Manages optimization and placement of connectors"""

    connector_cci: np.ndarray[int]
    connector_locations: np.ndarray[float]
    connectors: np.ndarray[Box]
    cc_area: list[float]
    cc_path: list[tuple[int]]
    n_connectors: int
    collisions: np.ndarray

    def __init__(self, tree: bsp_tree.BSPTree):
        self.cc_area = []
        connectors = []
        self.n_connectors = 0
        self.cc_path = []
        self.connector_cci = []
        if len(tree.nodes) < 2:
            raise ConnectorPlacerInputError()

        cap_manager = CollisionManager()
        conn_manager = CollisionManager()
        n_cc = 0
        n_conn = 0
        # create connectors
        logger.info("Creating connectors...")
        for node in tree.nodes:
            if node.cross_section is None:
                continue
            for ccp in node.cross_section.cc_polygons:
                ccc = _get_connectors(ccp, node.part, node.cross_section.xform, n_cc)
                if len(ccc) == 0:
                    raise NoConnectorSitesFoundError()
                connectors.append(ccc)
                self.cc_area.append(ccp.area)
                self.cc_path.append(node.path)
                cap_manager.add_object(f"{n_cc}", get_cap(ccp, node.cross_section.xform))
                for i, conn in enumerate(ccc):
                    conn_manager.add_object(f"{n_cc}_{n_conn + i}", conn)
                    self.connector_cci.append(n_cc)
                n_cc += 1
                n_conn += len(ccc)

        self.connectors = np.concatenate(connectors)
        self.connector_cci = np.array(self.connector_cci)
        logger.info("initial connectors: %s", n_conn)

        # get rid of bad connectors
        logger.info("removing bad connectors")
        _, collisions = cap_manager.in_collision_other(conn_manager, return_names=True)
        bad_con_idx = []
        for cap, con in collisions:
            cap_i, con_i = con.split("_")
            if cap == cap_i:
                continue
            bad_con_idx.append(int(con_i))
        good_con_mask = ~np.in1d(np.arange(self.connectors.shape[0]), bad_con_idx)
        self.connectors = self.connectors[good_con_mask]
        self.connector_cci = self.connector_cci[good_con_mask]
        self.connector_locations = np.array([b.centroid for b in self.connectors])

        # connector stats
        self.n_connectors = self.connectors.shape[0]
        logger.info("Connection placer stats")
        logger.info("connectors %s", self.n_connectors)
        logger.info("connected components %s", len(self.cc_area))

        # collisions
        new_idx = np.cumsum(good_con_mask) - 1
        self.collisions = np.zeros((self.n_connectors, self.n_connectors), dtype=bool)
        logger.info("determining connector collisions")
        _, collisions = conn_manager.in_collision_internal(return_names=True)
        for con_a, con_b in collisions:
            a = new_idx[int(con_a.split("_")[1])]
            b = new_idx[int(con_b.split("_")[1])]
            self.collisions[a, b] = True
            self.collisions[b, a] = True
        logger.info("Done setting up connector placer")

    def get_initial_state(self) -> np.ndarray:
        """gets initial random state vector, enables one connector for each connected
        component
        """
        state = np.zeros(self.n_connectors, dtype=bool)
        for i in range(len(self.cc_area)):
            mask = self.connector_cci == i
            idx = sample(mask)
            state[idx] = True
        return state

    def simulated_annealing_connector_placement(self) -> np.ndarray:
        """run simulated annealing to optimize placement of connectors"""
        initial_state = self.get_initial_state()
        initial_objective = evaluate_connector_objective(
            self.connector_locations,
            self.connector_cci,
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
            self.connector_locations,
            self.connector_cci,
            self.collisions,
            self.cc_area,
        )
        final_objective = evaluate_connector_objective(
            self.connector_locations,
            self.connector_cci,
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
            progress.update(connector_progress=0.3 + 0.7 * (ni) / len(tree.nodes))
            if node.plane is None:
                continue
            try:
                new_tree = bsp_tree.expand_node(
                    new_tree, node.path, node.plane, separate=False
                )
            except CcTooSmallError:
                logger.exception("why")
                raise
            new_node = new_tree.get_node(node.path)
            for child_node in new_node.children:
                c_dist = child_node.part.nearest.signed_distance(
                    self.connector_locations
                )
                mask = state & (abs(c_dist) < 1.0e-3)
                conn = sum(self.connectors[mask])
                child_node.part = insert_connector(child_node.part, conn, "difference")
        progress.update(connector_progress=1)
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
        utils.trimesh_repair(new_part)
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
        utils.trimesh_repair(new_part)
        if new_part.volume < part.volume:
            return new_part
        check = new_part.intersection(connector)
        logger.warning(
            "difference connector check failed, intersect volume %s", check.volume
        )
        if check.volume < 1:
            return new_part
    raise OperationFailedError()
