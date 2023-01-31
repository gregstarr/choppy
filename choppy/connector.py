from __future__ import annotations

import numpy as np
from numba import jit
from trimesh import Trimesh
from trimesh.primitives import Sphere

from choppy import bsp_tree, settings, utils
from choppy.logger import logger, progress

connector_t = np.dtype(
    [
        ("center", np.float32, (3,)),
        ("radius", np.float32),
        ("cci", np.uint32),
        ("variant", np.uint32),
    ]
)

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
            rc = 20 * cc_conn["radius"]
            rc = np.minimum(rc, np.sqrt(area))
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
        if ci < 0:
            objective -= ci / area

    return objective


@jit(nopython=True)
def sa_iteration(
    connectors: np.ndarray[connector_t],
    collisions: np.array,
    cc_area: list[float],
    state: np.ndarray,
    objective: float,
    temp: float,
    n_variants: int,
) -> tuple[np.ndarray, float]:
    """run a single simulated annealing iteration"""
    new_state = state.copy()
    if np.random.randint(0, 2) or not state.any() or state.all():
        idx = sample(
            connectors, np.ones(connectors.shape[0]).astype("bool"), n_variants
        )
        new_state[idx] = 0 if state[idx] else 1
    else:
        add = sample(connectors, ~state, n_variants)
        remove = sample(connectors, state, n_variants)
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
def sample(connectors: np.ndarray[connector_t], mask, n):
    r1 = np.random.randint(0, n)
    mask2 = mask & (connectors["variant"] == r1)
    if not mask2.any():
        mask2 = mask
    r2 = np.random.randint(0, mask2.sum())
    return np.flatnonzero(mask2)[r2]


@jit(nopython=True)
def sa_connector_placement(
    state: np.ndarray,
    connectors: np.ndarray[connector_t],
    collisions: np.array,
    cc_area: list[float],
    n_variants: int,
):
    objective = evaluate_connector_objective(connectors, collisions, cc_area, state)
    # initialization
    for _ in range(settings.SA_INITIALIZATION_ITERATIONS):
        state, objective = sa_iteration(
            connectors, collisions, cc_area, state, objective, 0, n_variants
        )

    initial_temp = objective / 2
    for temp in np.linspace(initial_temp, 0, settings.SA_ITERATIONS):
        state, objective = sa_iteration(
            connectors, collisions, cc_area, state, objective, temp, n_variants
        )

    return state


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
    n_variants: int

    def __init__(self, tree: bsp_tree.BSPTree):
        self.cc_area = []
        connectors = []
        caps = []
        self.n_connectors = 0
        self.cc_path = []
        if len(tree.nodes) < 2:
            raise ConnectorPlacerInputError()

        logger.info("Creating connectors...")
        for node in tree.nodes:
            if node.cross_section is None:
                continue
            # create global vector of connectors
            for cc in node.cross_section.connected_components:
                # register the ConnectedComponent's sites with the global array of
                # connectors
                for conn in cc.connectors:
                    connectors.append(
                        (conn["center"], conn["radius"], len(self.cc_area), 0)
                    )
                self.cc_area.append(cc.area)
                self.cc_path.append(node.path)
                caps.append(cc.mesh)

        self.connectors = np.array(connectors, dtype=connector_t)
        self.n_connectors = self.connectors.shape[0]
        if self.n_connectors == 0:
            raise NoConnectorSitesFoundError()

        unique_radii = np.unique(self.connectors["radius"])
        self.n_variants = unique_radii.shape[0]
        for i, rad in enumerate(unique_radii):
            mask = self.connectors["radius"] == rad
            self.connectors["variant"][mask] = i

        self.collisions = np.zeros((self.n_connectors, self.n_connectors), dtype=bool)

        logger.info("determining connector-cut intersections")
        intersections = np.zeros(self.n_connectors, dtype=int)
        for cap in caps:
            dist = cap.nearest.on_surface(self.connectors["center"])[1]
            mask = dist < self.connectors["radius"]
            intersections[mask] += 1
        self.collisions[intersections > 1, :] = True
        self.collisions[:, intersections > 1] = True

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
            idx = sample(self.connectors, mask, self.n_variants)
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
        state = sa_connector_placement(
            initial_state,
            self.connectors,
            self.collisions,
            self.cc_area,
            self.n_variants
        )
        final_objective = evaluate_connector_objective(
            self.connectors,
            self.collisions,
            self.cc_area,
            state
        )
        progress.update(connector_progress=0.3)
        logger.info("finished with connector placement, final objective %s", final_objective)
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
        num_inserts = state.sum()
        logger.info("inserting %s connectors", num_inserts)
        nc = 0
        if tree.nodes[0].plane is None:
            new_tree = bsp_tree.separate_starter(tree.nodes[0].part, printer_extents)
        else:
            new_tree = bsp_tree.BSPTree(tree.nodes[0].part, printer_extents)
        for node in tree.nodes:
            if node.plane is None:
                continue
            new_tree2 = bsp_tree.expand_node(new_tree, node.path, node.plane)
            new_tree = new_tree2
            new_node = new_tree.get_node(node.path)
            # reduce to connectors for this path (cross section)
            cci = [i for i, path in enumerate(self.cc_path) if path == node.path]
            xs_conn = self.connectors[np.isin(self.connectors["cci"], cci) & state]
            for cc in node.cross_section.connected_components:
                # reduce to connectors for this connected component
                cc_mask = (
                    abs(cc.mesh.nearest.signed_distance(xs_conn["center"])) < 1.0e-3
                )
                cc_conn = xs_conn[cc_mask]
                pi = cc.positive
                ni = cc.negative
                conn_m = []
                conn_f = []
                for center, radius, *_ in cc_conn:
                    nc += 1
                    conn_m.append(Sphere(radius=radius, center=center))
                    conn_f.append(
                        Sphere(
                            radius=radius + settings.CONNECTOR_TOLERANCE / 2,
                            center=center
                        )
                    )
                conn_m = sum(conn_m)
                conn_f = sum(conn_f)
                new_node.children[ni].part = insert_connector(
                    new_node.children[ni].part, conn_f, "difference"
                )
                new_node.children[pi].part = insert_connector(
                    new_node.children[pi].part, conn_m, "union"
                )
                progress.update(connector_progress=0.3 + 0.7 * nc / num_inserts)
        return new_tree


class InvalidOperationError(Exception):
    def __init__(self) -> None:
        super().__init__("operation not 'union' or 'difference'")

class OperationFailedError(Exception):
    def __init__(self) -> None:
        super().__init__("Couldn't insert connector")



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
