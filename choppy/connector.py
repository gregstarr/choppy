from __future__ import annotations
import numpy as np
from trimesh import Trimesh
from trimesh.primitives import Sphere

from choppy import settings, utils, bsp_tree
from choppy.logger import logger, progress


connector_t = np.dtype(
    [
        ("center", np.float32, (3, )),
        ("radius", np.float32),
        ("cci", np.uint32),
    ]
)


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
            raise Exception("input tree needs to have a chop")

        logger.info("Creating connectors...")
        for node in tree.nodes:
            if node.cross_section is None:
                continue
            # create global vector of connectors
            for cc in node.cross_section.connected_components:
                # register the ConnectedComponent's sites with the global array of
                # connectors
                for conn in cc.connectors:
                    connectors.append((conn["center"], conn["radius"], len(self.cc_area)))
                self.cc_area.append(cc.area)
                self.cc_path.append(node.path)
                caps.append(cc.mesh)
        
        self.connectors = np.array(connectors, dtype=connector_t)
        self.n_connectors = self.connectors.shape[0]
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
        distances = np.sqrt(np.sum((self.connectors["center"][:, None, :] - self.connectors["center"][None, :, :]) ** 2, axis=2))
        mask = (distances > 0) & (distances < (self.connectors["radius"][:, None] + self.connectors["radius"][None, :]))
        self.collisions = np.logical_or(self.collisions, mask)

    def evaluate_connector_objective(self, state: np.ndarray) -> float:
        """evaluates connector objective given a state vector"""
        objective = 0
        n_collisions = self.collisions[state, :][:, state].sum()
        objective += settings.CONNECTOR_COLLISION_PENALTY * n_collisions
        if n_collisions > 0:
            return objective

        for i, area in enumerate(self.cc_area):
            ci = 0
            cc_mask = self.connectors["cci"] == i
            cc_conn = self.connectors[cc_mask & state]
            if cc_conn.shape[0] > 0:
                rc = 20 * cc_conn["radius"]
                rc = np.minimum(rc, np.sqrt(area) / 2)
                ci = np.pi * np.sum(rc ** 2)
                distances = np.sqrt(
                    np.sum((cc_conn["center"][:, None, :] - cc_conn["center"][None, :, :]) ** 2, axis=2)
                )
                mask = distances > 0
                rc_sq = rc[:, None] + rc[None, :]
                mask &= distances < rc_sq
                ci -= np.sum(np.pi * (rc_sq[mask] - distances[mask] / 2) ** 2)
            objective += area / (settings.EMPTY_CC_PENALTY + max(0, ci))
            if ci < 0:
                objective -= ci / area

        return objective

    def get_initial_state(self) -> np.ndarray:
        """gets initial random state vector, enables one connector for each connected
        component"""
        state = np.zeros(self.n_connectors, dtype=bool)
        for i in range(len(self.cc_area)):
            idx = np.random.choice(np.argwhere(self.connectors["cci"] == i)[:, 0])
            state[idx] = True
        return state

    def simulated_annealing_connector_placement(self) -> np.ndarray:
        """run simulated annealing to optimize placement of connectors"""
        state = self.get_initial_state()
        objective = self.evaluate_connector_objective(state)
        logger.info("initial objective: %s", objective)
        # initialization
        for i in range(settings.SA_INITIALIZATION_ITERATIONS):
            state, objective = self.sa_iteration(state, objective, 0)
        progress.update(connector_progress=.1)
        logger.info("post initialization objective: %s", objective)
        initial_temp = objective / 2
        for i, temp in enumerate(np.linspace(initial_temp, 0, settings.SA_ITERATIONS)):
            if (i % (settings.SA_ITERATIONS // 20)) == 0:
                progress.update(connector_progress=0.1 + 0.4 * i / settings.SA_ITERATIONS)
            state, objective = self.sa_iteration(state, objective, temp)
        progress.update(connector_progress=0.5)
        return state

    def sa_iteration(
        self, state: np.ndarray, objective: float, temp: float
    ) -> tuple[np.ndarray, float]:
        """run a single simulated annealing iteration"""
        new_state = state.copy()
        if np.random.randint(0, 2) or not state.any() or state.all():
            e = np.random.randint(0, self.n_connectors)
            new_state[e] = 0 if state[e] else 1
        else:
            add = np.random.choice(np.argwhere(~state)[:, 0])
            remove = np.random.choice(np.argwhere(state)[:, 0])
            new_state[add] = 1
            new_state[remove] = 0

        new_objective = self.evaluate_connector_objective(new_state)
        if new_objective < objective:
            return new_state, new_objective
        elif temp > 0:
            if np.random.rand() < np.exp(-(new_objective - objective) / temp):
                return new_state, new_objective
        return state, objective

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
        NI = state.sum()
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
                cc_mask = abs(cc.mesh.nearest.signed_distance(xs_conn["center"])) < 1.0e-3
                cc_conn = xs_conn[cc_mask]
                pi = cc.positive
                ni = cc.negative
                for center, radius, _ in cc_conn:
                    nc += 1
                    progress.update(connector_progress=0.5 + 0.5 * nc / NI)
                    conn_m = Sphere(radius=radius, center=center)
                    conn_f = Sphere(radius=radius + settings.CONNECTOR_TOLERANCE, center=center)
                    new_node.children[ni].part = insert_connector(
                        new_node.children[ni].part, conn_f, "difference"
                    )
                    new_node.children[pi].part = insert_connector(
                        new_node.children[pi].part, conn_m, "union"
                    )
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
        raise ValueError("operation not 'union' or 'difference'")
    utils.trimesh_repair(part)
    if operation == "union":
        new_part = part.union(connector)
        if new_part.volume > part.volume:
            return new_part
        check = new_part.intersection(connector)
        logger.warning("union connector check failed, intersect volume diff %s", abs(check.volume - connector.volume))
        if abs(check.volume - connector.volume) < 1:
            return new_part
    else:
        new_part = part.difference(connector)
        if new_part.volume < part.volume:
            return new_part
        check = new_part.intersection(connector)
        logger.warning("difference connector check failed, intersect volume %s", check.volume)
        if check.volume < 1:
            return new_part
    raise Exception("Couldn't insert connector")
