from __future__ import annotations

from typing import NamedTuple

import numpy as np
from trimesh import Trimesh
from trimesh.transformations import angle_between_vectors, random_rotation_matrix

from choppy import section, settings


class ConvexHullError(Exception):
    """Convex hull error"""
class LowVolumeError(Exception):
    """low volume"""


class Plane(NamedTuple):
    """tuple of (origin, normal), both 3D numpy vectors"""

    origin: np.ndarray
    normal: np.ndarray


class OrientedBoundingBox(NamedTuple):
    vectors: np.ndarray
    extents: np.ndarray
    volume: np.ndarray


class BSPNode:
    """Binary Space Partition Node"""

    part: Trimesh
    parent: "BSPNode"
    children: list["BSPNode"]
    path: tuple[int]
    plane: Plane
    cross_section: section.CrossSection
    printer_extents: np.ndarray

    def __init__(
        self,
        part: Trimesh,
        parent: "BSPNode",
        printer_extents: np.ndarray = None,
        num: int = None,
    ):
        """Initialize a bspnode, determine n_parts objective and termination status.

        Args:
            part (Trimesh): mesh part associated with this node
            parent (BSPNode): BSPNode which was split to produce this node and
                this node's sibling nodes. Use `None` for root node.
            printer_extents (np.ndarray, optional): printer dimensions (mm), inherit
                from parent if None.
            num (int, optional): index of this node, e.g. 0 -> 0th child of this node's
                parent. Defaults to None.
        """
        self.part = part
        self.parent = parent
        self.children = []
        self.path = tuple()
        # this node will get a plane and a cross_section if and when it is split
        self.plane = None
        self.cross_section = None
        if printer_extents is None:
            self.printer_extents = parent.printer_extents
        else:
            self.printer_extents = printer_extents
        self.obb = self._get_obb()
        # determine n_parts and termination status
        self.n_parts = np.prod(
            np.ceil(self.obb.extents / self.printer_extents)
        )
        self.terminated = np.all(self.obb.extents <= self.printer_extents)
        # if this isn't the root node
        if self.parent is not None:
            self.path = (*self.parent.path, num)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.path=})"

    def _get_obb(self):
        rotations = random_rotation_matrix(num=100)[:, :3, :3]
        projections = np.matmul(self.part.vertices, rotations)
        extents = projections.max(axis=1) - projections.min(axis=1)
        volumes = np.prod(extents, axis=1)
        idx = np.argmin(volumes)
        return OrientedBoundingBox(rotations[idx], extents[idx], volumes[idx])

    @property
    def auxiliary_normals(self) -> np.ndarray:
        """(3 x 3) numpy array who's rows are the three unit vectors aligned with this
        node's part's oriented bounding box

        Returns:
            np.ndarray
        """
        return self.obb.vectors

    @property
    def connection_objective(self) -> float:
        """connection objective for this node

        Returns:
            float
        """
        return max(cc.objective for cc in self.cross_section.connected_components)

    def different_from(self, other_node: "BSPNode") -> bool:
        """determine if this node is different plane-wise from the same node on another
        tree (check if this node's plane is different from another node's plane)

        Args:
            other_node (BSPNode): corresponding node on another tree

        Returns:
            bool
        """
        other_o = other_node.plane.origin  # other plane origin
        # vector from this plane's origin to the others'
        delta = other_o - self.plane.origin
        # distance along this plane's normal to other plane
        dist = abs(self.plane.normal @ delta)
        # angle between this plane's normal vector and the other plane's normal vector
        angle = angle_between_vectors(self.plane.normal, other_node.plane.normal)
        # also consider angle between the vectors in the opposite direction
        angle = min(np.pi - angle, angle)
        # check if either the distance or the angle are above their respective
        # thresholds
        return (
            dist > settings.DIFFERENT_ORIGIN_TH or angle > settings.DIFFERENT_ANGLE_TH
        )


def split(node: BSPNode, plane: Plane) -> BSPNode:
    """Split a node with a plane

    Args:
        node (BSPNode): BSPNode to split
        plane (Plane): the cutting plane

    Returns:
        BSPNode
    """
    node.plane = plane

    # split the part
    node.cross_section = section.CrossSection(node.part, plane)
    parts = node.cross_section.split(node.part)

    for i, part in enumerate(parts):
        if part.volume < 0.1:  # make sure each part has some volume
            raise LowVolumeError()
        child = BSPNode(part, parent=node, num=i)  # potential convex hull failure
        node.children.append(child)  # The parts become this node's children
    return node
