"""main search functions"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from choppy import settings
from choppy.bsp_node import BSPNode
from choppy.bsp_tree import BSPTree, process_normal
from choppy.exceptions import InvalidChoppyInputError, ProcessFailureError
from choppy.logger import logger, progress


def all_at_goal(trees: list[BSPTree]) -> bool:
    """convenience / readability function which returns whether a list of trees are all
    terminated
    """
    for tree in trees:
        if not tree.terminated:
            return False
    return True


def not_at_goal_set(trees: list[BSPTree]) -> list[BSPTree]:
    """convenience / readability function which returns the non terminated trees from a
    list
    """
    not_at_goal = []
    for tree in trees:
        if not tree.terminated:
            not_at_goal.append(tree)
    return not_at_goal


def evaluate_cuts(
    base_tree: BSPTree, node: BSPNode, normals: np.ndarray
) -> list[BSPTree]:
    """returns a list of unique trees by splitting a specified node of an input tree
    along all planes as defined in the configuration

    Args:
        base_tree (BSPTree): tree to split at a particular node
        node (BSPNode): node of the input tree to split
        normals (np.ndarray): ): list of normal vectors to evaluate

    Returns:
        list[BSPTree]: _description_
    """

    test_normals = normals.copy()  # Collect predefined set of normal vectors
    # Append partition's bounding-box-aligned vectors as normals
    test_normals = np.append(test_normals, node.auxiliary_normals, axis=0)
    # Return sorted unique elements of input array_like
    test_normals = np.unique(np.round(test_normals, 3), axis=0)

    trees = []
    for i, normal in enumerate(test_normals):
        progress.update(normal=i, total_normals=len(test_normals))
        trees += process_normal(normal, node, base_tree)
    progress.update(normal=0, total_normals=len(test_normals))

    logger.info("total trees: %d", len(trees))
    # go through the list of trees, best ones first, and throw away any that are too
    # similar to another tree already in the result list
    result_set = []
    for tree in sorted(trees, key=lambda x: x.objective):
        if tree.sufficiently_different(node, result_set):
            result_set.append(tree)
    logger.info("%d valid trees", len(result_set))
    return result_set


def beam_search(starter: BSPTree, name: str, output_dir: Path) -> BSPTree:
    """executes the beam search to find a good BSPTree partitioning of the input object

    Args:
        starter (BSPTree)
        name (str): name for saving
        otuput_dir (Path): directory

    Raises:
        ValueError: Incorrect input
        ValueError: Mesh already fits within printer
        Exception: No valid chops found

    Returns:
        BSPTree: BSPTree which adequately partitions the input object
    """
    current_trees = [starter]

    logger.info("Starting beam search with an instance of %s", type(starter))
    logger.info("n_leaves: %s", len(starter.leaves))
    logger.info("Largest part trimesh stats:")
    logger.info(
        "verts: %d extents: %s",
        starter.largest_part.part.vertices.shape[0],
        starter.largest_part.part.extents,
    )

    if all_at_goal(current_trees):
        raise InvalidChoppyInputError()

    total_parts = int(sum([p.n_parts for p in current_trees[0].leaves]))
    progress.update(part=0, total_parts=total_parts, total_trees=settings.BEAM_WIDTH)
    # keep track of n_leaves, in each iteration we will only consider trees with the
    # same number of leaves
    # I think the trees become less comparable when they don't have the same number of
    # leaves
    n_leaves = 1
    # continue until we have at least {beam_width} trees
    while not all_at_goal(current_trees):
        new_bsps = []  # list of new bsps
        # look at all trees that haven't terminated
        active_trees = not_at_goal_set(current_trees)
        progress.update(part=n_leaves - 1)
        for i, tree in enumerate(active_trees):
            progress.update(tree=i)
            # only consider trees with a certain number of leaves
            if len(tree.leaves) != n_leaves:
                continue
            # remove the current tree (we will replace it with its best partition)
            current_trees.remove(tree)
            largest_node = tree.largest_part  # split the largest node
            # consider many different cutting planes for the node
            new_bsps += evaluate_cuts(tree, largest_node, settings.NORMALS)

        n_leaves += 1  # on the next iteration, look at trees with more leaves
        current_trees += new_bsps
        # sort all of the trees including the new ones
        current_trees = sorted(current_trees, key=lambda x: x.objective)
        # if we are considering part separation, some of the trees may have more leaves
        # put those away for later
        extra_leaves_trees = [t for t in current_trees if len(t.leaves) > n_leaves]
        # only keep the best {beam_width} trees
        current_trees = current_trees[: settings.BEAM_WIDTH]
        # add back in the trees with extra leaves
        current_trees += [t for t in extra_leaves_trees if t not in current_trees]

        if len(current_trees) == 0:  # all of the trees failed
            raise ProcessFailureError()

        total_parts = int(sum(p.n_parts for p in current_trees[0].leaves))
        progress.update(total_parts=total_parts)
        logger.info("Leaves: %d", n_leaves)
        logger.info("estimated number of parts: %s", total_parts)
        
        logger.info("top trees:")
        for tree in current_trees[: settings.BEAM_WIDTH]:
            for comp, val in tree.get_objective_components().items():
                logger.info("%s: %s", comp, val)
            logger.info("total objective: %s\n", tree.objective)

        # save progress
        for i, tree in enumerate(current_trees[: settings.BEAM_WIDTH]):
            tree.save(output_dir / f"{name}_beam{i}.json")
            tree.export_stls(output_dir, f"{name}_tree{i}")
    
    progress.update(tree=0, part=total_parts)
    return current_trees
