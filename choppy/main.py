"""
choppy - cli model chop utility
"""
from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path

import numpy as np
import trimesh
from trimesh.interfaces.blender import exists

from choppy import connector, settings, utils
from choppy.blender_ops import decimate
from choppy.bsp_tree import BSPTree, open_tree, separate_starter
from choppy.logger import logger
from choppy.search import beam_search


def run(meshpath: Path, printer_extents: np.ndarray, name: str, output_directory: Path):
    """Complete choppy process, including the beam search for the optimal cutting
    planes, determining the connector locations, adding the connectors to the part
    meshes, then saving the STLs, the tree json and the configuration file.

    Args:
        meshpath (Path): path to mesh stl
        printer_extents (np.ndarray): printer dims (mm)
        name (str): name for saving
        output_directory (Path)
    """
    # mark starting time
    t0 = time.time()
    starter = prepare_starter(meshpath, printer_extents)
    # complete the beam search using the starter, no search will take place if the
    # starter tree is already adequately partitioned
    trees = beam_search(starter, name, output_directory)
    logger.info("Best BSP-trees found in %s seconds", time.time() - t0)

    error = None
    for tree in trees:
        try:
            # save the tree now in case the connector placement fails
            tree.save(output_directory / "final_tree.json")
            # mark starting time
            t0 = time.time()
            logger.info("finding best connector arrangement")
            # create connector placer object, this creates all potential connectors and
            # determines their collisions
            connector_placer = connector.ConnectorPlacer(tree)
            # use simulated annealing to determine the best combination of connectors
            state = connector_placer.simulated_annealing_connector_placement()
            # save the final tree including the state
            logger.info("Saving tree with connector placement")
            tree.save(output_directory / "final_tree_with_connectors.json", state)
            # add the connectors / subtract the slots from the parts of the partitioned
            # input object
            original_tree = open_tree(
                output_directory / "final_tree.json", meshpath, printer_extents
            )
            tree = connector_placer.insert_connectors(
                original_tree, state, printer_extents
            )
        except Exception as exc:
            logger.info("failed connector insertion")
            logger.exception(exc)
            error = exc
        else:
            # export the parts of the partitioned object
            tree.export_stls(output_directory, name)
            logger.info("Finished")
            return

    raise error



def prepare_starter(mesh_fn: Path, printer_extents) -> BSPTree:
    """open trimesh, decimate, separate into pieces

    Args:
        mesh_fn (Path): path to STL

    Returns:
        BSPTree: starter
    """
    # open the input mesh as the starter
    starter = trimesh.load(mesh_fn, use_pyembree=True)
    starter.rezero()
    utils.trimesh_repair(starter)
    n_faces = len(starter.faces)
    n_verts = len(starter.vertices)
    ratio = settings.MAX_FACES / n_faces
    logger.info("n faces %s n verts %s ratio %s", n_faces, n_verts, ratio)
    if ratio < 1:
        logger.info("decimating")
        starter = decimate(starter, ratio)
        n_faces = len(starter.faces)
        n_verts = len(starter.vertices)
        logger.info("n faces %s n verts %s ratio %s", n_faces, n_verts, ratio)
    # separate pieces
    if starter.body_count > 1:
        starter = separate_starter(starter, printer_extents)
    else:
        starter = BSPTree(starter, printer_extents)
    return starter


def main():
    """Main script: argument parsing, logging setup, run process"""
    warnings.filterwarnings("ignore")
    # Read mesh filepath from argument
    import argparse  # pylint: disable=import-outside-toplevel
    import sys  # pylint: disable=import-outside-toplevel

    logger.info("Choppy called with command: %s", sys.argv)

    parser = argparse.ArgumentParser(description="choppy command line runner")
    parser.add_argument("mesh", type=str)
    parser.add_argument("printer_x", type=float)
    parser.add_argument("printer_y", type=float)
    parser.add_argument("printer_z", type=float)
    parser.add_argument("-n", "--name", type=str, default="chop")
    parser.add_argument("-o", "--output-dir", type=str, default=".")
    args = parser.parse_args()

    meshpath = Path(args.mesh)
    if not meshpath.exists:
        logger.error("Mesh path doesn't exist")
        sys.exit(1)
    logger.info("mesh: %s", meshpath)

    output_directory = Path(args.output_dir)
    if not output_directory.exists:
        logger.warning("output directory doesn't exist, creating")
        output_directory.mkdir()
    logger.info("output directory: %s", output_directory)

    file_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s (%(filename)s:%(lineno)3s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(output_directory / "info.log")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    printer_extents = np.array([args.printer_x, args.printer_y, args.printer_z])
    logger.info("printer dims: %s", printer_extents)
    logger.info("name: %s", args.name)

    assert exists, "blender executable not found"
    logger.info("blender executable found")
    try:
        run(meshpath, printer_extents, args.name, output_directory)
    except Exception as exc:
        logger.exception("$ERROR %s", exc)
        raise
