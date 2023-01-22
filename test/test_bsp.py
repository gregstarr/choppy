import trimesh
import numpy as np
import copy
from pathlib import Path
import pytest

from choppy.bsp_tree import BSPTree, get_planes, expand_node
from choppy.bsp_node import split, Plane
from choppy.section import CrossSection
from choppy import utils, settings


def test_get_planes(bunny_mesh):
    """verify that for the default bunny mesh, which is a single part, all planes returned by `bsp_tree.get_planes`
        cut through the mesh (they have a good cross section)
    """
    mesh = trimesh.load(bunny_mesh, validate=True)

    for _ in range(100):
        normal = trimesh.unitize(np.random.rand(3))
        planes = get_planes(mesh, normal)

        for origin, normal in planes:
            path3d = mesh.section(plane_origin=origin, plane_normal=normal)
            assert path3d is not None


def test_different_from():
    """verify that `BSPNode.different_from` has the expected behavior

    Get a list of planes. Split the object using the first plane, then for each of the other planes, split the object,
    check if the plane is far enough away given the config, then assert that `BSPNode.different_from` returns the
    correct value. This skips any splits that fail.
    """

    mesh = trimesh.primitives.Sphere(radius=50)
    printer_extents = np.ones(3) * 200

    tree = BSPTree(mesh, printer_extents)
    root = tree.nodes[0]
    normal = trimesh.unitize(np.random.rand(3))
    planes = get_planes(mesh, normal)
    base_node = copy.deepcopy(root)
    base_node = split(base_node, planes[0])

    for plane in planes[1:]:
        # smaller origin offset, should not be different
        test_node = copy.deepcopy(root)
        test_node = split(test_node, plane)
        if abs((plane[0] - planes[0][0]) @ planes[0][1]) > settings.DIFFERENT_ORIGIN_TH:
            assert base_node.different_from(test_node)
        else:
            assert not base_node.different_from(test_node)

    # smaller angle difference, should not be different
    test_node = copy.deepcopy(root)
    random_vector = trimesh.unitize(np.random.rand(3))
    axis = np.cross(random_vector, planes[0][1])
    rotation = trimesh.transformations.rotation_matrix(np.pi / 11, axis)
    normal = trimesh.transform_points(planes[0][1][None, :], rotation)[0]
    test_plane = Plane(planes[0][0], normal)
    test_node = split(test_node, test_plane)
    assert not base_node.different_from(test_node)

    # larger angle difference, should be different
    test_node = copy.deepcopy(root)
    random_vector = trimesh.unitize(np.random.rand(3))
    axis = np.cross(random_vector, planes[0][1])
    rotation = trimesh.transformations.rotation_matrix(np.pi / 9, axis)
    normal = trimesh.transform_points(planes[0][1][None, :], rotation)[0]
    test_plane = Plane(planes[0][0], normal)
    test_node = split(test_node, test_plane)
    assert base_node.different_from(test_node)


def test_copy_tree(bunny_mesh):
    """Now that objectives are calculated outside of the tree (using the objective function evaluators), verify
    that copying a tree doesn't modify its objectives dict
    """
    mesh = trimesh.load(bunny_mesh, validate=True)
    printer_extents = np.ones(3) * 200

    # make tree, get node, get random normal, pick a plane right through middle, make sure that the slice is good
    tree = BSPTree(mesh, printer_extents)
    node = tree.largest_part
    normal = np.array([0, 0, 1])
    planes = get_planes(node.part, normal)
    plane = planes[len(planes) // 2]
    tree = expand_node(tree, node.path, plane)
    new_tree = tree.copy()
    assert new_tree.objectives == tree.objectives


def test_expand_node(bunny_mesh):
    """no errors when using expand_node, need to think of better tests here"""
    mesh = trimesh.load(bunny_mesh, validate=True)
    printer_extents = np.ones(3) * 200

    # make tree, get node, get random normal, pick a plane right through middle, make sure that the slice is good
    tree = BSPTree(mesh, printer_extents)

    node = tree.largest_part
    normal = np.array([0, 0, 1])
    planes = get_planes(node.part, normal)
    plane = planes[0]
    tree1 = expand_node(tree, node.path, plane)
    print("tree objective: ", tree1.objective)

    node = tree1.largest_part
    planes = get_planes(node.part, normal)
    plane = planes[0]
    tree2 = expand_node(tree1, node.path, plane)
    print(tree2)


def test_grid_sample():
    """verify that when the cross section is barely larger than the connector diameter, only 1 sample is
    returned by `ConnectedComponent.grid_sample_polygon`"""
    plane = Plane(np.zeros(3), np.array([0, 0, 1]))

    # test
    mesh = trimesh.primitives.Box(extents=[10.1, 10.1, 40]).to_mesh()
    cross_section = CrossSection(mesh, plane)
    samples = cross_section.connected_components[0].connectors["radius"]
    assert samples.shape[0] == 1

    mesh.apply_translation([3, 0, 0])
    cross_section = CrossSection(mesh, plane)
    samples = cross_section.connected_components[0].connectors["radius"]
    assert samples.shape[0] == 1

    xform = trimesh.transformations.rotation_matrix(np.pi/4, np.array([0, 0, 1]))
    mesh.apply_transform(xform)
    cross_section = CrossSection(mesh, plane)
    samples = cross_section.connected_components[0].connectors["radius"]
    assert samples.shape[0] == 1


def test_basic_separation():
    mesh = trimesh.load(Path(__file__).parent / 'test_meshes' / 'separate_test.stl')
    printer_extents = np.ones(3) * 200
    tree = BSPTree(mesh, printer_extents)
    node = tree.largest_part
    plane = Plane(np.zeros(3), np.array([1, 0, 0]))
    tree = expand_node(tree, node.path, plane)
    # 1 root, three leaves come out of the split
    assert len(tree.nodes) == 4
