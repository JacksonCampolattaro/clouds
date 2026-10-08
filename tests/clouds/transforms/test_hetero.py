"""Heterogeneous (multi-level) support for the transforms."""

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from torch_geometric.typing import WITH_GRID_CLUSTER as HAS_PYG_GRID_CLUSTER

from clouds.data import MultiGridData
from clouds.transforms import (
    AttributeAsFeature,
    AttributeDropout,
    BallGraph,
    CenterPoints,
    EstimateDensity,
    ExtractCoords,
    ExtractHeights,
    GroundPoints,
    Identity,
    KNNSourceGraph,
    Mix3D,
    NormalizeScale,
    RandomColorAutoContrast,
    RandomJitter,
    RandomRotate,
    RandomScale,
    RandomScaleAndRotate,
    RandomShift,
    ScaleAttribute,
    UnpackSourceGraph,
    VoteAugmentations,
    VoxelCluster,
)


def make_multigrid(num_fine: int = 100, num_coarse: int = 40) -> MultiGridData:
    mg = MultiGridData()
    mg.set_level(
        'x0',
        Data(pos=torch.randn(num_fine, 3), x=torch.randn(num_fine, 4), color=torch.rand(num_fine, 3)),
    )
    mg.set_level(
        'x1',
        Data(pos=torch.randn(num_coarse, 3), x=torch.randn(num_coarse, 4), color=torch.rand(num_coarse, 3)),
    )
    return mg


def collate_multigrids() -> MultiGridData:
    return collate(cls=MultiGridData, data_list=[make_multigrid(), make_multigrid()])[0]


def test_scale_attribute_scales_every_level():
    mg = make_multigrid()
    original = {level: mg[level].x.clone() for level in mg.node_types}

    out = ScaleAttribute('x', factor=2.0, p=1.0)(mg)

    for level in out.node_types:
        assert torch.allclose(out[level].x, original[level] * 2.0)


def test_random_jitter_shifts_every_level():
    mg = make_multigrid()
    original = {level: mg[level].pos.clone() for level in mg.node_types}

    out = RandomJitter(sigma=1.0, clip=None)(mg)

    for level in out.node_types:
        assert not torch.allclose(out[level].pos, original[level])


def test_random_shift_uses_one_offset_and_adds_no_phantom_level():
    mg = make_multigrid()
    original = {level: mg[level].pos.clone() for level in mg.node_types}

    out = RandomShift(max_offset=0.0)(mg)

    # A zero offset must leave every level untouched...
    for level in out.node_types:
        assert torch.allclose(out[level].pos, original[level])
    # ...and must not create a stray 'pos' node type.
    assert out.node_types == ['x0', 'x1']


def test_extract_heights_and_coords_apply_to_every_level():
    mg = make_multigrid()

    heights = ExtractHeights()(mg)
    assert all(hasattr(heights[level], 'height') for level in heights.node_types)

    coords = ExtractCoords(dims=[0, 1])(mg)
    assert all(coords[level].coord.shape[-1] == 2 for level in coords.node_types)


def test_attribute_as_feature_handles_missing_level_attributes():
    mg = make_multigrid()
    del mg['x1'].color

    out = AttributeAsFeature(['color'], overwrite=True)(mg)

    assert out['x0'].x.shape == (100, 3)
    assert out['x1'].x.shape == (40, 0)


def test_attribute_dropout_zeros_every_level():
    mg = make_multigrid()

    out = AttributeDropout('x', p=1.0)(mg)

    for level in out.node_types:
        assert torch.count_nonzero(out[level].x) == 0


def test_random_color_auto_contrast_applies_to_every_level():
    mg = make_multigrid()
    original = {level: mg[level].color.clone() for level in mg.node_types}

    out = RandomColorAutoContrast(p=1.0, blend_factor=1.0)(mg)

    for level in out.node_types:
        assert not torch.allclose(out[level].color, original[level])


def test_random_color_auto_contrast_on_batched_multigrid():
    batch = collate_multigrids()
    original = {level: batch[level].color.clone() for level in batch.node_types}

    out = RandomColorAutoContrast(p=1.0, blend_factor=1.0)(batch)

    for level in out.node_types:
        assert not torch.allclose(out[level].color, original[level])


def test_voxel_cluster_assigns_clusters_to_every_level():
    if not HAS_PYG_GRID_CLUSTER:
        pytest.skip('pyg grid clustering not installed')

    mg = make_multigrid()

    out = VoxelCluster(voxel_size=1.0)(mg)

    for level in out.node_types:
        assert out[level].cluster_index.shape == (out[level].num_nodes,)


def test_estimate_density_assigns_density_to_every_level():
    mg = make_multigrid()

    out = EstimateDensity(pointwise=True)(mg)

    for level in out.node_types:
        assert out[level].density.shape == (out[level].num_nodes, 1)


def test_ball_graph_builds_intra_level_edges():
    mg = make_multigrid()

    out = BallGraph(r=1.0, max_num_neighbors=4)(mg)

    for level in out.node_types:
        assert out[level, 'to', level].edge_index.shape[0] == 2


def test_knn_source_graph_builds_dense_intra_level_edges():
    mg = make_multigrid()

    out = KNNSourceGraph(k=4)(mg)

    for level in out.node_types:
        assert out[level, 'to', level].edge_index.shape == (out[level].num_nodes, 4)


def test_unpack_source_graph_converts_every_edge_store():
    mg = make_multigrid()
    for level in mg.node_types:
        mg[level, 'to', level].edge_index = torch.randint(0, mg[level].num_nodes, (mg[level].num_nodes, 3))

    out = UnpackSourceGraph()(mg)

    for level in out.node_types:
        assert out[level, 'to', level].edge_index.shape[0] == 2


@pytest.mark.parametrize('transform', [CenterPoints(), GroundPoints()])
def test_center_and_ground_points_on_batched_multigrid(transform):
    batch = collate_multigrids()

    out = transform(batch)

    # The first level defines the shared offset, so it must be centered/grounded per graph.
    for graph in torch.unique(out['x0'].batch):
        pos = out['x0'].pos[out['x0'].batch == graph]
        if isinstance(transform, GroundPoints):
            assert torch.allclose(pos.amin(dim=0), torch.zeros(3), atol=1e-6)
        else:
            assert torch.allclose(pos.mean(dim=0), torch.zeros(3), atol=1e-6)


@pytest.mark.parametrize(
    'transform',
    [
        NormalizeScale(),
        RandomScale(),
        RandomRotate(),
        RandomScaleAndRotate(),
        Mix3D(),
        VoteAugmentations([Identity()]),
    ],
)
def test_group3_transforms_reject_heterogeneous_data(transform):
    with pytest.raises(NotImplementedError):
        transform(make_multigrid())
