"""Every transform must leave the tensors it reads byte-for-byte intact.

Datasets hand out ``Data`` objects whose attributes are views into one flat
cached tensor. An in-place write through such a view dirties that shared storage
(and every forked DataLoader worker), so transforms may rebind attributes but
must not write through them.
"""

import pytest
import torch
from torch_geometric.data import Data

from clouds.data import MultiGridData
from clouds.transforms import (
    ApplySelection,
    AttributeAsFeature,
    AttributeDropout,
    BallGraph,
    BlockSample,
    BlockSelect,
    BuildClusterMultiGrid,
    BuildPhantomMultigrid,
    BuildSelectionMultiGrid,
    CategoryClassMask,
    CenterPoints,
    ClampPos,
    ClusterSample,
    ClusterSelect,
    CombineVotes,
    CutCrop,
    CutSelect,
    CylinderSample,
    CylinderSelect,
    DiscardAttributes,
    EdgesToDownsampleNeighbors,
    EstimateDensity,
    ExtractCoords,
    ExtractHeights,
    FlattenMultigrid,
    FurthestPointSample,
    FurthestPointSelect,
    GroundPoints,
    Identity,
    InverseDensitySelect,
    KNNSourceGraph,
    KNNUpsampleNeighbors,
    Mix3D,
    MultiGridTransform,
    NearestSelectionCluster,
    NormalizeScale,
    RadiusSelect,
    RandomColorAutoContrast,
    RandomJitter,
    RandomPointDropout,
    RandomRotate,
    RandomSample,
    RandomScale,
    RandomScaleAndRotate,
    RandomSelect,
    RandomShift,
    RefinePartSegmentation,
    ScaleAttribute,
    SlabSample,
    SlabSelect,
    SphereCrop,
    SphereSelect,
    UnpackSourceGraph,
    VoteAugmentations,
    VoxelCluster,
    VoxelSample,
    VoxelSelect,
)

NUM_NODES = 64
POS = torch.arange(NUM_NODES * 3, dtype=torch.float).reshape(NUM_NODES, 3) / 10
POS2 = POS[:, :2]
NORM = torch.nn.functional.normalize(POS, dim=-1)
X = torch.arange(NUM_NODES * 4, dtype=torch.float).reshape(NUM_NODES, 4)
COLOR = POS / POS.amax()
EDGE_INDEX = torch.arange(NUM_NODES * 4).reshape(NUM_NODES, 4) % NUM_NODES
SELECTION = torch.arange(0, NUM_NODES, 2)
CLUSTER_INDEX = torch.arange(NUM_NODES) // 4


def _back_attrs(store, attrs):
    """Assign ``attrs`` to ``store``, making every tensor a view into a larger buffer."""
    storages = []
    for key, value in attrs.items():
        if isinstance(value, torch.Tensor):
            storage = value.new_zeros(value.numel() + 8)
            view = storage[4 : 4 + value.numel()].view(value.shape)
            view.copy_(value)
            store[key] = view
            storages.append(storage)
        else:
            setattr(store, key, value)
    return storages


def _data(**attrs):
    data = Data()
    return data, _back_attrs(data, attrs)


def _batched(**attrs):
    batch = torch.arange(2).repeat_interleave(NUM_NODES // 2)
    ptr = torch.tensor([0, NUM_NODES // 2, NUM_NODES])
    return _data(batch=batch, ptr=ptr, **attrs)


def _multigrid(levels, edges=()):
    mg = MultiGridData()
    storages = []
    for level, attrs in levels.items():
        storages += _back_attrs(mg[level], attrs)
    for key, attrs in edges:
        storages += _back_attrs(mg[key], attrs)
    return mg, storages


def _two_levels(edge_index=None):
    levels = {
        'x0': {'pos': POS, 'x': X},
        'x1': {'pos': POS[: NUM_NODES // 2], 'x': X[: NUM_NODES // 2]},
    }
    edges = [] if edge_index is None else [(('x0', 'to', 'x1'), {'edge_index': edge_index})]
    return _multigrid(levels, edges)


def _partseg():
    return _data(
        pos=POS,
        pred=torch.eye(5)[torch.arange(NUM_NODES) % 5],
        y=torch.zeros(NUM_NODES, dtype=torch.long),
        category=torch.zeros(1, dtype=torch.long),
        batch=torch.zeros(NUM_NODES, dtype=torch.long),
    )


CASES = [
    pytest.param(Identity(), _data, id='Identity'),
    pytest.param(AttributeAsFeature(['color']), lambda: _data(pos=POS, color=COLOR), id='AttributeAsFeature'),
    pytest.param(AttributeDropout('x', p=1.0), lambda: _data(x=X), id='AttributeDropout'),
    pytest.param(BallGraph(r=1.0, max_num_neighbors=4), lambda: _data(pos=POS), id='BallGraph'),
    pytest.param(BlockSelect(max_num_points=10), lambda: _data(pos=POS), id='BlockSelect'),
    pytest.param(BlockSample(max_num_points=10), lambda: _data(pos=POS), id='BlockSample'),
    pytest.param(CenterPoints(), lambda: _data(pos=POS), id='CenterPoints'),
    pytest.param(GroundPoints(), lambda: _data(pos=POS), id='GroundPoints'),
    pytest.param(ClusterSelect(deterministic=True), lambda: _data(pos=POS, cluster_index=CLUSTER_INDEX), id='ClusterSelect'),
    pytest.param(ClusterSample(deterministic=True), lambda: _data(pos=POS, cluster_index=CLUSTER_INDEX), id='ClusterSample'),
    pytest.param(NearestSelectionCluster(), lambda: _data(pos=POS, selection_index=SELECTION), id='NearestSelectionCluster'),
    pytest.param(RandomColorAutoContrast(p=1.0), lambda: _data(pos=POS, color=COLOR), id='RandomColorAutoContrast'),
    pytest.param(CylinderSelect(max_num_points=10), lambda: _data(pos=POS), id='CylinderSelect'),
    pytest.param(CylinderSample(max_num_points=10), lambda: _data(pos=POS), id='CylinderSample'),
    pytest.param(SlabSelect(max_num_points=10), lambda: _data(pos=POS), id='SlabSelect'),
    pytest.param(SlabSample(max_num_points=10), lambda: _data(pos=POS), id='SlabSample'),
    pytest.param(CutSelect(max_num_points=10), lambda: _data(pos=POS), id='CutSelect'),
    pytest.param(CutCrop(max_num_points=10), lambda: _data(pos=POS), id='CutCrop'),
    pytest.param(EstimateDensity(), lambda: _data(pos=POS), id='EstimateDensity'),
    pytest.param(InverseDensitySelect(), lambda: _data(pos=POS, density=torch.ones(NUM_NODES, 1)), id='InverseDensitySelect'),
    pytest.param(DiscardAttributes('color'), lambda: _data(pos=POS, color=COLOR), id='DiscardAttributes'),
    pytest.param(ExtractCoords(dims=[0, 1]), lambda: _data(pos=POS), id='ExtractCoords'),
    pytest.param(ExtractHeights(), lambda: _data(pos=POS), id='ExtractHeights'),
    pytest.param(FurthestPointSelect(max_num_points=10), lambda: _data(pos=POS), id='FurthestPointSelect'),
    pytest.param(FurthestPointSample(max_num_points=10), lambda: _data(pos=POS), id='FurthestPointSample'),
    pytest.param(KNNSourceGraph(k=4), lambda: _data(pos=POS), id='KNNSourceGraph'),
    pytest.param(Mix3D(p=1.0), lambda: _batched(pos=POS), id='Mix3D'),
    pytest.param(
        MultiGridTransform(Identity(), levels=[0, 1]),
        lambda: _two_levels(),
        id='MultiGridTransform',
    ),
    pytest.param(
        BuildSelectionMultiGrid([BlockSelect(max_num_points=16)]),
        lambda: _data(pos=POS),
        id='BuildSelectionMultiGrid',
    ),
    pytest.param(
        BuildClusterMultiGrid([VoxelCluster(voxel_size=2.0)], [ClusterSample(deterministic=True, pick=0)]),
        lambda: _data(pos=POS),
        id='BuildClusterMultiGrid',
    ),
    pytest.param(BuildPhantomMultigrid(), lambda: _data(pos=POS), id='BuildPhantomMultigrid'),
    pytest.param(
        FlattenMultigrid(),
        lambda: _two_levels(edge_index=torch.tensor([[0, 1, 2, 3], [32, 33, 34, 35]])),
        id='FlattenMultigrid',
    ),
    pytest.param(ClampPos(min=[-1, -1, -1], max=[1, 1, 1]), lambda: _data(pos=POS), id='ClampPos'),
    pytest.param(NormalizeScale(), lambda: _data(pos=POS), id='NormalizeScale'),
    pytest.param(
        CategoryClassMask({'category': [0, 1, 2, 3, 4]}),
        _partseg,
        id='CategoryClassMask',
    ),
    pytest.param(
        RefinePartSegmentation({'category': [0, 1, 2, 3, 4]}, k=16),
        _partseg,
        id='RefinePartSegmentation',
    ),
    pytest.param(RadiusSelect(max_num_points=10), lambda: _data(pos=POS), id='RadiusSelect'),
    pytest.param(SphereSelect(max_num_points=10), lambda: _data(pos=POS), id='SphereSelect'),
    pytest.param(SphereCrop(max_num_points=10), lambda: _data(pos=POS), id='SphereCrop'),
    pytest.param(RandomScale(p=1.0), lambda: _data(pos=POS, norm=NORM), id='RandomScale'),
    pytest.param(RandomScale(p=1.0), lambda: _batched(pos=POS, norm=NORM), id='RandomScale-batched'),
    pytest.param(RandomRotate(p=1.0), lambda: _data(pos=POS, norm=NORM), id='RandomRotate'),
    pytest.param(RandomRotate(p=1.0), lambda: _batched(pos=POS, norm=NORM), id='RandomRotate-batched'),
    pytest.param(RandomScaleAndRotate(), lambda: _data(pos=POS, norm=NORM), id='RandomScaleAndRotate'),
    pytest.param(RandomScaleAndRotate(), lambda: _batched(pos=POS, norm=NORM), id='RandomScaleAndRotate-batched'),
    pytest.param(RandomJitter(), lambda: _data(pos=POS), id='RandomJitter'),
    pytest.param(RandomSelect(selection_factor=0.5), lambda: _data(pos=POS), id='RandomSelect'),
    pytest.param(RandomSample(selection_factor=0.5), lambda: _data(pos=POS), id='RandomSample'),
    pytest.param(RandomPointDropout(), lambda: _data(pos=POS), id='RandomPointDropout'),
    pytest.param(RandomShift(), lambda: _data(pos=POS), id='RandomShift'),
    pytest.param(ScaleAttribute('x', 2.0, p=1.0), lambda: _data(x=X), id='ScaleAttribute'),
    pytest.param(
        EdgesToDownsampleNeighbors(),
        lambda: _multigrid(
            {'x0': {'pos': POS, 'x': X}, 'x1': {'pos': POS2, 'x': X[: NUM_NODES // 2]}},
            [(('x0', 'to', 'x0'), {'edge_index': EDGE_INDEX}), ('x0', {'selection_index': SELECTION})],
        ),
        id='EdgesToDownsampleNeighbors',
    ),
    pytest.param(KNNUpsampleNeighbors(k=2), lambda: _two_levels(), id='KNNUpsampleNeighbors'),
    pytest.param(UnpackSourceGraph(), lambda: _data(pos=POS, edge_index=EDGE_INDEX), id='UnpackSourceGraph'),
    pytest.param(VoteAugmentations([RandomJitter()]), lambda: _data(pos=POS), id='VoteAugmentations'),
    pytest.param(
        CombineVotes(),
        lambda: _data(pos=torch.rand(NUM_NODES, 3), pred=torch.rand(NUM_NODES, 5), num_votes=2),
        id='CombineVotes',
    ),
    pytest.param(VoxelCluster(voxel_size=2.0), lambda: _data(pos=POS), id='VoxelCluster'),
    pytest.param(VoxelSelect(voxel_size=2.0), lambda: _data(pos=POS), id='VoxelSelect'),
    pytest.param(VoxelSample(voxel_size=2.0), lambda: _data(pos=POS), id='VoxelSample'),
    pytest.param(
        ApplySelection(),
        lambda: _data(pos=POS, x=X, edge_index=EDGE_INDEX, selection_index=SELECTION),
        id='ApplySelection',
    ),
]


@pytest.mark.parametrize(('transform', 'make_input'), CASES)
def test_transform_does_not_modify_input_storage(transform, make_input):
    data, storages = make_input()
    before = [storage.clone() for storage in storages]

    transform(data)

    for storage, snapshot in zip(storages, before, strict=True):
        assert torch.equal(storage, snapshot)
