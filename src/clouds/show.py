import math

import polyscope as ps
import torch
from torch_geometric import Index
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.storage import EdgeStorage, NodeStorage

# Coloropt colors http://tsitsul.in/blog/coloropt/
COLOROPT_NORMAL = (
    torch.FloatTensor(
        [
            [235, 172, 35],
            [184, 0, 88],
            [0, 140, 249],
            [0, 110, 0],
            [0, 187, 173],
            [209, 99, 230],
            [178, 69, 2],
            [255, 146, 135],
            [89, 84, 214],
            [0, 198, 248],
            [135, 133, 0],
            [0, 167, 108],
            [189, 189, 189],
        ]
    )
    / 255.0
)
COLOROPT_BRIGHT = (
    torch.FloatTensor(
        [
            [239, 230, 69],
            [233, 53, 161],
            [0, 227, 255],
            [225, 86, 44],
            [83, 126, 255],
            [0, 203, 133],
            [238, 238, 238],
        ]
    )
    / 255.0
)
COLOROPT_DARK = (
    torch.FloatTensor(
        [
            [0, 89, 0],
            [0, 0, 120],
            [73, 13, 0],
            [138, 3, 79],
            [0, 90, 138],
            [68, 53, 0],
            [88, 88, 88],
        ]
    )
    / 255.0
)

# Brewer colors
BREWER_SET3 = (
    torch.FloatTensor(
        [
            [141, 211, 199],
            [255, 255, 179],
            [190, 186, 218],
            [251, 128, 114],
            [128, 177, 211],
            [253, 180, 98],
            [179, 222, 105],
            [252, 205, 229],
            [217, 217, 217],
            [188, 128, 189],
            [204, 235, 197],
            [255, 237, 111],
        ]
    )
    / 255.0
)
BREWER_PAIRED = (
    torch.FloatTensor(
        [
            [66, 206, 227],
            [31, 120, 180],
            [178, 223, 138],
            [51, 160, 44],
            [251, 154, 153],
            [227, 26, 28],
            [253, 191, 111],
            [255, 127, 0],
            [202, 178, 214],
            [106, 61, 154],
            [255, 255, 153],
            [177, 89, 40],
        ]
    )
    / 255.0
)

COLORS = torch.cat(
    [
        COLOROPT_NORMAL,
        BREWER_SET3,
        COLOROPT_BRIGHT,
        COLOROPT_DARK,
        BREWER_PAIRED,
        # todo: more colors!
    ],
    dim=0,
)

ID_COLORSCHEME = 'spectral'
SCALAR_COLORSCHEME = 'viridis'


def register_nodes(
    name: str,
    nodes: NodeStorage,
    enabled: bool = True,
    scaling_factor: float = 1.0,
    color: tuple[float] | None = None,
    **kwargs,
) -> ps.PointCloud:

    if not hasattr(nodes, 'pos'):
        return None

    # Draw the nodes themselves
    num_nodes = int(nodes.num_nodes)
    pos = nodes.pos
    cloud = ps.register_point_cloud(
        name,
        pos.cpu(),
        enabled=enabled,
        color=color[:3] if color else None,
        transparency=color[-1] if color and len(color) == 4 else None,
        **kwargs,
    )

    # Draw faces (if present)
    if hasattr(nodes, 'face'):
        ps.register_surface_mesh(name + '-face', pos.cpu(), nodes.face.T.numpy())

    # Draw each node feature type
    for key, item in nodes.items():
        if any(n in key for n in ['face', 'edge', 'ptr', 'pos']) or not isinstance(item, torch.Tensor):
            continue

        # Broadcast batch-global features to all points (maybe unnecessary?)
        if len(item.shape) == 0:
            item = item.unsqueeze(0).repeat(num_nodes)

        # Broadcast instance-global features to all points
        if hasattr(nodes, 'batch') and item.size(0) == (nodes.batch.amax() + 1):
            item = item[nodes.batch]

        # Convert selection indices to masks
        # FIXME: maybe remove this?
        if isinstance(item, Index):
            mask = torch.zeros([num_nodes], dtype=item.dtype, device=nodes.pos.device)
            mask[item] = 1
            item = mask
            cloud.add_scalar_quantity(key, item.cpu(), cmap='reds', enabled=False)
            continue

        if 'norm' in key:
            # Norms are drawn as vectors
            cloud.add_vector_quantity(key, item.cpu())

        elif len(item.shape) == 1 or (len(item.shape) == 2 and item.shape[-1] == 1):
            # 1D elements are drawn based on their type
            item = item.flatten()
            if not torch.is_floating_point(item):
                item = item.long()
                cloud.add_scalar_quantity(f"{key}-id", item.cpu(), cmap=ID_COLORSCHEME)
                if key == 'y' and item.amax() < len(COLORS):
                    # Labels are drawn with colors, if possible
                    y_colors = COLORS[item.cpu(), :]
                    cloud.add_color_quantity(key, y_colors.cpu())
                else:
                    # Everything else is drawn with a selected color scheme
                    # TODO: make the color scheme a parameter!
                    cloud.add_scalar_quantity(key, item.cpu(), cmap=SCALAR_COLORSCHEME)
            else:
                # Continuous values are drawn with a selected color scheme
                cloud.add_scalar_quantity(key, item.cpu(), cmap=SCALAR_COLORSCHEME)

        elif 'color' in key and item.shape[1] == 3:
            # Assumes colors values are between 0..1
            cloud.add_color_quantity(key, item.cpu())

        else:
            # High-dimensional values will be drawn based on mixed colors

            prediction = item.argmax(dim=-1)
            probabilities = item.softmax(dim=-1)
            zeros = (item == 0).all(dim=-1)

            if item.shape[-1] <= COLORS.shape[0]:
                colors = COLORS[: item.shape[-1]]
                prediction_colors = colors[prediction.cpu(), :]
                # label_weights = torch.softmax(x, dim=-1)
                # weighted_colors = label_weights.unsqueeze(-1) * label_colors.unsqueeze(0)
                # colors = weighted_colors.sum(dim=1, keepdim=False)
                probability_colors = (probabilities.cpu().unsqueeze(-1) * colors.unsqueeze(0)).sum(dim=1)
                # zero predictions should be black
                prediction_colors[zeros, :] = 0
                probability_colors[zeros, :] = 0
                cloud.add_color_quantity(f"{key}-max", prediction_colors.cpu())
                cloud.add_color_quantity(f"{key}-prob", probability_colors.cpu(), enabled=True)

            cloud.add_scalar_quantity(f"{key}-id", prediction.cpu(), cmap=ID_COLORSCHEME)

    return cloud


def register_edges(
    name: str,
    source_nodes: NodeStorage,
    dest_nodes: NodeStorage,
    edges: EdgeStorage,
    color: tuple[float] | None = None,
    enabled=False,
    **kwargs,
) -> ps.CurveNetwork:

    if not hasattr(source_nodes, 'pos') or not hasattr(dest_nodes, 'pos'):
        return None
    if not hasattr(edges, 'edge_index'):
        return None

    # TODO: handle adj_t

    # Convert kNN edges to pairwise edges
    if edges.edge_index.size(0) != 2:
        edges.edge_index = torch.stack(
            [
                edges.edge_index.flatten(),
                torch.arange(
                    edges.edge_index.size(0),
                    dtype=edges.edge_index.dtype,
                    device=edges.edge_index.device,
                ).repeat_interleave(edges.edge_index.size(1)),
            ]
        )

    if source_nodes is not dest_nodes:
        # Edges connecting one point cloud to another require a combined point cloud
        all_pos = torch.cat([source_nodes.pos, dest_nodes.pos], dim=0)
        remapped_edge_index = edges.edge_index.cpu() + torch.tensor([0, source_nodes.pos.shape[0]]).unsqueeze(-1)
    else:
        all_pos = source_nodes.pos
        remapped_edge_index = edges.edge_index

    curve = ps.register_curve_network(
        name + '-edges',
        all_pos.cpu(),
        remapped_edge_index.T.cpu(),
        enabled=enabled,
        color=color,
        transparency=color[-1] if color and len(color) == 4 else None,
        **kwargs,
    )

    # TODO: generalize to all other edge properties!
    if hasattr(edges, 'edge_weight'):
        curve.add_scalar_quantity('weight', edges.edge_weight.cpu(), defined_on='edges', enabled=True)

    # Compute & render edge lengths
    edge_lengths = torch.linalg.vector_norm(
        all_pos[remapped_edge_index[1], :] - all_pos[remapped_edge_index[0], :],
        dim=-1,
    )
    curve.add_scalar_quantity('length', edge_lengths.cpu(), defined_on='edges', enabled=True)

    return curve

def register_pyg_data(
    data: Data,
    node_color: dict | None = None,
    node_radius: dict | None = None,
    edge_color: dict | None = None,
    edge_radius: dict | None = None,
) -> None:
    data = data.clone()

    # Determine the number of batches
    batch_size = 1
    if hasattr(data.node_stores[-1], 'batch'):
        batch_size = torch.max(data.node_stores[-1].batch) + 1

    # Apply offsets to each instance so a whole batch can be viewed at once
    # TODO: use a nicer packing heuristic, for pretty layouts!
    for store in data.node_stores:
        if hasattr(store, 'pos'):
            bbox_size = store.pos.max(dim=0).values - store.pos.min(dim=0).values
            break
    batch_rows = 4
    for store in data.node_stores:
        if hasattr(store, 'pos') and hasattr(store, 'batch'):
            y_offset_index = torch.remainder(store.batch, batch_rows)
            x_offset_index = (store.batch - y_offset_index) // batch_rows
            store.pos[:, 0] += x_offset_index * bbox_size[0] * 1.1
            store.pos[:, 1] += y_offset_index * bbox_size[1] * 1.1


    if isinstance(data, HeteroData):
        node_color, node_radius = node_color or {}, node_radius or {}
        edge_color, edge_radius = edge_color or {}, edge_radius or {}
        show_scale = True
        scaling_factor = 1.0
        for scale, item in data.node_items():
            for k, v in data._global_store.items():
                item[k] = v
            register_nodes(
                scale,
                nodes=item,
                color=node_color.get(scale, None),
                # TODO: how to choose radius?
                radius=node_radius.get(scale, (0.005 / math.sqrt(batch_size)) * scaling_factor),
                enabled=show_scale,
            )
            scaling_factor *= 1.05
            if hasattr(item, 'pos'):
                # Only show the first scale with position, by default
                show_scale = False  

        for (source, _, dest), item in data.edge_items():
            register_edges(
                f"{source}-to-{dest}",
                source_nodes=data[source],
                dest_nodes=data[dest],
                edges=item,
                color=edge_color.get(f'{source}__to__{dest}', None),
                # TODO: how to choose radius?
                radius=edge_radius.get(f'{source}__to__{dest}', 0.005 / batch_size),
            )

    else:
        assert not isinstance(node_color, dict)
        assert not isinstance(edge_color, dict)
        if hasattr(data, 'pos'):
            register_nodes(
                'data',
                data.node_stores[0],
                color=node_color or None,
                radius=node_radius or (0.005 / math.sqrt(batch_size)),
            )
            register_edges(
                'data',
                data.node_stores[0],
                data.node_stores[0],
                data.edge_stores[0],
                color=edge_color or None,
                radius=edge_radius or (0.005 / batch_size),
            )

@torch.compiler.disable(recursive=True)
def show_data(
    data: Data,
    **kwargs,
):
    ps.init()

    # These defaults are appropriate for the datasets in this library
    ps.set_up_dir('z_up')
    ps.set_front_dir('x_front')
    ps.set_ground_plane_mode('shadow_only')
    ps.set_SSAA_factor(1)

    with torch.no_grad():
        print(data)
        register_pyg_data(data, **kwargs)

    ps.show()
    ps.remove_all_structures()

# TODO: add utility for rendering to an image!
@torch.compiler.disable(recursive=True)
def render_data(
    filename: str,
    data: Data,
    **kwargs,
):
    ps.init()

    ps.set_up_dir('z_up')
    ps.set_front_dir('x_front')
    ps.set_ground_plane_mode('shadow_only')
    ps.set_SSAA_factor(4)  # Nicer visual fidelity!

    with torch.no_grad():
        print(data)
        register_pyg_data(data, **kwargs)

    ps.screenshot(filename)
    ps.remove_all_structures()
