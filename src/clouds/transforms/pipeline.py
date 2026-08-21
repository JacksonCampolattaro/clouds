from typing import Callable

from torch_geometric.transforms import BaseTransform, Compose


def unpack_pipeline(pipeline: list[Callable, str, dict]) -> dict[str, dict[str, BaseTransform | None]]:

    out = dict(
        transform=dict(train=[], val=[], test=[]),
        batch_transform=dict(train=[], val=[], test=[]),
        finalize_transform=dict(train=[], val=[], test=[]),
        post_transform=dict(train=[], val=[], test=[]),
    )
    current_stage = 'transform'
    for potential_transform in pipeline:

        # State machine determines which stage of the pipeline we're in
        if potential_transform == 'collate':
            current_stage = 'batch_transform'
        elif potential_transform == 'device':
            current_stage = 'finalize_transform'
        elif potential_transform == 'model':
            current_stage = 'post_transform'
        else:
            # When we encounter a transform, add it in the appropriate place
            if isinstance(potential_transform, dict):
                for split, transform in potential_transform.items():
                    if isinstance(transform, list):
                        out[current_stage][split].extend(transform)
                    else:
                        out[current_stage][split].append(transform)
            else:
                for split in ['train', 'val', 'test']:
                    out[current_stage][split].append(potential_transform)

    # Convert to the desired output format 
    for stage, split_transforms in out.items():
        for split, t in split_transforms.items():
            if not t:
                out[stage][split] = None
            elif len(t) == 1:
                out[stage][split] = t[0]
            else:
                out[stage][split] = Compose(t)

    return out

