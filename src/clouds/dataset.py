from torch_geometric.transforms import BaseTransform, Compose


def unpack_transforms(transforms: list[BaseTransform, str, dict]) -> dict:

    # TODO: example of the expected output format:
    [
        # This applies to all data
        BaseTransform(),
        # These apply only on the train set (e.g. regularization transforms)
        dict(train=[BaseTransform(), ...]),  #
        "collate",
        # This applies to data after batching
        BaseTransform(),
        "device",
        # This applies to data after transferring to the GPU
        BaseTransform(),
        "model",
        # This applies to the prediction produced by the model
        dict(test=BaseTransform()),
    ]

    # Transforms that apply to a single Data object (before collation)
    # TODO: take from list until 'collate'
    # TODO: dict(train=..., ) enables split-specific transforms

    # Data is collated

    # Data is transferred to the device (e.g. GPU)
    
    # Data is finalized (e.g. unpack, combine into features, etc.)
    
    # Model is applied
    
    # Output data is postprocessed


    # NOTE: vote transforms could produce elements in a new batch dim?

    # TODO: example of the expected output format:
    return dict(
        train=dict(
            transform=Compose([BaseTransform(), BaseTransform(), ...]),
            batch_transform=...,  # After collation
            finalize_transform=...,  # On the GPU
        ),
        val=dict(transform=...),
        test=dict(
            transform=...,
            post_transform=...,  # After applying to the model
        ),
    )

