
from torch_geometric.transforms import Compose

from clouds.transforms import Identity
from clouds.transforms.pipeline import unpack_pipeline


class MockTransform(Identity):
    def __init__(self, name=""):
        self.name = name

class TestUnpackPipeline:
    def test_unpack_pipeline_full_pipeline(self):
        """Test unpack_pipeline with all stages present."""
    
        # Create mock transforms
        t1 = MockTransform("t1")
        t2 = MockTransform("t2")
        t3 = MockTransform("batch")
        t4 = MockTransform("finalize")
        t5 = MockTransform("post")
    
        # Build pipeline
        pipeline = [
            t1,
            dict(train=[t2]),
            "collate",
            t3,
            "device",
            t4,
            "model",
            dict(test=t5),
        ]
    
        result = unpack_pipeline(pipeline)
    
        expected = {
            "transform": {
                "train": Compose([t1, t2]),
                "val": t1,
                "test": t1,
            },
            "batch_transform": {
                "train": t3,
                "val": t3,
                "test": t3,
            },
            "finalize_transform": {
                "train": t4,
                "val": t4,
                "test": t4,
            },
            "post_transform": {
                "train": None,
                "val": None,
                "test": t5,
            },
        }

        assert result['transform'].pop('train').transforms == expected['transform'].pop('train').transforms
        assert result == expected


