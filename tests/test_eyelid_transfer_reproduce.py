import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from src.eyelid_transfer_neural import AnemiaMobileNet, prefix_cache, runtime
from src.eyelid_transfer_reproduce import reproduce


class TransferReproductionTests(unittest.TestCase):
    def test_checksum_failure_precedes_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / 'payload.pt').write_text('changed')
            (output / 'checksum_manifest.json').write_text(json.dumps({'payload.pt': 'unexpected'}))
            with patch('torch.load', side_effect=AssertionError('must not deserialize')):
                with self.assertRaises(ValueError): reproduce(output, output)

    def test_same_geometry_prefix_reproduction_is_exact_after_tail_changes(self):
        runtime(91); model = AnemiaMobileNet().eval()
        x = torch.rand(7, 3, 224, 224)
        original = prefix_cache(model, x, batch_size=3)
        with torch.no_grad():
            for p in model.features[-2:].parameters(): p.add_(.01)
            model.head.weight.add_(.01)
        fresh = prefix_cache(model, x, batch_size=3)
        np.testing.assert_array_equal(original.numpy(), fresh.numpy())

if __name__ == '__main__': unittest.main()
