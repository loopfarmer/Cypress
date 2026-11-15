##
# @file   segment_clearance_unittest.py
# @brief  edaroute: the segment-clearance term against hand-worked cases, and its gradient against finite differences.
#
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dreamplace.ops.segment_clearance.segment_clearance import SegmentClearance  # noqa: E402
sys.path.pop()


def build(stub=0.0):
    # pins 0 and 1 are a segment's ends on parts 0 and 1; pin 2 (part 2) and pin 3 (part 0) are obstacles
    doc = {"stub": stub,
           "segments": [{"a": "P0", "b": "P1", "net": "RF", "kind": "rf", "clearance": 2.0}],
           "obstacles": {f"P{i}": {"part": f"N{n}", "net": "RF" if i < 2 else "", "side": "top", "r": 1.0}
                         for i, n in enumerate([0, 1, 2, 0])}}
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(doc, f)
    f.close()
    db = SimpleNamespace(net_name2id_map={f"P{i}": i for i in range(4)}, net2pin_map=[np.array([i]) for i in range(4)],
                         pin_offset_x=np.zeros(4), pin_offset_y=np.array([0.0, 0, 0, 1.0]), pin2node_map=np.array([0, 1, 2, 0]),
                         num_pins=4)
    return SegmentClearance(f.name, db, torch.device("cpu"), torch.float64)


class SegmentClearanceTest(unittest.TestCase):
    def test_hand_cases(self):
        sc = build()
        # segment from (0,0) to (10,0); pin 2 at (5,1): reach 2 + 1 = 3, distance 1, shortfall 2, cost 4
        x = torch.tensor([0.0, 10, 5, 50, 0, 0, 1, 50], dtype=torch.float64)
        self.assertAlmostEqual(float(sc(x)), 4.0)
        x[6] = 5.0                       # pin 2 now 5 away: clear
        self.assertAlmostEqual(float(sc(x)), 0.0)

    def test_stub_hides_the_end_parts_own_neighbours(self):
        x = torch.tensor([0.0, 10, 50, 0.5, 0, 0, 50, 1.0], dtype=torch.float64)   # pin 3, on part 0, beside pin 0
        self.assertGreater(float(build(stub=0.0)(x)), 0.0)
        self.assertAlmostEqual(float(build(stub=2.0)(x)), 0.0)                       # within the stub of its own end

    def test_gradient(self):
        sc = build()
        x = torch.tensor([0.0, 10, 5, 3, 0, 0.4, 1.2, 0.9], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(sc, (x,), eps=1e-6, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
