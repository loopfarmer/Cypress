##
# @file   pin_pos_unitest.py
# @author Yibo Lin
# @date   Aug 2019
#

import os
import sys
import time
import numpy as np
import unittest
#import pickle
import gzip
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
from dreamplace.ops.pin_pos import pin_pos
sys.path.pop()

import pdb

import torch
from torch.autograd import Function, Variable


def build_pin_pos(pos, pin_offset_x, pin_offset_y, pin2node_map,
                  num_physical_nodes):
    num_nodes = pos.numel() // 2
    pin_x = pin_offset_x.add(
        torch.index_select(pos[0:num_physical_nodes],
                           dim=0,
                           index=pin2node_map.long()))
    pin_y = pin_offset_y.add(
        torch.index_select(pos[num_nodes:num_nodes + num_physical_nodes],
                           dim=0,
                           index=pin2node_map.long()))
    pin_pos = torch.cat([pin_x, pin_y], dim=0)
    return pin_pos


class PinPosOpTest(unittest.TestCase):
    def test_pin_pos_random(self):
        dtype = torch.float32
        pos = np.array([[1, 10], [2, 20], [3, 30]], dtype=np.float32)
        node2pin_map = np.array([np.array([0, 4]), np.array([1, 2, 3])], dtype=object)  # edaroute: ragged, for NumPy 2
        num_physical_nodes = len(node2pin_map)
        num_pins = 0
        for pins in node2pin_map:
            num_pins += len(pins)
        pin2node_map = np.zeros(num_pins, dtype=np.int32)
        for node_id, pins in enumerate(node2pin_map):
            for pin in pins:
                pin2node_map[pin] = node_id

        pin_offset_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=pos.dtype)
        pin_offset_y = np.array([0.01, 0.02, 0.03, 0.04, 0.05],
                                dtype=pos.dtype)

        # construct flat_node2pin_map and flat_node2pin_start_map
        # flat nodepin map, length of #pins
        flat_node2pin_map = np.zeros(num_pins, dtype=np.int32)
        # starting index in nodepin map for each node, length of #nodes+1, the last entry is #pins
        flat_node2pin_start_map = np.zeros(len(node2pin_map) + 1,
                                           dtype=np.int32)
        count = 0
        for i in range(len(node2pin_map)):
            flat_node2pin_map[count:count +
                              len(node2pin_map[i])] = node2pin_map[i]
            flat_node2pin_start_map[i] = count
            count += len(node2pin_map[i])
        flat_node2pin_start_map[len(node2pin_map)] = len(pin2node_map)

        print("flat_node2pin_map = ", flat_node2pin_map)
        print("flat_node2pin_start_map = ", flat_node2pin_start_map)

        pos_var = Variable(torch.from_numpy(pos).reshape([-1]),
                           requires_grad=True)

        golden_value = build_pin_pos(pos_var, torch.from_numpy(pin_offset_x),
                                     torch.from_numpy(pin_offset_y),
                                     torch.from_numpy(pin2node_map),
                                     num_physical_nodes)
        golden_loss = golden_value.sum()
        print("golden_value = ", golden_value)
        print("golden_loss = ", golden_loss)
        golden_loss.backward()
        golden_grad = pos_var.grad.clone()
        print("golden grad = ", golden_grad)
        golden_value = golden_value.detach().numpy()
        golden_grad = golden_grad.detach().numpy()

        # test cpu
        print(pos_var)
        # clone is very important, because the custom op cannot deep copy the data
        custom = pin_pos.PinPos(
            pin_offset_x=torch.from_numpy(pin_offset_x),
            pin_offset_y=torch.from_numpy(pin_offset_y),
            pin2node_map=torch.from_numpy(pin2node_map),
            flat_node2pin_map=torch.from_numpy(flat_node2pin_map),
            flat_node2pin_start_map=torch.from_numpy(flat_node2pin_start_map),
            num_physical_nodes=num_physical_nodes,
            # edaroute: upstream's op takes sizes and an orientation since rotation; theta = 0 is unrotated
            h=torch.ones(num_physical_nodes), w=torch.ones(num_physical_nodes),
            best_theta=torch.zeros(num_physical_nodes))
        result = custom.forward(pos_var)
        custom_loss = result.sum()
        print("custom = ", result)
        pos_var.grad.zero_()
        custom_loss.backward()
        grad = pos_var.grad.clone()
        print("custom_grad = ", grad)

        np.testing.assert_allclose(result.data.detach().numpy(),
                                   golden_value,
                                   atol=1e-6)
        np.testing.assert_allclose(grad.data.detach().numpy(),
                                   golden_grad,
                                   atol=1e-6)

        # test gpu
        if torch.cuda.device_count():
            pos_var.grad.zero_()
            custom_cuda = pin_pos.PinPos(
                pin_offset_x=torch.from_numpy(pin_offset_x).cuda(),
                pin_offset_y=torch.from_numpy(pin_offset_y).cuda(),
                pin2node_map=torch.from_numpy(pin2node_map).cuda(),
                flat_node2pin_map=torch.from_numpy(flat_node2pin_map).cuda(),
                flat_node2pin_start_map=torch.from_numpy(
                    flat_node2pin_start_map).cuda(),
                num_physical_nodes=num_physical_nodes,
                h=torch.ones(num_physical_nodes).cuda(), w=torch.ones(num_physical_nodes).cuda(),
                best_theta=torch.zeros(num_physical_nodes).cuda())
            result_cuda = custom_cuda.forward(pos_var.cuda())
            custom_cuda_loss = result_cuda.sum()
            print("custom_cuda_result = ", result_cuda.data.cpu())
            custom_cuda_loss.backward()
            grad_cuda = pos_var.grad.clone()
            print("custom_grad_cuda = ", grad_cuda.data.cpu())

            np.testing.assert_allclose(result_cuda.data.cpu().numpy(),
                                       golden_value,
                                       atol=1e-6)
            np.testing.assert_allclose(grad_cuda.data.cpu().numpy(),
                                       grad.data.numpy(),
                                       rtol=1e-6,
                                       atol=1e-6)


class PinPosRotationTest(unittest.TestCase):
    """edaroute: rotation about each part's centre, and its position and angle gradients against finite differences."""

    def setUp(self):
        torch.manual_seed(0)
        self.nodes, self.phys = 7, 5                    # the last 2 are fillers, with no pins
        pins_per = [3, 1, 4, 2, 2]
        self.pin2node = torch.tensor([n for n, k in enumerate(pins_per) for _ in range(k)], dtype=torch.long)
        P = len(self.pin2node)
        self.w = torch.rand(self.nodes, dtype=torch.double) * 10 + 2
        self.h = torch.rand(self.nodes, dtype=torch.double) * 6 + 1
        self.ofx = torch.rand(P, dtype=torch.double) * self.w[self.pin2node]
        self.ofy = torch.rand(P, dtype=torch.double) * self.h[self.pin2node]
        self.flat = torch.arange(P, dtype=torch.int32)
        self.start = torch.tensor([0] + torch.cumsum(torch.tensor(pins_per + [0, 0]), 0).tolist(), dtype=torch.int32)
        self.pos = torch.rand(2 * self.nodes, dtype=torch.double) * 100

    def pins(self, pos, theta):
        return pin_pos.PinPosFunction.apply(pos, self.ofx, self.ofy, theta, self.pin2node, self.flat, self.start,
                                            self.phys, self.h, self.w)

    def devices(self):
        return ["cpu"] + (["cuda"] if torch.cuda.device_count() else [])

    def test_gradients(self):
        for device in self.devices():
            for theta in [torch.rand(self.phys, dtype=torch.double) * 2 * np.pi,
                          torch.tensor([0, 1, 2, 3, 1], dtype=torch.double) * np.pi / 2]:
                args = [t.to(device) for t in (self.ofx, self.ofy, self.pin2node, self.flat, self.start)]
                f = lambda p, t: pin_pos.PinPosFunction.apply(p, args[0], args[1], t, args[2], args[3], args[4],
                                                              self.phys, self.h.to(device), self.w.to(device))
                pos = self.pos.to(device).requires_grad_()
                self.assertTrue(torch.autograd.gradcheck(f, (pos, theta.to(device).requires_grad_()),
                                                         eps=1e-6, atol=1e-6))

    def test_unrotated_is_exact(self):
        out = self.pins(self.pos, torch.zeros(self.phys, dtype=torch.double))
        n = self.nodes
        ref = torch.cat([self.pos[:n][self.pin2node] + self.ofx, self.pos[n:][self.pin2node] + self.ofy])
        self.assertTrue(torch.equal(out, ref))

    def test_quarter_turn(self):
        theta = torch.zeros(self.phys, dtype=torch.double)
        theta[0] = np.pi / 2
        out, P, n = self.pins(self.pos, theta), len(self.pin2node), self.nodes
        cx, cy = self.pos[0] + self.w[0] / 2, self.pos[n] + self.h[0] / 2
        for k in range(3):                               # node 0's pins
            self.assertAlmostEqual(float(out[k]), float(cx - (self.ofy[k] - self.h[0] / 2)), places=9)
            self.assertAlmostEqual(float(out[P + k]), float(cy + (self.ofx[k] - self.w[0] / 2)), places=9)


if __name__ == '__main__':
    unittest.main()
