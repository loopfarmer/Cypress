##
# @file   segment_clearance.py
# @brief  edaroute: keep pins of other nets clear of the straight line between two connected pins.
#
# A segment is two pins that should be joined by a short straight track: an RF hop from its entry pad to its exit pad,
# or an IC pin's escape to the part on its net. Any other pin (an obstacle) closer to the segment than the segment's
# clearance plus the pin's own radius costs (clearance + radius - distance)^2. That is the placement critic's (edaloop)
# "other pads in the way" and "the escape crosses other pins", as a smooth term the placer can follow.
#
# Positions come from the pin-position op, so the term turns with the parts. It's plain PyTorch, so it runs on any
# device, and its gradient comes from autograd.
#

import json
import logging

import numpy as np
import torch
from torch import nn


class SegmentClearance(nn.Module):
    def __init__(self, path, placedb, device, dtype):
        """
        @param path a segments JSON written by edaroute export --segments: segments (a, b, net, kind, clearance) name
               pins by their one-pin obstacle nets, and obstacles (r, net, part) give each pin's radius and real net
        """
        super().__init__()
        doc = json.load(open(path))
        name2id = {(k.decode() if isinstance(k, bytes) else str(k)): v for k, v in placedb.net_name2id_map.items()}

        def pin(net):
            return int(placedb.net2pin_map[name2id[net]][0])

        obstacles = list(doc["obstacles"].items())
        obst_pin = np.array([pin(n) for n, _ in obstacles], dtype=np.int64)
        obst_r = np.array([o["r"] for _, o in obstacles], dtype=np.float64)
        obst_net = [o["net"] for _, o in obstacles]
        obst_side = [o.get("side", "top") for _, o in obstacles]
        index = {n: i for i, (n, _) in enumerate(obstacles)}
        segs = doc["segments"]
        a = np.array([pin(s["a"]) for s in segs], dtype=np.int64)
        b = np.array([pin(s["b"]) for s in segs], dtype=np.int64)
        c = np.array([s["clearance"] for s in segs], dtype=np.float64)
        # a pin may sit near a segment it belongs to: its ends, and any pin on the segment's own net. And a track leaves
        # a pin by a short stub straight out of it before it turns (the critic's escape model), so the end parts' own
        # pins within `stub` of an end don't count: a fine-pitch IC's neighbouring pins would otherwise read as in the
        # way of every escape that leaves at an angle. Distances within a part don't change as it moves or turns.
        stub = float(doc.get("stub", 0.0))
        offx, offy = np.asarray(placedb.pin_offset_x, np.float64), np.asarray(placedb.pin_offset_y, np.float64)
        p2n = np.asarray(placedb.pin2node_map)
        mask = np.ones((len(segs), len(obstacles)), dtype=bool)
        for i, s in enumerate(segs):
            mask[i, index[s["a"]]] = mask[i, index[s["b"]]] = False
            for j, net in enumerate(obst_net):
                if net and net == s["net"]:
                    mask[i, j] = False
            for end in (a[i], b[i]):
                near = (p2n[obst_pin] == p2n[end]) & (np.hypot(offx[obst_pin] - offx[end], offy[obst_pin] - offy[end]) <= stub)
                mask[i, near] = False
            if s["kind"] == "escape":
                # as the critic checks an escape: only the IC's own pins are in its way; a track goes round other parts
                mask[i, p2n[obst_pin] != p2n[a[i]]] = False
            else:
                # an RF hop runs on its parts' side: pads on the other side aren't in its way
                mask[i, np.array([o["side"] != obst_side[index[s["a"]]] for _, o in obstacles])] = False
        # each end on an IC moves `stub` outwards, from the centroid of the IC's pins through the end pin, so a hop or
        # escape starts where a track leaves the pin rather than at the pin's centre over the IC's own pads
        part_pins = {}
        for j, pid in enumerate(obst_pin):
            part_pins.setdefault(int(p2n[pid]), []).append(int(pid))
        groups = sorted({int(p2n[e]) for i, sg in enumerate(segs) for e, k in ((a[i], "ext_a"), (b[i], "ext_b")) if sg.get(k)})
        width = max((len(part_pins[g]) for g in groups), default=1)
        self.group_pins = torch.full((max(len(groups), 1), width), -1, dtype=torch.long)
        for gi, g in enumerate(groups):
            self.group_pins[gi, :len(part_pins[g])] = torch.tensor(part_pins[g])
        gid = {g: gi for gi, g in enumerate(groups)}
        self.ext_a = torch.tensor([gid[int(p2n[a[i]])] if sg.get("ext_a") else -1 for i, sg in enumerate(segs)], dtype=torch.long).to(device)
        self.ext_b = torch.tensor([gid[int(p2n[b[i]])] if sg.get("ext_b") else -1 for i, sg in enumerate(segs)], dtype=torch.long).to(device)
        self.group_pins = self.group_pins.to(device)
        self.stub = stub
        self.kinds = np.array([s["kind"] for s in segs])
        self.a = torch.from_numpy(a).to(device)
        self.b = torch.from_numpy(b).to(device)
        self.obst = torch.from_numpy(obst_pin).to(device)
        self.reach = torch.from_numpy(c[:, None] + obst_r[None, :]).to(device=device, dtype=dtype)
        self.mask = torch.from_numpy(mask).to(device)
        self.num_pins = placedb.num_pins
        logging.info("segment clearance: %d segments (%s) against %d pins"
                     % (len(segs), ", ".join("%d %s" % (int((self.kinds == k).sum()), k) for k in np.unique(self.kinds)),
                        len(obstacles)))

    def violations(self, pin_pos):
        """Each (segment, pin) pair's shortfall from its clearance, zero where it's clear."""
        n = self.num_pins
        px, py = pin_pos[:n], pin_pos[n:]
        ax, ay, bx, by = px[self.a], py[self.a], px[self.b], py[self.b]
        if self.stub > 0 and len(self.group_pins):
            valid = self.group_pins >= 0
            gp = self.group_pins.clamp_min(0)
            cnt = valid.sum(1).clamp_min(1)
            gx, gy = (px[gp] * valid).sum(1) / cnt, (py[gp] * valid).sum(1) / cnt

            def out(x, y, ext):
                on = ext >= 0
                e = ext.clamp_min(0)
                ux, uy = x - gx[e], y - gy[e]
                norm = torch.sqrt(ux * ux + uy * uy + 1e-12)
                return x + on * self.stub * ux / norm, y + on * self.stub * uy / norm
            ax, ay = out(ax, ay, self.ext_a)
            bx, by = out(bx, by, self.ext_b)
        ox, oy = px[self.obst], py[self.obst]
        dx, dy = (bx - ax)[:, None], (by - ay)[:, None]
        len2 = (dx * dx + dy * dy).clamp_min(1e-9)
        t = (((ox[None, :] - ax[:, None]) * dx + (oy[None, :] - ay[:, None]) * dy) / len2).clamp(0, 1)
        qx, qy = ax[:, None] + t * dx - ox[None, :], ay[:, None] + t * dy - oy[None, :]
        dist = torch.sqrt(qx * qx + qy * qy + 1e-12)
        return torch.relu(self.reach - dist) * self.mask

    def forward(self, pin_pos):
        v = self.violations(pin_pos)
        return (v * v).sum()
