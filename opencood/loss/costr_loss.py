# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from opencood.loss.point_pillar_loss import PointPillarLoss


class CoSTrLoss(nn.Module):
    """
    Wrapper loss for CoSTr reproduction:
    - Base detection loss: PointPillarLoss
    - Optional MI + sparsity terms from model outputs:
        - output_dict['mi_loss']
        - output_dict['mi_l1']
    """

    def __init__(self, args: Dict):
        super().__init__()
        self.det = PointPillarLoss(args["det"])
        self.mi_weight = float(args.get("mi_weight", 1.0))
        self.mi_l1_weight = float(args.get("mi_l1_weight", 0.0))
        self.loss_dict = {}

    def forward(self, output_dict: Dict, target_dict: Dict, suffix: str = "") -> torch.Tensor:
        det_loss = self.det(output_dict, target_dict, suffix=suffix)
        total = det_loss

        mi_loss = output_dict.get("mi_loss", None)
        mi_l1 = output_dict.get("mi_l1", None)

        if suffix == "":
            if mi_loss is not None:
                total = total + self.mi_weight * mi_loss
            if (mi_l1 is not None) and self.mi_l1_weight > 0:
                total = total + self.mi_l1_weight * mi_l1

        # merge loss dicts for logging
        self.loss_dict = dict(self.det.loss_dict)
        self.loss_dict.update(
            {
                "total_loss": float(total.detach().cpu().item()),
                "mi_loss": float(mi_loss.detach().cpu().item()) if mi_loss is not None else 0.0,
                "mi_l1": float(mi_l1.detach().cpu().item()) if mi_l1 is not None else 0.0,
            }
        )
        return total

    def logging(self, epoch, batch_id, batch_len, writer=None, suffix: str = ""):
        self.det.logging(epoch, batch_id, batch_len, writer=writer, suffix=suffix)
        if suffix == "":
            mi_loss = self.loss_dict.get("mi_loss", 0.0)
            mi_l1 = self.loss_dict.get("mi_l1", 0.0)
            print(f"[epoch {epoch}][{batch_id+1}/{batch_len}] || MI: {mi_loss:.4f} || MI_L1: {mi_l1:.4f}")
            if writer is not None:
                writer.add_scalar("MI_loss", mi_loss, epoch * batch_len + batch_id)
                writer.add_scalar("MI_L1", mi_l1, epoch * batch_len + batch_id)


