"""Lightning module for the relative-motion, structured-loss experiment."""

from typing import Dict, Tuple

import torch

from limo.src.models.limo_model import LimoModel


class RelativeStructuredLimoModel(LimoModel):
    """Keep the trajectory forward API while supervising decoder motions."""

    _COMPONENT_NAMES = ("xy", "yaw", "motion", "endpoint", "progress", "smooth")

    def model_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        targets = batch["path"]
        preds, pred_motion = self.net.forward_with_motion(batch)
        components = self.loss.compute_components(preds, targets, pred_motion)
        self._last_loss_components = components
        self._last_pred_motion = pred_motion
        return components["total"], preds, targets

    def _log_loss_components(self, split: str) -> None:
        for name in self._COMPONENT_NAMES:
            self.log(
                f"{split}/loss_{name}",
                self._last_loss_components[name],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        output = super().training_step(batch, batch_idx)
        self._log_loss_components("train")
        output["motion"] = self._last_pred_motion
        return output

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        output = super().validation_step(batch, batch_idx)
        self._log_loss_components("val")
        output["motion"] = self._last_pred_motion
        return output

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        super().test_step(batch, batch_idx)
        self._log_loss_components("test")
