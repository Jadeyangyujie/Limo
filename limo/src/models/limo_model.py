from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import wandb
import numpy as np
import yaml
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
from torchmetrics.regression import MeanAbsoluteError

from limo.src.utils.visualization import create_combined_visualization


class LimoModel(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        camera_info: Optional[str] = None,
        train_scalar_log_interval: int = 20,
        wandb_num_images: int = 4,
        wandb_train_log_interval: int = 500,
        wandb_log_worst: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.net = net
        self.loss = loss

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

        self.val_loss_best = MinMetric()
        self.camera_info = self._load_camera_info(camera_info)

    def _load_camera_info(
        self, camera_info_path: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        if camera_info_path is None:
            return None

        path = Path(camera_info_path)
        if not path.exists():
            self.print(
                f"camera_info file not found: {camera_info_path}. Falling back to image-only visualization."
            )
            return None

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(batch)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_mae.reset()
        self.val_loss_best.reset()

    def model_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path = batch["path"]
        preds = self(batch)
        loss = self.loss(preds, path)
        return loss, preds, path

    @staticmethod
    def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
        """
        Wrap angle to [-pi, pi].

        这样可以避免 179° 和 -179° 被误认为差了 358°。
        """
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    @classmethod
    def _compute_path_metrics(
        cls, preds: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        preds, targets: [B, N, 3]
        每个 waypoint 是 [x, y, yaw]。

        返回：
            ade: 每条路径所有 waypoint 的平均位置误差
            fde: 最后一个 waypoint 的位置误差
            mean_yaw_deg: 所有 waypoint 的平均 yaw 误差，单位 degree
            final_yaw_deg: 最后一个 waypoint 的 yaw 误差，单位 degree
        """
        pos_err = torch.linalg.norm(preds[..., :2] - targets[..., :2], dim=-1)
        yaw_err = cls._wrap_angle(preds[..., 2] - targets[..., 2]).abs()

        ade = pos_err.mean(dim=-1)
        fde = pos_err[:, -1]

        mean_yaw_deg = yaw_err.mean(dim=-1) * 180.0 / np.pi
        final_yaw_deg = yaw_err[:, -1] * 180.0 / np.pi

        return {
            "ade": ade,
            "fde": fde,
            "mean_yaw_deg": mean_yaw_deg,
            "final_yaw_deg": final_yaw_deg,
        }

    @staticmethod
    def _to_np_image(t: torch.Tensor) -> np.ndarray:
        """
        Convert image tensor [B, C, H, W] to uint8 numpy [B, H, W, C].

        兼容两种情况：
        1. 图像已经在 [0, 1]
        2. 图像被 normalize 到其他范围
        """
        x = t.detach().cpu().float()

        if x.min() < 0.0 or x.max() > 1.0:
            x_min = x.amin(dim=(1, 2, 3), keepdim=True)
            x_max = x.amax(dim=(1, 2, 3), keepdim=True)
            x = (x - x_min) / (x_max - x_min + 1e-6)

        x = x.clamp(0.0, 1.0)
        x = x.permute(0, 2, 3, 1).numpy()
        return (x * 255).astype(np.uint8)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        loss, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        self.train_mae(preds, targets)

        metrics = self._compute_path_metrics(preds, targets)

        # --------------------------------------------------
        # 1. epoch-level metrics
        # 保留原来的 train/loss、train/mae 命名
        # --------------------------------------------------
        self.log(
            "train/loss",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/mae",
            self.train_mae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log(
            "train/ade",
            metrics["ade"].mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/fde",
            metrics["fde"].mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/mean_yaw_deg",
            metrics["mean_yaw_deg"].mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train/final_yaw_deg",
            metrics["final_yaw_deg"].mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        # --------------------------------------------------
        # 2. step-level metrics
        # 只新增日志，不参与 loss，不影响反向传播
        # --------------------------------------------------
        interval = self.hparams.train_scalar_log_interval

        if interval > 0 and self.global_step % interval == 0:
            step_mae = torch.mean(torch.abs(preds - targets))

            self.log(
                "train/loss_step",
                loss.detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            self.log(
                "train/mae_step",
                step_mae.detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )
            self.log(
                "train/ade_step",
                metrics["ade"].mean().detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            self.log(
                "train/fde_step",
                metrics["fde"].mean().detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            self.log(
                "train/mean_yaw_deg_step",
                metrics["mean_yaw_deg"].mean().detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )
            self.log(
                "train/final_yaw_deg_step",
                metrics["final_yaw_deg"].mean().detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )

            if len(self.trainer.optimizers) > 0:
                lr = self.trainer.optimizers[0].param_groups[0]["lr"]
                self.log(
                    "train/lr",
                    lr,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                )

        # 保持原始逻辑：preds 不 detach
        return {"loss": loss, "preds": preds}

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss)
        self.val_mae(preds, targets)

        metrics = self._compute_path_metrics(preds, targets)

        # 保留原来的 val/loss、val/mae 命名
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True)

        self.log(
            "val/ade",
            metrics["ade"].mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/fde",
            metrics["fde"].mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/mean_yaw_deg",
            metrics["mean_yaw_deg"].mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "val/final_yaw_deg",
            metrics["final_yaw_deg"].mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        # 保持原始逻辑：loss 和 preds 不 detach
        return {"loss": loss, "preds": preds}

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        preds = self(batch)
        uuids = batch["uuid"]
        paths = preds.detach().cpu().numpy()
        return {"uuid": uuids, "path_pred": paths}

    def on_validation_batch_end(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if wandb.run is None:
            return

        if not self.trainer.is_global_zero:
            return

        # val 每个 epoch 只记录第一个 batch，压力比较小
        if batch_idx == 0:
            self.log_images_wandb(
                outputs,
                batch,
                split="val",
                num_imgs=self.hparams.wandb_num_images,
                sort_by_error=self.hparams.wandb_log_worst,
            )

    def on_train_batch_end(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if wandb.run is None:
            return

        if not self.trainer.is_global_zero:
            return

        interval = self.hparams.wandb_train_log_interval

        # train 图片不要每个 batch 都传，每隔 interval 个 step 传一次
        if interval > 0 and self.global_step > 0 and self.global_step % interval == 0:
            self.log_images_wandb(
                outputs,
                batch,
                split="train",
                num_imgs=self.hparams.wandb_num_images,
                sort_by_error=self.hparams.wandb_log_worst,
            )

    def on_validation_epoch_end(self) -> None:
        cur_val_loss = self.val_loss.compute()
        self.val_loss_best(cur_val_loss)
        self.log(
            "val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True
        )

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        self.test_loss(loss)
        self.test_mae(preds, targets)

        metrics = self._compute_path_metrics(preds, targets)

        self.log(
            "test/loss",
            self.test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True)

        self.log(
            "test/ade",
            metrics["ade"].mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/fde",
            metrics["fde"].mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/mean_yaw_deg",
            metrics["mean_yaw_deg"].mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "test/final_yaw_deg",
            metrics["final_yaw_deg"].mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def setup(self, stage: str) -> None:
        self.net.setup()
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        if self.hparams.scheduler is not None:
            # Lightning 在初始化 optimizer/scheduler 时会先调用 configure_optimizers。
            # 有些版本里，此时 datamodule.setup("fit") 还没执行；
            # 直接访问 estimated_stepping_batches 会触发 train_dataloader，
            # 从而报 "Train dataset not loaded. Call setup() first."。
            # 这里仅补一次 datamodule.setup("fit")，不改变 optimizer、scheduler、loss、forward 等训练逻辑。
            try:
                total_steps = self.trainer.estimated_stepping_batches
            except RuntimeError as e:
                if "Train dataset not loaded" not in str(e):
                    raise

                datamodule = getattr(self.trainer, "datamodule", None)
                if datamodule is None:
                    raise

                datamodule.setup(stage="fit")
                total_steps = self.trainer.estimated_stepping_batches

            scheduler = self.hparams.scheduler(
                optimizer=optimizer, total_steps=total_steps
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}

    def log_images_wandb(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        split: str,
        num_imgs: int = 4,
        sort_by_error: bool = True,
    ) -> None:
        """
        Log predicted paths as combined visualizations to wandb.

        轻量版：
        1. 不上传 table
        2. 不上传 histogram
        3. 只上传少量图片
        4. 默认优先上传当前 batch 中 FDE 最大的样本
        5. 每张图带 caption，方便定位问题
        """
        assert wandb.run is not None, "This can only be used with wandb active"

        if "preds" not in outputs:
            return

        preds = outputs["preds"]
        targets = batch["path"]

        metrics = self._compute_path_metrics(preds, targets)

        batch_size = preds.shape[0]
        num_imgs = min(num_imgs, batch_size)

        if sort_by_error:
            selected_indices = torch.argsort(metrics["fde"], descending=True)[:num_imgs]
        else:
            selected_indices = torch.arange(num_imgs, device=preds.device)

        predicted_paths = preds[selected_indices].detach().cpu().numpy()
        ground_truth_paths = targets[selected_indices].detach().cpu().numpy()

        images = self._to_np_image(batch["image_front"][selected_indices])

        images_left = None
        images_right = None

        if "image_left" in batch:
            images_left = self._to_np_image(batch["image_left"][selected_indices])

        if "image_right" in batch:
            images_right = self._to_np_image(batch["image_right"][selected_indices])

        goals = batch["goal"][selected_indices].detach().cpu().numpy()

        selected_indices_cpu = selected_indices.detach().cpu().tolist()

        ade_np = metrics["ade"].detach().cpu().numpy()
        fde_np = metrics["fde"].detach().cpu().numpy()
        mean_yaw_np = metrics["mean_yaw_deg"].detach().cpu().numpy()
        final_yaw_np = metrics["final_yaw_deg"].detach().cpu().numpy()

        uuids = batch.get("uuid", None)

        visualizations = []

        for rank, original_idx in enumerate(selected_indices_cpu):
            if uuids is not None:
                try:
                    uuid = uuids[original_idx]
                except Exception:
                    uuid = str(original_idx)
            else:
                uuid = str(original_idx)

            combined_img = create_combined_visualization(
                images[rank],
                [ground_truth_paths[rank], predicted_paths[rank]],
                goals[rank : rank + 1],
                camera_info=self.camera_info,
                image_left=images_left[rank] if images_left is not None else None,
                image_right=images_right[rank] if images_right is not None else None,
            )

            goal_x, goal_y, goal_yaw = goals[rank]

            caption = (
                f"{split} | epoch={self.current_epoch} | step={self.global_step} | "
                f"idx={original_idx} | uuid={uuid} | "
                f"ADE={ade_np[original_idx]:.3f}m | "
                f"FDE={fde_np[original_idx]:.3f}m | "
                f"yaw={mean_yaw_np[original_idx]:.2f}deg | "
                f"final_yaw={final_yaw_np[original_idx]:.2f}deg | "
                f"goal=({goal_x:.2f}, {goal_y:.2f}, {goal_yaw:.2f})"
            )

            visualizations.append(wandb.Image(combined_img, caption=caption))

        wandb.log({f"{split}/predictions": visualizations})