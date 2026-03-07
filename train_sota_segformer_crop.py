import os
import argparse
import random
import logging
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from transformers import SegformerForSemanticSegmentation
from dataset_ce2ch_crop import get_loader, test_dataset

warnings.filterwarnings(action="ignore")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _two_class_logits_to_pred(logits_2ch: torch.Tensor) -> torch.Tensor:
    if logits_2ch.ndim != 4 or logits_2ch.size(1) != 2:
        raise ValueError(f"Expected logits shape (B,2,H,W), got {tuple(logits_2ch.shape)}")
    return torch.argmax(logits_2ch, dim=1)


def _binary_metrics_from_pred(
    gt_long: torch.Tensor,
    pred_long: torch.Tensor,
    ignore_index: int = 255,
):
    if gt_long.ndim != 3:
        raise ValueError(f"Expected gt shape (B,H,W), got {tuple(gt_long.shape)}")
    if pred_long.ndim != 3:
        raise ValueError(f"Expected pred shape (B,H,W), got {tuple(pred_long.shape)}")

    valid = gt_long != ignore_index
    if valid.sum() == 0:
        return {"object_iou": 0.0, "background_iou": 0.0, "miou": 0.0, "dice": 0.0}

    gt = gt_long[valid]
    pred = pred_long[valid]

    gt_obj = gt == 1
    pred_obj = pred == 1
    gt_bg = ~gt_obj
    pred_bg = ~pred_obj

    tp = (pred_obj & gt_obj).sum().float()
    tn = (pred_bg & gt_bg).sum().float()
    fp = (pred_obj & gt_bg).sum().float()
    fn = (pred_bg & gt_obj).sum().float()

    obj_union = tp + fp + fn
    bg_union = tn + fp + fn
    obj_iou = (tp / obj_union).item() if obj_union > 0 else 0.0
    bg_iou = (tn / bg_union).item() if bg_union > 0 else 0.0

    obj_denom = 2 * tp + fp + fn
    dice = (2 * tp / obj_denom).item() if obj_denom > 0 else 0.0
    miou = 0.5 * (obj_iou + bg_iou)

    return {
        "object_iou": float(obj_iou),
        "background_iou": float(bg_iou),
        "miou": float(miou),
        "dice": float(dice),
    }


def _meta_scalar(meta, key: str, idx: int) -> int:
    value = meta[key]
    if isinstance(value, torch.Tensor):
        return int(value[idx].item())
    if isinstance(value, (list, tuple)):
        item = value[idx]
        if isinstance(item, torch.Tensor):
            return int(item.item())
        return int(item)
    return int(value)


class Trainer:
    def __init__(self, opt, model):
        super().__init__()

        self.opt = opt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        os.makedirs(opt.save_path, exist_ok=True)

        self.model = model.to(self.device)
        self.num_epoch = opt.num_epoch
        self.start_epoch = 0

        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        if opt.use_adamw_split_lr:
            backbone_params, head_params = [], []
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if "decode_head" in n:
                    head_params.append(p)
                else:
                    backbone_params.append(p)

            self.optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": float(opt.lr)},
                    {"params": head_params, "lr": float(opt.lr) * float(opt.head_lr_mult)},
                ],
                weight_decay=float(opt.weight_decay),
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=float(opt.lr), weight_decay=float(opt.weight_decay)
            )

        self.trainloader = get_loader(
            opt.train_root, opt.batch_size, opt.train_size, shuffle=True, drop_last=True
        )
        valset = test_dataset(opt.valid_root, opt.train_size)
        self.valloader = DataLoader(dataset=valset, batch_size=1, shuffle=False, drop_last=False)

        self.ckpoint_path = os.path.join(opt.save_path, "checkpoint")
        self.images_path = os.path.join(opt.save_path, "images")
        self.log_path = os.path.join(opt.save_path, "log")

        os.makedirs(self.ckpoint_path, exist_ok=True)
        os.makedirs(self.images_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

        self.best_miou = 0.0
        self._logging_init(opt)

        if opt.resume_option:
            print("############# resume training #############")
            ckpt_path = os.path.join(self.ckpoint_path, "latest.pth")
            if os.path.exists(ckpt_path):
                self.start_epoch, optim_sd, self.best_miou = self._load_checkpoint(ckpt_path)
                self.optimizer.load_state_dict(optim_sd)
                logging.info("-----------------------------------------------------------------------------")
                logging.info(
                    f"Resume Epoch [{self.start_epoch}] - "
                    f"current_lr: {self.optimizer.param_groups[0]['lr']:.6f}, "
                    f"Best miou : {self.best_miou:.4f}"
                )
                logging.info("-----------------------------------------------------------------------------")
            else:
                logging.info(">>> resume_option=True but latest.pth not found. Start new training.")
        else:
            logging.info(">>> Not resuming. Start new training.")

    def _logging_init(self, opt):
        log_file = os.path.join(self.log_path, "log.log")
        logging.basicConfig(
            filename=log_file,
            format="[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]",
            level=logging.INFO,
            filemode="a",
            datefmt="%Y-%m-%d %I:%M:%S %p",
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger().addHandler(console)

        logging.info(">>> current mode: network-train/val")
        logging.info(">>> config: {}".format(opt))

    def _save_checkpoint(self, epoch, filename):
        path = os.path.join(self.ckpoint_path, filename)
        torch.save(
            {
                "network": self.model.state_dict(),
                "epoch": epoch,
                "optimizer": self.optimizer.state_dict(),
                "best_miou": self.best_miou,
            },
            path,
        )

    def _load_checkpoint(self, ckpt_path):
        chkpoint = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(chkpoint["network"])
        return chkpoint["epoch"], chkpoint["optimizer"], chkpoint["best_miou"]

    @staticmethod
    def save_concat_results3(image, gt, pred, save_dir, epoch, idx, normalize=True):
        os.makedirs(save_dir, exist_ok=True)

        if normalize:
            mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
            image = image * std + mean
            image = torch.clamp(image, 0, 1)

        gt = (gt > 0.5).float()
        pred = (pred > 0.5).float()

        gt_3ch = gt.repeat(1, 3, 1, 1)
        pred_3ch = pred.repeat(1, 3, 1, 1)

        row = torch.cat([image, gt_3ch, pred_3ch], dim=3)

        for b in range(row.size(0)):
            save_path = os.path.join(save_dir, f"epoch_{epoch:03d}_idx_{idx:04d}.png")
            save_image(row[b], save_path, normalize=False, padding=0)

    def fit(self):
        summary = SummaryWriter(f"{self.log_path}/tensorboard")
        best_score = {"epoch": self.start_epoch, "miou": self.best_miou, "loss": 0.0}

        try:
            for epoch in range(self.start_epoch, self.num_epoch):
                self.model.train()
                tr_losses, tr_obj_ious, tr_mious, tr_dices = [], [], [], []

                for images, gts in tqdm(self.trainloader, desc=f"[Train {epoch}/{self.num_epoch}]"):
                    images = images.to(self.device)
                    gts = gts.to(self.device)

                    self.optimizer.zero_grad()

                    out = self.model(pixel_values=images)
                    logits = F.interpolate(
                        out.logits,
                        size=gts.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                    loss = self.criterion(logits, gts)
                    loss.backward()
                    self.optimizer.step()

                    pred = _two_class_logits_to_pred(logits)
                    metrics = _binary_metrics_from_pred(gts, pred, ignore_index=255)

                    tr_losses.append(loss.item())
                    tr_obj_ious.append(metrics["object_iou"])
                    tr_mious.append(metrics["miou"])
                    tr_dices.append(metrics["dice"])

                avg_train_loss = float(np.mean(tr_losses)) if tr_losses else 0.0
                avg_train_obj_iou = float(np.mean(tr_obj_ious)) if tr_obj_ious else 0.0
                avg_train_miou = float(np.mean(tr_mious)) if tr_mious else 0.0
                avg_train_dice = float(np.mean(tr_dices)) if tr_dices else 0.0

                self.model.eval()
                te_losses, te_obj_ious, te_mious, te_dices = [], [], [], []

                with torch.no_grad():
                    for vidx, batch in enumerate(tqdm(self.valloader, desc=f"[Valid {epoch}/{self.num_epoch}]")):
                        if len(batch) >= 4:
                            image, gt, _, meta = batch
                        elif len(batch) == 2:
                            image, gt = batch
                            meta = None
                        else:
                            image, gt = batch[0], batch[1]
                            meta = None

                        image = image.to(self.device)
                        gt = gt.to(self.device)

                        out = self.model(pixel_values=image)
                        logits = F.interpolate(
                            out.logits,
                            size=gt.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )

                        loss_v = self.criterion(logits, gt)
                        pred_v = _two_class_logits_to_pred(logits)

                        if meta is not None:
                            pad_x = _meta_scalar(meta, "pad_x", 0)
                            pad_y = _meta_scalar(meta, "pad_y", 0)
                            new_w = _meta_scalar(meta, "new_w", 0)
                            new_h = _meta_scalar(meta, "new_h", 0)

                            y1 = pad_y
                            y2 = pad_y + new_h
                            x1 = pad_x
                            x2 = pad_x + new_w

                            gt_eval = gt[:, y1:y2, x1:x2]
                            pred_eval = pred_v[:, y1:y2, x1:x2]
                        else:
                            gt_eval = gt
                            pred_eval = pred_v

                        metrics_v = _binary_metrics_from_pred(gt_eval, pred_eval, ignore_index=255)

                        te_losses.append(loss_v.item())
                        te_obj_ious.append(metrics_v["object_iou"])
                        te_mious.append(metrics_v["miou"])
                        te_dices.append(metrics_v["dice"])

                        if self.opt.save_vis_every == 0 or epoch % self.opt.save_vis_every == 0:
                            gt_vis = (gt == 1).float().unsqueeze(1)
                            pred_vis = (pred_v == 1).float().unsqueeze(1)
                            self.save_concat_results3(
                                image=image,
                                gt=gt_vis,
                                pred=pred_vis,
                                save_dir=self.images_path,
                                epoch=epoch,
                                idx=vidx,
                                normalize=True,
                            )

                avg_valid_loss = float(np.mean(te_losses)) if te_losses else 0.0
                avg_valid_obj_iou = float(np.mean(te_obj_ious)) if te_obj_ious else 0.0
                avg_valid_miou = float(np.mean(te_mious)) if te_mious else 0.0
                avg_valid_dice = float(np.mean(te_dices)) if te_dices else 0.0

                if best_score["miou"] <= avg_valid_miou:
                    best_score["epoch"] = epoch
                    best_score["miou"] = avg_valid_miou
                    best_score["loss"] = avg_valid_loss
                    self.best_miou = avg_valid_miou
                    self._save_checkpoint(epoch, "best_miou.pth")

                logging.info("-----------------------------------------------------------------------------")
                logging.info(
                    f"Epoch [{epoch}/{self.num_epoch}] "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"ObjIoU: {avg_train_obj_iou:.4f}, "
                    f"mIoU: {avg_train_miou:.4f}, "
                    f"Dice: {avg_train_dice:.4f}"
                )
                logging.info(
                    f"Epoch [{epoch}/{self.num_epoch}] "
                    f"Valid Loss: {avg_valid_loss:.4f}, "
                    f"ObjIoU: {avg_valid_obj_iou:.4f}, "
                    f"mIoU: {avg_valid_miou:.4f}, "
                    f"Dice: {avg_valid_dice:.4f}"
                )
                logging.info(
                    f"Best mIoU epoch: {best_score['epoch']}, "
                    f"Best mIoU: {best_score['miou']:.4f}"
                )
                logging.info("-----------------------------------------------------------------------------")

                summary.add_scalar("train/loss", avg_train_loss, epoch)
                summary.add_scalar("train/obj_iou", avg_train_obj_iou, epoch)
                summary.add_scalar("train/miou", avg_train_miou, epoch)
                summary.add_scalar("train/dice", avg_train_dice, epoch)

                summary.add_scalar("valid/loss", avg_valid_loss, epoch)
                summary.add_scalar("valid/obj_iou", avg_valid_obj_iou, epoch)
                summary.add_scalar("valid/miou", avg_valid_miou, epoch)
                summary.add_scalar("valid/dice", avg_valid_dice, epoch)
                summary.flush()

                self._save_checkpoint(epoch, "latest.pth")

        except KeyboardInterrupt:
            print("Keyboard Interrupt: save model and exit.")
            self._save_checkpoint(epoch, f"epoch_{epoch + 1}.pth")
            raise
        finally:
            summary.close()


def main():
    seed_everything(2024)

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_root", type=str, default="B:/3_exp/uda/data/MAS3K/fold1/train/")
    parser.add_argument("--valid_root", type=str, default="B:/3_exp/uda/data/MAS3K/fold1/valid/")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epoch", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--train_size", type=int, default=512)
    parser.add_argument("--resume_option", action="store_true")
    parser.add_argument(
        "--save_path",
        type=str,
        default="B:/3_exp/uda/my_code/checkpoints_sota/MAS3K_f1/segformerb3_crop_b4",
    )

    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument(
        "--use_adamw_split_lr",
        action="store_true",
        help="Use AdamW and split LR: backbone=lr, head=lr*head_lr_mult",
    )
    parser.add_argument("--head_lr_mult", type=float, default=10.0)

    parser.add_argument(
        "--save_vis_every",
        type=int,
        default=50,
        help="save visualizations every N epochs (0=every epoch)",
    )

    opt = parser.parse_args()

    if os.path.abspath(opt.train_root) == os.path.abspath(opt.valid_root):
        logging.warning(
            "train_root and valid_root are identical. Validation metrics may be over-optimistic due to data leakage."
        )

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b3", num_labels=2, ignore_mismatched_sizes=True
    )

    trainer = Trainer(opt, model)
    trainer.fit()


if __name__ == "__main__":
    main()
