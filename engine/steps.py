# -*- coding = utf-8 -*-
# @File : steps.py
"""StepRunner/EpochRunner únicos (multilabel), parametrizados por um
forward_fn (ver engine/forward_adapters.py). Substitui as duas cópias que
existiam em train.py e train_finetuning.py.
"""
import sys

import torch
import torchmetrics
from tqdm import tqdm


class StepRunner:
    def __init__(self, model, loss_fn, forward_fn, device, stage="train", optimizer=None):
        self.net = model
        self.loss_fn = loss_fn
        self.forward_fn = forward_fn
        self.device = device
        self.stage = stage
        self.optimizer = optimizer

    def step(self, batch):
        logits, targets = self.forward_fn(self.net, batch, self.device)
        # BCEWithLogitsLoss espera targets float multi-hot
        loss = self.loss_fn(logits, targets.float())

        if self.optimizer is not None and self.stage == "train":
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item(), logits, targets

    def __call__(self, batch):
        self.net.train() if self.stage == "train" else self.net.eval()
        return self.step(batch)


class EpochRunner:
    def __init__(self, steprunner, num_labels, threshold=0.5):
        self.steprunner = steprunner
        self.stage = steprunner.stage
        self.device = steprunner.device

        metrics_cfg = {
            "task": "multilabel",
            "num_labels": num_labels,
            "threshold": threshold,
            "average": "macro",
        }
        self.acc_metric = torchmetrics.Accuracy(**metrics_cfg).to(self.device)
        self.prec_metric = torchmetrics.Precision(**metrics_cfg).to(self.device)
        self.rec_metric = torchmetrics.Recall(**metrics_cfg).to(self.device)
        self.f1_metric = torchmetrics.F1Score(**metrics_cfg).to(self.device)
        self.auroc_metric = torchmetrics.AUROC(
            task="multilabel", num_labels=num_labels, average="macro"
        ).to(self.device)

    def __call__(self, dataloader):
        total_loss, step = 0, 0

        self.acc_metric.reset()
        self.prec_metric.reset()
        self.rec_metric.reset()
        self.f1_metric.reset()
        self.auroc_metric.reset()

        loop = tqdm(dataloader, file=sys.stdout)
        for batch in loop:
            loss, logits, targets = self.steprunner(batch)

            probs = torch.sigmoid(logits.detach())
            targets_int = targets.to(self.device).long()

            self.acc_metric.update(probs, targets_int)
            self.prec_metric.update(probs, targets_int)
            self.rec_metric.update(probs, targets_int)
            self.f1_metric.update(probs, targets_int)
            self.auroc_metric.update(probs, targets_int)

            step_log = {f"{self.stage}_loss": loss}
            total_loss += loss
            step += 1
            loop.set_postfix(**step_log)

        return {
            f"{self.stage}_loss": total_loss / step,
            f"{self.stage}_acc": self.acc_metric.compute().item(),
            f"{self.stage}_prec": self.prec_metric.compute().item(),
            f"{self.stage}_rec": self.rec_metric.compute().item(),
            f"{self.stage}_f1": self.f1_metric.compute().item(),
            f"{self.stage}_auroc": self.auroc_metric.compute().item(),
        }
