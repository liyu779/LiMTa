# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import logging
from typing import Any, Callable, Dict, List, Tuple, Union

import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.utils.lars import LARS
from solo.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from solo.utils.metrics import accuracy_at_k, weighted_mean,weighted_add
from solo.utils.misc import (
    omegaconf_select,
    param_groups_layer_decay,
    remove_bias_and_norm_from_weight_decay,
)
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score

class Conv1DModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv1DModel, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)  # 用于在时间维度上压缩

    def forward(self, x):
        # x: (batch_size, timewindow, feature)
        x = x.transpose(1, 2)  #  (batch_size, feature, timewindow)
        x = self.conv1d(x)  
        x = self.relu(x)  
        x = self.pool(x)  
        x = x.squeeze(-1)  
        return x


class LoRALayer(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.encoder_cfg = kwargs
        self.num_features = self.encoder_cfg['d_model']
        self.trans = nn.Linear(self.encoder_cfg['d_model'], self.encoder_cfg['d_model'])
        self.A = nn.Parameter(torch.randn(self.encoder_cfg['d_model'], self.encoder_cfg['r']) * 0.01)  # 高斯分布初始化
        self.B = nn.Parameter(torch.zeros(self.encoder_cfg['r'], self.encoder_cfg['d_model']))  # 初始化为零矩阵
        self.scale = self.encoder_cfg['alpha']/self.encoder_cfg['r']
        self.dropout = nn.Dropout(self.encoder_cfg['drop'])
        self.trans_out = nn.Linear(self.encoder_cfg['d_model'], self.encoder_cfg['d_model'])
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0)
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    nn.init.constant_(m.out_proj.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # x = self.trans(x)
        # W = self.scale * torch.matmul(self.A, self.B)
        # W = self.dropout(W) 
        # x = torch.matmul(x, W.T)
        x = self.trans_out(x)
        return x

class AdapterLayer(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.encoder_cfg = kwargs
        self.lora1 = LoRALayer(kwargs)
        self.lora2 = LoRALayer(kwargs)
        

    def forward(self, x):
        # x = self.trans(x)
        x = self.lora1(x)
        x = self.lora2(x)
        # W = self.scale * torch.matmul(self.A, self.B)
        # W = self.dropout(W) 
        # x = torch.matmul(x, W.T)
        # x = self.trans_out(x)
        return x



# 定义一个包含多个自定义层的模型
class Shell(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        layers = []
        self.encoder_cfg = kwargs
        self.trans = nn.Linear(self.encoder_cfg['input_dim'], self.encoder_cfg['d_model'])
        for i in range(self.encoder_cfg["lora_n_layer"]):
            layers.append(LoRALayer(kwargs))
        self.model = nn.Sequential(*layers)
        self.projector = Conv1DModel(self.encoder_cfg['d_model'], self.encoder_cfg['d_model'], self.encoder_cfg['kernel_size'], 1, 1)


    def forward(self, x,features):
        x = self.trans(x)
        for i, layer in enumerate(self.model):
            x = x + features[i]
            x = layer(x)
        x = x + features[-1]
        # x = self.model(x)
        x = self.projector(x)
        return x



class ShellModel(pl.LightningModule):
    _OPTIMIZERS = {
        "sgd": torch.optim.SGD,
        "lars": LARS,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    _SCHEDULERS = [
        "reduce",
        "warmup_cosine",
        "step",
        "exponential",
        "none",
    ]

    def __init__(
        self,
        patch_embed :nn.Module,
        backbone: nn.Module,
        cfg: omegaconf.DictConfig,
        loss_func: Callable = None,
        mixup_func: Callable = None,
    ):
        """Implements linear and finetune evaluation.

        .. note:: Cfg defaults are set in init by calling `cfg = add_and_assert_specific_cfg(cfg)`

        backbone (nn.Module): backbone architecture for feature extraction.
        Cfg basic structure:
            data:
                num_classes (int): number of classes in the dataset.
            max_epochs (int): total number of epochs.

            optimizer:
                name (str): name of the optimizer.
                batch_size (int): number of samples in the batch.
                lr (float): learning rate.
                weight_decay (float): weight decay for optimizer.
                kwargs (Dict): extra named arguments for the optimizer.
            scheduler:
                name (str): name of the scheduler.
                min_lr (float): minimum learning rate for warmup scheduler. Defaults to 0.0.
                warmup_start_lr (float): initial learning rate for warmup scheduler.
                    Defaults to 0.00003.
                warmup_epochs (float): number of warmup epochs. Defaults to 10.
                lr_decay_steps (Sequence, optional): steps to decay the learning rate
                    if scheduler is step. Defaults to None.
                interval (str): interval to update the lr scheduler. Defaults to 'step'.

            finetune (bool): whether or not to finetune the backbone. Defaults to False.

            performance:
                disable_channel_last (bool). Disables channel last conversion operation which
                speeds up training considerably. Defaults to False.
                https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models

        loss_func (Callable): loss function to use (for mixup, label smoothing or default).
        Defaults to None mixup_func (Callable, optional). function to convert data and targets
        with mixup/cutmix. Defaults to None.
        """

        super().__init__()

        # add default values and assert that config has the basic needed settings
        cfg = self.add_and_assert_specific_cfg(cfg)

        # backbone
        self.patch_embed = patch_embed
        self.backbone = backbone
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embeding.dim))
        if hasattr(self.backbone, "inplanes"):
            features_dim = self.backbone.inplanes
        else:
            features_dim = self.backbone.num_features

        # classifier
        self.classifier = nn.Linear(features_dim, cfg.data.num_classes)  # type: ignore

        # mixup/cutmix function
        self.mixup_func: Callable = mixup_func

        if loss_func is None:
            loss_func = nn.CrossEntropyLoss()
        self.loss_func = loss_func

        # shell **cfg.backbone.kwargs
        self.shellmodel = Shell(cfg.backbone.kwargs)

        # training related
        self.max_epochs: int = cfg.max_epochs
        self.accumulate_grad_batches: Union[int, None] = cfg.accumulate_grad_batches

        # optimizer related
        self.optimizer: str = cfg.optimizer.name
        self.batch_size: int = cfg.optimizer.batch_size
        self.lr: float = cfg.optimizer.lr
        self.weight_decay: float = cfg.optimizer.weight_decay
        self.extra_optimizer_args: Dict[str, Any] = cfg.optimizer.kwargs
        self.exclude_bias_n_norm_wd: bool = cfg.optimizer.exclude_bias_n_norm_wd
        self.layer_decay: float = cfg.optimizer.layer_decay

        # scheduler related
        self.scheduler: str = cfg.scheduler.name
        self.lr_decay_steps: Union[List[int], None] = cfg.scheduler.lr_decay_steps
        self.min_lr: float = cfg.scheduler.min_lr
        self.warmup_start_lr: float = cfg.scheduler.warmup_start_lr
        self.warmup_epochs: int = cfg.scheduler.warmup_epochs
        self.scheduler_interval: str = cfg.scheduler.interval
        assert self.scheduler_interval in ["step", "epoch"]
        if self.scheduler_interval == "step":
            logging.warn(
                f"Using scheduler_interval={self.scheduler_interval} might generate "
                "issues when resuming a checkpoint."
            )

        # if finetuning the backbone
        self.finetune: bool = cfg.finetune

        # for performance
        self.no_channel_last = cfg.performance.disable_channel_last

        if not self.finetune:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # keep track of validation metrics
        self.validation_step_outputs = []
        # test reuse
        # self.feats = torch.randn(256, 500).to('cuda')
        # self.hidden_features =  [torch.randn(256, 401, 500).to('cuda') for _ in range(6)]
        

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        # default parameters for optimizer
        cfg.optimizer.exclude_bias_n_norm_wd = omegaconf_select(
            cfg, "optimizer.exclude_bias_n_norm_wd", False
        )
        # default for extra optimizer kwargs (use pytorch's default if not available)
        cfg.optimizer.kwargs = omegaconf_select(cfg, "optimizer.kwargs", {})
        cfg.optimizer.layer_decay = omegaconf_select(cfg, "optimizer.layer_decay", 0.0)

        # whether or not to finetune the backbone
        cfg.finetune = omegaconf_select(cfg, "finetune", False)

        # default for acc grad batches
        cfg.accumulate_grad_batches = omegaconf_select(cfg, "accumulate_grad_batches", 1)

        # default parameters for the scheduler
        cfg.scheduler.lr_decay_steps = omegaconf_select(cfg, "scheduler.lr_decay_steps", None)
        cfg.scheduler.min_lr = omegaconf_select(cfg, "scheduler.min_lr", 0.0)
        cfg.scheduler.warmup_start_lr = omegaconf_select(cfg, "scheduler.warmup_start_lr", 3e-5)
        cfg.scheduler.warmup_epochs = omegaconf_select(cfg, "scheduler.warmup_epochs", 10)
        cfg.scheduler.interval = omegaconf_select(cfg, "scheduler.interval", "step")

        # default parameters for performance optimization
        cfg.performance = omegaconf_select(cfg, "performance", {})
        cfg.performance.disable_channel_last = omegaconf_select(
            cfg, "performance.disable_channel_last", False
        )

        return cfg

    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        if self.layer_decay > 0:
            assert self.finetune, "Only with use layer weight decay with finetune on."
            msg = (
                "Method should implement no_weight_decay() that returns "
                "a set of parameter names to ignore from weight decay"
            )
            assert hasattr(self.backbone, "no_weight_decay"), msg

            learnable_params = param_groups_layer_decay(
                self.backbone,
                self.weight_decay,
                no_weight_decay_list=self.backbone.no_weight_decay(),
                layer_decay=self.layer_decay,
            )
            learnable_params.append({"name": "classifier", "params": self.classifier.parameters()})
        else:
            # 根据需要调整 优化的参数
            learnable_params = (
                [
                    {"name": "shell", "params": self.shellmodel.parameters()},
                    {"name": "classifier", "params": self.classifier.parameters()},
                ]
                if not self.finetune
                else [
                    {"name": "backbone", "params": self.backbone.parameters()},
                    {"name": "classifier", "params": self.classifier.parameters()},
                ]
            )

        # exclude bias and norm from weight decay
        if self.exclude_bias_n_norm_wd:
            learnable_params = remove_bias_and_norm_from_weight_decay(learnable_params)

        assert self.optimizer in self._OPTIMIZERS
        optimizer = self._OPTIMIZERS[self.optimizer]

        optimizer = optimizer(
            learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )

        # select scheduler
        if self.scheduler == "none":
            return optimizer

        if self.scheduler == "warmup_cosine":
            max_warmup_steps = (
                self.warmup_epochs * (self.trainer.estimated_stepping_batches / self.max_epochs)
                if self.scheduler_interval == "step"
                else self.warmup_epochs
            )
            max_scheduler_steps = (
                self.trainer.estimated_stepping_batches
                if self.scheduler_interval == "step"
                else self.max_epochs
            )
            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=max_warmup_steps,
                    max_epochs=max_scheduler_steps,
                    warmup_start_lr=self.warmup_start_lr if self.warmup_epochs > 0 else self.lr,
                    eta_min=self.min_lr,
                ),
                "interval": self.scheduler_interval,
                "frequency": 1,
            }
        elif self.scheduler == "reduce":
            scheduler = ReduceLROnPlateau(optimizer)
        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps, gamma=0.1)
        elif self.scheduler == "exponential":
            scheduler = ExponentialLR(optimizer, self.weight_decay)
        else:
            raise ValueError(
                f"{self.scheduler} not in (warmup_cosine, cosine, reduce, step, exponential)"
            )

        return [optimizer], [scheduler]

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs forward pass of the frozen backbone and the linear layer for evaluation.

        Args:
            X (torch.tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing features and logits.
        """
        # X = self.patch_embed(X)
        # if not self.no_channel_last:
        #     X = X.to(memory_format=torch.channels_last)

        with torch.set_grad_enabled(self.finetune):
            feats, hidden_features = self.backbone(X, return_intermediates = True)
            # feats = self.feats
            # hidden_features =  self.hidden_features
        if self.finetune:
            logits = self.classifier(feats)
        else:
            add_feats = self.shellmodel(X, hidden_features)
        
            logits = self.classifier(feats + add_feats)
        # print(feats)
        
        
        return {"logits": logits, "feats": feats}

    def shared_step(
        self, batch: Tuple, batch_idx: int
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs operations that are shared between the training nd validation steps.

        Args:
            batch (Tuple): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
                batch size, loss, accuracy @1 and accuracy @5.
        """

        X, target = batch

        metrics = {"batch_size": X.size(0)}
        if self.training and self.mixup_func is not None:
            X, target = self.mixup_func(X, target)
            out = self(X)["logits"]
            loss = self.loss_func(out, target)
            metrics.update({"loss": loss})
        else:
            # import time
            # start_time = time.time()
            out = self(X)["logits"]
            loss = F.cross_entropy(out, target)
            acc1, acc5 = accuracy_at_k(out, target, top_k=(1, 5))
            # end_time = time.time()
            _, pred = out.topk(1, 1, True, True)
            pred = pred.t()
            # print("AAAA",end_time-start_time,X.shape)
            acc = accuracy_score(target.cpu(),pred[0].cpu())
            f1 = f1_score(target.cpu(),pred[0].cpu(),average="micro")
            metrics.update({"loss": loss, "acc1": acc1, "acc5": acc5,"acc":acc,"f1" :f1,"times": 0})

        return metrics

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Performs the training step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            torch.Tensor: cross-entropy loss between the predictions and the ground truth.
        """

        # set backbone to eval mode
        if not self.finetune:
            self.backbone.eval()

        X = batch[0]
        B, C, H, W = X.shape
        x = self.patch_embed(X.reshape(B, C, -1))
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((x, cls_tokens), dim=1)
        batch[0] = x

        out = self.shared_step(batch, batch_idx)

        log = {"train_loss": out["loss"]}
        if self.mixup_func is None:
            log.update({"train_acc1": out["acc1"], "train_acc5": out["acc5"]})

        self.log_dict(log, on_epoch=True, sync_dist=True)
        return out["loss"]

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        """Performs the validation step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Dict[str, Any]:
                dict with the batch_size (used for averaging),
                the classification loss and accuracies.
        """
        # import time
        X = batch[0]
        # X = X.to('cpu')
        B, C, H, W = X.shape
        # patch embeding
        # start  = time.time()
        # X = X.to('cuda')
        # model_mem = torch.cuda.memory_allocated() / (1024**2)
        # print("mem",model_mem)
        # torch.cuda.reset_peak_memory_stats()
        # torch.cuda.reset_max_memory_allocated()
        
        x = self.patch_embed(X.reshape(B, C, -1))
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((x, cls_tokens), dim=1)
        batch[0] = x
        # X, target = batch
        # out = self(X)["logits"]
        # _, pred = out.topk(1, 1, True, True)
        # pred = pred.to('cpu')
        # end = time.time()
        # print("AAAA",B/(end-start),X.shape)
                # max_mem = torch.cuda.max_memory_allocated() / (1024**2)
        # print("max",max_mem)
        # torch.cuda.empty_cache()
        out = self.shared_step(batch, batch_idx)

        metrics = {
            "batch_size": out["batch_size"],
            "val_loss": out["loss"],
            "val_acc1": out["acc1"],
            "val_acc5": out["acc5"],
            "val_acc": out["acc"],
            "val_f1": out["f1"],
            "val_times": out["times"]
        }
        self.validation_step_outputs.append(metrics)
        return metrics

    def on_validation_epoch_end(self):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.
        """

        val_loss = weighted_mean(self.validation_step_outputs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(self.validation_step_outputs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(self.validation_step_outputs, "val_acc5", "batch_size")
        val_acc = weighted_mean(self.validation_step_outputs, "val_acc", "batch_size")
        val_f1 = weighted_mean(self.validation_step_outputs, "val_f1", "batch_size")
        val_times = weighted_add(self.validation_step_outputs, "val_times", "batch_size")

        self.validation_step_outputs.clear()

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5,
               "val_acc": val_acc,"val_f1": val_f1,"val_times":val_times}
        self.log_dict(log, sync_dist=True)
