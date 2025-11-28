from typing import Any, Dict, List, Sequence, Tuple
import random
import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.byol import byol_loss_func
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params
class StrideEmbed(nn.Module):
    def __init__(self, img_height=40, img_width=40, stride_size=4, in_chans=1, embed_dim=192):
        super().__init__()
        self.num_patches = img_height * img_width // stride_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=stride_size, stride=stride_size)

    def forward(self, x):
        x = self.proj(x).transpose(1, 2)
        return x


class TrafficFFTRC(BaseMomentumMethod):
    def __init__(self, cfg: omegaconf.DictConfig):

        super().__init__(cfg)

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        pred_hidden_dim: int = cfg.method_kwargs.pred_hidden_dim

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        # momentum projector
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        initialize_momentum_params(self.projector, self.momentum_projector)

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, proj_output_dim),
        )
        # patch_embed
        embed_dim = cfg.embeding.dim
        self.patch_embed = StrideEmbed(embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        # num_cls_token = 1
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(TrafficFFTRC, TrafficFFTRC).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.pred_hidden_dim")

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [
            {"name": "projector", "params": self.projector.parameters()},
            {"name": "predictor", "params": self.predictor.parameters()},
        ]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """
        out = super().forward(X)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        out.update({"z": z, "p": p})
        return out

    def multicrop_forward(self, X) -> Dict[str, Any]:
        """Performs the forward pass for the multicrop views.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[]: a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().multicrop_forward(X)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        out.update({"z": z, "p": p})
        return out
    
    

    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict:
        """Performs the forward pass of the momentum backbone and projector.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of
                the parent and the momentum projected features.
        """

        out = super().momentum_forward(X)
        z = self.momentum_projector(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for BYOL reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of BYOL and classification loss.
        """
        
        # B, C, H, W = X.shape
        # # patch embeding
        # x = self.patch_embed(X.reshape(B, C, -1))
        # # positation embedding
        # x = x + self.pos_embed[:, :-1, :]
        # append cls token
        # cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((x, cls_tokens), dim=1)
        
        batch = self.embedding_sample(batch)
        x = batch[0]
        # build frequency features-missing samples
        fft_x = self.time_to_frequency(x)
        # batch[0] = torch.cat(([x],[fft_x]), dim = 0)
        batch[0] = [x,fft_x] # type: ignore

        out = super().training_step(batch, batch_idx) # type: ignore
        class_loss = out["loss"]
        Z = out["z"]
        P = out["p"]
        Z_momentum = out["momentum_z"]

        # ------- negative consine similarity loss -------
        neg_cos_sim = 0
        for v1 in range(self.num_large_crops):
            for v2 in np.delete(range(self.num_crops), v1):
                neg_cos_sim += byol_loss_func(P[v2], Z_momentum[v1])

        # calculate std of features
        with torch.no_grad():
            z_std = F.normalize(torch.stack(Z[: self.num_large_crops]), dim=-1).std(dim=1).mean()

        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return neg_cos_sim + class_loss
    def validation_step(
        self,
        batch: List[torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = None, # type: ignore
        update_validation_step_outputs: bool = True,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Validation step for pytorch lightning. It performs all the shared operations for the
        momentum backbone and classifier, such as forwarding a batch of images in the momentum
        backbone and classifier and computing statistics.

        Args:
            batch (List[torch.Tensor]): a batch of data in the format of [X, Y].
            batch_idx (int): index of the batch.
            update_validation_step_outputs (bool): whether or not to append the
                metrics to validation_step_outputs

        Returns:
            Tuple(Dict[str, Any], Dict[str, Any]): tuple of dicts containing the batch_size (used
                for averaging), the classification loss and accuracies for both the online and the
                momentum classifiers.
        """
        batch = self.embedding_sample(batch)
        return super().validation_step(batch, batch_idx)
    
    def embedding_sample(self, batch):
        X = batch[0]
        B, C, H, W = X.shape
        # patch embeding
        x = self.patch_embed(X.reshape(B, C, -1))
        x = x + self.pos_embed[:, :, :]  # position embedding 需要调整
        batch[0] = x
        return batch

    def time_to_frequency(self, data):
        """
        Convert time series data to frequency domain using PyTorch.
        Args:
        data (torch.Tensor): The time series data, expected shape (batch_size, sequence_length)
        Returns:
        torch.Tensor: The frequency domain representation of the data.
        """
        B,P,F = data.shape
        data_fft = torch.zeros_like(data)
        for b in range(B):
            k = int(P* 0.75) 
            m = int(F* 0.75)
            mask_k = [random.randint(0,P-1) for i in range(k)]
            mask_m = [random.randint(0,F-1) for i in range(m)]
            # time domin mask
            # data_b = data[b]
            # data_b[mask_k,:] = 0
            # data_b[:,mask_m] = 0
            # data_fft[b,:,:] = data_b
            # frequnecy mask
            frequency_data = torch.fft.fft(data[b])
            frequency_data[mask_k,:] = 0
            frequency_data[:,mask_m] = 0
            data_fft[b,:,:] = torch.fft.ifft(frequency_data)
        return data_fft