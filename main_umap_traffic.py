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

import json
import os
from pathlib import Path
import torch
from omegaconf import DictConfig, OmegaConf
from solo.args.linear import parse_cfg
from solo.args.umap import parse_args_umap
from solo.data.classification_traffic_dataloader import prepare_data
from solo.methods import METHODS
from solo.utils.auto_umap import OfflineUMAP
from solo.methods.traffic_fft_rc import StrideEmbed
from solo.methods.base import BaseMethod
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import torch.nn as nn
import hydra
class LinearModelMap(nn.Module):
    def __init__(self, submodel1, submodel2):
        super(LinearModelMap, self).__init__()
        self.submodel1 = submodel1
        self.submodel2 = submodel2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 500))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.submodel1(x.reshape(B, C, -1))
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((x, cls_tokens), dim=1)
        x = self.submodel2(x)
        return x
@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)
    backbone_model = BaseMethod._BACKBONES[cfg.backbone.name]
    traffic_embdding = StrideEmbed(embed_dim=cfg.embeding.dim)
    # initialize backbone
    backbone = backbone_model(method=cfg.pretrain_method, **cfg.backbone.kwargs)
    
    ckpt_path = cfg.pretrained_feature_extractor
    assert ckpt_path.endswith(".ckpt") or ckpt_path.endswith(".pth") or ckpt_path.endswith(".pt")
    patch_embed_state = {}
    backbone_state = {}
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    for k in list(state.keys()):
        # if "encoder" in k:
        #     state[k.replace("encoder", "backbone")] = state[k]
        #     logging.warn(
        #         "You are using an older checkpoint. Use a new one as some issues might arrise."
        #     )
        if "backbone." in k:
            backbone_state[k.replace("backbone.", "")] = state[k]
        if "patch_embed" in k:
            patch_embed_state[k.replace("patch_embed.", "")] = state[k]
        del state[k]
    backbone.load_state_dict(backbone_state, strict=False)
    traffic_embdding.load_state_dict(patch_embed_state, strict=False)
    # logging.info(f"Loaded {ckpt_path}")
    print(f"Loaded {ckpt_path}")

    model = LinearModelMap(traffic_embdding,backbone)
    print("Loaded model")
    # args = parse_args_umap()

    # build paths
    # ckpt_dir = Path(args.pretrained_checkpoint_dir)
    # args_path = ckpt_dir / "args.json"
    # ckpt_path = [ckpt_dir / ckpt for ckpt in os.listdir(ckpt_dir) if ckpt.endswith(".ckpt")][0]

    # prepare data
    train_loader, val_loader = prepare_data(
        cfg.data.dataset,
        train_data_path=cfg.data.train_path,
        val_data_path=cfg.data.val_path,
        data_format=cfg.data.format,
        batch_size=cfg.optimizer.batch_size,
        num_workers=cfg.data.num_workers,
        auto_augment=cfg.auto_augment,
    )
    print(train_loader)
    umap = OfflineUMAP()

    # move model to the gpu
    device = "cuda:0"
    model = model.to(device)

    umap.plot(device, model, train_loader, "i100_train_umap.pdf")
    umap.plot(device, model, val_loader, "i100_val_umap.pdf")


if __name__ == "__main__":
    main()
