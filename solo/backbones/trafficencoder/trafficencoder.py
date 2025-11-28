import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        # 创建一个位置编码矩阵（max_len, d_model）
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))
        # 注册位置编码为一个持久的缓冲区
        # self.register_buffer('pe', pe)

    def forward(self, x):
        # x的形状为 (batch_size, seq_len, d_model)
        x = x + self.pe[:x.size(0), :] # type: ignore
        return self.dropout(x)

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

class TrafficEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_cfg = kwargs
        self.num_features = self.encoder_cfg['d_model']
        self.trans = nn.Linear(self.encoder_cfg['input_dim'], self.encoder_cfg['d_model'])
        self.pos_emb = PositionalEncoding(self.encoder_cfg['d_model'],self.encoder_cfg['seq_length'])
        
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_cfg['d_model'],
            nhead=self.encoder_cfg['n_head'],
            dim_feedforward= self.encoder_cfg['n_head'] * self.encoder_cfg['d_model'],
            batch_first=True,
            dropout=0.2,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.encoder_cfg['n_layer'])

        self.proj = Conv1DModel(self.encoder_cfg['d_model'], self.encoder_cfg['d_model'], self.encoder_cfg['kernel_size'], 1, 1)
        # x_pos = pos_encoder(x)
        # init model parameters
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

    def forward(self, x, return_intermediates = False):
        # 返回 features
        if return_intermediates:
            features = []
            x = self.trans(x)
            x = self.pos_emb(x)
            features.append(x)
            for i, layer in enumerate(self.encoder.layers):
                x = layer(x)
                features.append(x)
            # x = self.encoder(x)
            hidden_x = self.proj(x)
            # features.append(hidden_x)
            return hidden_x, features  
        else:
            x = self.trans(x)
            x = self.pos_emb(x)
            x = self.encoder(x)
            hidden_x = self.proj(x)
            return hidden_x



class Attention(nn.Module):
    def __init__(self, embed_dim=192, num_heads=3):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  
        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features=192, hidden_features=768):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(0.0)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(0.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Block(nn.Module):
    def __init__(self, embed_dim=192, drop_path_prob=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim)
        self.drop_path1 = nn.Identity() if drop_path_prob == 0 else nn.Dropout(drop_path_prob)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim)
        self.drop_path2 = nn.Identity() if drop_path_prob == 0 else nn.Dropout(drop_path_prob)

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class TrafficTransformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_cfg = kwargs
        embed_dim = self.encoder_cfg['input_dim']
        output_dim = self.encoder_cfg['d_model']
        num_layers=self.encoder_cfg['n_layer']
        self.num_features = output_dim
        # self.patch_embed = PatchEmbed(in_channels=1, embed_dim=embed_dim)
        # self.pos_drop = nn.Dropout(0.0)
        drop_probs = [0.0, 0.033, 0.067, 0.1,0.125]  
        # self.blocks = nn.Sequential(*[Block(embed_dim, drop_probs[i]) for i in range(num_layers)])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_cfg['d_model'],
            nhead=self.encoder_cfg['n_head'],
            dim_feedforward= 256, #1024, #self.encoder_cfg['n_head'] * self.encoder_cfg['d_model'],
            batch_first=True,
            dropout=0.2,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.encoder_cfg['n_layer'])

        
        
        self.fc_norm = nn.LayerNorm(embed_dim)
        self.head_drop = nn.Dropout(0.2)
        self.head = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        # x = self.patch_embed(x)
        # x = self.pos_drop(x)
        x = self.encoder(x)
        x = self.fc_norm(x.mean(dim=1))
        x = self.head_drop(x)
        x = self.head(x)
        return x

class TrafficEncoderSimple(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_cfg = kwargs
        self.num_features = self.encoder_cfg['out_dim']
        self.trans = nn.Linear(self.encoder_cfg['input_dim'], self.encoder_cfg['d_model'])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_cfg['d_model'],
            nhead=self.encoder_cfg['n_head'],
            dim_feedforward= 256, #1024, #self.encoder_cfg['n_head'] * self.encoder_cfg['d_model'],
            batch_first=True,
            dropout=0.2,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.encoder_cfg['n_layer'])

        self.proj = Conv1DModel(self.encoder_cfg['d_model'], self.encoder_cfg['d_model'], self.encoder_cfg['kernel_size'], 1, 1)
        # x_pos = pos_encoder(x)
        # init model parameters
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         nn.init.xavier_normal_(m.weight.data)
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     elif isinstance(m, nn.MultiheadAttention):
        #         nn.init.xavier_uniform_(m.in_proj_weight)
        #         if m.in_proj_bias is not None:
        #             nn.init.constant_(m.in_proj_bias, 0)
        #         nn.init.xavier_uniform_(m.out_proj.weight)
        #         if m.out_proj.bias is not None:
        #             nn.init.constant_(m.out_proj.bias, 0)
        #     elif isinstance(m, nn.Embedding):
        #         nn.init.xavier_uniform_(m.weight)
        self.fc_norm = nn.LayerNorm([self.encoder_cfg['d_model']])
        self.head_drop = nn.Dropout(0.2)
        self.head = nn.Linear(self.encoder_cfg['d_model'], self.num_features)

    def forward(self, x, return_intermediates = False):
        # 返回 features
        if return_intermediates:
            features = []
            x = self.trans(x)
            features.append(x)
            for i, layer in enumerate(self.encoder.layers):
                x = layer(x)
                features.append(x)
            # x = self.encoder(x)
            hidden_x = self.proj(x)
            # features.append(hidden_x)
            return hidden_x, features  
        else:
            x = self.trans(x)
            x = self.encoder(x)
            x = self.proj(x)
            # x = self.fc_norm(x)
            # x = self.head_drop(x)
            # x = self.head(x)
            return x

@register_model
def traffic_encoder(**kwargs):
    encoder = TrafficEncoder(**kwargs)
    return encoder

@register_model
def traffic_encoder_simple(**kwargs):
    encoder = TrafficEncoderSimple(**kwargs)
    return encoder


@register_model
def traffic_transformer(**kwargs):
    encoder = TrafficTransformer(**kwargs)
    return encoder