import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn

from pl.util.misc import get_parameters_from_state_dict

class DenseGrid(nn.Module):
    def __init__(self, channels, world_size, xyz_min, xyz_max, **kwargs):
        super(DenseGrid, self).__init__()
        self.channels = channels
        self.world_size = world_size
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.grid = nn.Parameter(torch.zeros([1, channels, *world_size]))

    def forward(self, xyz):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        out = F.grid_sample(self.grid, ind_norm, mode='bilinear', align_corners=True)
        out = out.reshape(self.channels,-1).T.reshape(*shape,self.channels)
        if self.channels == 1:
            out = out.squeeze(-1)
        return out

    def scale_volume_grid(self, new_world_size):
        if self.channels == 0:
            self.grid = nn.Parameter(torch.zeros([1, self.channels, *new_world_size]))
        else:
            self.grid = nn.Parameter(
                F.interpolate(self.grid.data, size=tuple(new_world_size), mode='trilinear', align_corners=True))
    
    def init_from_numpy(self, init_parm):
        assert (self.channels, *self.world_size) == init_parm.shape, f"no matching parameter {(self.channels, *self.world_size)} and {init_parm.shape}"
        
        self.grid = nn.Parameter(torch.as_tensor(init_parm[np.newaxis, ...]))
        
    def init_from_file(self, init_path):

        grid = torch.load(init_path)
        if grid.dim() == 3:
            grid = torch.unsqueeze(grid, dim = 0)
        else:
            grid = torch.permute(grid, dims = [3, 0, 1, 2])

        self.grid = nn.Parameter(grid[None])

    def get_dense_grid(self):
        return self.grid

    @torch.no_grad()
    def __isub__(self, val):
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self.world_size.tolist()}'


class TSDFVol(nn.Module):
    def __init__(self, feat_grid_dim = 48,  pe_view=2, pe_feat=2, dim_mlp_density=256, dim_mlp_color=256, num_semantic_classes=0, tsdf_grid_size = 512, feat_grid_size = 256, densitynet_weight_path = None, output_mlp_semantics=torch.nn.Softmax(dim=-1), dim_mlp_instance=256, dim_feature_instance=None, use_semantic_mlp=False, use_feature_reg=False, xyz_min = [-1],  xyz_max = [1], invalid_tsdf_val = 1., **kwargs):
        super().__init__()
        
        self.dim_feature_instance = dim_feature_instance
        self.num_semantic_classes = num_semantic_classes
        self.use_feature_reg = use_feature_reg and use_semantic_mlp
        self.pe_view, self.pe_feat = pe_view, pe_feat
        self.dim_mlp_color = dim_mlp_color
        self.feat_grid_dim = feat_grid_dim
        self.invalid_tsdf_val = invalid_tsdf_val

        # self.feat_grid = DenseGrid(
        #     channels=self.feat_grid_dim, world_size=[feat_grid_size]*3, xyz_min=xyz_min, xyz_max=xyz_max)
        self.feat_grid = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Grid",
                "type": "hash",
                "n_levels": 16,
                "n_features_per_level": 8,
                "log2_hash_map_size": 19,
                "bash_resolution": 16,
                "per_level_scale": 2.,
                "interpolation": "Linear"
            }
        )
        self.feat_grid_dim = 16 * 8

        self.tsdf_grid = DenseGrid(
            channels=1, world_size=[tsdf_grid_size]*3, xyz_min=xyz_min, xyz_max=xyz_max)
        self.tsdf_grid.init_from_file(densitynet_weight_path)
        
        # self.densitynet = nn.Sequential(
        #     nn.Linear(3, dim_mlp_density), nn.ReLU(inplace=True),
        #     *[
        #         nn.Sequential(nn.Linear(dim_mlp_density, dim_mlp_density), nn.ReLU(inplace=True))
        #         for _ in range(2)
        #     ],
        #     nn.Linear(dim_mlp_density, 1),
        # )
        # if densitynet_weight_path is not None:
        #     self.densitynet.load_state_dict(torch.load(densitynet_weight_path))

        self.render_appearance_mlp = MLPRenderFeature(3 + self.feat_grid_dim, 3, pe_view, pe_feat, dim_mlp_color)
        self.render_instance_mlp = MLPRenderInstanceFeature(3 + self.feat_grid_dim, dim_feature_instance, num_mlp_layers=4, dim_mlp=dim_mlp_instance, output_activation=torch.nn.Identity())
        self.render_semantic_mlp = MLPRenderSemanticFeature (3 + self.feat_grid_dim, num_semantic_classes, output_activation=output_mlp_semantics, num_mlp_layers = 4)  

    @torch.no_grad()
    def compute_sdf(self, xyz_sampled):
        tsdf_val = self.tsdf_grid(xyz_sampled)
        return tsdf_val.detach()
        
    def compute_appearance_feature(self, xyz_sampled):
        return self.feat_grid(xyz_sampled)

    def compute_semantic_feature(self, xyz_sampled):
        return self.feat_grid(xyz_sampled)

    def compute_instance_feature(self, xyz_sampled, feature):
        return self.render_instance_mlp(xyz_sampled, feature)

    def get_optimizable_parameters(self, lr_grid, lr_net, weight_decay=0):
        grad_vars = [{'params': self.feat_grid.parameters(), 'lr': lr_grid}, {'params': self.render_appearance_mlp.parameters(), 'lr': lr_net},
                    {'params': self.render_semantic_mlp.parameters(), 'lr': lr_net},
                     {'params': self.render_instance_mlp.parameters(), 'lr': lr_net}]
        return grad_vars

    def get_optimizable_segment_parameters(self, lr_grid, lr_net, _weight_decay=0):

        grad_vars = [{'params': self.render_semantic_mlp.parameters(), 'lr': lr_net}]
        return grad_vars

    def get_optimizable_instance_parameters(self, lr_grid, lr_net):
        return [
            {'params': self.render_instance_mlp.parameters(), 'lr': lr_net, 'weight_decay': 1e-3}
        ]


class MLPRenderFeature(torch.nn.Module):

    def __init__(self, in_channels, out_channels=3, pe_view=2, pe_feat=2, dim_mlp_color=128, output_activation=torch.sigmoid):
        super().__init__()
        self.pe_view = pe_view
        self.pe_feat = pe_feat
        self.output_channels = out_channels
        self.view_independent = self.pe_view == 0 and self.pe_feat == 0
        self.in_feat_mlp = 2 * pe_view * 3 + 2 * pe_feat * 3 + in_channels + (3 if not self.view_independent else 0)
        self.output_activation = output_activation
        layer1 = torch.nn.Linear(self.in_feat_mlp, dim_mlp_color)
        layer2 = torch.nn.Linear(dim_mlp_color, dim_mlp_color)
        layer3 = torch.nn.Linear(dim_mlp_color, out_channels)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, xyz, viewdirs, features):
        indata = [xyz, features]
        if not self.view_independent:
            indata.append(viewdirs)
        if self.pe_feat > 0:
            indata += [MLPRenderFeature.positional_encoding(xyz, self.pe_feat)]
        if self.pe_view > 0:
            indata += [MLPRenderFeature.positional_encoding(viewdirs, self.pe_view)]
        mlp_in = torch.cat(indata, dim=-1)
        out = self.mlp(mlp_in)
        out = self.output_activation(out)
        return out

    @staticmethod
    def positional_encoding(positions, freqs):
        freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)
        pts = (positions[..., None] * freq_bands).reshape(positions.shape[:-1] + (freqs * positions.shape[-1],))
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts


class MLPRenderInstanceFeature(torch.nn.Module):

    def __init__(self, in_channels, out_channels, num_mlp_layers=5, dim_mlp=256, output_activation=torch.nn.Softmax(dim=-1)):
        super().__init__()
        self.output_channels = out_channels
        self.output_activation = output_activation
        layers = [torch.nn.Linear(in_channels, dim_mlp)]
        for i in range(num_mlp_layers - 2):
            layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.Linear(dim_mlp, dim_mlp))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(dim_mlp, out_channels))
        self.mlp = torch.nn.Sequential(*layers)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, xyz, feat_xyz):
        indata = torch.cat([xyz, feat_xyz], dim = -1)
        out = self.mlp(indata)
        out = self.output_activation(out)
        return out


class MLPRenderSemanticFeature(torch.nn.Module):

    def __init__(self, in_channels, out_channels, pe_feat=0, num_mlp_layers=5, dim_mlp=256, output_activation=torch.nn.Identity()):
        super().__init__()
        self.output_channels = out_channels
        self.output_activation = output_activation
        self.pe_feat = pe_feat
        self.in_feat_mlp = in_channels + 2 * pe_feat * 3
        layers = [torch.nn.Linear(self.in_feat_mlp, dim_mlp)]
        for i in range(num_mlp_layers - 2):
            layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.Linear(dim_mlp, dim_mlp))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(dim_mlp, out_channels))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, xyz, feat_xyz):
        indata = [xyz, feat_xyz]
        if self.pe_feat > 0:
            indata += [MLPRenderFeature.positional_encoding(xyz, self.pe_feat)]
        mlp_in = torch.cat(indata, dim=-1)
        out = self.mlp(mlp_in)
        out = self.output_activation(out)
        return out



def render_features_direct(_viewdirs, appearance_features):
    return appearance_features


def render_features_direct_with_softmax(_viewdirs, appearance_features):
    return torch.nn.Softmax(dim=-1)(appearance_features)
