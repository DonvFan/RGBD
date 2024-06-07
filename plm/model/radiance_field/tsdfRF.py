import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn

from plm.util.misc import get_parameters_from_state_dict

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
    def __init__(self, feat_grid_dim = 48,  pe_view=2, pe_feat=2, dim_mlp_density=256, dim_mlp_color=256, num_semantic_classes=0, tsdf_grid_size = 512, feat_grid_size = 256, grid_weight_path = None, densitynet_weight_path = None, output_mlp_semantics=torch.nn.Softmax(dim=-1), dim_mlp_instance=256, dim_feature_instance=None, use_semantic_mlp=False, use_feature_reg=False, xyz_min = [-1],  xyz_max = [1], invalid_tsdf_val = 1., **kwargs):
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
        self.sdf_feat_grid = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Grid",
                "type": "hash",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hash_map_size": 19,
                "bash_resolution": 16,
                "per_level_scale": 2.,
                "interpolation": "Linear"
            }
        )

        # self.instance_feat_grid = tcnn.Encoding(
        #     n_input_dims=3,
        #     encoding_config={
        #         "otype": "Grid",
        #         "type": "hash",
        #         "n_levels": 16,
        #         "n_features_per_level": 2,
        #         "log2_hash_map_size": 19,
        #         "bash_resolution": 16,
        #         "per_level_scale": 2.,
        #         "interpolation": "Linear"
        #     }
        # )

        self.semantic_feat_grid = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Grid",
                "type": "hash",
                "n_levels": 16,
                "n_features_per_level": 4,
                "log2_hash_map_size": 19,
                "bash_resolution": 16,
                "per_level_scale": 2.,
                "interpolation": "Linear"
            }
        )

        self.color_feat_grid = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Grid",
                "type": "hash",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hash_map_size": 19,
                "bash_resolution": 16,
                "per_level_scale": 2.,
                "interpolation": "Linear"
            }
        )

        self.feat_grid_dim = 16 * 2

        self.tsdf_grid = DenseGrid(
            channels=1, world_size=[tsdf_grid_size]*3, xyz_min=xyz_min, xyz_max=xyz_max)
        self.tsdf_grid.init_from_file(grid_weight_path)
        
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
        self.feat_dim = dim_mlp_density
        self.render_appearance_mlp = MLPRenderFeature(3 + self.feat_dim + self.feat_grid_dim , 3, pe_view, pe_feat, dim_mlp_color)
        self.render_instance_mlp = MLPRenderInstanceFeature(3 + self.feat_dim  + self.feat_grid_dim * 2, dim_feature_instance, num_mlp_layers=4, dim_mlp=dim_mlp_instance, output_activation=torch.nn.Identity())
        self.render_semantic_mlp = MLPRenderSemanticFeature (3 + self.feat_dim + self.feat_grid_dim * 2, num_semantic_classes, output_activation=output_mlp_semantics, num_mlp_layers = 4)  
        self.density_mlp = DensityResNet(3 + self.feat_grid_dim, out_channels = 1 + self.feat_dim, dim_mlp = self.feat_dim)
        self.tsdf_weight_mlp = nn.Sequential(
            *[
                nn.Linear(2, 64), nn.ReLU(),
                nn.Linear(64, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 2), nn.Softmax(dim = -1)
            ]
        )

        # self.tsdf_weight_mlp.apply(DensityResNet.init_zero)
        # if densitynet_weight_path is not None:
        #     print('densitynet_weight_path:', densitynet_weight_path)
        #     self.load_density_weight(densitynet_weight_path)


    def train_density_mlp(self, xyz_sampled):
        assert xyz_sampled.dim() == 2
        feat = self.sdf_feat_grid(xyz_sampled)
        res = self.density_mlp(xyz_sampled, feat)
        return res
    
    
    def load_density_weight(self, path):
        weight = torch.load(path, map_location=next(self.density_mlp.parameters()).device)
        self.density_mlp.load_state_dict(weight['mlp_params'])
        self.sdf_feat_grid.load_state_dict(weight['grid_params'])


    def compute_tsdf(self, xyz_sampled):
        with torch.no_grad():
            tsdf_val = self.tsdf_grid(xyz_sampled).detach()
        return tsdf_val

    # def compute_sdf(self, xyz_sampled):
    #     with torch.no_grad():
    #         tsdf_val = self.tsdf_grid(xyz_sampled).detach()
    #     # import debugpy
    #     # debugpy.listen(17456)
    #     # debugpy.wait_for_client()
    #     B, N, _ = xyz_sampled.shape
    #     feat = self.sdf_feat_grid(xyz_sampled.flatten(end_dim = -2))
    #     feat = feat.reshape((B, N, -1))
    #     res = self.density_mlp(xyz_sampled, feat).clamp(min = -1, max=1.)
    #     mlp_sdf, mlp_feat = res[..., 0], res[..., 1:]
    #     ind = torch.logical_and(tsdf_val < 1.0, tsdf_val > -1.0)
    #     weight = self.tsdf_weight_mlp(torch.stack([tsdf_val[ind], mlp_sdf[ind]], dim = -1))
    #     mlp_sdf[ind] = weight[..., 0] * tsdf_val[ind] + weight[..., 1] * mlp_sdf[ind]

    #     return mlp_sdf, mlp_feat, tsdf_val
    
    # def compute_occ(self, xyz_sampled):
    #     mlp_sdf, mlp_feat, _ = self.compute_sdf(xyz_sampled)
    #     mlp_sdf = 1. - (mlp_sdf + 1.) / 2. 
    #     # tsdf_val = torch.clamp(mlp_sdf, 0.0, 1.0)
    #     inv_tsdf = -0.1 * torch.log((1 / (mlp_sdf + 1e-8)) - 1 + 1e-7) #0.1
    #     # inv_tsdf = torch.clamp(inv_tsdf, -100.0, 100.0)
    #     return inv_tsdf, mlp_feat

    def compute_occ(self, xyz_sampled):
        with torch.no_grad():
            tsdf_val = self.tsdf_grid(xyz_sampled)
            mid_val = 1. - (tsdf_val + 1.) / 2. #0,1
            mid_val = torch.clamp(mid_val, 0.0, 1.0)
            inv_tsdf = -0.1 * torch.log((1 / (mid_val + 1e-8)) - 1 + 1e-7)
            # inv_tsdf = torch.clamp(inv_tsdf, -100.0, 100.0)

        inv_tsdf = inv_tsdf.detach()
        B, N, _ = xyz_sampled.shape
        feat = self.sdf_feat_grid(xyz_sampled.flatten(end_dim = -2))
        feat = feat.reshape((B, N, -1))
        res = self.density_mlp(xyz_sampled, feat).clamp(min = -1, max=1.)
        mlp_occ, mlp_feat = res[..., 0], res[..., 1:]
        ind = torch.logical_and(tsdf_val < 1.0, tsdf_val > -1.0)
        weight = self.tsdf_weight_mlp(torch.stack([inv_tsdf[ind], mlp_occ[ind]], dim = -1))
        mlp_occ[ind] = weight[..., 0] * inv_tsdf[ind] + weight[..., 1] * mlp_occ[ind]

        return mlp_occ, mlp_feat

    
    def compute_appearance_feature(self, xyz_sampled, viewdir, feat):
        B, N, _ = xyz_sampled.shape
        grid_feat = self.color_feat_grid(xyz_sampled.flatten(end_dim = -2))
        grid_feat = grid_feat.reshape((B, N, -1))
        return self.render_appearance_mlp(xyz_sampled, viewdir, feat, grid_feat)

    def compute_semantic_feature(self, xyz_sampled, feat):
        B, N, _ = xyz_sampled.shape
        grid_feat = self.semantic_feat_grid(xyz_sampled.flatten(end_dim = -2))
        grid_feat = grid_feat.reshape((B, N, -1))
        return self.render_semantic_mlp(xyz_sampled, feat, grid_feat)

    def compute_instance_feature(self, xyz_sampled, feature):
        B, N, _ = xyz_sampled.shape
        # grid_feat = self.instance_feat_grid(xyz_sampled.flatten(end_dim = -2))
        grid_feat = self.semantic_feat_grid(xyz_sampled.flatten(end_dim=-2)).detach()
        grid_feat = grid_feat.reshape((B, N, -1))

        return self.render_instance_mlp(xyz_sampled, feature, grid_feat)

    def get_optimizable_parameters(self, lr_grid, lr_net, weight_decay=0):
        grad_vars = [{'params': self.color_feat_grid.parameters(), 'lr': lr_grid}, {'params': self.render_appearance_mlp.parameters(), 'lr': lr_net},
                    {'params': self.render_semantic_mlp.parameters(), 'lr': lr_net},
                     {'params': self.render_instance_mlp.parameters(), 'lr': lr_net}, 
                     {'params': self.density_mlp.parameters(), 'lr': lr_net  },
                     {
                         'params':self.tsdf_weight_mlp.parameters(), 
                         'lr':lr_net 
                     },
                    #  {'params': self.instance_feat_grid.parameters(), 'lr': lr_grid},
                     {'params': self.semantic_feat_grid.parameters(), 'lr': lr_grid},
                     {'params': self.sdf_feat_grid.parameters(), 'lr': lr_grid
                     }
        ]
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

    def forward(self, xyz, viewdirs, features, grid_feat):
        indata = [xyz, features, grid_feat]
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


class DensityResNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels = 1, num_mlp_layers = 3, dim_mlp = 256, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.out_channels = out_channels
        layers = [torch.nn.Linear(in_channels, dim_mlp)]
        for i in range(num_mlp_layers - 2):
            layers.extend([torch.nn.ReLU(), 
                           torch.nn.Linear(dim_mlp, dim_mlp)])
        layers.append(torch.nn.Linear(dim_mlp, out_channels))
        self.mlp = torch.nn.Sequential(*layers)
        # self.mlp.apply(DensityResNet.init_zero)

    def forward(self,  xyz, feat):
        input = torch.cat([xyz, feat], dim = -1)
        out = self.mlp(input)
        return out
    
    @staticmethod
    def init_zero(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.constant_(m.weight, 0.)
            torch.nn.init.constant_(m.bias, 0.)
    

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

    def forward(self, xyz, feat_xyz, grid_feat):
        indata = torch.cat([ xyz, feat_xyz, grid_feat], dim = -1)
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

    def forward(self, xyz, feat_xyz, grid_feat):
        indata = [xyz, feat_xyz, grid_feat]
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
