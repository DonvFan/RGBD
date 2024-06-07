# MIT License
#
# Copyright (c) 2022 Anpei Chen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import random
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch_efficient_distloss import eff_distloss

from plm.util.distinct_colors import DistinctColors
from plm.util.misc import visualize_points
from plm.util.transforms import tr_comp, dot, trs_comp


class TsdfRFRenderer(nn.Module):

    def __init__(self, bbox_aabb, grid_dim, stop_semantic_grad=True, semantic_weight_mode="none", step_ratio=0.5, distance_scale=25, raymarch_weight_thres=0.0001, alpha_mask_threshold=0.0075, parent_renderer_ref=None, instance_id=0, n_samples = 128, surface_n_samples = 16):
        super().__init__()
        self.register_buffer("bbox_aabb", bbox_aabb)
        self.register_buffer("grid_dim", torch.LongTensor(grid_dim))
        self.register_buffer("inv_box_extent", torch.zeros([3]))
        self.register_buffer("units", torch.zeros([3]))
        self.semantic_weight_mode = semantic_weight_mode
        self.parent_renderer_ref = parent_renderer_ref
        self.step_ratio = step_ratio
        self.distance_scale = distance_scale
        self.raymarch_weight_thres = raymarch_weight_thres
        self.alpha_mask_threshold = alpha_mask_threshold
        self.step_size = None
        self.n_samples = n_samples
        self.surface_n_samples = surface_n_samples
        self.stop_semantic_grad = stop_semantic_grad
        self.instance_id = instance_id
        self.truncationd_dis = torch.max(bbox_aabb[1]-bbox_aabb[0]).abs() / self.grid_dim.max() * 4
        # self.register_buffer("surface_sample_vals", torch.linspace(0., 1., steps = surface_n_samples + 1) * self.truncationd_dis)
        self.register_buffer("surface_sample_space", torch.arange(0, surface_n_samples + 1))
        # self.update_step_size(self.grid_dim)
    

    @staticmethod
    def raw2outputs_nerf_color(raw, z_vals, occupancy=True, device='cuda:0'):
        """
        Transforms model's predictions to semantically meaningful values.

        Args:
            raw (tensor, N_rays*N_samples): prediction from model.
            z_vals (tensor, N_rays*N_samples): integration time.
            rays_d (tensor, N_rays*3): direction of each ray.
            occupancy (bool, optional): occupancy or volume density. Defaults to False.
            device (str, optional): device. Defaults to 'cuda:0'.

        Returns:
            depth_map (tensor, N_rays): estimated distance to object.
            depth_var (tensor, N_rays): depth variance/uncertainty.
            rgb_map (tensor, N_rays*3): estimated RGB color of a ray.
            weights (tensor, N_rays*N_samples): weights assigned to each sampled color.
        """

        def raw2alpha(raw, dists, act_fn=F.relu): return 1. - \
            torch.exp(-act_fn(raw)*dists)
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = dists.float()
        dists = torch.cat([dists, torch.Tensor([1e10]).float().to(
            device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

        # different ray angle corresponds to different unit length
        # dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
        if occupancy:
    
            alpha = torch.sigmoid(10 * raw) #sigmoid tanh -1,1 # when occ do belong to 0 - 1
            alpha_theta = 0
        else:
            # original nerf, volume density
            alpha = raw2alpha(raw, dists)  # (N_rays, N_samples)

        weights = alpha.float() * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(
            device).float(), (1.-alpha + 1e-10).float()], -1).float(), -1)[:, :-1]    
        
        # return depth_map, depth_var, rgb_map, weights
        return alpha, weights
        

    @staticmethod
    def sdf2weights(sdf, z_vals, truncation = 2 / 512. * 4, sc_factor = 1.0):
            alpha = torch.sigmoid(sdf / truncation) * torch.sigmoid(-sdf / truncation)

            signs = sdf[:, 1:] * sdf[:, :-1]
            mask = torch.where(signs < 0.0, torch.ones_like(signs), torch.zeros_like(signs))
            inds = torch.argmax(mask, axis=1)
            inds = inds[..., None]
            z_min = torch.gather(z_vals, 1, inds) # The first surface
            mask = torch.where(z_vals < z_min + sc_factor * truncation, torch.ones_like(z_vals), torch.zeros_like(z_vals))
            
            alpha = alpha * mask
            weights = alpha / (torch.sum(alpha, axis=-1, keepdims=True) + 1e-8)
            # weights = torch.exp(alpha) ** 3
            # weights = weights / torch.sum(weights, dim = -1, keepdim=True)
            return alpha, weights, weights[:, -1:]



    def gen_samples(self, tensorf, rays, perturb = True, n_samples = 64, n_importance = None, n_guassian = None):
        if n_importance is None:
            n_importance = n_samples
        if n_guassian is None:
            n_guassian = n_samples

        rays_o, rays_d, nears, fars = rays[:, 0:3], rays[:, 3:6], rays[:, 6], rays[:, 7]
        xyz_sampled_u, z_vals_u = sample_points_uniform_in_box(rays_o, rays_d, nears, fars, self.bbox_aabb, n_samples * 2, perturb)
        occ, _= tensorf.compute_occ(xyz_sampled_u)
        _, weights = TsdfRFRenderer.raw2outputs_nerf_color(occ, z_vals_u)
        z_vals_mid = .5 * (z_vals_u[..., 1:] + z_vals_u[..., :-1])
        z_vals_h = sample_pdf(
            z_vals_mid, weights[..., 1:-1], n_importance, det=~perturb)
        z_vals_h = z_vals_h

        z_vals_g = torch.zeros([len(rays), n_guassian], dtype = rays_o.dtype, device = rays_o.device)
        ind = fars > 1e-5
        
        _, z_vals = sample_points_gaussian(rays_o[ind], rays_d[ind], nears[ind], fars[ind], n_samples=n_guassian, scale = 1.1, std_scale=6)
        z_vals_g[ind] = z_vals
        z_vals_g[~ind] = z_vals_u[~ind][..., ::n_samples * 2 // n_guassian]
        z_vals = torch.cat([z_vals_g, z_vals_h], dim = -1)
        z_vals, _ =  torch.sort(z_vals, dim = -1)
        xyz_sampled = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]
        return z_vals.detach(), xyz_sampled.detach()

    @torch.no_grad()
    def test_rendering(self, tensorf, rays):
        rays_o, rays_d, nears, fars = rays[:, 0:3], rays[:, 3:6], rays[:, 6], rays[:, 7]
        z_vals = torch.linspace(0, 1, 20, dtype=rays.dtype, device=rays.device)
        z_vals  = (fars[..., None] - nears[..., None]) * z_vals[None]
        temp_z_vals = nears[..., None] + z_vals
        xyz_sampled = rays_o[..., None, :] + rays_d[..., None, :] * temp_z_vals[..., None]

        occ, feat = tensorf.compute_occ(xyz_sampled)
        _, weights = TsdfRFRenderer.raw2outputs_nerf_color(occ, z_vals)
        viewdirs = rays[:, 3:6].view(-1, 1, 3).expand(xyz_sampled.shape)

        rgbs = tensorf.compute_appearance_feature(xyz_sampled, viewdirs, feat)
        semantics = tensorf.compute_semantic_feature(xyz_sampled, feat.detach())
        instances = tensorf.compute_instance_feature(xyz_sampled, feat.detach())

        rgb_map = torch.sum(weights[..., None] * rgbs, -2)  # (N_rays, 3)
        depth_map = torch.sum(weights * z_vals, -1)  # (N_rays)
        tmp = (z_vals-depth_map.unsqueeze(-1))  # (N_rays, N_samples)
        depth_var = torch.sum(weights*tmp*tmp, dim=1)  # (N_rays)
       
        w = weights[..., None]
        if self.semantic_weight_mode == "argmax":
            w = torch.nn.functional.one_hot(w.argmax(dim=1)[:, 0], num_classes=w.shape[1]).unsqueeze(-1)
        if self.stop_semantic_grad:
            w = w.detach()
            semantic_map = torch.sum(w * semantics, -2)
            instance_map = torch.sum(w * instances, -2)
            # if tensorf.use_feature_reg:
            #     regfeat_map = torch.sum(w * regfeats, -2)
        else:
            semantic_map = torch.sum(w * semantics, -2)
            instance_map = torch.sum(w * instances, -2)
            # if tensorf.use_feature_reg:
            #     regfeat_map = torch.sum(w * regfeats, -2)

        if self.semantic_weight_mode == "softmax":
            semantic_map = semantic_map / (semantic_map.sum(-1).unsqueeze(-1) + 1e-8)
            semantic_map = torch.log(semantic_map + 1e-8)

        rgb_map = rgb_map.clamp(0, 1)
        regfeat_map = torch.zeros([1, 1], device=rgb_map.device)
        return rgb_map, semantic_map, instance_map, depth_map, regfeat_map, depth_var




    def forward(self, tensorf, rays, perturb, white_bg, is_train):
        
        z_vals, xyz_sampled = self.gen_samples(tensorf, rays, perturb > 0., self.n_samples)
        occ, feat = tensorf.compute_occ(xyz_sampled)
        _, weights = TsdfRFRenderer.raw2outputs_nerf_color(occ, z_vals)
        viewdirs = rays[:, 3:6].view(-1, 1, 3).expand(xyz_sampled.shape)

        rgbs = tensorf.compute_appearance_feature(xyz_sampled, viewdirs, feat)
        semantics = tensorf.compute_semantic_feature(xyz_sampled, feat.detach())
        instances = tensorf.compute_instance_feature(xyz_sampled, feat.detach())

        rgb_map = torch.sum(weights[..., None] * rgbs, -2)  # (N_rays, 3)
        depth_map = torch.sum(weights * z_vals, -1)  # (N_rays)
        tmp = (z_vals-depth_map.unsqueeze(-1))  # (N_rays, N_samples)
        depth_var = torch.sum(weights*tmp*tmp, dim=1)  # (N_rays)
       
        w = weights[..., None]
        if self.semantic_weight_mode == "argmax":
            w = torch.nn.functional.one_hot(w.argmax(dim=1)[:, 0], num_classes=w.shape[1]).unsqueeze(-1)
        if self.stop_semantic_grad:
            w = w.detach()
            semantic_map = torch.sum(w * semantics, -2)
            instance_map = torch.sum(w * instances, -2)
            # if tensorf.use_feature_reg:
            #     regfeat_map = torch.sum(w * regfeats, -2)
        else:
            semantic_map = torch.sum(w * semantics, -2)
            instance_map = torch.sum(w * instances, -2)
            # if tensorf.use_feature_reg:
            #     regfeat_map = torch.sum(w * regfeats, -2)

        if self.semantic_weight_mode == "softmax":
            semantic_map = semantic_map / (semantic_map.sum(-1).unsqueeze(-1) + 1e-8)
            semantic_map = torch.log(semantic_map + 1e-8)

        rgb_map = rgb_map.clamp(0, 1)
        regfeat_map = torch.zeros([1, 1], device=rgb_map.device)
        return rgb_map, semantic_map, instance_map, depth_map, regfeat_map, depth_var

    def forward_instance_feature(self, tensorf, rays, perturb, is_train):
        z_vals, xyz_sampled = self.gen_samples(tensorf, rays, perturb > 0., self.n_samples)
        occ, feat = tensorf.compute_occ(xyz_sampled)
        _, weights = TsdfRFRenderer.raw2outputs_nerf_color(occ, z_vals)

        instances = tensorf.compute_instance_feature(xyz_sampled, feat)
        instance_map = torch.sum(weights[..., None].detach() * instances, -2)
        return instance_map

    def forward_segment_feature(self, tensorf, rays, perturb, is_train):

        z_vals, xyz_sampled = self.gen_samples(tensorf, rays, perturb > 0., self.n_samples)
        occ, feat = tensorf.compute_occ(xyz_sampled)
        _, weights = TsdfRFRenderer.raw2outputs_nerf_color(occ, z_vals)
        viewdirs = rays[:, 3:6].view(-1, 1, 3).expand(xyz_sampled.shape)

        
        semantics = tensorf.compute_semantic_feature(xyz_sampled, feat)

        w = weights[..., None]
        w = w.detach()
        segment_map = torch.sum(w * semantics, -2)

        if self.semantic_weight_mode == "softmax":
            segment_map = segment_map / (segment_map.sum(-1).unsqueeze(-1) + 1e-8)
            segment_map = torch.log(segment_map + 1e-8)

        return segment_map


def split_points_minimal(xyz, extents, positions, orientations):
    split_xyz = []
    point_flags = []
    for i in range(extents.shape[0]):
        inverse_transform = torch.linalg.inv(trs_comp(positions[i], orientations[i], torch.ones([1], device=xyz.device)))
        inverse_transformed_xyz = (inverse_transform @ torch.cat([xyz, torch.ones([xyz.shape[0], 1], device=xyz.device)], 1).T).T[:, :3]
        t0 = torch.logical_and(inverse_transformed_xyz[:, 0] <= extents[i, 0] / 2, inverse_transformed_xyz[:, 0] >= -extents[i, 0] / 2)
        t1 = torch.logical_and(inverse_transformed_xyz[:, 1] <= extents[i, 1] / 2, inverse_transformed_xyz[:, 1] >= -extents[i, 1] / 2)
        t2 = torch.logical_and(inverse_transformed_xyz[:, 2] <= extents[i, 2] / 2, inverse_transformed_xyz[:, 2] >= -extents[i, 2] / 2)
        selection = torch.logical_and(torch.logical_and(t0, t1), t2)
        point_flags.append(selection)
        split_xyz.append(xyz[selection, :])
    return split_xyz, point_flags


def sample_points_in_box(rays, bbox_aabb, n_samples, step_size, perturb, is_train):
    rays_o, rays_d, nears, fars = rays[:, 0:3], rays[:, 3:6], rays[:, 6], rays[:, 7]
    vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
    rate_a = (bbox_aabb[1] - rays_o) / vec
    rate_b = (bbox_aabb[0] - rays_o) / vec
    t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=nears, max=fars)

    rng = torch.arange(n_samples)[None].float() #[1, Ns]
    if is_train and perturb != 0:
        rng = rng.repeat(rays_d.shape[-2], 1) #[N, Ns]
        rng = rng + perturb * torch.rand_like(rng[:, [0]]) 
    step = step_size * rng.to(rays_o.device) #[N, Ns]
    interpx = (t_min[..., None] + step) #[N, Ns]

    rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
    mask_outbbox = ((bbox_aabb[0] > rays_pts) | (rays_pts > bbox_aabb[1])).any(dim=-1)

    return rays_pts, interpx, ~mask_outbbox


def sample_points_uniform_in_box(rays_o, rays_d, nears, fars, bbox_aabb, n_samples, perturb):
    vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
    rate_a = (bbox_aabb[1] - rays_o) / vec
    rate_b = (bbox_aabb[0] - rays_o) / vec
    rate_a[rate_a < 0] = torch.inf
    rate_b[rate_b < 0] = torch.inf
    t_min = torch.minimum(rate_a, rate_b).amin(-1)
    fars = torch.minimum(fars, t_min)

    rng = torch.arange(n_samples, device=rays_d.device)[None].float()
    if perturb != 0:
        rng = rng.repeat(rays_d.shape[-2], 1) #[N, Ns]
        rng = rng + perturb * torch.rand_like(rng)

    rng = rng / n_samples
    # print(nears.device, fars.device, rng.device)
    z_vals = nears[..., None]* (1 - rng) + fars[..., None] * rng # [N, Ns]
    rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]
    return rays_pts, z_vals


def sample_points_gaussian(rays_o, rays_d, nears, fars, n_samples, scale = 1.1, std_scale = 3):
    dis = fars / scale
    std = torch.clamp((dis - nears) / std_scale, min = 0) + 1e-5
    normal_sampling = torch.randn([len(rays_o), n_samples], device=fars.device)
    z_vals = normal_sampling * std[..., None] + dis[..., None]
    
    rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]
    return rays_pts, z_vals


def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device = weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device = weights.device)

    # Pytest, overwrite u with numpy's fixed random numbers
    

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
    
    

