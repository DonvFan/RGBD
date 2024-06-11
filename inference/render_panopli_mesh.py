# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import sys
import random
import omegaconf
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
import numpy as np
import os


import open3d as o3d
import open3d.core as o3c
import imgviz
import json


def render_panopli_checkpoint(run_path):
    run_path = Path(run_path)
    res_path = run_path / 'results'
    res_path.mkdir()
    config = run_path / 'config.yaml'
    config = omegaconf.OmegaConf.load(config)
    checkpoints = list((run_path / 'checkpoints').iterdir())
    checkpoints.sort()
    config.resume = str(checkpoints.pop().absolute())
    config.mesh_path = str(Path(config.dataset_root) / 'o3d_mesh.ply')

    import sys

    sys.path.append(str(run_path / 'code' / 'pf'))

    from dataset import PanopLiDataset, create_segmentation_data_panopli
    from model.radiance_field.tensoRF import TensorVMSplit
    from model.renderer.panopli_tensoRF_renderer import TensoRFRenderer
    from trainer import visualize_panoptic_outputs
    from util.camera import distance_to_depth
    from util.misc import get_parameters_from_state_dict
    from util.camera import compute_world2normscene
   
    device = torch.device("cuda:0")
    test_set = PanopLiDataset(Path(config.dataset_root), "test", (config.image_dim[0], config.image_dim[1]), config.max_depth, overfit=config.overfit, semantics_dir='m2f_semantics', instance_dir='m2f_instance',
                              instance_to_semantic_key='m2f_instance_to_semantic', create_seg_data_func=create_segmentation_data_panopli, subsample_frames=config.subsample_frames,depth_scale=config.depth_scale)
    # H, W, alpha = config.image_dim[0], config.image_dim[1], 0.65
    # whether to render the test set or a predefined trajectory through the scene
    
    
    instance_mesh = o3d.io.read_triangle_mesh(config.mesh_path)
    color_mesh = o3d.geometry.TriangleMesh(instance_mesh)
    semantic_mesh = o3d.geometry.TriangleMesh(instance_mesh)

    if not semantic_mesh.has_vertex_normals():
        semantic_mesh = semantic_mesh.compute_vertex_normals()
    # points, normals = torch.from_numpy(np.asarray(pcd.points)).float(), torch.from_numpy(np.asarray(pcd.normals)).float()
    points, normals = torch.from_numpy(np.asarray(semantic_mesh.vertices,  dtype=np.float32)), torch.from_numpy(np.asarray(semantic_mesh.vertex_normals, dtype=np.float32))
    
    ins_mat = torch.from_numpy(np.loadtxt(Path(config.dataset_root) / 'intrinsic' / 'intrinsic_depth.txt', ndmin=2)).float()
    pose_dir = Path(config.dataset_root) / 'pose'
    depth_dir = Path(config.dataset_root) / 'depth'
    depth_files = list(depth_dir.iterdir())
    pose_files = list(pose_dir.iterdir())
    pose_files.sort()
    depth_files.sort()

    depth_scale = 1000.
    max_depth = config.max_depth

    poses, depths = [], []
    W, H = None, None
    tar_w, tar_h = config.image_dim[0], config.image_dim[1]
    for pf, df in zip(pose_files, depth_files):
        if W is None:
            H,W = np.asarray(Image.open(df)).shape
        poses.append(torch.from_numpy(np.loadtxt(pf)))
        depths.append(torch.from_numpy(np.asarray(Image.open(df).resize((tar_w, tar_h), resample=Image.NEAREST))))

    poses = torch.stack(poses).float()
    ins_mat = torch.diag(torch.tensor([tar_w / W, tar_h / H, 1.])) @ ins_mat[:3, :3]
    img_num = len(poses)
    dims = torch.tensor([tar_h, tar_w]).expand([img_num, -1])
    ins_mats = ins_mat.expand([img_num, -1, -1])

    scene2norm = compute_world2normscene(dims.float(), ins_mats[..., :3, :3], poses, max_depth)

    R, T = scene2norm[:3, :3], scene2norm[:3, 3:]
    norm_points = (R @ points.T + T).T

    checkpoint = torch.load(config.resume, map_location="cpu")
    state_dict = checkpoint['state_dict']
    total_classes = len(test_set.segmentation_data.bg_classes) + len(test_set.segmentation_data.fg_classes)
    output_mlp_semantics = torch.nn.Identity() if config.semantic_weight_mode != "softmax" else torch.nn.Softmax(dim=-1)
    model = TensorVMSplit([config.min_grid_dim, config.min_grid_dim, config.min_grid_dim], num_semantics_comps=(32, 32, 32),
                           num_semantic_classes=total_classes, dim_feature_instance=config.max_instances,
                           output_mlp_semantics=output_mlp_semantics, use_semantic_mlp=config.use_mlp_for_semantics)
    

    renderer = TensoRFRenderer(test_set.scene_bounds, [config.min_grid_dim, config.min_grid_dim, config.min_grid_dim],
                                    semantic_weight_mode=config.semantic_weight_mode, stop_semantic_grad=config.stop_semantic_grad)
    renderer.load_state_dict(get_parameters_from_state_dict(state_dict, "renderer"))

    for epoch in config.grid_upscale_epochs[::-1]:
        if checkpoint['epoch'] >= epoch:
            model.upsample_volume_grid(renderer.grid_dim)
            renderer.update_step_size(renderer.grid_dim)
            break
    
    model.load_state_dict(get_parameters_from_state_dict(state_dict, "model"))
    model = model.to(device)
    model = model.eval()
    renderer = renderer.to(device)

    rays = torch.cat([
         norm_points, normals, 
         -0.01 * torch.ones([len(norm_points), 1]),
         0.01 * torch.ones([len(norm_points), 1])
    ], dim = 1)
    # disable this for fast rendering (just add more steps along the ray)
    min_z = norm_points[..., -1].min()
    ind = norm_points[..., -1] < (min_z + 1e-2)
    renderer.update_step_ratio(renderer.step_ratio * 0.5)
    with torch.no_grad():
            concated_outputs = []
            outputs = []
            # infer semantics and surrogate ids
            for i in tqdm(range(0, len(rays), config.chunk)):
                out_rgb_, out_semantics_, out_instances_, out_depth_, _, _ = renderer(model, rays[i:i+config.chunk].to(device), config.perturb, test_set.white_bg, False)
                outputs.append([out_rgb_, out_semantics_, out_instances_, out_depth_])
            for i in range(len(outputs[0])):
                concated_outputs.append(torch.cat([outputs[j][i] for j in range(len(outputs))], dim=0))
            p_rgb, p_semantics, p_instances, p_dist = concated_outputs
            # p_depth = distance_to_depth(test_set.intrinsics[0], p_dist.view(H, W))
            # create surrogate ids
            p_instances = create_instances_from_semantics(p_instances, p_semantics, test_set.segmentation_data.fg_classes)

            semantic_ids = p_semantics.argmax(dim = -1, keepdim = True)
            semantic_ids_np = semantic_ids.cpu().numpy()
            points_out = []
            with open("./inference/label_map.json") as rf:
                d = json.load(rf)
                bim_class = d['bim_class']
                scannet_class = d['scannet_class']
            
            for k, v in scannet_class.items():
                temp = []
                for id in v:
                    temp.append(points[semantic_ids_np[..., -1] == id])
                temp = np.concatenate(temp, axis=0)
                temp = np.concatenate([temp, np.full((len(temp), 1), bim_class[k])], axis=-1)
                points_out.append(temp)
            
            points_out = np.concatenate(points_out, axis=0)
            np.savetxt(res_path / "labeled_pcd.txt", points_out, fmt='%.4f')




def create_instances_from_semantics(instances, semantics, thing_classes):
    stuff_mask = ~torch.isin(semantics.argmax(dim=1), torch.tensor(thing_classes).to(semantics.device))
    padded_instances = torch.ones((instances.shape[0], instances.shape[1] + 1), device=instances.device) * -float('inf')
    padded_instances[:, 1:] = instances
    padded_instances[stuff_mask, 0] = float('inf')
    return padded_instances



