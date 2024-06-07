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
import sys
# sys.path.append('/home/fgm/disk1/Focus/code/panoptic-lifting')
from dataset import PanopLiDataset, create_segmentation_data_panopli
from model.radiance_field.tensoRF import TensorVMSplit
from model.renderer.panopli_tensoRF_renderer import TensoRFRenderer
from trainer import visualize_panoptic_outputs
from util.camera import distance_to_depth
from util.misc import get_parameters_from_state_dict
from util.camera import compute_world2normscene, frustum_world_bounds
import open3d as o3d
import open3d.core as o3c
import imgviz
import tsdf_torch as tsdf
import marching_cubes as mcubes
import trimesh
# def get_mesh(config):
#     train_set = PanopLiDataset(Path(config.dataset_root), "train", (config.image_dim[0], config.image_dim[1]), config.max_depth, overfit=config.overfit, semantics_dir='m2f_semantics', instance_dir='m2f_instance',
#                             instance_to_semantic_key='m2f_instance_to_semantic', create_seg_data_func=create_segmentation_data_panopli, subsample_frames=config.subsample_frames)
#     c2w = train_set.cam2normscene
#     frame_names = train_set.all_frame_names
#     root_dir = train_set.root_dir

#     device = o3d.core.Device("CUDA:0")
#     vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf', 'weight'),
#                                     attr_dtypes=(o3c.float32,
#                                                     o3c.float32),
#                                     attr_channels=((1), (1)),
#                                     voxel_size=3.0 / 512,
#                                     block_resolution=16,
#                                     block_count=50000,
#                                     device=device)
#     intrinsics = np.loadtxt('/home/fgm/disk1/Focus/code/panoptic-lifting/data/scannet/scene0423_02/intrinsic/intrinsic_depth.txt', dtype=np.float32, ndmin=2)

#     max_depth = train_set.max_depth 
#     scale = 1000. / train_set.normscene_scale
#     for i, name in tqdm(enumerate(frame_names)):
#         depth = o3d.t.io.read_image(root_dir / "depth" / "%s.png"%name.str()).to(device)
#         w2c = np.linalg.inv(c2w[i])
#         frustum_block_coords = vbg.compute_unique_block_coordinates(
#             depth, intrinsics, w2c, scale,
#             max_depth)
        
#         vbg.integrate(frustum_block_coords, depth, intrinsics,
#                           w2c, scale, max_depth)
#     mesh = vbg.extract_triangle_mesh().to_legacy()
#     o3d.io.write_triangle_mesh('test.ply', mesh, write_vertex_normals = True)



def train_net(config):
    
    ins_mat = torch.from_numpy(np.loadtxt(Path(config.dataset_root) / 'intrinsic' / 'intrinsic_color.txt', ndmin=2)).float()
    pose_dir = Path(config.dataset_root) / 'pose'
    depth_dir = Path(config.dataset_root) / 'depth'
    color_dir = Path(config.dataset_root) / 'color'
    depth_files = list(depth_dir.iterdir())
    pose_files = list(pose_dir.iterdir())
    color_files = list(color_dir.iterdir())
    pose_files.sort()
    depth_files.sort()
    color_files.sort()

    depth_scale = 4000.
    max_depth = 3.
    transf = np.array([
            [1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,1.],
        ])

    # transf = np.eye(4)

    poses, depths, rgbs = [], [], []
    W, H = None, None
    tar_w, tar_h = 512, 512
    for pf, df, cf in tqdm(zip(pose_files, depth_files, color_files)):
        if W is None:
            H,W = np.asarray(Image.open(df)).shape

        poses.append(torch.from_numpy(np.loadtxt(pf)) @ transf)
        depths.append(torch.from_numpy(np.asarray(Image.open(df).resize((tar_w, tar_h), resample=Image.NEAREST))))
        rgbs.append(torch.from_numpy(np.asarray(Image.open(cf).resize(
            (tar_w, tar_h), resample = Image.LANCZOS
        ))))


    ins_mat = torch.diag(torch.tensor([tar_w / W, tar_h / H, 1.])) @ ins_mat[:3, :3]
    poses = torch.stack(poses).float()
    depths=  torch.stack(depths).float() / depth_scale
    rgbs = torch.stack(rgbs).float() / 255.
    print(depths.shape)
    # H, W = 1024, 1280
    # H, W = 768, 1024
    img_num = len(poses)
    dims = torch.tensor([tar_h, tar_w]).expand([img_num, -1])
    ins_mats = ins_mat.expand([img_num, -1, -1])

    # scene2norm = compute_world2normscene(dims.float(), ins_mat[..., :3, :3], poses, max_depth)

    
    '''
    get pcd
    '''
    # pixel2cam = torch.linalg.inv(ins_mat)
    # W_grid, H_grid = torch.meshgrid([torch.arange(W), torch.arange(H)])
    # pcd = []
    # for p, d in tqdm(zip(poses, depths)):
    #     pixels = torch.stack([W_grid, H_grid, torch.ones_like(H_grid)], dim = -1) * d.T[..., None]
    #     points = pixel2cam[:3, :3] @ pixels.flatten(end_dim=-2).T
    #     points = p[:3, :3] + p[:3, 3:]
    #     points = points.T
    #     points = points[torch.randint(0, len((points)), (100,))]
    #     pcd.append(points)
 
    # pcd = torch.cat(pcd).numpy()
    # # pcd = pixels.flatten(end_dim = -2).numpy()
    # o3d_pcd = o3d.geometry.PointCloud()
    # o3d_pcd.points = o3d.utility.Vector3dVector(pcd)
    # o3d.io.write_point_cloud('test_pcd.ply', o3d_pcd)

    # poses = torch.matmul(scene2norm, poses)
    # vol_bnds = torch.ones((2, 3)).float()
    # vol_bnds[0] *= -1

    vol_bnds = frustum_world_bounds(dims, ins_mats[..., :3, :3], poses, max_depth)
    vox_len = torch.max(vol_bnds[1] - vol_bnds[0]).item() / 512.
    print('vl', vox_len)
    grid_tsdf, _ = tsdf.fusion(depths.cuda(), rgbs.cuda(), poses.cuda(), ins_mat[:3, :3].float().cuda(), vol_bnds.cuda(), vox_len, 4.*vox_len, 1.0)

    vertices, triangles = mcubes.marching_cubes(grid_tsdf.cpu().numpy(), 0., truncation=3.0)
    print(vertices.shape)
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    mesh.export(Path(config.dataset_root) / './density_test.ply')


    shape = torch.tensor(grid_tsdf.shape)
    max_dim = torch.max(shape)
    grid = torch.ones([max_dim] * 3, device = grid_tsdf.device)
    x, y, z = grid_tsdf.shape
    x_shift, y_shift, z_shift = (max_dim - shape) // 2
    grid[x_shift:x_shift + x, y_shift:y_shift+y, z_shift:z_shift + z] = grid_tsdf
    torch.save(grid, Path(config.dataset_root) / 'tsdf_grid.pt')


    # xi, yi, zi = grid_tsdf.shape
    # voxel_xyz = torch.meshgrid(torch.arange(xi), torch.arange(yi), torch.arange(zi))
    # voxel_xyz = torch.stack(voxel_xyz, -1)
    # voxel_xyz = voxel_xyz.flatten(end_dim = -2)

    # train_data = torch.cat([voxel_xyz, grid_tsdf.cpu().flatten()[..., None]], dim = -1)
    # train_idx = torch.randperm(len(train_data))
    # batch_size = 8192

    # model = torch.nn.Sequential(
    #     torch.nn.Linear(3, 256),
    #     torch.nn.ReLU(),
    # )
    
    # for i in range(2):
    #     model.append(torch.nn.Linear(256, 256))
    #     model.append(torch.nn.ReLU())

    # model.append(torch.nn.Linear(256, 1))

    # model = model.cuda()
    # model = model.train()
    # op = torch.optim.Adam(model.parameters(), lr = 1e-3)
    # loss_func = torch.nn.MSELoss()

    # epoch = 10
    # for i in tqdm(range(epoch), desc = 'training'):
    #     for j in tqdm(range(0, len(train_idx), batch_size), desc = 'train_epoch'):
    #         ids = train_idx[j:j + batch_size]
    #         train_samples = train_data[ids]
    #         input, target = train_samples[..., :3], train_samples[..., 3:]
    #         coords = input * 0.05 + vol_bnds[0][None]
    #         coords, target = coords.cuda(), target.cuda()
    #         output = model(coords)
    #         loss_val = loss_func(output, target)
    #         op.zero_grad()
    #         loss_val.backward()
    #         op.step()
    #     print('loss:', loss_val.cpu().item())

    # torch.save(model.state_dict(), 'densitynet.pt')
    # new_grid = []
    # model = model.eval()
    # with torch.no_grad():
    #     for i in tqdm(range(0, len(voxel_xyz), batch_size * 2), desc = 'inference'):
    #         ids = voxel_xyz[i:i + batch_size * 2]
    #         inputs = ids * 0.05 + vol_bnds[0]
    #         outputs = model(inputs.cuda())
    #         new_grid.append(outputs.cpu())
    
    # new_grid = torch.cat(new_grid, dim = 0).reshape(*grid_tsdf.shape)
    
    # vertices, triangles = mcubes.marching_cubes(new_grid.cpu().numpy(), 0., truncation=3.0)
    # print(vertices.shape)
    # mesh = trimesh.Trimesh(vertices, triangles, process=False)
    # mesh.export('./density_net_test.ply')

   
def create_instances_from_semantics(instances, semantics, thing_classes):
    stuff_mask = ~torch.isin(semantics.argmax(dim=1), torch.tensor(thing_classes).to(semantics.device))
    padded_instances = torch.ones((instances.shape[0], instances.shape[1] + 1), device=instances.device) * -float('inf')
    padded_instances[:, 1:] = instances
    padded_instances[stuff_mask, 0] = float('inf')
    return padded_instances


if __name__ == "__main__":
    # needs a predefined trajectory named trajectory_blender in case test_only = False
    
    cfg = omegaconf.OmegaConf.load('/home/fgm/disk1/Focus/code/panoptic-lifting/config/panopli.yaml')
    train_net(cfg)
