import zipfile
import typing
import time    
from pathlib import Path
import numpy as np
import json
import subprocess
import os

def data_ingp2pl(src_folder, json_name = 'transforms.json', flip = True):
    json_file = Path(src_folder) / json_name
    with open(json_file) as f:
        transforms = json.load(f)
    if 'fl_x' not in transforms:
        ins = transforms['frames'][0]
        fl_x, fl_y, cx, cy = ins['fl_x'], ins['fl_y'], ins['cx'], ins['cy']
    else:
        fl_x, fl_y, cx, cy = transforms['fl_x'], transforms['fl_y'], transforms['cx'], transforms['cy']
    
    ins_mat = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]])
    Path(src_folder, "intrinsic").mkdir(exist_ok=True)
    Path(src_folder, "intrinsic", "intrinsic_color.txt").write_text(
        f"""{ins_mat[0, 0]} {ins_mat[0, 1]} {ins_mat[0, 2]} 0.00\n{ins_mat[1, 0]} {ins_mat[1, 1]} {ins_mat[1, 2]} 0.00\n{ins_mat[2, 0]} {ins_mat[2, 1]} {ins_mat[2, 2]} 0.00\n0.00 0.00 0.00 1.00""")
    Path(src_folder, "intrinsic", "intrinsic_depth.txt").write_text(
        f"""{ins_mat[0, 0]} {ins_mat[0, 1]} {ins_mat[0, 2]} 0.00\n{ins_mat[1, 0]} {ins_mat[1, 1]} {ins_mat[1, 2]} 0.00\n{ins_mat[2, 0]} {ins_mat[2, 1]} {ins_mat[2, 2]} 0.00\n0.00 0.00 0.00 1.00""")

    Path(src_folder, "pose").mkdir(exist_ok=True)

    if flip:
        flip_mat = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
    else:
        flip_mat = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    for frame in transforms['frames']:
        filepath = Path(frame['file_path'])
        RT = np.array(frame['transform_matrix']) @ flip_mat
        Path(src_folder, "pose", f'{filepath.stem}.txt').write_text(f"""{RT[0, 0]} {RT[0, 1]} {RT[0, 2]} {RT[0, 3]}\n{RT[1, 0]} {RT[1, 1]} {RT[1, 2]} {RT[1, 3]}\n{RT[2, 0]} {RT[2, 1]} {RT[2, 2]} {RT[2, 3]}\n0.00 0.00 0.00 1.00""")



# def data_polycam2ingp(data_folder_path: str, format: str = "ingp"):
#     folder = CaptureFolder(data_folder_path)
#     if format.lower() == "ingp" or format.lower() == "instant-ngp":
#         convertor = InstantNGPConvertor()
#     else:
#         logger.error("Format {} is not curently supported. Consider adding a convertor for it".format(format))
#         exit(1)
#     convertor.convert(folder)


def data_zip2ns(data_path):
    out_dir = 'data_path'
    data_zip_path = os.path.join(out_dir, 'source.zip')
    cmd = 'ns-process-data polycam --data {0} --output-dir {1} --use-depth --max-dataset-size -1 --min-blur-score -1  --num-downscales 0'.format(data_zip_path, out_dir)
    os.system(cmd)
    dest = Path(out_dir)
    img_path = dest / 'color'
    img_path.symlink_to((dest / 'images').absolute(), target_is_directory=True)
    return out_dir


def label_m2f(root_dir:str, available_cuda = 0):
    image_dir = Path(root_dir) / 'images'
    output_dir = Path(root_dir) / 'panoptic'
    output_dir.mkdir()
    cmd = 'CUDA_VISIBLE_DEVICES={0} python demo.py --config-file ../configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml  --input {1} --output {2} --predictions {2} --n 1 --p 0 --opts MODEL.WEIGHTS ../checkpoints/model_final_f07440.pkl'.format(str(available_cuda), image_dir.absolute(), output_dir.absolute())
    subprocess.check_call(cmd, cwd='preprocess/mask2former/demo', shell=True)


def unzip_polycam_src(zip_data_path, output_dir = None)-> str:
    if output_dir is None:
        output_dir = Path('./data/') / str(time.time_ns())
    
    with zipfile.ZipFile(zip_data_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    output_dir = list(output_dir.iterdir()).pop()
    output_dir = output_dir.rename(output_dir.parent / 'src')
    
    return str(output_dir)

from preprocess.pl.preprocess_m3d import *
def label_m2f_pl(dest:str, sc_classes = 'super'):
    # copy distorted images from raw to root
    # copy_color(dest, fraction=1)
    # copy transforms json
    # rename_and_copy_transforms(dest)
    # create segmentation data stub
    create_segmentation_data(dest, sc_classes=sc_classes)
    # # undistort images
    # create_undistorted_images(dest)
    # create_poses_without_undistortion(dest)
    # # create validation set (15% to 25%)
    # # create_validation_set(dest, 0.15)
    # # make sure to run mask2former before this step
    # # run mask2former segmentation data mapping
    map_panoptic_coco(dest, sc_classes=sc_classes)
    # # # visualize xlabels
    visualize_mask_folder(dest / "m2f_semantics")
    visualize_mask_folder(dest / "m2f_instance")
    visualize_mask_folder(dest / "m2f_notta_semantics")
    visualize_mask_folder(dest / "m2f_notta_instance")
    # copy predicted labels as GT, since we don't have GT
    shutil.copytree(dest / "m2f_semantics", dest / "semantics")
    shutil.copytree(dest / "m2f_instance", dest / "instance")
    shutil.copytree(dest / "m2f_semantics", dest / "rs_semantics")
    shutil.copytree(dest / "m2f_instance", dest / "rs_instance")
          

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
from pl.util.camera import compute_world2normscene, frustum_world_bounds
import open3d as o3d
import open3d.core as o3c
import imgviz
import tsdf_torch as tsdf
import marching_cubes as mcubes
import trimesh
import time


def train_get_tsdf_grid(data_root, depth_scale = 1000., max_depth = 5., tsdf_dim = 512):
    ins_mat = torch.from_numpy(np.loadtxt(Path(data_root) / 'intrinsic' / 'intrinsic_color.txt', ndmin=2)).float()
    pose_dir = Path(data_root) / 'pose'
    depth_dir = Path(data_root) / 'depth'
    color_dir = Path(data_root) / 'color'
    depth_files = list(depth_dir.iterdir())
    pose_files = list(pose_dir.iterdir())
    color_files = list(color_dir.iterdir())
    pose_files.sort()
    depth_files.sort()
    color_files.sort()

    depth_scale = depth_scale
    max_depth = max_depth
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
    # print(depths.shape)
    img_num = len(poses)
    dims = torch.tensor([tar_h, tar_w]).expand([img_num, -1])
    ins_mats = ins_mat.expand([img_num, -1, -1])
    

    vol_bnds = frustum_world_bounds(dims, ins_mats[..., :3, :3], poses, max_depth)
    vox_len = torch.max(vol_bnds[1] - vol_bnds[0]).item() / tsdf_dim
    # print('vl', vox_len)
    grid_tsdf, _ = tsdf.fusion(depths.cuda(), rgbs.cuda(), poses.cuda(), ins_mat[:3, :3].float().cuda(), vol_bnds.cuda(), vox_len, 4.*vox_len, 1.0)
    # print(grid_tsdf.shape)
    vertices, triangles = mcubes.marching_cubes(grid_tsdf.cpu().numpy(), 0., truncation=3.0)
    # print(vertices.shape)
    vertices = vertices * vox_len + vol_bnds[0].cpu().numpy()
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    mesh.export(Path(data_root) / ('./tsdf_mesh.ply'))
    # exit()

    shape = torch.tensor(grid_tsdf.shape)
    max_dim = torch.max(shape)
    grid = torch.ones([max_dim] * 3, device = grid_tsdf.device)
    x, y, z = grid_tsdf.shape
    x_shift, y_shift, z_shift = (max_dim - shape) // 2
    grid[x_shift:x_shift + x, y_shift:y_shift+y, z_shift:z_shift + z] = grid_tsdf
    torch.save(grid, Path(data_root) / ('tsdf_grid.pt'))


import omegaconf
import pf.trainer.train_panopli_tensorf as train_pf
# import plm.trainer.train_panopli_tensorf as train_pf
def train_model(data_dir, max_ep = 10):
    name = Path(data_dir).stem
    output_dir = Path('./runs') / name
    output_dir.mkdir(exist_ok= True)
    
    config = omegaconf.OmegaConf.load('./pf/config/config.yaml')
    config.experiment = name
    config.dataset_root = data_dir
    config.max_epoch = max_ep
    # config.grid_weight_path = str(Path(data_dir) / 'tsdf_grid.pt')
    train_pf.main(config)
    return str(output_dir)


from inference.render_panopli_mesh import render_panopli_checkpoint as infer_pf

def inference_model(weight_dir):
   infer_pf(weight_dir)



import tqdm
import numpy as np
from PIL import Image
import os
import glob
import json
import open3d as o3d

from math import cos, sin


def o3d_tsdf_fusion(data_dir):
    json_path = Path(data_dir) / 'transforms.json'
    with open(json_path) as rf:
        data = json.load(rf)

    frames = data['frames']
    

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length = 0.04,
        sdf_trunc = 0.2,
        color_type = o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )


    transf = np.array([
            [1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,1.],
        ])
    cnt = 0
    for frame in tqdm.tqdm(frames):
        fl_x, fl_y, cx, cy = frame['fl_x'], frame['fl_y'], frame['cx'], frame['cy']
        W, H = frame['w'], frame['h']
        '''
        Compute TSDF & mesh
        '''
        # d_img = o3d.io.read_image(frame['depth_file_path'])
        d_img = o3d.io.read_image(str(Path(data_dir)/frame['depth_file_path']))

        s_img = o3d.io.read_image(str(Path(data_dir) / frame['file_path']))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            s_img, d_img, depth_trunc=10., convert_rgb_to_intensity=False, depth_scale=1000.)
        
        # RT = np.asarray(pose['transform_matrix'] @ )
        # RT = np.linalg.inv(RT)
        pose = np.asarray(frame['transform_matrix'] @  transf)
        RT = np.linalg.inv(pose)
        # RT = pose
        
        
        volume.integrate(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                W, H, fl_x, fl_y, cx, cy
            ),
            RT)
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(os.path.join(data_dir, "o3d_mesh.ply"), mesh)

