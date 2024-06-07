# Copyright (c) Meta Platforms, Inc. All Rights Reserved

from pathlib import Path
from pl.dataset.base import BaseDataset, create_segmentation_data_sem, InconsistentBaseDataset, InconsistentSingleBaseDataset
from pl.dataset.panopli import PanopLiDataset, InconsistentPanopLiDataset, InconsistentPanopLiSingleDataset, create_segmentation_data_panopli, SegmentPanopLiDataset


def get_dataset(config, load_only_val=False):
    if config.dataset_class == "panopli":
        train_set = None
        if not load_only_val:
            train_set = PanopLiDataset(Path(config.dataset_root), "train", (config.image_dim[0], config.image_dim[1]), config.max_depth, overfit=config.overfit, semantics_dir='m2f_semantics', load_depth=True,
                                       load_feat=config.use_feature_regularization, instance_dir='m2f_instance', instance_to_semantic_key='m2f_instance_to_semantic',
                                       create_seg_data_func=create_segmentation_data_panopli, subsample_frames=config.subsample_frames, depth_scale = config.depth_scale,
                                       depth_z2d=config.depth_z2d)
        val_set = PanopLiDataset(Path(config.dataset_root), "val", (config.image_dim[0], config.image_dim[1]), config.max_depth, overfit=config.overfit, semantics_dir='m2f_semantics',
                                 instance_dir='m2f_instance', instance_to_semantic_key='m2f_instance_to_semantic', 
                                 create_seg_data_func=create_segmentation_data_panopli, load_depth=True,
                                 subsample_frames=config.subsample_frames, depth_scale = config.depth_scale,
                                       depth_z2d=config.depth_z2d)
        return train_set, val_set
    raise NotImplementedError


def get_inconsistent_single_dataset(config):
    if config.dataset_class == "panopli":
        return InconsistentPanopLiSingleDataset(Path(config.dataset_root), "train", (256, 256), config.max_depth, overfit=config.overfit,
                                                max_rays=config.max_rays_instances, semantics_dir='m2f_semantics', instance_dir='m2f_instance', instance_to_semantic_key='m2f_instance_to_semantic',
                                                create_seg_data_func=create_segmentation_data_panopli, subsample_frames=config.subsample_frames,depth_scale=config.depth_scale, depth_z2d = config.depth_z2d)
    raise NotImplementedError


def get_segment_dataset(config):
    if config.dataset_class == "panopli":
        return SegmentPanopLiDataset(Path(config.dataset_root), "train", (256, 256), config.max_depth, overfit=config.overfit,
                                     max_rays=config.max_rays_segments, semantics_dir='m2f_semantics', instance_dir='m2f_instance', instance_to_semantic_key='m2f_instance_to_semantic',
                                     create_seg_data_func=create_segmentation_data_panopli, subsample_frames=config.subsample_frames, depth_scale=config.depth_scale, depth_z2d = config.depth_z2d)
    raise NotImplementedError
