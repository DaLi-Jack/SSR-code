import os, sys
sys.path.append(os.getcwd())
import copy

import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.data
import numpy as np
import json, gzip
import skimage
import imageio

from utils.sdf_util import *

category_label_mapping = {
    "table": 0, "sofa": 1, "wardrobe": 2, "tool": 3,
    "chair": 4, "bookcase": 5, "bed": 6, "desk": 7, "misc": 8
}


class Pix3d_Recon_Dataset(Dataset):
    def __init__(self, config, mode):
        super(Pix3d_Recon_Dataset, self).__init__()
        self.mode = mode
        self.config = config
        self.use_depth = self.config['data']['use_depth']
        self.use_normal = self.config['data']['use_normal']
        self.use_sdf = self.config['data']['use_sdf']
        # self.img_res = self.config['data']['img_res']
        self.num_pixels = self.config['data']['num_pixels'][mode]
        self.resize_img = self.config['data']['resize_img']
        if self.resize_img:
            self.resize_res = self.config['data']['resize_res']
        else:
            self.resize_res = None

        self.mask_filter = self.config['data']['mask_filter']
        self.bdb2d_filter = self.config['data']['bdb2d_filter']
        self.soft_pixels = self.config['data']['soft_pixels']
        if mode=="train":
            classnames = self.config['data']['train_class_name']
        elif mode == 'val':
            classnames = self.config['data']['train_class_name']        # val is in training time
        else:
            classnames = self.config['data']['test_class_name']
        
        dataset_name = self.config['data']['dataset']       # now, just for front3d
        if isinstance(classnames, list):
            self.multi_class = True
            self.split = []
            for class_name in classnames:
                self.split_path = os.path.join(self.config['data']['split_dir'], dataset_name, 'split', mode, class_name + ".json")
                with open(self.split_path, 'rb') as f:
                    self.split += json.load(f)
        else:
            self.multi_class = False
            self.split = []
            class_name = classnames
            self.split_path = os.path.join(self.config['data']['split_dir'], dataset_name, 'split', mode, class_name + ".json")
            with open(self.split_path, 'r') as f:
                self.split = json.load(f)
        # if trial, only use 200 sample
        if self.config['data']['trial']:
            self.split = self.split[:200]

        self.vis_mask_loss = self.config['loss']['vis_mask_loss']

        self.add_bdb3d_points = self.config['model']['ray_sampler']['add_bdb3d_points']
        if self.add_bdb3d_points:
            self.total_add_points = self.config['model']['ray_sampler']['total_add_points']
            self.use_surface_points = self.config['model']['ray_sampler']['use_surface_points']

        # use object bdb2d global feature
        self.use_global_encoder = self.config['model']['latent_feature']['use_global_encoder']

        self.use_cls_encoder = self.config['model']['latent_feature']['use_cls_encoder']


    def __len__(self):
        return len(self.split)
    

    def load_rgb(self, path, resize_img=False, resize_res=None, normalize_rgb = False):

        img = imageio.imread(path)          # [H, W, C]

        if resize_img:
            img = cv2.resize(img, (resize_res[1], resize_res[0]))         # resize_res [H, W]

        img = skimage.img_as_float32(img)

        if normalize_rgb: # [-1,1] --> [0,1]
            img -= 0.5
            img *= 2.
        img = img.transpose(2, 0, 1)        # [C, H, W]
        return img


    def __getitem__(self, data_idx):
        success_flag=False
        while success_flag==False:

            imgid, objid, cname = self.split[data_idx]
            data_idx = np.random.randint(0,self.__len__())

            # load data
            img_path = os.path.join(self.config['data']['data_path'], imgid)
            post_fix = img_path.split('.')[-1]      # avoid '.png' '.jpg' '.jpeg'

            if not os.path.exists(img_path):
                continue
            
            img_np = self.load_rgb(img_path, self.resize_img, self.resize_res)         # load image

            _, height, width = img_np.shape     # [C, H, W]

            self.img_res = np.array([height, width])
            self.total_pixels = self.img_res[0] * self.img_res[1]

            # load annotation
            anno_path = img_path.replace('img', 'annotation').replace(f'.{post_fix}', '.json')
            if not os.path.exists(anno_path):
                continue
            with open(anno_path, 'r') as f:
                sequence = json.load(f)

            # resize image need to modify camera intrinsics
            if self.resize_img:
                ori_res = sequence['img_res']
                ori_height = ori_res[0]
                ori_width = ori_res[1]
                radio_h = height / ori_height
                radio_w = width / ori_width

                rM = np.array([
                    [radio_w, 0, 0],
                    [0, radio_h, 0],
                    [0, 0, 1]
                ])
            
            sequence['rgb_img'] = img_np

            # load mask (full mask)
            mask_path = img_path.replace('img', 'mask').split('.')[0] + '.png'
            segm = cv2.imread(mask_path)        # [H, W, 3]
            if self.resize_img:
                segm = cv2.resize(segm, (self.resize_res[1], self.resize_res[0]))
            
            sequence['all_mask'] = segm

            if self.use_depth:
                # load depth
                depth_path = img_path.replace('img', 'depth').replace(f'.{post_fix}', '.npy.gz')
                with gzip.GzipFile(depth_path, 'r') as f:
                    depth = np.load(f)
                    depth = 1 - depth               # omnidata fix
                
                if self.resize_img:
                    depth = cv2.resize(depth, (self.resize_res[1], self.resize_res[0]))

                sequence['depth'] = depth

            if self.use_normal:
                # load normal
                normal_path = img_path.replace('img', 'normal').replace(f'.{post_fix}', '.npy.gz')
                with gzip.GzipFile(normal_path, 'r') as f:
                    normal = np.load(f)

                if self.resize_img:
                    normal = cv2.resize(normal, (self.resize_res[1], self.resize_res[0]))
                    
                sequence['normal'] = normal

            # category label
            cid = category_label_mapping[cname]
            object_ind = objid            # white means object, pix3d one image only have one object

            # camera pose (from camera to world)
            camera_pose_tran = np.array(sequence['camera_pose_tran'])
            camera_pose_rot = np.array(sequence['camera_pose_rot'])
            camera_pose = np.eye(4)
            camera_pose[0:3, 3] = camera_pose_tran
            camera_pose[0:3, 0:3] = camera_pose_rot
            
            camera_intrinsics = np.array(sequence['camera_intrinsics'])
            if self.resize_img:
                camera_intrinsics = rM @ camera_intrinsics
                # for padding
                camera_intrinsics[0][2] = width/2
                camera_intrinsics[1][2] = height/2

            # camera extrinsics (from world to camera)
            camera_extrinsics_raw = np.array(sequence['camera_extrinsics'])	# ndarray 3*4
            camera_extrinsics = np.eye(4)
            camera_extrinsics[0:3, :] = camera_extrinsics_raw

            segm = sequence['all_mask'][:, :, 0]
            segm_index = np.argwhere(segm == int(object_ind))
            px = [index[0] for index in segm_index]                 # height    uv[1]
            py = [index[1] for index in segm_index]                 # width     uv[0]
            obj_map = np.zeros((height, width), dtype=np.uint8)
            obj_map[px, py] = 1

            full_mask_array = np.zeros(height*width, dtype=bool)         # [H*W, ]
            for full_index in segm_index:
                full_mask_array[full_index[0]*width + full_index[1]] = True

            # full bbox
            xmin, xmax = int(np.min(py)), int(np.max(py))
            ymin, ymax = int(np.min(px)), int(np.max(px))
            full_bbox_2d = [xmin, ymin, xmax, ymax]

            if self.vis_mask_loss:
                """
                for pix3d, vis mask is full mask
                """
                vis_mask_array = np.zeros(height*width, dtype=bool)         # [H*W, ]
                vis_mask_index = segm_index
                for index in vis_mask_index:
                    vis_mask_array[index[0]*width + index[1]] = True

            # load 2D bbox, 3D bdb
            bdb_2d = np.array(full_bbox_2d)
            bdb_3d = np.array(sequence['obj_dict'][object_ind]['bbox3d_world'])
            bdb_3d_center = np.array(sequence['obj_dict'][object_ind]['bbox3d_world_center'])
            half_length = np.array(sequence['obj_dict'][object_ind]['half_length'])
            obj_rot = np.array(sequence['obj_dict'][object_ind]['obj_rot'])
            obj_tran = np.array(sequence['obj_dict'][object_ind]['obj_tran'])               # obj_tran is different with bdb_3d_center

            obj_to_world = np.eye(4)
            obj_to_world[0:3, 0:3] = obj_rot
            obj_to_world[0:3, 3] = obj_tran
            world_to_obj = np.linalg.inv(obj_to_world)          # from world coords to obj coords

            if self.use_sdf:
                # load object SDF
                scene_scale = np.array([1.0, 1.0, 1.0])         # none scene scale
                model_file_name = sequence['obj_dict'][object_ind]['model_file_name']
                voxel_path = os.path.join(self.config['data']['sdf_path'], cname, model_file_name)
                sdf_path = os.path.join(voxel_path, 'voxels.npy.gz')
                spacing_path = os.path.join(voxel_path, 'spacing_centroid.json')

                if not os.path.exists(sdf_path):
                    continue

                with open(spacing_path, 'r') as f:
                    spacing_dic = json.load(f)
                spacing = np.array(spacing_dic['spacing'])
                padding = float(spacing_dic['padding'])
                centroid = np.array(spacing_dic['centroid'])
                voxel_range = np.array([1.0, 1.0, 1.0])                          # voxel_range include padding : voxel_range is the range of voxel after none_equal_scale coords transfer
                none_equal_scale = np.array(spacing_dic['none_equal_scale'])     # none_equal_scale = (2 - padding) / mesh.bounding_box.extents
                with gzip.GzipFile(sdf_path, 'r') as f:
                    voxels = np.load(f)                             # [R, R, R]

            # !!! numpy array index is different with image uv coordinate 
            if self.mask_filter:
                uv = torch.from_numpy(np.array([py, px])).transpose(1, 0)
                real_total_pixels = uv.shape[0]

            else:
                if self.bdb2d_filter:
                    [xmin, ymin, xmax, ymax] = bdb_2d       # x -> img_W -> py -> uv[0], y -> img_H -> px -> uv[1]
                    # apply soft bdb2d
                    xmin = max(xmin-self.soft_pixels, 0)
                    xmax = min(xmax+self.soft_pixels, self.img_res[1])
                    ymin = max(ymin-self.soft_pixels, 0)
                    ymax = min(ymax+self.soft_pixels, self.img_res[0])
                    uv = np.mgrid[ymin:ymax, xmin:xmax].astype(np.int32)
                    uv = torch.from_numpy(np.flip(uv, axis=0).copy())           # flip
                    uv = uv.reshape(2, -1).transpose(1, 0)                      # uv [x, y] x->img_W, y->img_H
                    real_total_pixels = uv.shape[0]

                else:
                    uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
                    uv = torch.from_numpy(np.flip(uv, axis=0).copy())
                    uv = uv.reshape(2, -1).transpose(1, 0)                          # uv [x, y] x->img_W, y->img_H
                    real_total_pixels = self.total_pixels

            # sample pixels
            if self.num_pixels != -1:       # -1 represent not sampler in inference
                if real_total_pixels < self.num_pixels:             # mask pixels less than num_pixels
                    sampling_idx = torch.randperm(real_total_pixels)
                    for i in range((self.num_pixels-1)//real_total_pixels):
                        sample_num = min(real_total_pixels, self.num_pixels-sampling_idx.shape[0])
                        temp_sampling_idx = torch.randperm(real_total_pixels)[:sample_num]
                        sampling_idx = torch.cat((sampling_idx, temp_sampling_idx), dim=0)
                else:
                    sampling_idx = torch.randperm(real_total_pixels)[:self.num_pixels]
                uv = uv[sampling_idx]

            
            # load object image
            image = sequence['rgb_img']             # [C, H, W]
            _, height, width = image.shape

            uv_sampling_idx = torch.tensor([xy[1]*width+xy[0] for xy in uv])              # img_H * W + img_W

            image_gt = copy.deepcopy(image)         # for calculate loss
            image_gt = image_gt.reshape(3, -1).transpose(1, 0)
            image_gt = image_gt[uv_sampling_idx]

            # ground_truth.image for calculate loss
            ground_truth = {
                'rgb': image_gt,
                'mask': obj_map,
                'bdb_2d': bdb_2d,
                'bdb_3d': bdb_3d,
                'bdb_3d_center': bdb_3d_center,
                'half_length': half_length,
                'obj_rot': obj_rot,
                'obj_tran': obj_tran,
                'world_to_obj': world_to_obj,
                'img_id': imgid,
                'object_id': int(objid),
                'cname': str(cname)
            }

            if self.vis_mask_loss:
                vis_pixel = vis_mask_array[uv_sampling_idx]
                ground_truth['vis_pixel'] = torch.tensor(vis_pixel)

            full_mask_pixel = full_mask_array[uv_sampling_idx]
            ground_truth['full_mask_pixel'] = torch.tensor(full_mask_pixel)
            
            if self.use_depth:
                depth_gt = copy.deepcopy(sequence['depth'])         # [H, W]
                depth_gt = depth_gt.reshape(-1, 1)                  
                depth_gt = depth_gt[uv_sampling_idx]
                # modify error depth (cause: mask edge near to window)
                depth_error_idx = np.argwhere(depth_gt > 1000)
                depth_gt[depth_error_idx] = -1
                depth_gt[depth_error_idx] = max(depth_gt)       # modify to maximum

                ground_truth['depth'] = depth_gt

            if self.use_normal:
                normal_gt = copy.deepcopy(sequence['normal'])       # [H, W, 3]

                # fix normal gt
                normal_gt_fix = np.zeros_like(normal_gt)
                normal_gt_fix[:, :, 0] = normal_gt[:, :, 0]
                normal_gt_fix[:, :, 1] = 255 - normal_gt[:, :, 1]
                normal_gt_fix[:, :, 2] = 255 - normal_gt[:, :, 2]
                normal_gt = normal_gt_fix

                normal_gt = normal_gt.reshape(-1, 3)
                normal_gt = normal_gt[uv_sampling_idx]

                ground_truth['normal'] = normal_gt * 2.0 - 1.0      # [0, 1] --> [-1, 1]

            if self.use_sdf:
                ground_truth['voxel_sdf'] = np.expand_dims(voxels, axis=-1).transpose(3, 0, 1, 2)     # (1, R, R, R)      for F.grid_sample
                ground_truth['voxel_spacing'] = spacing
                ground_truth['voxel_padding'] = padding
                ground_truth['centroid'] = centroid
                ground_truth['voxel_range'] = voxel_range
                ground_truth['none_equal_scale'] = none_equal_scale             # model scale to cube
                ground_truth['scene_scale'] = scene_scale                       # in pix3d, scene_scale = 1.0

            # sample.image for extractor image feature
            sample = {
                "image": image,                     # [C, H, W]
                "uv": uv,
                "intrinsics": camera_intrinsics,
                "pose": camera_pose,
                "extrinsics": camera_extrinsics,
                'obj_rot': obj_rot,
                'obj_tran': obj_tran,
                'world_to_obj': world_to_obj,
                'centroid': centroid,
                'none_equal_scale': none_equal_scale,
                'scene_scale': scene_scale,
                'voxel_range': voxel_range,
            }

            # add bdb3d world points
            if self.add_bdb3d_points:
                addpoints_total = self.total_add_points             
                
                if self.use_surface_points:         # add object surface points
                    voxel_index = np.argwhere((voxels < 0.1) & (voxels > -0.1))
                    surface_points = voxel_index2obj_coordinate(voxel_index, centroid, voxel_range, spacing, none_equal_scale)            # [N, 3], in model object coords
                    surface_points = surface_points * scene_scale                                                       # [N, 3], in scene object coords
                    surface_sample_count = surface_points.shape[0]
                    points = surface_points + np.random.normal(scale=0.0025, size=(surface_sample_count, 3))

                else:                               # random add object bdb3d points
                    raise ValueError('not use surface points')

                if points.shape[0] < addpoints_total:
                    add_points_sampling_idx = np.random.permutation(points.shape[0])
                    for i in range((addpoints_total-1)//points.shape[0]):
                        add_sample_num = min(points.shape[0], addpoints_total-add_points_sampling_idx.shape[0])
                        temp_add_sample_idx = np.random.permutation(points.shape[0])[:add_sample_num]
                        add_points_sampling_idx = np.concatenate((add_points_sampling_idx, temp_add_sample_idx), axis=0)
                else:
                    add_points_sampling_idx = np.random.permutation(points.shape[0])[:addpoints_total]    

                add_points_flat = points[add_points_sampling_idx]                           # in scene object coords
                # transfer to world coords
                add_points_world_flat = obj2world_numpy(add_points_flat, obj_rot, obj_tran)
                add_points_world = add_points_world_flat.reshape(100, addpoints_total // 100, 3)     # (100, addpoints_total // 100, 3), similar to ray 
                sample['add_points_world'] = add_points_world

            # use object bdb2d global feature
            if self.use_global_encoder:
                [xmin, ymin, xmax, ymax] = bdb_2d           # x -> img_W -> py -> uv[0], y -> img_H -> px -> uv[1]
                bdb_x = np.linspace(xmin, xmax, 64)
                bdb_y = np.linspace(ymin, ymax, 64)
                bdb_X, bdb_Y = np.meshgrid(bdb_x, bdb_y)
                bdb_X = (bdb_X - width/2) / width*2 #-1 ~ 1
                bdb_Y = (bdb_Y - height/2) / height*2 #-1 ~ 1
                bdb_grid = np.concatenate([bdb_X[:, :, np.newaxis], bdb_Y[:, :, np.newaxis]], axis=-1)          # [64, 64, 2]

                sample["bdb_grid"] = bdb_grid

            if self.use_cls_encoder:
                cls_codes = np.zeros([9])
                cls_codes[cid] = 1
                sample['cls_encoder'] = cls_codes.astype(np.float32)

            success_flag=True

        return data_idx, sample, ground_truth
    

def worker_init_fn(worker_id):
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)

def Pix3D_Recon_dataloader(config, mode='train'):
    dataloader = DataLoader(
                    dataset=Pix3d_Recon_Dataset(config, mode),
                    num_workers=config['data']['num_workers'],
                    batch_size=config['data']['batch_size'][mode],
                    shuffle=(mode == 'train'),
                    worker_init_fn=worker_init_fn, pin_memory=True
                )
    return dataloader
