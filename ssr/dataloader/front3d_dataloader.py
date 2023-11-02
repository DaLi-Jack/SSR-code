import os, sys
sys.path.append(os.getcwd())
import copy

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.data
from torchvision import transforms
import numpy as np
import json, gzip
from tqdm import tqdm

from utils.sdf_util import *
from utils.rend_util import load_rgb


category_label_mapping = {
    "table": 0, "sofa": 1, "cabinet": 2, "night_stand": 3,
    "chair": 4, "bookshelf": 5, "bed": 6, "desk": 7, "dresser": 8
}

data_transforms_mask = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
])


class Front3D_Recon_Dataset(Dataset):
    def __init__(self, config, mode):
        super(Front3D_Recon_Dataset, self).__init__()
        self.mode = mode
        self.config = config
        self.use_depth = self.config['data']['use_depth']
        self.use_normal = self.config['data']['use_normal']
        self.use_sdf = self.config['data']['use_sdf']
        self.use_instance_mask = self.config['data']['use_instance_mask']
        self.img_res = self.config['data']['img_res']
        self.num_pixels = self.config['data']['num_pixels'][mode]
        self.total_pixels = self.img_res[0] * self.img_res[1]
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
        # if trial, only use 200 samples
        if self.config['data']['trial']:
            self.split = self.split[:200]

        # if evaluation, just 2000 object
        if self.mode == 'test':
            self.split = self.split[:2000]

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

    def __getitem__(self, index):
        imgid, objid, cname = self.split[index]
        '''load the data dynamically or store them in the memory firstly'''
        if self.config['data']['load_dynamic'] == True:
            img_path = os.path.join(self.config['data']['data_path'], imgid)
            post_fix = img_path.split('.')[-1]      # avoid '.png' '.jpg' '.jpeg'
            img_np = load_rgb(img_path)         # load image

            anno_path = img_path.replace('rgb', 'annotation').replace(f'.{post_fix}', '.json')
            with open(anno_path, 'r') as f:
                sequence = json.load(f)             # load annotation
            
            sequence['rgb_img'] = img_np

            # load mask
            mask_path = img_path.replace('rgb', 'mask').replace(f'.{post_fix}', '.npy.gz')           
            with gzip.GzipFile(mask_path, 'r') as f:
                segm = np.load(f)                   # load full mask, later this is mask predicted by 2D mask branch
            _, height, width = img_np.shape     # [C, H, W]
            segm = segm[100:100+height, 100:100+width, :]       # axis = 0 is height, axis = 1 is width
            sequence['all_mask'] = segm             # all objects mask

            if self.vis_mask_loss:          # load visible mask
                vis_mask_path = img_path.replace('rgb', 'segm').replace(f'.{post_fix}', '.npy.gz')
                with gzip.GzipFile(vis_mask_path, 'r') as f:
                    vis_mask = np.load(f)
                sequence['vis_mask'] = vis_mask

            if self.use_depth:
                # load depth
                depth_path = img_path.replace('rgb', 'depth').replace(f'.{post_fix}', '.npy.gz')
                with gzip.GzipFile(depth_path, 'r') as f:
                    depth = np.load(f)
                sequence['depth'] = depth

            if self.use_normal:
                # load normal
                normal_path = img_path.replace('rgb', 'normal').replace(f'.{post_fix}', '.npy.gz')
                with gzip.GzipFile(normal_path, 'r') as f:
                    normal = np.load(f)
                sequence['normal'] = normal

            cid = category_label_mapping[cname]
            
        else:
            sequence = self.anno_dict[imgid]

            if self.use_sdf:
                jid = sequence['obj_dict'][objid]['model_file_name'][0]        # ['xxxxxxxxx'], a list
                cid = self.cid_dict[jid]

        object_ind = objid

        # camera pose (from camera to world)
        camera_pose_tran = np.array(sequence['camera_pose_tran'])
        camera_pose_rot = np.array(sequence['camera_pose_rot'])
        camera_pose = np.eye(4)
        camera_pose[0:3, 3] = camera_pose_tran
        camera_pose[0:3, 0:3] = camera_pose_rot
        
        camera_intrinsics = np.array(sequence['camera_intrinsics'])

        # camera extrinsics (from world to camera)
        camera_extrinsics_raw = np.array(sequence['camera_extrinsics'])	# ndarray 3*4
        camera_extrinsics = np.eye(4)
        camera_extrinsics[0:3, :] = camera_extrinsics_raw

        # an object full mask, obj_id is semantic id in front3d mask map; objid is object index in this image
        obj_id = sequence['obj_dict'][object_ind]['obj_id'][0]              # [xxx], a list
        segm = sequence['all_mask']
        segm_index = np.argwhere(segm == obj_id)
        px = [index[0] for index in segm_index]                 # height    uv[1]
        py = [index[1] for index in segm_index]                 # width     uv[0]
        obj_map = np.zeros((height, width), dtype=np.uint8)
        obj_map[px, py] = 1

        height, width, _ = segm.shape
        full_mask_array = np.zeros(height*width, dtype=bool)         # [H*W, ]
        for full_index in segm_index:
            full_mask_array[full_index[0]*width + full_index[1]] = True

        # full bbox (get from full mask)
        xmin, xmax = int(np.min(py)), int(np.max(py))
        ymin, ymax = int(np.min(px)), int(np.max(px))
        full_bbox_2d = [xmin, ymin, xmax, ymax]

        if self.vis_mask_loss:
            vis_mask = sequence['vis_mask']                 # [H, W]
            height, width = vis_mask.shape
            vis_mask_array = np.zeros(height*width, dtype=bool)         # [H*W, ]
            vis_mask_index = np.argwhere(vis_mask == obj_id)
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

        # set sample bound according to bbox3d_camera 
        bdb_3d_camera = np.array(sequence['obj_dict'][object_ind]['bbox3d_camera'])

        if self.use_sdf:
            # load object SDF
            jid = sequence['obj_dict'][object_ind]['model_file_name'][0]        # ['xxxxxxxxx'], a list
            scene_scale = np.array(sequence['obj_dict'][object_ind]['obj_scale'])
            # scale_name = format(scale[0], '.6f') + '_' + format(scale[1], '.6f') + '_' + format(scale[2], '.6f')
        
            if self.config['data']['load_dynamic'] == True:
                spacing_path = os.path.join(self.config['data']['sdf_path'], cname, jid, 'spacing_centroid.json')
                sdf_path = os.path.join(self.config['data']['sdf_path'], cname, jid, 'voxels.npy.gz')
                with open(spacing_path, 'r') as f:
                    spacing_dic = json.load(f)
                spacing = np.array(spacing_dic['spacing'])
                padding = float(spacing_dic['padding'])
                centroid = np.array(spacing_dic['centroid'])
                voxel_range = np.array([1.0, 1.0, 1.0])                          # voxel_range include padding : voxel_range is the range of voxel after none_equal_scale coords transfer
                none_equal_scale = np.array(spacing_dic['none_equal_scale'])     # none_equal_scale = (2 - padding) / mesh.bounding_box.extents
                with gzip.GzipFile(sdf_path, 'r') as f:
                    voxels = np.load(f)                             # [R, R, R]

            else:
                raise ValueError('Not support load static sdf now!')


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

        # width, height = image.size
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
            'bdb_3d_camera': bdb_3d_camera,
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

        if self.use_instance_mask:
            crop_mask = obj_map[ymin:ymax, xmin:xmax]       # [H, W], y --> img_H, x --> img_W
            crop_mask = data_transforms_mask(crop_mask * 255)       # transforms.ToTensor divide 255

            ground_truth['instance_mask'] = crop_mask

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
            ground_truth['scene_scale'] = scene_scale                       # model in different scene, may have different scene scale

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

        return index, sample, ground_truth

def worker_init_fn(worker_id):
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)

def Front3D_Recon_dataloader(config, mode='train'):
    dataloader = DataLoader(
                    dataset=Front3D_Recon_Dataset(config, mode),
                    num_workers=config['data']['num_workers'],
                    batch_size=config['data']['batch_size'][mode],
                    shuffle=(mode == 'train'),
                    worker_init_fn=worker_init_fn, pin_memory=True
                )
    return dataloader
