import os
import copy
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
import pickle
from PIL import Image
from PIL import ImageDraw
import cv2
import numpy as np
import torch
# from ssr.ssr_utils.bins import *
from utils import rend_util


# init bin data
bin = {}
NUM_ORI_BIN=6
ORI_BIN_WIDTH = float(2 * np.pi / NUM_ORI_BIN)
ori_bin=[[(i - NUM_ORI_BIN / 2) * ORI_BIN_WIDTH, (i - NUM_ORI_BIN / 2 + 1) * ORI_BIN_WIDTH] for i
                          in range(NUM_ORI_BIN)]
bin["ori_bin"]=ori_bin
NUM_DEPTH_BIN = 10
DEPTH_WIDTH = 1.0
# centroid_bin = [0, 6]
bin['centroid_bin'] = [[i * DEPTH_WIDTH, (i + 1) * DEPTH_WIDTH] for i in
                       range(NUM_DEPTH_BIN)]

NUM_LAYOUT_ORI_BIN = 2
ORI_LAYOUT_BIN_WIDTH = np.pi / 4
bin['layout_ori_bin'] = [[np.pi / 4 + i * ORI_LAYOUT_BIN_WIDTH, np.pi / 4 + (i + 1) * ORI_LAYOUT_BIN_WIDTH] for i in range(NUM_LAYOUT_ORI_BIN)]

'''camera bin'''
PITCH_NUMBER_BINS = 2
PITCH_WIDTH = 40 * np.pi / 180
ROLL_NUMBER_BINS = 2
ROLL_WIDTH = 20 * np.pi / 180

# pitch_bin = [[-60 * np.pi/180, -20 * np.pi/180], [-20 * np.pi/180, 20 * np.pi/180]]
bin['pitch_bin'] = [[-60.0 * np.pi / 180 + i * PITCH_WIDTH, -60.0 * np.pi / 180 + (i + 1) * PITCH_WIDTH] for
                    i in range(PITCH_NUMBER_BINS)]
# roll_bin = [[-20 * np.pi/180, 0 * np.pi/180], [0 * np.pi/180, 20 * np.pi/180]]
bin['roll_bin'] = [[-20.0 * np.pi / 180 + i * ROLL_WIDTH, -20.0 * np.pi / 180 + (i + 1) * ROLL_WIDTH] for i in
                   range(ROLL_NUMBER_BINS)]



front3d_category_label_mapping = {
    "table": 0, "sofa": 1, "cabinet": 2, "night_stand": 3,
    "chair": 4, "bookshelf": 5, "bed": 6, "desk": 7, "dresser": 8
}



sunrgbd_layout_path = "./dataset/SUNRGBD/preprocessed/layout_avg_file.pkl"
with open(sunrgbd_layout_path,'rb') as f:
    sunrgbd_avg_layout = pickle.load(f)
sunrgbd_avgsize_path = "./dataset/SUNRGBD/preprocessed/size_avg_category.pkl"
with open(sunrgbd_avgsize_path,'rb') as f:
    sunrgbd_avgsize = pickle.load(f)


def get_centroid_from_proj(centroid_depth, proj_centroid, K):
    x_temp = (proj_centroid[0] - K[0, 2]) / K[0, 0]
    y_temp = (proj_centroid[1] - K[1, 2]) / K[1, 1]
    z_temp = 1
    ratio = centroid_depth / np.sqrt(x_temp ** 2 + y_temp ** 2 + z_temp ** 2)
    x_cam = x_temp * ratio
    y_cam = y_temp * ratio
    z_cam = z_temp * ratio
    p = np.stack([x_cam, y_cam, z_cam])
    return p


def R_from_yaw_pitch_roll(yaw, pitch, roll):
    '''
    get rotation matrix from predicted camera yaw, pitch, roll angles.
    :param yaw: batch_size x 1 tensor
    :param pitch: batch_size x 1 tensor
    :param roll: batch_size x 1 tensor
    :return: camera rotation matrix
    '''
    Rp = np.zeros((3, 3))
    Ry = np.zeros((3, 3))
    Rr = np.zeros((3, 3))
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cr = np.cos(roll)
    sr = np.sin(roll)
    Rp[0, 0] = 1
    Rp[1, 1] = cp
    Rp[1, 2] = -sp
    Rp[2, 1] = sp
    Rp[2, 2] = cp

    Ry[0, 0] = cy
    Ry[0, 2] = sy
    Ry[1, 1] = 1
    Ry[2, 0] = -sy
    Ry[2, 2] = cy

    Rr[0, 0] = cr
    Rr[0, 1] = -sr
    Rr[1, 0] = sr
    Rr[1, 1] = cr
    Rr[2, 2] = 1

    R = np.dot(np.dot(Rr, Rp), Ry)
    return R



class REALIMG_Recon_Dataset(Dataset):
    def __init__(self, config, mode):
        super(REALIMG_Recon_Dataset, self).__init__()
        self.mode = mode
        self.config = config
        self.data_root_path = config['data']['data_path']
        self.split_dir = config['data']['split_dir']
        self.dataset = config['data']['dataset']
        classname = self.config['data']['test_class_name']
        self.split_path = os.path.join(self.split_dir, self.dataset, classname+'.json')
        with open(self.split_path, 'r') as f:
            self.split = json.load(f)


    def __len__(self):
        return len(self.split)
    

    def __getitem__(self, data_idx):
        success_flag=False
        while success_flag==False:

            split_name, object_id, category = self.split[data_idx]
            data_idx = np.random.randint(0,self.__len__())
            split_path = os.path.join(self.data_root_path, 'Im3d_det', split_name)
            with open(split_path, 'rb') as f:
                sequence = pickle.load(f)

            img_path = split_path.replace('Im3d_det', 'rgb').replace('.pkl', '.png')        # load img
            image = Image.open(img_path)
            sequence['rgb_img'] = np.array(image)
            width,height=image.size
            # depth = Image.fromarray(sequence['depth_map'])
            # camera = sequence['camera']
            boxes = sequence['bboxes']

            camera_intrinsics = sequence['K']

            bdb2D=boxes[object_id]['bdb2D']

            
            # ###### test bdb2D
            # im = copy.deepcopy(image)
            # draw = ImageDraw.Draw(im)
            # draw.rectangle(bdb2D.tolist(), outline='red')
            # im.show()

            # delta2d = boxes['delta_2D'][object_id]
            # project_center = np.zeros([2])
            # project_center[0] = (bdb2D[0] + bdb2D[2]) / 2 - delta2d[0] * (bdb2D[2] - bdb2D[0])
            # project_center[1] = (bdb2D[1] + bdb2D[3]) / 2 - delta2d[1] * (bdb2D[3] - bdb2D[1])
            project_center = boxes[object_id]['project_center']
            centroid_depth = boxes[object_id]['centroid_depth']
            # centroid_cls, centroid_reg = boxes['centroid_cls'][object_id], boxes['centroid_reg'][object_id]
            # centroid_depth = np.mean(bin['centroid_bin'][centroid_cls]) + centroid_reg * DEPTH_WIDTH
            obj_cam_center=get_centroid_from_proj(centroid_depth, project_center, camera_intrinsics)
            

            cls_codes = np.zeros([9])
            cls_codes[front3d_category_label_mapping[category]] = 1

            # size_reg=boxes['size_reg'][object_id]
            # avg_size=sunrgbd_avgsize[boxes['size_cls'][object_id]]
            # bbox_size=(1+size_reg)*avg_size*2

            bbox_size = boxes[object_id]['size']


            none_equal_scale = 2.0 / bbox_size
            scene_scale = np.array([1.0, 1.0, 1.0])
            voxel_range = np.array([1.0, 1.0, 1.0])                          # voxel_range include padding : voxel_range is the range of voxel after none_equal_scale coords transfer


            # pitch_cls,pitch_reg=camera['pitch_cls'],camera['pitch_reg']
            # roll_cls,roll_reg=camera['roll_cls'],camera['roll_reg']
            # ori_cls,ori_reg=boxes['ori_cls'][object_id],boxes['ori_reg'][object_id]


            pitch = sequence["layout"]['pitch']
            roll = sequence["layout"]['roll']
            yaw = boxes[object_id]['yaw']
            rot_matrix = R_from_yaw_pitch_roll(-yaw, pitch, roll)

            # pitch=np.mean(bin['pitch_bin'][pitch_cls])+pitch_reg*PITCH_WIDTH
            # roll=np.mean(bin['roll_bin'][roll_cls])+roll_reg*ROLL_WIDTH
            # yaw=np.mean(bin['ori_bin'][ori_cls])+ori_reg*ORI_BIN_WIDTH-np.pi/2

            # yaw=yaw
            # rot_matrix=R_from_yaw_pitch_roll(yaw,pitch,-roll)

            rot_transfer_matrix = np.array([
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0]
            ])
            rot = np.dot(rot_transfer_matrix, rot_matrix)
            trans_transfer_matrix = np.array([
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0]
            ])
            trans = np.dot(trans_transfer_matrix, obj_cam_center)
            camera_extrinsics = np.eye(4)
            camera_extrinsics[:3, :3] = rot
            camera_extrinsics[:3, 3] = trans

            camera_pose = np.linalg.inv(camera_extrinsics)

            obj_rot = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ])
            obj_tran = np.array([0.0, 0.0, 0.0])
            obj_centroid = np.array([0.0, 0.0, 0.0])
            obj_to_world = np.eye(4)
            obj_to_world[0:3, 0:3] = obj_rot
            obj_to_world[0:3, 3] = obj_tran
            world_to_obj = np.linalg.inv(obj_to_world)          # from world coords to obj coords



            # # test projection
            # device = 'cpu'
            # marching_cube_resolution = 32
            # x_coor = torch.linspace(-1.0, 1.0, steps=marching_cube_resolution).float().to(device)
            # y_coor = torch.linspace(-1.0, 1.0, steps=marching_cube_resolution).float().to(device)
            # z_coor = torch.linspace(-1.0, 1.0, steps=marching_cube_resolution).float().to(device)
            # X, Y, Z = torch.meshgrid(x_coor, y_coor, z_coor)
            # samples_incan = torch.cat([X[:, :, :, None], Y[:, :, :, None], Z[:, :, :, None]], dim=3)
            # pnts_world = (samples_incan.view(-1, 3)*bbox_size/2).float()                                      # bbox size scale to original size
            # camera_extrinsics = torch.from_numpy(camera_extrinsics).unsqueeze(0).float()
            # camera_intrinsics = torch.from_numpy(camera_intrinsics).unsqueeze(0).float()

            # prt_img = cv2.cvtColor(sequence['rgb_img'], cv2.COLOR_RGB2BGR)
            # bbox_img = copy.deepcopy(prt_img)
            # uv_pro, _ = rend_util.get_uv_world(pnts_world, camera_intrinsics, camera_extrinsics)
            # uv_pro = uv_pro.squeeze(0)
            # uv_pro = uv_pro.numpy()
            # for p_2d in uv_pro:
            #     p_2d = p_2d.astype(int)
            #     cv2.circle(prt_img, p_2d, 1, (255, 0, 0), -1)
            
            # # prt_img.save('prt.png')
            # # cv2.imwrite('prt.png', prt_img)
            # cv2.rectangle(prt_img, (int(bdb2D[0]), int(bdb2D[1])), (int(bdb2D[2]), int(bdb2D[3])), (0, 255, 0), 2)
            # prt_img = cv2.cvtColor(prt_img, cv2.COLOR_BGR2RGB)
            # prt_img = Image.fromarray(prt_img)
            # # prt_img.show()
            # prt_img.save('prt.png')
            # # # cv2.imwrite(f'bbox_{category}.png', bbox_img)
            # # bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB)
            # # bbox_img = Image.fromarray(bbox_img)
            # # bbox_img.show()
            # # # bbox_img.save(f'bbox_{category}.png')
            # print('done')
            # import time
            # time.sleep(100)

            # exit()



            # save bbox image
            bbox_img = cv2.cvtColor(sequence['rgb_img'], cv2.COLOR_RGB2BGR)
            cv2.rectangle(bbox_img, (int(bdb2D[0]), int(bdb2D[1])), (int(bdb2D[2]), int(bdb2D[3])), (0, 255, 0), 2)
            # cv2.imwrite(f'bbox_{category}.png', bbox_img)

            
            img_gt = sequence['rgb_img']
            img_gt = img_gt / 255.0
            img_gt = np.transpose(img_gt, (2, 0, 1))


            sample = {
                "image": img_gt,
                "intrinsics": camera_intrinsics,
                "pose": camera_pose,
                "extrinsics": camera_extrinsics,
                'obj_rot': obj_rot,
                'obj_tran': obj_tran,
                'world_to_obj': world_to_obj,
                'none_equal_scale': none_equal_scale,
                'scene_scale': scene_scale,
                'voxel_range': voxel_range,
                'centroid': obj_centroid,
                'cls_encoder': cls_codes.astype(np.float32)
            }

            # use object bdb2d global feature
            [xmin, ymin, xmax, ymax] = bdb2D           # x -> img_W -> py -> uv[0], y -> img_H -> px -> uv[1]
            bdb_x = np.linspace(xmin, xmax, 64)
            bdb_y = np.linspace(ymin, ymax, 64)
            bdb_X, bdb_Y = np.meshgrid(bdb_x, bdb_y)
            bdb_X = (bdb_X - width/2) / width*2 #-1 ~ 1
            bdb_Y = (bdb_Y - height/2) / height*2 #-1 ~ 1
            bdb_grid = np.concatenate([bdb_X[:, :, np.newaxis], bdb_Y[:, :, np.newaxis]], axis=-1)          # [64, 64, 2]

            sample["bdb_grid"] = bdb_grid

            # pesudo uv, just for suitable for the code
            uv = np.zeros((64, 2))
            sample["uv"] = uv


            ground_truth = {
                'rgb': img_gt,
                'split_name': split_name.split('.')[0],
                'object_id': object_id,
                'bdb_2d': bdb2D,
                'obj_rot': obj_rot,
                'obj_tran': obj_tran,
                'world_to_obj': world_to_obj,
                'category': category,
                'voxel_padding': 0.0,
                'centroid': obj_centroid,
                'bbox_img': bbox_img,
            }

            success_flag=True


        return data_idx, sample, ground_truth



def worker_init_fn(worker_id):
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)

def REALIMG_Recon_dataloader(config, mode='train'):
    dataloader = DataLoader(
                    dataset=REALIMG_Recon_Dataset(config, mode),
                    num_workers=config['data']['num_workers'],
                    batch_size=config['data']['batch_size'][mode],
                    shuffle=(mode == 'train'),
                    worker_init_fn=worker_init_fn, pin_memory=True
                )
    return dataloader

