import numpy as np
import imageio
import skimage
import torch
from torch.nn import functional as F

from utils import sdf_util


def get_psnr(img1, img2, normalize_rgb=False):
    if normalize_rgb: # [-1,1] --> [0,1]
        img1 = (img1 + 1.) / 2.
        img2 = (img2 + 1. ) / 2.

    mse = torch.mean((img1 - img2) ** 2)
    psnr = -10. * torch.log(mse) / torch.log(torch.Tensor([10.]).cuda())

    return psnr


def load_rgb(path, normalize_rgb = False):
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)

    if normalize_rgb: # [-1,1] --> [0,1]
        img -= 0.5
        img *= 2.
    img = img.transpose(2, 0, 1)        # [C, H, W]
    return img


def get_camera_params_cam(uv, intrinsics, get_obj_dirs=False, model_input=None):
    """
    get ray_dirs, cam_loc in camera coords
    """
    batch_size, num_samples, _ = uv.shape

    cam_loc = torch.tensor([0, 0, 0]).cuda().float()                        # cam is the origin point
    cam_loc = torch.repeat_interleave(cam_loc.unsqueeze(0), batch_size, dim=0)

    depth = torch.ones((batch_size, num_samples)).cuda()
    x_cam = uv[:, :, 0].view(batch_size, -1)
    y_cam = uv[:, :, 1].view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)     # camera coordinate, [B, N, 4]

    # permute for solve camera coords conflict in blender
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)                    # [B, 4, N]

    ###### solve camera coords conflict in blender
    pixel_points_cam = camera_coords_transfer(pixel_points_cam)             # [B, 4, N]

    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)[:, :, :3]          # [B, N, 3]
    ray_dirs = pixel_points_cam - cam_loc[:, None, :]
    ray_dirs = F.normalize(ray_dirs, dim=2)

    
    if get_obj_dirs:
        pose = model_input['pose']
        obj_rot = model_input['obj_rot']
        rot1 = pose[:, :3, :3]
        rot2 = obj_rot.permute(0, 2, 1)
        ray_world = torch.bmm(rot1, ray_dirs.permute(0, 2, 1))              # [B, 3, N]
        ray_obj = torch.bmm(rot2, ray_world)                                # [B, 3, N]
        ray_obj = ray_obj.permute(0, 2, 1)                                  # [B, N, 3]

        ray_world = ray_world.permute(0, 2, 1)                              # [B, N, 3]

        ###### test ray_dirs_obj (get cam_loc_obj)
        world_to_obj = model_input['world_to_obj']

        cam_loc_temp = cam_loc[:, None, None, :]                        # [B, 1, 1, 3]
        
        cam_loc_world = camera_to_world(cam_loc_temp, pose)                                   # (B, 1, 1, 3)

        cam_loc_obj = sdf_util.world2obj(cam_loc_world, world_to_obj)              # (B, 1, 1, 3)
        
        cam_loc_obj = cam_loc_obj.squeeze(2)                                # [B, 1, 3]
        cam_loc_obj = cam_loc_obj.squeeze(1)                                # [B, 3]

        cam_loc_world = cam_loc_world.squeeze(2)                                # [B, 1, 3]
        cam_loc_world = cam_loc_world.squeeze(1)                                # [B, 3]

    else:
        return ray_dirs, cam_loc

    return ray_dirs, cam_loc, ray_obj, cam_loc_obj, ray_world, cam_loc_world


def get_camera_params_world(uv, pose, intrinsics):
    """
    get ray_dirs, cam_loc in world coords
    """
    cam_loc = pose[:, :3, 3]
    p = pose

    batch_size, num_samples, _ = uv.shape

    depth = torch.ones((batch_size, num_samples)).cuda()
    x_cam = uv[:, :, 0].view(batch_size, -1)
    y_cam = uv[:, :, 1].view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)     # camera coordinate

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)                    # [B, 4, N]

    ###### solve camera coords conflict in blender
    pixel_points_cam = camera_coords_transfer(pixel_points_cam)

    world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]
    ray_dirs = world_coords - cam_loc[:, None, :]
    ray_dirs = F.normalize(ray_dirs, dim=2)

    return ray_dirs, cam_loc


def lift(x, y, z, intrinsics):
    # parse intrinsics
    intrinsics = intrinsics.cuda()
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    x_lift = (x - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z).cuda()), dim=-1)


def camera_to_world(points_cam, pose):
    """
    from camera coords to world coords
    :params points_cam, [B, N, M, 3] (in use, N is ray number, M is points on a ray)
    :params pose, [B, 4, 4]
    """
    B, N, M, _ = points_cam.shape
    ones = torch.ones((B, N, M)).cuda()          # homogeneous
    x_cam = points_cam[:, :, :, 0]
    y_cam = points_cam[:, :, :, 1]
    z_cam = points_cam[:, :, :, 2]
    points_cam_homo = torch.stack((x_cam, y_cam, z_cam, ones), dim=-1)      # [B, N, M, 4]
    
    points_cam_homo = points_cam_homo.reshape(B, -1, 4)     # [B, N*M, 4]
    points_cam_homo = points_cam_homo.permute(0, 2, 1)      # [B, 4, N*M]

    points_world = torch.bmm(pose, points_cam_homo).permute(0, 2, 1)[:, :, :3]      # [B, N*M, 3]

    return points_world.reshape(B, N, M, 3)


def world_to_camera(pnts, extrinsics):
    """
    from world coords to camera coords
    :params pnts, [N, 3] or [B, N_uv, N_ray, 3], in world coords
    :params camera extrinsics, [B, 4, 4] (in use, B is 1, only one obj)
    """
    if len(pnts.shape) == 2:                    # inference shape [N, 3]
        pnts = pnts[None, None, ...]            # [1, 1, N, 3]

    B, num_pixels, ray_points, _ = pnts.shape
    R = extrinsics[:, 0:3, 0:3]                 # [B, 3, 3]
    T = extrinsics[:, 0:3, 3]                   # [B, 3]
    # world to camera
    points_re = (pnts.reshape(B, -1, 3)).permute(0, 2, 1)           # [B, 3, num_pixels*ray_points]
    pnts_cam = torch.bmm(R, points_re)+ T.unsqueeze(2)              # [B, 3, num_pixels*ray_points]

    # don't need ''rend_util.camera_coords_transfer(pnts_camera)''
    # because this is from world to camera
    # when from uv (image coords) to camera, need solve camera coords conflict in blender

    return pnts_cam                         # [B, 3, num_pixels*ray_points]


def camera_coords_transfer(points):
    """
    solve camera coords conflict in blender
    y --> -y, z --> -z
    :params points, [B, 3, N] or [B, 4, N] (homogeneous camera coords), [3, N] or [4, N]
    """
    points_cam = points.clone()
    if len(points_cam.shape) == 3:                      # [B, 3, N] or [B, 4, N]
        points_cam[:, 1, :] = -points_cam[:, 1, :]        # y --> -y
        points_cam[:, 2, :] = -points_cam[:, 2, :]        # z --> -z
    elif len(points_cam.shape) == 2:                    # [3, N] or [4, N]
        points_cam[1, :] = -points_cam[1, :]              # y --> -y
        points_cam[2, :] = -points_cam[2, :]              # z --> -z
    else:
        raise ValueError('points_cam shape error!')

    return points_cam


def get_latent_feature(model, pnts, intrinsics, extrinsics, input):
    
    uv_align, z_vals_pnts = get_uv_world(pnts, intrinsics, extrinsics)
    latent_feature, cat_feature = model.get_feature(input, uv_align, z_vals_pnts)
    return latent_feature, cat_feature


def get_uv_world(pnts, intrinsics, extrinsics):
    """
    Get pixel-aligned uv coords at 3D world coordinate
    :param pnts (B, N_uv, N_ray, 3) world coordinate points (x, y, z)
    :param intrinsics (B, 3, 3)   camera intrinsics
    :param extrinsics (B, 4, 4)    camera extrinsics  from world to camera coordinate
    """
    pnts_cam_temp = world_to_camera(pnts, extrinsics)       # [B, 3, num_pixels*ray_points], camera coords, if transfer to image coords, need camera coords transfer
    z_vals_pnts = torch.norm(pnts_cam_temp, p=2, dim=1)

    uv = get_uv_cam(pnts_cam_temp.permute(0, 2, 1), intrinsics)
    return uv, z_vals_pnts


def get_uv_cam(pnts_cam, intrinsics):
    """
    points in camera coords get uv
    :params pnts_cam_input, [B, num_pixels*ray_points, 3]
    """
    pnts_cam = pnts_cam.permute(0, 2, 1)                            # [B, 3, num_pixels*ray_points]

    # NOTE: if transfer to image coords use camera intrinsics, need camera coords transfer
    pnts_camera = camera_coords_transfer(pnts_cam)                  # [B, 3, num_pixels*ray_points] solve camera coords conflict in blender
    
    # trans pnts to homogeneous camera coordinates
    pnts_cam_homo = pnts_camera / (pnts_camera[:, 2, :].unsqueeze(1))   # [B, 3, num_pixels*ray_points]
    pnts_pix = torch.bmm(intrinsics, pnts_cam_homo)
    uv = pnts_pix[:, :2, :]                                             # [B, 2, num_pixels*ray_points]
    uv = uv.permute(0, 2, 1)                                            # [B, num_pixels*ray_points, 2]

    return uv


def rot_angle(rot, axis, angle):
    """
    refer https://blog.csdn.net/zsq306650083/article/details/8773996
    :params rot, original rotation matrix (3, 3)
    :params axis, coordinate axis x, y, z
    :params angle, rotation angle (-180, 180)
    """
    angle = angle / 180.0 * np.pi
    cos = np.cos(angle)
    sin = np.sin(angle)

    rot = rot.cpu().numpy()

    if axis == 'x':
        rot_max = np.array([
            [1, 0, 0],
            [0, cos, sin],
            [0, -sin, cos]
        ])

    elif axis == 'y':
        rot_max = np.array([
            [cos, 0, -sin],
            [0, 1, 0],
            [sin, 0, cos]
        ])

    else:
        rot_max = np.array([
            [cos, sin, 0],
            [-sin, cos, 0],
            [0, 0, 1]
        ])

    rot_new = rot @ rot_max             # rotation local coordinate, right multiply

    return torch.from_numpy(rot_new).cuda()


def compose_matrix(rot, trans):
    """
    :params rot, [3, 3]
    :params trans, [3, 1]
    """
    M = torch.eye(4)
    M[:3, :3] = rot
    M[:3, 3] = trans

    return M.cuda()


def rot_camera_pose(pose, obj_bdb_3d_camera, angle, axis='y'):
    """
    :params pose, [4, 4], original camera pose
    :params obj_bdb_3d_camera, [3, 8], object center in world coords
    :params angle, a scalar
    :params axis, 'x, y, z', rot axis, NOTE: rot in camera coords, y is up
    """
    obj_to_cam_trans = torch.mean(obj_bdb_3d_camera, dim=1)
    obj_to_cam = compose_matrix(torch.eye(3), obj_to_cam_trans)
    obj2_to_obj = compose_matrix(rot_angle(torch.eye(3), axis, angle), torch.tensor([0, 0, 0]))
    new_pose = pose @ obj_to_cam @ obj2_to_obj @ torch.inverse(obj_to_cam)

    return new_pose


def trans_camera_pose(pose, x_dist, y_dist, z_dist):
    """
    :params pose, [4, 4], original camera pose
    :params x_dist, y_dist, z_dist, NOTE: translation in world coords, z is up
    """

    rot = pose[:3, :3]
    trans = pose[:3, 3]

    dist = torch.tensor([x_dist, y_dist, z_dist], dtype=torch.float32).cuda()
    new_trans = trans + dist

    new_pose = torch.eye(4, dtype=torch.float32).cuda()
    new_pose[:3, :3] = rot
    new_pose[:3, 3] = new_trans

    return new_pose
