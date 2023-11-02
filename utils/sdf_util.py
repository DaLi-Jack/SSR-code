import os
import torch
import numpy as np
import torch.nn.functional as F

from utils import rend_util


def obj_coordinate2voxel_index(obj_coordinate, centroid, voxel_range, spacing, none_equal_scale):
    """
    obj coordinate (x, y, z) to voxel index (a, b, c)
    voxel_range include padding : voxel_range = (mesh.bounding_box.extents + padding)/2
    obj coords -voxel_range   --->   voxel index (0, 0, 0)
    obj coords  voxel_range   --->   voxel index (R, R, R)
    :params obj_coordinate, [B, N_ray, N_points, 3]
    :params spacing, centroid, voxel_range, [B, 3]
    """
    # input is in model object coords
    # map mesh centroid to origin point
    obj_coordinate = obj_coordinate - centroid.unsqueeze(1).unsqueeze(1)
    # none equal scale to cube
    obj_coordinate = obj_coordinate * none_equal_scale.unsqueeze(1).unsqueeze(1)
    # voxel grid range from -voxel_range to voxel_range
    obj_coordinate = obj_coordinate + voxel_range.unsqueeze(1).unsqueeze(1)
    # map obj coordinate to voxel index
    voxel_index = obj_coordinate / spacing.unsqueeze(1).unsqueeze(1)

    return voxel_index


def voxel_index2obj_coordinate(voxel_index, centroid, voxel_range, spacing, none_equal_scale):
    """
    voxel index (a, b, c) to obj coordinate (x, y, z)
    voxel_range include padding : voxel_range = (mesh.bounding_box.extents + padding)/2
    obj coords -voxel_range   --->   voxel index (0, 0, 0)
    obj coords  voxel_range   --->   voxel index (R, R, R)
    :params voxel_index, [N, 3]
    """
    obj_coordinate = (voxel_index*spacing - voxel_range) / none_equal_scale + centroid

    return obj_coordinate           # return model object coords


def voxel_index2grid_sample(voxel_index, voxel_resolution):
    """
    for F.grid_sample, index range from -1 to 1    R: voxel_resolution
    voxel index (a, b, c) to grid sample (l, m, n)
    l = (a / voxel_resolution - 0.5) * 2
    m = (b / voxel_resolution - 0.5) * 2
    n = (c / voxel_resolution - 0.5) * 2

    voxel index (0, 0, 0) --->  grid sample (-1, -1, -1)
    voxel index (R, R, R) --->  grid sample (1, 1, 1)

    :params voxel_index, [B, Num_pixels, Num_points_a_ray, 3]
    """

    grid_sample = (voxel_index/voxel_resolution - 0.5) * 2

    # transpose x and z
    # NOTE: F.grid_sample need (z, y, x) order
    grid_sample = grid_sample[:, :, :, [2, 1, 0]]

    return grid_sample


def scene_obj2model_obj(scene_obj_coords, scene_scale):
    """
    scene coordinate to model coordinate
    model in different scene, may have different scene scale
    :params scene_obj_coords, [B*num_pixels*ray_points, 3]
    :params scene_scale, [B, 3]
    """
    batch_size = scene_scale.shape[0]
    scene_obj_coords = scene_obj_coords.reshape(batch_size, -1, 3)          # [B, num_pixels*ray_points, 3]
    model_obj_coords = scene_obj_coords / scene_scale.unsqueeze(1)          # [B, num_pixels*ray_points, 3]
    model_obj_coords = model_obj_coords.reshape(-1, 3)

    return model_obj_coords


def model_obj2scene_obj(model_obj_coords, scene_scale):
    """
    model coordinate to scene coordinate
    model in different scene, may have different scene scale
    :params model_obj_coords, [B*num_pixels*ray_points, 3]
    :params scene_scale, [B, 3]
    """
    batch_size = scene_scale.shape[0]
    model_obj_coords = model_obj_coords.reshape(batch_size, -1, 3)          # [B, num_pixels*ray_points, 3]
    scene_obj_coords = model_obj_coords * scene_scale.unsqueeze(1)          # [B, num_pixels*ray_points, 3]
    scene_obj_coords = scene_obj_coords.reshape(-1, 3)

    return scene_obj_coords


def model_obj2cube_coords(model_obj_coords, centroid, none_equal_scale):
    """
    obj coordinate (x, y, z) to voxel index (a, b, c)
    voxel_range include padding : voxel_range = (mesh.bounding_box.extents + padding)/2
    obj coords -voxel_range   --->   voxel index (0, 0, 0)
    obj coords  voxel_range   --->   voxel index (R, R, R)
    :params model_obj_coords, [B, N_ray*N_points, 3]
    :params centroid, none_equal_scale, [B, 3]
    """
    batch_size = none_equal_scale.shape[0]
    model_obj_coords = model_obj_coords.reshape(batch_size, -1, 3)          # [B, num_pixels*ray_points, 3]
    # map mesh centroid to origin point
    cube_coords = model_obj_coords - centroid.unsqueeze(1)
    # none equal scale to cube
    cube_coords = cube_coords * none_equal_scale.unsqueeze(1)
    cube_coords = cube_coords.reshape(-1, 3)

    return cube_coords


def scene_obj2cube_coords(scene_obj_coords, scene_scale, centroid, none_equal_scale):
    model_obj_coords = scene_obj2model_obj(scene_obj_coords, scene_scale)
    cube_coords = model_obj2cube_coords(model_obj_coords, centroid, none_equal_scale)

    return cube_coords


def world2obj(world_coords_points, world_to_obj):
    """
    from world coordinate to object coordinate (local coordinate)
    :params world_coords_points, points in world coordinate, [B, num_pixels, ray_points, 3]
    :params world_to_obj, object bdb_3d rotation matrix, [B, 4, 4]
    """
    B, num_pixels, ray_points, _ = world_coords_points.shape                    # [B, num_pixels, ray_points, 3]
    R = world_to_obj[:, 0:3, 0:3]                                               # [B, 3, 3]
    T = world_to_obj[:, 0:3, 3]                                                 # [B, 3]
    points_re = (world_coords_points.reshape(B, -1, 3)).permute(0, 2, 1)        # [B, 3, num_pixels*ray_points]
    obj_coords = (torch.bmm(R, points_re)).permute(0, 2, 1) + T.unsqueeze(1)    # [B, num_pixels*ray_points, 3]
    obj_coords = obj_coords.reshape(B, num_pixels, ray_points, 3)               # [B, num_pixels, ray_points, 3]

    return obj_coords


def obj2world(obj_coords, obj_rot, obj_tran):
    """
    from object coordinate to world coordinate
    :params obj_coords, points in object coordinate, [B, num_pixels, ray_points, 3]
    :params obj_rot, [B, 3, 3]
    :parmas obj_tran, [B, 3]
    """
    B, num_pixels, ray_points, _ = obj_coords.shape
    points_re = (obj_coords.reshape(B, -1, 3)).permute(0, 2, 1)                                 # [B, 3, num_pixels*ray_points]
    world_points = (torch.bmm(obj_rot, points_re)).permute(0, 2, 1) + obj_tran.unsqueeze(1)     # [B, num_pixels*ray_points, 3]
    world_points = world_points.reshape(B, num_pixels, ray_points, 3)                           # [B, num_pixels, ray_points, 3]

    return world_points


def camera2obj(camera_coords_points, pose, world_to_obj):
    """
    from camera coords to obj coords
    :params camera_coords_points, [B, N, M, 3] (in use, N is ray number, M is points on a ray)
    :params pose, [B, 4, 4]
    :params world_to_obj, [B, 4, 4], world to object matrix
    """

    points_world = rend_util.camera_to_world(camera_coords_points, pose)                        # [B, N, M, 3]
    points_obj = world2obj(points_world, world_to_obj)       # [B, N, M, 3]

    return points_obj


def obj2world_numpy(obj_coords, obj_rot, obj_tran):
    """
    from object coordinate to world coordinate
    for use in data loader, in numpy
    :params obj_coords, points in object coordinate, [N, 3]
    :params obj_rot, [3, 3]
    :parmas obj_tran, [3,]
    """

    obj_coords = obj_coords.transpose(1, 0)                                     # [3, N]
    world_points = (obj_rot @ obj_coords).transpose(1, 0) + obj_tran            # [N, 3]

    return world_points


def get_sample_sdf(voxels_sdf, point_index):

    samples = F.grid_sample(
        voxels_sdf,                         # (N, C, D_in, H_in, W_in)          d->x, h->y, w->z
        point_index,                        # (N, D_out, H_out, W_out, 3)       range from -1 to 1
        align_corners=True,                 # feature align with index corner
        mode='bilinear',                    # when input is 3D-grid, 'bilinear' is exactly trilinear
        padding_mode='border',              # use border value padding
    )

    return samples                          # (N, C, D_out, H_out, W_out)


def get_sdf_gt_objcoords(points, ground_truth):
    """
    get gt sdf
    :params points, [B, Num_pixels, Num_points_a_ray, 3], points in object coords
    :params ground_truth, list of gt
    """
    voxel_sdf_gt = ground_truth['voxel_sdf'].cuda()             # (B, 1, R, R, R)
    voxel_resolution = voxel_sdf_gt.shape[-1]                   # R: voxel_resolution
    centroid = ground_truth['centroid'].cuda()                  # (B, 3)
    voxel_range = ground_truth['voxel_range'].cuda()            # (B, 3)
    spacing = ground_truth['voxel_spacing'].cuda()              # (B, 3)
    none_equal_scale = ground_truth['none_equal_scale'].cuda()  # (B, 3)
    scene_scale = ground_truth['scene_scale'].cuda()            # (B, 3)

    batch_size, num_pixels, num_points_a_ray, _ = points.shape
    # transfer to model object coords
    model_obj = scene_obj2model_obj(points.reshape(-1, 3), scene_scale)     # [B*Num_pixels*Num_points_a_ray, 3]
    model_obj = model_obj.reshape(batch_size, num_pixels, num_points_a_ray, 3)

    # map points to voxels index
    voxel_index = obj_coordinate2voxel_index(model_obj, centroid, voxel_range, spacing, none_equal_scale)        # [B, Num_pixels, Num_points_a_ray, 3]
    # map voxel index to grid sample
    grid_sample = voxel_index2grid_sample(voxel_index, voxel_resolution)                    # [B, Num_pixels, Num_points_a_ray, 3], NOTE: F.grid_sample need (z, y, x) order
    grid_sample = grid_sample.unsqueeze(1).to(torch.float32)                                # [B, 1, Num_pixels, Num_points_a_ray, 3]
    # grid sample to sdf
    sdf_gt = get_sample_sdf(voxel_sdf_gt, grid_sample)                                      # [B, C, 1, Num_pixels, Num_points_a_ray]  C=1
    sdf_gt = sdf_gt.reshape(-1, 1)                                                          # [N, 1]

    return sdf_gt


def get_sdf_gt_worldcoords(points, ground_truth):
    """
    get gt sdf
    :params points, [B, Num_pixels, Num_points_a_ray, 3], points in world coords
    :params ground_truth, list of gt
    """
    if len(points.shape) == 2:                    # inference shape [N, 3]
        points = points[None, None, ...]            # [1, 1, N, 3]

    world_to_obj = ground_truth['world_to_obj'].float().cuda()
    obj_coords = world2obj(points, world_to_obj)
    sdf_gt = get_sdf_gt_objcoords(obj_coords, ground_truth)

    return sdf_gt


def vis_sdf(points_flat, points_sdf, mode):
    """
    visual sdf during train
    :params points_flat, [N, 3], points in world coords
    :params points_sdf, [N, 1]
    :params mode, 'all points', 'only pos', 'only neg'
    """
    import pyrender
    import copy
    points = copy.deepcopy(points_flat.cpu().detach().numpy())
    sdf = copy.deepcopy(points_sdf.cpu().detach().numpy())
    sdf = sdf.reshape(-1)

    colors = np.zeros(points.shape)     # [red, green, blue]

    if mode == 'all points':
        ######## all points
        colors[sdf < 0, 2] = 1              # blue
        colors[sdf > 0, 0] = 1              # red

    elif mode == 'only pos':
        ######### only sdf>0 points
        colors[sdf > 0, 0] = 1              # red
        colors = colors[sdf > 0]
        points = points[sdf > 0]

    elif mode == 'only neg':
        ######### only sdf<0 points
        colors[sdf < 0, 2] = 1              # blue
        colors = colors[sdf < 0]
        points = points[sdf < 0]

    else:
        raise ValueError(f'mode {mode} not support!')

    cloud = pyrender.Mesh.from_points(points, colors=colors)
    scene = pyrender.Scene()
    scene.add(cloud)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=7)
