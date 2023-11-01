import numpy as np
import torch
from skimage import measure
import torchvision
import trimesh
from PIL import Image

from utils import rend_util
from utils.sdf_util import *


avg_pool_3d = torch.nn.AvgPool3d(2, stride=2)
upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')


@torch.no_grad()
def get_surface_sliding(path, epoch, model, img, intrinsics, extrinsics, model_input, ground_truth, resolution=512, grid_boundary=[-2.0, 2.0], return_mesh=False, delta=0, level=0, eval_gt=False, export_color_mesh=False):
    assert resolution % 256 == 0

    model.encoder(img)                           # img: (B, C, H, W)
    image_shape = torch.empty(2).cuda()          # (W, H)
    image_shape[0] = img.shape[-1]               # W
    image_shape[1] = img.shape[-2]               # H

    batch_size = img.shape[0]

    resN = resolution
    cropN = 256
    level = 0.0
    N = resN // cropN

    # grid_min = grid_boundary.min(dim=1)[0]          # .min -> (values, indices)   .min[0] -> values
    # grid_max = grid_boundary.max(dim=1)[0]
    grid_min = np.array([-1, -1, -1])
    grid_max = np.array([1, 1, 1])
    xs = np.linspace(grid_min[0]-delta, grid_max[0]+delta, N+1)
    ys = np.linspace(grid_min[1]-delta, grid_max[1]+delta, N+1)
    zs = np.linspace(grid_min[2]-delta, grid_max[2]+delta, N+1)

    # for evaluation, align InstPIFu size
    bbox_scale_value = 2.0 / (2.0 - ground_truth['voxel_padding'][0])


    meshes = []
    for i in range(N):
        for j in range(N):
            for k in range(N):

                x_min, x_max = xs[i], xs[i+1]
                y_min, y_max = ys[j], ys[j+1]
                z_min, z_max = zs[k], zs[k+1]

                x = np.linspace(x_min, x_max, cropN)
                y = np.linspace(y_min, y_max, cropN)
                z = np.linspace(z_min, z_max, cropN)

                xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
                points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda().to(torch.float32)          # in cube coords

                def evaluate(points):
                    z = []
                    for _, pnts in enumerate(torch.split(points, 100000, dim=0)):
                        # get model object coords
                        model_obj = pnts / model_input['none_equal_scale'] + model_input['centroid']
                        scene_obj = model_obj * model_input['scene_scale']                                                          # [N, 3]
                        scene_obj = scene_obj[None, None, ...]                                                                      # [1, 1, N, 3]
                        world_coords = obj2world(scene_obj, model_input['obj_rot'], model_input['obj_tran'])                        # [1, 1, N, 3]

                        latent_feature, cat_feature = rend_util.get_latent_feature(model, world_coords.reshape(-1, 3), intrinsics, extrinsics, model_input)

                        sdf = model.implicit_network(pnts, latent_feature, cat_feature)[:, 0]
                        z.append(sdf)
                    z = torch.cat(z, axis=0)
                    return z

                def evaluate_gt(points, ground_truth):
                    model_obj = points / model_input['none_equal_scale'] + model_input['centroid']
                    scene_obj = model_obj * model_input['scene_scale']                                                          # [N, 3]
                    scene_obj = scene_obj[None, None, ...]                                                                      # [1, 1, N, 3]
                    world_coords = obj2world(scene_obj, model_input['obj_rot'], model_input['obj_tran'])                        # [1, 1, N, 3]

                    sdf_gt = get_sdf_gt_worldcoords(world_coords, ground_truth)

                    z = sdf_gt.squeeze(1)
                    return z

                # construct point pyramids
                points = points.reshape(cropN, cropN, cropN, 3).permute(3, 0, 1, 2)
                points_pyramid = [points]
                for _ in range(3):            
                    points = avg_pool_3d(points[None])[0]
                    points_pyramid.append(points)
                points_pyramid = points_pyramid[::-1]

                # evalute pyramid with mask
                mask = None
                threshold = 2 * (x_max - x_min)/cropN * 8
                for pid, pts in enumerate(points_pyramid):
                    coarse_N = pts.shape[-1]
                    pts = pts.reshape(3, -1).permute(1, 0).contiguous()
                    
                    if mask is None:    
                        if eval_gt:
                            pts_sdf = evaluate_gt(pts, ground_truth)
                        else:
                            pts_sdf = evaluate(pts)
                    else:                    
                        mask = mask.reshape(-1)
                        pts_to_eval = pts[mask]
                        #import pdb; pdb.set_trace()
                        if pts_to_eval.shape[0] > 0:
                            if eval_gt:
                                pts_sdf_eval = evaluate_gt(pts_to_eval.contiguous(), ground_truth)
                            else:
                                pts_sdf_eval = evaluate(pts_to_eval.contiguous())
                            pts_sdf[mask] = pts_sdf_eval
                        # print("ratio", pts_to_eval.shape[0] / pts.shape[0])

                    if pid < 3:
                        # update mask
                        mask = torch.abs(pts_sdf) < threshold
                        mask = mask.reshape(coarse_N, coarse_N, coarse_N)[None, None]
                        mask = upsample(mask.float()).bool()

                        pts_sdf = pts_sdf.reshape(coarse_N, coarse_N, coarse_N)[None, None]
                        pts_sdf = upsample(pts_sdf)
                        pts_sdf = pts_sdf.reshape(-1)               # [N, ]

                    threshold /= 2.


                z = pts_sdf.detach().cpu().numpy()                  # [N,]

                if (not (np.min(z) > level or np.max(z) < level)):
                    z = z.astype(np.float32)

                    verts, faces, normals, values = measure.marching_cubes(
                        volume=z.reshape(cropN, cropN, cropN), #.transpose([1, 0, 2]),
                        level=level,
                        spacing=(
                                (x_max - x_min)/(cropN-1),
                                (y_max - y_min)/(cropN-1),
                                (z_max - z_min)/(cropN-1)))

                    # print(np.array([x_min, y_min, z_min]))
                    # print(verts.min(), verts.max())
                    verts = verts + np.array([x_min, y_min, z_min])     # in cube coords
                    
                    if not export_color_mesh:
                        # for evaluation
                        verts = verts * bbox_scale_value.detach().cpu().numpy()
                    
                    meshcrop = trimesh.Trimesh(verts, faces, normals)

                    #meshcrop.export(f"{i}_{j}_{k}.ply")
                    meshes.append(meshcrop)

    combined = trimesh.util.concatenate(meshes)

    if return_mesh:
        return combined
    else:
        combined.export('{0}/surface_{1}.ply'.format(path, epoch), 'ply')    
        

def plot_normal_maps(normal_maps, ground_true, path, epoch, img_res, indices, ray_mask):

    normal_maps = (normal_maps[0].view(img_res[0], img_res[1], -1)).cpu().detach().numpy()
    ray_mask_map = (ray_mask[0].view(img_res[0], img_res[1], -1)).cpu().detach().numpy()

    normal_maps = (normal_maps * 255).astype(np.uint8)
    normal_maps_temp = Image.fromarray(normal_maps)
    normal_maps = normal_maps_temp.convert('RGBA')

    # ray_mask_map = Image.fromarray(ray_mask_map[:, :, 0].astype(np.uint8) * 255)
    ray_mask_map = Image.fromarray((ray_mask_map[:, :, 0] * 255).astype(np.uint8))
    ray_mask_map = ray_mask_map.convert('L')
    normal_maps.putalpha(ray_mask_map)

    ground_true = ground_true.cuda()            # [B, N, 3]

    normal_maps_plot = lin2img(ground_true, img_res)

    tensor = torchvision.utils.make_grid(normal_maps_plot,
                                         scale_each=False,
                                         normalize=False).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    # tensor[971:1171,357:450] = [0,0,0]

    img = Image.fromarray(tensor)
    return img, normal_maps


def plot_images(rgb_points, ground_true, path, epoch, img_res, indices, exposure=False, ray_mask=None):

    rgb_map = (rgb_points[0].view(img_res[0], img_res[1], -1)).cpu().detach().numpy()
    ray_mask_map = (ray_mask[0].view(img_res[0], img_res[1], -1)).cpu().detach().numpy()

    rgb_map = (rgb_map * 255).astype(np.uint8)
    rgb_map_temp = Image.fromarray(rgb_map)
    rgb_map = rgb_map_temp.convert('RGBA')

    # ray_mask_map = Image.fromarray(ray_mask_map[:, :, 0].astype(np.uint8) * 255)
    ray_mask_map = Image.fromarray((ray_mask_map[:, :, 0] * 255).astype(np.uint8))
    ray_mask_map = ray_mask_map.convert('L')
    rgb_map.putalpha(ray_mask_map)


    ground_true = ground_true.cuda()
    ground_true = lin2img(ground_true, img_res)

    tensor = torchvision.utils.make_grid(ground_true,
                                         scale_each=False,
                                         normalize=False,).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    return img, rgb_map, rgb_map_temp


def plot_depth_maps(depth_maps, ground_true, path, epoch, img_res, indices, ray_mask):

    depth_maps = (depth_maps[0].view(img_res[0], img_res[1])).cpu().detach().numpy()
    ray_mask_map = (ray_mask[0].view(img_res[0], img_res[1], -1)).cpu().detach().numpy()

    # depth normalize
    max_depth = np.max(depth_maps)
    depth_maps = depth_maps / max_depth

    depth_maps = (depth_maps * 150).astype(np.uint8)
    depth_maps_temp = Image.fromarray(depth_maps)
    depth_maps = depth_maps_temp.convert('RGB')

    ray_mask_map = Image.fromarray((ray_mask_map[:, :, 0] * 255).astype(np.uint8))
    ray_mask_map = ray_mask_map.convert('L')
    depth_maps.putalpha(ray_mask_map)


    ground_true = ground_true.numpy()

    # depth normalize
    max_depth = np.max(ground_true)
    ground_true = ground_true / max_depth

    ground_true = ground_true[0].reshape(img_res[0], img_res[1])
    ground_true = (ground_true * 150).astype(np.uint8)
    ground_true = Image.fromarray(ground_true)
    ground_true = ground_true.convert('RGB')


    return ground_true, depth_maps


def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])


def split_input(model_input, total_pixels, n_pixels=10000):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of n_pixels in case of cuda out of memory error.
     '''
    split = []
    for i, indx in enumerate(torch.split(torch.arange(total_pixels).cuda(), n_pixels, dim=0)):
        data = model_input.copy()
        data['uv'] = torch.index_select(model_input['uv'], 1, indx)
        if 'object_mask' in data:
            data['object_mask'] = torch.index_select(model_input['object_mask'], 1, indx)
        if 'depth' in data:
            data['depth'] = torch.index_select(model_input['depth'], 1, indx)
        split.append(data)
    return split


def merge_output(res, total_pixels, batch_size):
    ''' Merge the split output. '''
    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, 1) for r in res],
                                             1).reshape(batch_size * total_pixels)
        else:
            model_outputs[entry] = torch.cat([r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res],
                                             1).reshape(batch_size * total_pixels, -1)

    return model_outputs
