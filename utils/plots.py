import numpy as np
import torch
from skimage import measure
import torchvision
import trimesh
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from utils import rend_util
from utils.sdf_util import *


def plot(implicit_network, indices, plot_data, path, epoch, img_res, plot_nimgs, resolution, grid_boundary,  level=0):

    if plot_data is not None:
        cam_loc, cam_dir = rend_util.get_camera_for_plot(plot_data['pose'])

        # plot images
        plot_images(plot_data['rgb_eval'], plot_data['rgb_gt'], path, epoch, plot_nimgs, img_res, indices)

        # plot normal maps
        plot_normal_maps(plot_data['normal_map'], plot_data['normal_gt'], path, epoch, plot_nimgs, img_res, indices)

        # plot depth maps
        plot_depth_maps(plot_data['depth_map'], plot_data['depth_gt'], path, epoch, plot_nimgs, img_res, indices)

        # concat output images to single large image
        images = []
        for name in ["rendering", "depth", "normal"]:
            images.append(cv2.imread('{0}/{1}_{2}_{3}.png'.format(path, name, epoch, indices[0])))        

        images = np.concatenate(images, axis=1)
        cv2.imwrite('{0}/merge_{1}_{2}.png'.format(path, epoch, indices[0]), images)

    surface_traces = get_surface_sliding(path=path,
                                         epoch=epoch,
                                         sdf=lambda x: implicit_network(x)[:, 0],
                                         resolution=resolution,
                                         grid_boundary=grid_boundary,
                                         level=level
                                         )

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
        

def get_3D_scatter_trace(points, name='', size=3, caption=None):
    assert points.shape[1] == 3, "3d scatter plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d scatter plot input points are not correctely shaped "

    trace = go.Scatter3d(
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        mode='markers',
        name=name,
        marker=dict(
            size=size,
            line=dict(
                width=2,
            ),
            opacity=1.0,
        ), text=caption)

    return trace


def get_3D_quiver_trace(points, directions, color='#bd1540', name=''):
    assert points.shape[1] == 3, "3d cone plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d cone plot input points are not correctely shaped "
    assert directions.shape[1] == 3, "3d cone plot input directions are not correctely shaped "
    assert len(directions.shape) == 2, "3d cone plot input directions are not correctely shaped "

    trace = go.Cone(
        name=name,
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        u=directions[:, 0].cpu(),
        v=directions[:, 1].cpu(),
        w=directions[:, 2].cpu(),
        sizemode='absolute',
        sizeref=0.125,
        showscale=False,
        colorscale=[[0, color], [1, color]],
        anchor="tail"
    )

    return trace


def get_surface_trace(path, epoch, sdf, resolution=100, grid_boundary=[-2.0, 2.0], return_mesh=False, level=0):
    grid = get_grid_uniform(resolution, grid_boundary)
    points = grid['grid_points']

    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts.cuda()).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    if (not (np.min(z) > level or np.max(z) < level)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                             grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=level,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1]))

        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])
        '''
        I, J, K = faces.transpose()

        traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, name='implicit_surface',
                            color='#ffffff', opacity=1.0, flatshading=False,
                            lighting=dict(diffuse=1, ambient=0, specular=0),
                            lightposition=dict(x=0, y=0, z=-1), showlegend=True)]
        '''
        meshexport = trimesh.Trimesh(verts, faces, normals)
        meshexport.export('{0}/surface_{1}.ply'.format(path, epoch), 'ply')

        if return_mesh:
            return meshexport
        #return traces
    return None


def get_surface_high_res_mesh(sdf, resolution=100, grid_boundary=[-2.0, 2.0], level=0, take_components=True):
    # get low res mesh to sample point cloud
    grid = get_grid_uniform(100, grid_boundary)
    z = []
    points = grid['grid_points']

    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    z = z.astype(np.float32)

    verts, faces, normals, values = measure.marching_cubes(
        volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                         grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
        level=level,
        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1]))

    verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

    mesh_low_res = trimesh.Trimesh(verts, faces, normals)
    if take_components:
        components = mesh_low_res.split(only_watertight=False)
        areas = np.array([c.area for c in components], dtype=np.float)
        mesh_low_res = components[areas.argmax()]

    recon_pc = trimesh.sample.sample_surface(mesh_low_res, 10000)[0]
    recon_pc = torch.from_numpy(recon_pc).float().cuda()

    # Center and align the recon pc
    s_mean = recon_pc.mean(dim=0)
    s_cov = recon_pc - s_mean
    s_cov = torch.mm(s_cov.transpose(0, 1), s_cov)
    vecs = torch.view_as_real(torch.linalg.eig(s_cov)[1].transpose(0, 1))[:, :, 0]
    if torch.det(vecs) < 0:
        vecs = torch.mm(torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]]).cuda().float(), vecs)
    helper = torch.bmm(vecs.unsqueeze(0).repeat(recon_pc.shape[0], 1, 1),
                       (recon_pc - s_mean).unsqueeze(-1)).squeeze()

    grid_aligned = get_grid(helper.cpu(), resolution)

    grid_points = grid_aligned['grid_points']

    g = []
    for i, pnts in enumerate(torch.split(grid_points, 100000, dim=0)):
        g.append(torch.bmm(vecs.unsqueeze(0).repeat(pnts.shape[0], 1, 1).transpose(1, 2),
                           pnts.unsqueeze(-1)).squeeze() + s_mean)
    grid_points = torch.cat(g, dim=0)

    # MC to new grid
    points = grid_points
    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    meshexport = None
    if (not (np.min(z) > level or np.max(z) < level)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid_aligned['xyz'][1].shape[0], grid_aligned['xyz'][0].shape[0],
                             grid_aligned['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=level,
            spacing=(grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1]))

        verts = torch.from_numpy(verts).cuda().float()
        verts = torch.bmm(vecs.unsqueeze(0).repeat(verts.shape[0], 1, 1).transpose(1, 2),
                   verts.unsqueeze(-1)).squeeze()
        verts = (verts + grid_points[0]).cpu().numpy()

        meshexport = trimesh.Trimesh(verts, faces, normals)

    return meshexport


def get_surface_by_grid(grid_params, sdf, resolution=100, level=0, higher_res=False):
    grid_params = grid_params * [[1.5], [1.0]]

    # params = PLOT_DICT[scan_id]
    input_min = torch.tensor(grid_params[0]).float()
    input_max = torch.tensor(grid_params[1]).float()

    if higher_res:
        # get low res mesh to sample point cloud
        grid = get_grid(None, 100, input_min=input_min, input_max=input_max, eps=0.0)
        z = []
        points = grid['grid_points']

        for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
            z.append(sdf(pnts).detach().cpu().numpy())
        z = np.concatenate(z, axis=0)

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                             grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=level,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1]))

        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

        mesh_low_res = trimesh.Trimesh(verts, faces, normals)
        components = mesh_low_res.split(only_watertight=False)
        areas = np.array([c.area for c in components], dtype=np.float)
        mesh_low_res = components[areas.argmax()]

        recon_pc = trimesh.sample.sample_surface(mesh_low_res, 10000)[0]
        recon_pc = torch.from_numpy(recon_pc).float().cuda()

        # Center and align the recon pc
        s_mean = recon_pc.mean(dim=0)
        s_cov = recon_pc - s_mean
        s_cov = torch.mm(s_cov.transpose(0, 1), s_cov)
        vecs = torch.view_as_real(torch.linalg.eig(s_cov)[1].transpose(0, 1))[:, :, 0]
        if torch.det(vecs) < 0:
            vecs = torch.mm(torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]]).cuda().float(), vecs)
        helper = torch.bmm(vecs.unsqueeze(0).repeat(recon_pc.shape[0], 1, 1),
                           (recon_pc - s_mean).unsqueeze(-1)).squeeze()

        grid_aligned = get_grid(helper.cpu(), resolution, eps=0.01)
    else:
        grid_aligned = get_grid(None, resolution, input_min=input_min, input_max=input_max, eps=0.0)

    grid_points = grid_aligned['grid_points']

    if higher_res:
        g = []
        for i, pnts in enumerate(torch.split(grid_points, 100000, dim=0)):
            g.append(torch.bmm(vecs.unsqueeze(0).repeat(pnts.shape[0], 1, 1).transpose(1, 2),
                               pnts.unsqueeze(-1)).squeeze() + s_mean)
        grid_points = torch.cat(g, dim=0)

    # MC to new grid
    points = grid_points
    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    meshexport = None
    if (not (np.min(z) > level or np.max(z) < level)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid_aligned['xyz'][1].shape[0], grid_aligned['xyz'][0].shape[0],
                             grid_aligned['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=level,
            spacing=(grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1]))

        if higher_res:
            verts = torch.from_numpy(verts).cuda().float()
            verts = torch.bmm(vecs.unsqueeze(0).repeat(verts.shape[0], 1, 1).transpose(1, 2),
                       verts.unsqueeze(-1)).squeeze()
            verts = (verts + grid_points[0]).cpu().numpy()
        else:
            verts = verts + np.array([grid_aligned['xyz'][0][0], grid_aligned['xyz'][1][0], grid_aligned['xyz'][2][0]])

        meshexport = trimesh.Trimesh(verts, faces, normals)

        # CUTTING MESH ACCORDING TO THE BOUNDING BOX
        if higher_res:
            bb = grid_params
            transformation = np.eye(4)
            transformation[:3, 3] = (bb[1,:] + bb[0,:])/2.
            bounding_box = trimesh.creation.box(extents=bb[1,:] - bb[0,:], transform=transformation)

            meshexport = meshexport.slice_plane(bounding_box.facets_origin, -bounding_box.facets_normal)

    return meshexport

def get_grid_uniform(resolution, grid_boundary=[-2.0, 2.0]):
    x = np.linspace(grid_boundary[0], grid_boundary[1], resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    return {"grid_points": grid_points,
            "shortest_axis_length": 2.0,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}

def get_grid(points, resolution, input_min=None, input_max=None, eps=0.1):
    if input_min is None or input_max is None:
        input_min = torch.min(points, dim=0)[0].squeeze().numpy()
        input_max = torch.max(points, dim=0)[0].squeeze().numpy()

    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)
    if (shortest_axis == 0):
        x = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(input_min[1] - eps, input_max[1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(input_min[1] - eps, input_max[1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()
    return {"grid_points": grid_points,
            "shortest_axis_length": length,
            "xyz": [x, y, z],
            "shortest_axis_index": shortest_axis}


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
