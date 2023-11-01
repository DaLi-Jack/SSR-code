import os

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import trimesh

from .resnet import resnet18_small_stride
from .encoder import make_encoder
from .mlp import ImplicitNetwork, RenderingNetwork
from .density import LaplaceDensity
from .ray_sampler import ErrorBoundSampler, UniformSampler
from utils import rend_util, sdf_util
from ssr.ssr_utils.utils import repeat_interleave
from ssr.model.Attention_module import Attention_RoI_Module


class SSRNet(torch.nn.Module):                                                 # modify from MonoSDFNetwork
    def __init__(self, conf):
        """
        :param conf PyHocon config subtree 'model'
        """
        super().__init__()
        self.config = conf
        self.show_rendering = conf['show_rendering']                                # whether render novel view images
        self.feature_vector_size = conf['model']['feature_vector_size']
        self.scene_bounding_sphere = conf['model']['scene_bounding_sphere']
        self.white_bkgd = conf['model']['white_bkgd']
        self.depth_norm = conf['model']['depth_norm']
        self.ray_noise = conf['model']['ray_noise']

        if 'fusion_scene' in conf['eval']:
            self.fusion_scene = conf['eval']['fusion_scene']
        else:
            self.fusion_scene = False

        self.bg_color = torch.tensor(conf['model']['bg_color']).float().cuda()

        self.encoder = make_encoder(conf['model']['latent_feature']['encoder'])
        self.use_encoder = conf['model']['latent_feature']['use_encoder']           # whether use image features

        self.stop_encoder_grad = conf['model']['stop_encoder_grad']                 # Stop ConvNet gradient (freeze weights)

        # use num_layers calculate latent_size, i.e. latent_size=[0, 64, 128, 256, 512, 1024][num_layers]
        d_latent = self.encoder.latent_size if self.use_encoder else 0
        
        # use object bdb2d global image features
        self.use_global_encoder = conf['model']['latent_feature']['use_global_encoder']

        if self.use_global_encoder:
            # object bdb2d global image feature
            self.global_encoder = nn.Sequential(
                nn.Conv2d(in_channels=256,out_channels=64,kernel_size=3,padding=1),
                resnet18_small_stride(pretrained=False,input_channels=64),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256)
            )
        
        self.use_cls_encoder = conf['model']['latent_feature']['use_cls_encoder']

        self.latent_size = conf['model']['latent_feature']['latent_feature_dim']
        if d_latent != self.latent_size:
            raise ValueError('d_latent must be similar to latent_feature_dim !')    # maybe should modify num_layers

        # whether use attention filter
        if 'use_atten' in conf['model']:
            self.use_atten = conf['model']['use_atten']
        else:
            self.use_atten = False
        if self.use_atten:
            self.post_op=Attention_RoI_Module(img_feat_channel=256, global_dim=256+9)

        conf_implicit = conf['model']['implicit_network']
        self.implicit_network = ImplicitNetwork(
            config=conf,                                    feature_vector_size=self.feature_vector_size,                   
            sdf_bounding_sphere=0.0 if self.white_bkgd else self.scene_bounding_sphere, 
            d_in=conf_implicit['d_in'],                     d_out=conf_implicit['d_out'],
            dims=conf_implicit['dims'],                     geometric_init=conf_implicit['geometric_init'],
            bias=conf_implicit['bias'],                     skip_in=conf_implicit['skip_in'],
            weight_norm=conf_implicit['weight_norm'],       multires=conf_implicit['multires'],
            sphere_scale=conf_implicit['sphere_scale'],     inside_outside=conf_implicit['inside_outside']
        )

        conf_rendering = conf['model']['rendering_network']
        self.rendering_network = RenderingNetwork(
            feature_vector_size=self.feature_vector_size,   mode=conf_rendering['mode'],
            d_in=conf_rendering['d_in'],                    d_out=conf_rendering['d_out'],
            dims=conf_rendering['dims'],                    weight_norm=conf_rendering['weight_norm'],
            multires_view=conf_rendering['multires_view'],  per_image_code=conf_rendering['per_image_code']
        )                                                

        self.density = LaplaceDensity(conf['model']['density']['params_init'], conf['model']['density']['beta_min'])
        
        self.sampling_method = conf['model']['sampling_method']
        conf_ray_sampler = conf['model']['ray_sampler']
        self.take_sphere_intersection = conf_ray_sampler['take_sphere_intersection']
        self.add_bdb3d_points = conf_ray_sampler['add_bdb3d_points']
        
        self.sample_near = conf_ray_sampler['near']
        self.sample_far = conf_ray_sampler['far']

        if self.sampling_method == "errorbounded":
            self.ray_sampler = ErrorBoundSampler(
                scene_bounding_sphere=self.scene_bounding_sphere,       near=conf_ray_sampler['near'],
                N_samples=conf_ray_sampler['N_samples'],                N_samples_eval=conf_ray_sampler['N_samples_eval'],
                N_samples_extra=conf_ray_sampler['N_samples_extra'],    eps=conf_ray_sampler['eps'],
                beta_iters=conf_ray_sampler['beta_iters'],              max_total_iters=conf_ray_sampler['max_total_iters'],
                take_sphere_intersection=self.take_sphere_intersection, far=conf_ray_sampler['far'], 
                encoder=self.encoder,
            )
        elif self.sampling_method == "uniform":
            self.ray_sampler = UniformSampler(
                scene_bounding_sphere=self.scene_bounding_sphere, 
                near=conf_ray_sampler['near'], 
                N_samples=conf_ray_sampler['N_samples'], 
                take_sphere_intersection=self.take_sphere_intersection,
                far=conf_ray_sampler['far'], 
            )
        else:
            raise ValueError('only support errorbounded or uniform!')

        self.use_instance_mask = conf['data']['use_instance_mask']
        if self.use_instance_mask:
            self.mask_decoder=nn.Sequential(
                nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=256,out_channels=128,kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
                nn.Sigmoid()
            )

    
    def get_feature(self, input, uv, z_vals_pnts):
        image = input["image"]                              # [B, 3, H, W]
        self.image_shape = torch.empty(2).cuda()
        self.image_shape[0] = image.shape[-1]               # W
        self.image_shape[1] = image.shape[-2]               # H

        cat_feature = None

        if self.use_global_encoder:
            bdb_grid = input["bdb_grid"].cuda().to(torch.float32)                                               # [B, 64, 64, 2]
            bdb_roi_feature = F.grid_sample(self.encoder.latent, bdb_grid, align_corners=True, mode='bilinear')     # [B, latent_size, 64, 64]
            global_latent = self.global_encoder(bdb_roi_feature)                        # [B, 256]

            cat_feature = global_latent

        if self.use_cls_encoder:
            cls_encoder = input['cls_encoder'].cuda().to(torch.float32)             # [B, 9]
            
            if cat_feature is None:
                cat_feature = cls_encoder
            else:
                cat_feature = torch.cat([cat_feature, cls_encoder], dim=1)          # [B, +9]


        roi_feat = None
        if self.use_atten:
            assert cat_feature.shape[1] == 256 + 9                          # must global_latent + cls_encoder
            ret_dict=self.post_op(self.encoder.latent, cat_feature, bdb_grid)
            roi_feat=ret_dict["roi_feat"]

        if self.use_encoder:
            latent = self.encoder.index(
                uv, None, self.image_shape, roi_feat=roi_feat
            )  # (B, latent_size, N_ray)
            if self.stop_encoder_grad:
                latent = latent.detach()
            latent = latent.transpose(1, 2).reshape(
                -1, self.latent_size
            )  # (B * N_ray, latent_size)
            latent_feature = latent                              ###### now, default use_encoder is True

        return latent_feature, cat_feature


    def forward(self, input, indices, new_pose=None, mesh=None, mesh_coords='canonical'):

        intrinsics = input["intrinsics"]
        image = input["image"]                              # [B, 3, H, W]
        uv = input["uv"]
        pose = input["pose"]                                # camera to world
        extrinsics = input["extrinsics"]                    # world to camera
        obj_rot = input['obj_rot']
        obj_tran = input['obj_tran']
        world_to_obj = input['world_to_obj']

        # if get mesh, predict every point color for colormesh
        if mesh != None:
            batch_size = image.shape[0]
            verts = mesh.vertices
            faces = mesh.faces
            normals = mesh.face_normals
            verts = torch.from_numpy(verts).cuda().to(torch.float32)

            verts_rgb = []
            pnts_obj_list = []
            pnts_world_list = []
            pnts_camera_list = []

            for _, pnts in enumerate(torch.split(verts, 100000, dim=0)):
                # get obj points
                model_obj = pnts / input['none_equal_scale'] + input['centroid']                                            
                scene_obj = model_obj * input['scene_scale']                                                                # [N, 3]
                scene_obj = scene_obj[None, None, ...]                                                                      # [1, 1, N, 3]
                world_coords = sdf_util.obj2world(scene_obj, input['obj_rot'], input['obj_tran'])                        # [1, 1, N, 3]

                pnts_camera = rend_util.world_to_camera(world_coords.reshape(-1, 3), extrinsics)                          # [1, 3, N]
                pnts_camera_list.append(pnts_camera.permute(0, 2, 1).reshape(-1, 3).detach().cpu().numpy())
                pnts_world_list.append(world_coords.reshape(-1, 3).detach().cpu().numpy())

                latent_feature, cat_feature = rend_util.get_latent_feature(self, world_coords.reshape(-1, 3), intrinsics, extrinsics, input)

                # get obj dirs
                cam_loc_incam = torch.tensor([0, 0, 0]).cuda().float()
                cam_loc_temp = cam_loc_incam[None, None, None, :]                                                   # [1, 1, 1, 3]
                cam_loc_obj = sdf_util.camera2obj(cam_loc_temp, input['pose'], input['world_to_obj'])               # [1, 1, 1, 3]
                cam_loc_obj = cam_loc_obj.squeeze(0).squeeze(0)                 # [1, 3]
                ray_dirs_obj = scene_obj.reshape(-1, 3) - cam_loc_obj           # [N, 3]
                ray_dirs_obj = F.normalize(ray_dirs_obj, dim=1)                 # [N, 3]

                sdf, feature_vectors, gradients = self.implicit_network.get_outputs(pnts.reshape(-1, 3), latent_feature, cat_feature)
                
                # color use scene obj
                rgb_flat = self.rendering_network(scene_obj.reshape(-1, 3), gradients, ray_dirs_obj, feature_vectors, indices=None)

                verts_rgb.append(rgb_flat.detach().cpu().numpy())
                pnts_obj_list.append(scene_obj.reshape(-1, 3).detach().cpu().numpy())

            verts = verts.detach().cpu().numpy()

            vertex_colors = np.concatenate(verts_rgb, axis=0)
            verts_obj = np.concatenate(pnts_obj_list, axis=0)
            verts_world = np.concatenate(pnts_world_list, axis=0)
            verts_camera = np.concatenate(pnts_camera_list, axis=0)

            if mesh_coords == 'canonical':
                # in obj coordinate
                meshcolor = trimesh.Trimesh(verts_obj, faces, normals, vertex_colors=vertex_colors)
                meshnonecolor = trimesh.Trimesh(verts_obj, faces, normals)
            
            elif mesh_coords == 'camera':
                # in camera coordinate
                meshcolor = trimesh.Trimesh(verts_camera, faces, normals, vertex_colors=vertex_colors)
                meshnonecolor = trimesh.Trimesh(verts_camera, faces, normals)

            else:
                # in world coordinate
                meshcolor = trimesh.Trimesh(verts_world, faces, normals, vertex_colors=vertex_colors)
                meshnonecolor = trimesh.Trimesh(verts_world, faces, normals)

            return meshcolor, meshnonecolor


        batch_size, num_pixels, _ = uv.shape

        self.image_shape = torch.empty(2).cuda()
        self.image_shape[0] = image.shape[-1]               # W
        self.image_shape[1] = image.shape[-2]               # H

        cat_feature = None
        
        # encoder the image
        self.encoder(image)           # [B, latent_size, H', W']

        if self.use_global_encoder:
            bdb_grid = input["bdb_grid"].cuda().to(torch.float32)                                               # [B, 64, 64]
            bdb_roi_feature = F.grid_sample(self.encoder.latent, bdb_grid, align_corners=True, mode='bilinear')     # [B, latent_size, 64, 64]
            global_latent = self.global_encoder(bdb_roi_feature)                        # [B, 256]

            cat_feature = global_latent

        if self.use_cls_encoder:
            cls_encoder = input['cls_encoder'].cuda().to(torch.float32)             # [B, 9]
            
            if cat_feature is None:
                cat_feature = cls_encoder
            else:
                cat_feature = torch.cat([cat_feature, cls_encoder], dim=1)          # [B, +9]

        roi_feat = None
        if self.use_atten:
            assert cat_feature.shape[1] == 256 + 9                          # must global_latent + cls_encoder
            ret_dict=self.post_op(self.encoder.latent, cat_feature, bdb_grid)
            roi_feat=ret_dict["roi_feat"]

        if self.use_instance_mask:
            if roi_feat is not None:
                bdb_roi_feat = F.grid_sample(roi_feat, bdb_grid, align_corners=True, mode='bilinear')        # [B, latent_size, 64, 64]
                pred_mask = self.mask_decoder(bdb_roi_feat)                                                  # [B, 1, 64, 64]
            else:
                pred_mask = self.mask_decoder(bdb_roi_feature)                                                  # [B, 1, 64, 64]

        if self.use_encoder:
            latent = self.encoder.index(
                uv, None, self.image_shape, roi_feat=roi_feat
            )  # (B, latent_size, N_ray)
            if self.stop_encoder_grad:
                latent = latent.detach()
            latent = latent.transpose(1, 2).reshape(
                -1, self.latent_size
            )  # (B * N_ray, latent_size)
            latent_feature = latent                              ###### now, default use_encoder is True

        # get camera location and ray direction in camera coordinate
        ray_dirs, cam_loc, ray_dirs_obj, cam_loc_obj, ray_dirs_world, cam_loc_world = rend_util.get_camera_params_cam(uv, intrinsics, get_obj_dirs=True, model_input=input)
        
        if self.depth_norm:
            # we should use unnormalized ray direction for depth
            pose_std = repeat_interleave(torch.eye(4).to(pose.device)[None], uv.shape[0])   # standard pose  [B, 4, 4]
            ray_dirs_tmp, _ = rend_util.get_camera_params_world(uv, pose_std, intrinsics)
            depth_scale = torch.abs(ray_dirs_tmp[:, :, 2:])                                            # modify depth_scale
            depth_scale = depth_scale.reshape(-1, 1)
        
        # repeat for all pixels
        cam_loc = repeat_interleave(cam_loc, num_pixels)
        ray_dirs = ray_dirs.reshape(-1, 3)

        cam_loc_obj = repeat_interleave(cam_loc_obj, num_pixels)
        ray_dirs_obj = ray_dirs_obj.reshape(-1, 3)

        cam_loc_world = repeat_interleave(cam_loc_world, num_pixels)
        ray_dirs_world = ray_dirs_world.reshape(-1, 3)

        if self.sampling_method == 'errorbounded':          # in canonical coords
            z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs_obj, cam_loc_obj, self, latent_feature, cat_feature, global_latent, uv, self.image_shape, input)        # z_vals: (B*N_uv, N_pts_per_ray)
        elif self.sampling_method == 'uniform':
            z_vals, _, _ = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self)
        else:
            raise NotImplementedError
        N_samples = z_vals.shape[1]

        # get points in scene object coordinate
        points_obj = cam_loc_obj.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs_obj.unsqueeze(1)         # canonical coordinate, [B*N_uv, N_pts_per_ray, 3]
        points_obj = points_obj.reshape(-1, 3)          # obj coords

        if self.ray_noise:
            points_obj = points_obj + (torch.rand_like(points_obj) - 0.5) * 0.01 

        # get points in world coordinate
        if self.show_rendering:
            # novel pose, can't use ray points

            points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)         # camera coordinate, [B*N_uv, N_pts_per_ray, 3]
            points_flat = points.reshape(-1, 3)             # camera coords
            points_temp = points_flat.reshape(-1, num_pixels, points.shape[1], 3)               # (B, N_uv, N_pts_per_ray, 3)
            points_world = rend_util.camera_to_world(points_temp, new_pose)                 # new camera pose, get world coords

            # get points feature in original camera pose
            latent_feature, cat_feature = rend_util.get_latent_feature(self, points_world, intrinsics, extrinsics, input)     # (B*N_uv*N_pts_per_ray, 256)
        else:
            points_world = cam_loc_world.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs_world.unsqueeze(1)           # camera coordinate, [B*N_uv, N_pts_per_ray, 3]
            points_world = points_world.reshape(batch_size, num_pixels, -1, 3)                                      # [B, N_uv, N_pts_per_ray, 3] world coords


        # ray in obj coords
        # NOTE: though points input implicit network is object coords normalize (object points - centroid)
        #       but translation don't impact ray direction
        dirs_obj = ray_dirs_obj.unsqueeze(1).repeat(1,N_samples,1)          # obj coords
        dirs_obj_flat = dirs_obj.reshape(-1, 3)                             # (B*N_uv*N_pts_per_ray, 3)

        ###### points_obj is the obj coordinate, gradients in obj coords too
        # transfer to cube coords
        cube_coords = sdf_util.scene_obj2cube_coords(points_obj, input['scene_scale'], input['centroid'], input['none_equal_scale'])    # (B*N_uv*N_pts_per_ray, 3)
        sdf, feature_vectors, gradients = self.implicit_network.get_outputs(cube_coords, latent_feature, cat_feature)

        ###### indices just is __getitem__ index
        # use canonical coords also
        rgb_flat = self.rendering_network(points_obj, gradients, dirs_obj_flat, feature_vectors, indices)
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        # cube sdf scale to volume rendering sdf
        mean_none_equal_scale = torch.mean(input['none_equal_scale'], dim=1, keepdim=True).unsqueeze(-1)
        mean_scene_scale = torch.mean(input['scene_scale'], dim=1, keepdim=True).unsqueeze(-1)
        scale_sdf = sdf.reshape(batch_size, -1, 1) / mean_none_equal_scale * mean_scene_scale

        if self.fusion_scene:
            # fusion one image
            weights, sort_idx, z_vals, ray_mask = self.fusion_volume_rendering(z_vals, sdf, batch_size)
            rgb = self.cat_rgb(rgb, sort_idx, batch_size, num_pixels)
        else:
            weights, ray_mask = self.volume_rendering(z_vals, scale_sdf.reshape(-1, 1))
        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)

        depth_values = torch.sum(weights * z_vals, 1, keepdims=True) / (weights.sum(dim=1, keepdims=True) +1e-8)        # pixel depth values
        depth_vals = z_vals             # points depth vals
        if self.depth_norm:
            # we should scale rendered distance to depth along z direction
            if self.fusion_scene:
                depth_scale = depth_scale.reshape(batch_size, -1, 1)
                depth_scale = depth_scale[0]

            depth_values = depth_scale * depth_values
            depth_vals = depth_scale * depth_vals
        
        # white background assumption
        # To composite onto a white background, use the accumulated alpha map.
        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)

        output = {
            'rgb':rgb,
            'rgb_values': rgb_values,
            'depth_values': depth_values,
            'z_vals': z_vals,
            'depth_vals': depth_vals,
            'sdf': sdf.reshape(z_vals.shape),
            'sample_points': points_world,       # world coords, (B, N_uv, N_pts_per_ray, 3)
            'weights': weights,
            'ray_mask': ray_mask,
        }

        if self.use_instance_mask:
            output['pred_mask'] = pred_mask

        if self.training and self.sampling_method == 'errorbounded':
            # Sample points for the eikonal loss
            # add some of the near surface points
            eikonal_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)

            # add some neighbour points as unisurf
            neighbour_points = eikonal_points + (torch.rand_like(eikonal_points) - 0.5) * 0.01   
            eikonal_points = torch.cat([eikonal_points, neighbour_points], 0)

            eik_latent_feature = torch.cat([latent_feature, latent_feature], 0)

            # get object coords
            eikonal_points = eikonal_points.reshape(batch_size, -1, 3).unsqueeze(2)                                         # (B, 2*N_uv, 1, 3)
            eikonal_points_obj = sdf_util.camera2obj(eikonal_points, pose, world_to_obj)                                    # (B, 2*N_uv, 1, 3)

            # transfer to cube coords
            eikonal_points_cube = sdf_util.scene_obj2cube_coords(eikonal_points_obj.reshape(-1, 3), input['scene_scale'], input['centroid'], input['none_equal_scale'])
            grad_theta = self.implicit_network.gradient(eikonal_points_cube, eik_latent_feature, cat_feature)     # (B*2*N_uv, 3)
            
            # split gradient to eikonal points and neighbour points
            output['grad_theta'] = grad_theta[:grad_theta.shape[0]//2]
            output['grad_theta_nei'] = grad_theta[grad_theta.shape[0]//2:]
        
        # gradient from obj coords to camera coords, gradient [N, 3]
        gradients = gradients.reshape(batch_size, -1, 3)                    # [B, N_uv*N_pts_per_ray, 3]
        rot1 = obj_rot
        if new_pose != None:
            new_extrinsics = torch.linalg.inv(new_pose)
            rot2 = new_extrinsics[:, :3, :3]
        else:
            rot2 = extrinsics[:, :3, :3]
        gradients_world = torch.bmm(rot1, gradients.permute(0, 2, 1))       # [B, 3, N_uv*N_pts_per_ray]
        gradients_cam = torch.bmm(rot2, gradients_world)                    # [B, 3, N_uv*N_pts_per_ray]
        gradients_cam = gradients_cam.permute(0, 2, 1).contiguous()         # [B, N_uv*N_pts_per_ray, 3]
        
        # compute normal map, camera coords
        normals = gradients_cam / (gradients_cam.norm(2, -1, keepdim=True) + 1e-6)
        normals = normals.reshape(-1, N_samples, 3)

        if self.fusion_scene:
            # fusion one image
            normals = self.cat_rgb(normals, sort_idx, batch_size, num_pixels)

        normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)

        output['normal_map'] = normal_map

        if self.add_bdb3d_points:
            add_points_world = input["add_points_world"].cuda().to(torch.float32)        # [B, N_uv, N_add, 3]
            add_latent_feature, add_cat_feature = rend_util.get_latent_feature(self, add_points_world, intrinsics, extrinsics, input)

            # get obj coords
            add_points_obj = sdf_util.world2obj(add_points_world, world_to_obj)                                                 # [B, N_uv, N_add, 3]

            # transfer to cube coords
            add_points_cube = sdf_util.scene_obj2cube_coords(add_points_obj.reshape(-1, 3), input['scene_scale'], input['centroid'], input['none_equal_scale'])
            add_sdf = self.implicit_network(add_points_cube, add_latent_feature, add_cat_feature)[:, 0]       # (B * N * N_add_points, 1)

            output['add_points_world'] = add_points_world
            output['add_sdf'] = add_sdf.reshape(-1, 1)

        return output

    def volume_rendering(self, z_vals, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1])  # (batch_size * num_pixels) x N_samples

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now

        ray_mask = 1 - transmittance[:, -1]

        weights = alpha * transmittance # probability of the ray hits something here

        return weights, ray_mask
    
    def fusion_volume_rendering(self, z_vals, sdf, batch_size):
        density_flat = self.density(sdf)
        density = density_flat.reshape(batch_size, -1, z_vals.shape[1])  # (batch_size, num_pixels, N_samples)

        _, N_uv, N_pnts = density.shape

        z_vals = z_vals.reshape(density.shape)

        # fusion one image
        cat_z_vals = self.cat_tensor(z_vals)                # (num_pixels, N_samples*batch_size)
        cat_density = self.cat_tensor(density)              # (num_pixels, N_samples*batch_size)

        # get sort index
        sort_idx = torch.argsort(cat_z_vals)
        cat_z_vals = torch.gather(cat_z_vals, 1, sort_idx)
        cat_density = torch.gather(cat_density, 1, sort_idx)

        dists = cat_z_vals[:, 1:] - cat_z_vals[:, :-1]
        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)

        # LOG SPACE
        free_energy = dists * cat_density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now

        ray_mask = 1 - transmittance[:, -1]

        weights = alpha * transmittance # probability of the ray hits something here

        return weights, sort_idx, cat_z_vals, ray_mask

    def cat_tensor(self, tensor):
        """
        :params tensor, [B, N_uv, N_pnts]
        return cat_tensor, [N_uv, N_pnts*B]
        """
        B, N_uv, N_pnts = tensor.shape
        tensor = tensor.permute(1, 2, 0)            # [N_uv, N_pnts, B]
        cat_tensor = tensor.reshape(N_uv, -1)       # [N_uv, N_pnts*B]

        return cat_tensor              # [N_uv, N_pnts*B]


    def cat_rgb(self, rgb, sort_idx, batch_size, num_pixels):
        """
        cat rgb and normal
        """
        rgb = rgb.reshape(batch_size, num_pixels, -1, 3)
        rgb = rgb.permute(1, 2, 0, 3)
        rgb = rgb.reshape(num_pixels, -1, 3)                    # cat rgb

        rgb_r = rgb[:, :, 0].squeeze(-1)
        rgb_r = torch.gather(rgb_r, 1, sort_idx).unsqueeze(-1)

        rgb_g = rgb[:, :, 1].squeeze(-1)
        rgb_g = torch.gather(rgb_g, 1, sort_idx).unsqueeze(-1)

        rgb_b = rgb[:, :, 2].squeeze(-1)
        rgb_b = torch.gather(rgb_b, 1, sort_idx).unsqueeze(-1)

        rgb = torch.cat((rgb_r, rgb_g, rgb_b), dim=-1)

        return rgb
