import os, time
import argparse

import cv2
import torch
from tqdm import tqdm

from utils import rend_util
from utils.plots import *
from configs.config_utils import CONFIG
from ssr.ssr_utils.utils import load_device, get_model, get_dataloader, CheckpointIO, load_checkpoint

dirname = os.path.dirname(cv2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('ssr inference')
    parser.add_argument('--config', type=str, required=True, help='configure file for training or testing.')
    return parser.parse_args()


def run(cfg):
    torch.set_default_dtype(torch.float32)
    config = cfg.config
    evals_folder_name = os.path.join(config['save_root_path'], config['exp_name'])
    evals_output_name = os.path.join(evals_folder_name, 'out')
    os.makedirs(evals_output_name, exist_ok=True)

    cfg.log_string('Data save path: %s' % (cfg.save_path))
    checkpoint=CheckpointIO(cfg)

    cfg.log_string('Loading device settings.')
    device = load_device(cfg)

    cfg.log_string('Loading dataset.')
    infer_loader = get_dataloader(cfg.config, mode='test')

    cfg.log_string('Loading model.')
    model = get_model(cfg.config, device=device).cuda().float()
    checkpoint.register_modules(net=model)

    cfg.log_string('Loading weight.')
    ckpt_path = os.path.join(evals_folder_name, config['weight'])
    load_checkpoint(ckpt_path, model)

    cfg.log_string('Inference begin.')
    model.eval()

    extract_mesh = config['eval']['extract_mesh']

    if 'fusion_scene' in config['eval']:
        fusion_scene = config['eval']['fusion_scene']
    else:
        fusion_scene = False

    img_res = config['data']['img_res']
    if img_res == 'None':
        img_res = config['data']['resize_res']
    total_pixels = img_res[0] * img_res[1]
    split_n_pixels = config['eval']['split_n_pixels']

    total_number = infer_loader.dataset.__len__()

    # whether render mesh with color
    if 'export_color_mesh' not in config['eval']:
        export_color_mesh = False
    else:
        export_color_mesh = config['eval']['export_color_mesh']

    if 'mesh_coords' not in config['eval']:
        mesh_coords = 'camera'
    else:
        mesh_coords = config['eval']['mesh_coords']


    angle_list = [-40, 0, 40]

    for angle_idx in range(len(angle_list)):
        angle = angle_list[angle_idx]
        for batch_id, (indices, model_input, ground_truth) in enumerate(infer_loader):
            img_id = ground_truth['img_id'][0].split('.')[0].split('/')[-1]
            obj_id = ground_truth['object_id'][0].numpy()
            cname = ground_truth['cname'][0]
            print(img_id)
            model_input["image"] = model_input["image"].cuda().to(torch.float32)
            model_input["intrinsics"] = model_input["intrinsics"].cuda().to(torch.float32)      # cpu -> gpu
            model_input["uv"] = model_input["uv"].cuda().to(torch.float32)                      # [B, N, 2]
            model_input['pose'] = model_input['pose'].cuda().to(torch.float32)
            model_input['extrinsics'] = model_input['extrinsics'].cuda().to(torch.float32)
            model_input['obj_rot'] = model_input['obj_rot'].cuda().to(torch.float32)
            model_input['obj_tran'] = model_input['obj_tran'].cuda().to(torch.float32)
            model_input['world_to_obj'] = model_input['world_to_obj'].cuda().to(torch.float32)
            model_input['centroid'] = model_input['centroid'].cuda().to(torch.float32)
            model_input['none_equal_scale'] = model_input['none_equal_scale'].cuda().to(torch.float32)
            model_input['scene_scale'] = model_input['scene_scale'].cuda().to(torch.float32)
            model_input['voxel_range'] = model_input['voxel_range'].cuda().to(torch.float32)

            batch_size, num_samples, _ =  model_input["uv"].shape
            total_pixels = min(total_pixels, num_samples)                        # if mask filter, although there is not pixel sample, num_samples is less than total pixels

            mesh_folder = os.path.join(evals_output_name, cname, f'{str(img_id)}_{str(obj_id)}')
            os.makedirs(mesh_folder, exist_ok=True)

            # not export mesh, only render images
            images_dir = '{0}/rendering'.format(mesh_folder)
            os.makedirs(images_dir, exist_ok=True)
            depth_dir = '{0}/depth'.format(mesh_folder)
            os.makedirs(depth_dir, exist_ok=True)
            normal_dir = '{0}/normal'.format(mesh_folder)
            os.makedirs(normal_dir, exist_ok=True)

            # check exist
            check_path = '{0}/eval_000_fusion_{1}.png'.format(normal_dir, angle)
            if os.path.exists(check_path):
                print(f'{img_id}_{obj_id} {angle} exist')
                continue

            pose = model_input['pose']

            # rot total objs
            total_bbox = ground_truth['bdb_3d_camera'][0]

            for obj_idx in range(batch_size):
                # rot one obj
                new_obj_camera_pose = rend_util.rot_camera_pose(pose[obj_idx], total_bbox, angle, 'y')
                pose[obj_idx] = new_obj_camera_pose

            render_poses = []
            render_poses.append(pose)
            for render_id in tqdm(range(len(render_poses))):
                new_pose = render_poses[render_id]

                split = split_input(model_input, total_pixels, n_pixels=split_n_pixels)
                res = []
                first_sta = 1                       # the first split
                for s in tqdm(split):
                    # torch.cuda.empty_cache()        # maybe slow speed, but free gpu cache
                    out = model(s, indices, new_pose=new_pose)
                    d = {
                        'rgb_values': out['rgb_values'].detach(),
                        'depth_values': out['depth_values'].detach(),
                        'normal_map': out['normal_map'].detach(),
                        'ray_mask': out['ray_mask'].detach(),
                    }
                    res.append(d)

                if fusion_scene:
                    batch_size = 1                  # after fusion only one image

                model_outputs = merge_output(res, total_pixels, batch_size)
                rgb_eval = model_outputs['rgb_values'].reshape(batch_size, num_samples, 3)
                depth_eval = model_outputs['depth_values'].reshape(batch_size, num_samples, 1)
                normal_eval = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
                ray_mask = model_outputs['ray_mask'].reshape(batch_size, num_samples, 1)

                depth_eval = depth_eval * ray_mask

                # plot rendering images and gt
                img, rgb_map, rgb_map_temp = plot_images(rgb_eval, ground_truth['rgb'], path=evals_folder_name, epoch=0, img_res=img_res, indices=indices, ray_mask=ray_mask)
                img.save('{0}/eval_{1}_gt_{2}.png'.format(images_dir,'%03d' % render_id, angle))
                rgb_map.save('{0}/eval_{1}_fusion_{2}.png'.format(images_dir,'%03d' % render_id, angle))

                # plot depth images and gt
                depth_gt_map, depth_map = plot_depth_maps(depth_eval, ground_truth['depth'], path=evals_folder_name, epoch=0, img_res=img_res, indices=indices, ray_mask=ray_mask)
                depth_gt_map.save('{0}/eval_{1}_gt_{2}.png'.format(depth_dir,'%03d' % render_id, angle))
                depth_map.save('{0}/eval_{1}_fusion_{2}.png'.format(depth_dir,'%03d' % render_id, angle))
                
                # plot normal images and gt
                normal_gt = (ground_truth['normal'].cuda() + 1.0) / 2.0
                normal_pred = (normal_eval + 1.0) / 2.0             # from [-1, 1] to [0, 1]

                normal_gt_map, normal_map = plot_normal_maps(normal_pred, normal_gt, path=evals_folder_name, epoch=0, img_res=img_res, indices=indices, ray_mask=ray_mask)
                normal_gt_map.save('{0}/eval_{1}_gt_{2}.png'.format(normal_dir,'%03d' % render_id, angle))
                normal_map.save('{0}/eval_{1}_fusion_{2}.png'.format(normal_dir,'%03d' % render_id, angle))

    cfg.log_string('Inference finished.')

if __name__=="__main__":
    args=parse_args()

    import yaml
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    cfg['model']['ray_sampler']['add_bdb3d_points'] = False          # in inference, not add points
    if 'ray_noise' not in cfg['model']:
        cfg['model']['ray_noise'] = False             # old version yaml not set this item

    if cfg['data']['mask_filter'] or cfg['data']['bdb2d_filter']:
        print('mask_filter and bdb2d_filter must be False in inference')
        cfg['data']['mask_filter'] = False
        cfg['data']['bdb2d_filter'] = False

    cfg=CONFIG(cfg, mode='test')
    cfg.update_config(args.__dict__)

    cfg.log_string('Loading configuration')
    cfg.log_string(cfg.config)
    # cfg.write_config()

    t1 = time.time()
    run(cfg)
    t2 = time.time()
    print(f'total time {t2 - t1}s')
