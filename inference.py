import os, time
import argparse

import cv2
import torch

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

    img_res = config['data']['img_res']
    if img_res == 'None':
        img_res = config['data']['resize_res']
    total_pixels = img_res[0] * img_res[1]

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
        total_pixels = min(total_pixels, num_samples)  # if mask filter, although there is not pixel sample, num_samples is less than total pixels

        if extract_mesh:
            grid_boundary = ground_truth['bdb_3d'][0]                       # just one object
            mesh_folder = os.path.join(evals_output_name, cname, f'{str(img_id)}_{str(obj_id)}')
            os.makedirs(mesh_folder, exist_ok=True)

            if os.path.exists(os.path.join(mesh_folder, f'label_cube.ply')):
                msg = 'continue {}/{}'.format(batch_id, total_number)
                print(msg)
                continue

            # else:
            mesh = get_surface_sliding(
                path="", epoch="",
                model=model, img=model_input["image"],
                intrinsics=model_input["intrinsics"],
                extrinsics=model_input["extrinsics"],
                model_input=model_input,
                ground_truth=ground_truth,
                resolution=256,
                grid_boundary=grid_boundary,
                return_mesh=True,
                delta=0.03,
                export_color_mesh=export_color_mesh,
            )

            # if render mesh, do not export cube mesh, cube mesh is for evaluation
            # Beacuse InstPIFu use cube mesh for evaluation
            if not export_color_mesh:
                try:
                    mesh.export(os.path.join(mesh_folder, f'pred_cube.ply'))
                    cfg.log_string('Pred mesh export successfully!')
                except:
                    cfg.log_string(f'{str(img_id)}_{str(obj_id)} pred mesh failed!')
                    continue

                mesh_gt = get_surface_sliding(
                    path="", epoch="",
                    model=model, img=model_input["image"],
                    intrinsics=model_input["intrinsics"],
                    extrinsics=model_input["extrinsics"],
                    model_input=model_input,
                    ground_truth=ground_truth,
                    resolution=256,
                    grid_boundary=grid_boundary,
                    return_mesh=True,
                    delta=0.03,
                    eval_gt=True,
                    export_color_mesh=export_color_mesh,
                )
                try:
                    mesh_gt.export(os.path.join(mesh_folder, f'label_cube.ply'))
                    cfg.log_string('Label mesh export successfully!')
                except:
                    cfg.log_string(f'{str(img_id)}_{str(obj_id)} gt mesh failed!')
                    continue

        # for visualization, export mesh with color and in original size of object
        if export_color_mesh:
            meshcolor, meshnonecolor = model(model_input, indices, mesh=mesh, mesh_coords=mesh_coords)  # mesh coords: camera, world, canonical(default)
            meshcolor.export(os.path.join(mesh_folder, f'mesh_color.ply'))              # original size
            meshnonecolor.export(os.path.join(mesh_folder, f'mesh_none_color.ply'))     # original size
            cfg.log_string('mesh export successfully!')

        cfg.log_string(f'inference {batch_id}/{total_number}')

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
