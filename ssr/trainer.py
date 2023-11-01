import os
import time
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.rend_util import get_psnr

def Recon_trainer(cfg,model,loss,optimizer,scheduler,train_loader,test_loader,device,checkpoint):
    start_t = time.time()
    config = cfg.config

    log_dir = cfg.save_path
    os.makedirs(log_dir, exist_ok=True)

    cfg.write_config()
    tb_logger = SummaryWriter(log_dir)

    start_epoch = 0
    iter = 0
    if config["resume"] == True:
        checkpoint.load(config["weight"])
        # start_epoch = scheduler.last_epoch
        start_epoch = checkpoint.module_dict['epoch']
        iter = checkpoint.module_dict['iter']
    if config['finetune']==True:
        start_epoch=0
    scheduler.last_epoch = start_epoch

    model.train()
    
    min_eval_loss = 10000
    for e in range(start_epoch, config['other']['nepoch']):
        torch.cuda.empty_cache()
        cfg.log_string("Switch Phase to Train")
        model.train()
        for batch_id, (indices, model_input, ground_truth) in enumerate(train_loader):
            '''
            indices: [B*1]
            model_input: 
                1.image: [B, 3, H, W]
                2.uv: [B, H*W, 2]                           image coordinate(pixel)
                3.intrinsics: [B, 3, 3]                     image to camera
                4.pose: [B, 4, 4]                           camera to world
                5*.add_points_world: [B, N_uv, N_add, 3]     add points in world coords
            ground_truth:
                1.rgb: [B, num_pixels, 3]
                2.depth: [B, num_pixels, 1]
                3.normal: [B, num_pixels, 3]
            '''
            model_input["image"] = model_input["image"].cuda().to(torch.float32)
            model_input["intrinsics"] = model_input["intrinsics"].cuda().to(torch.float32)        # cpu -> gpu
            model_input["uv"] = model_input["uv"].cuda().to(torch.float32)
            model_input['pose'] = model_input['pose'].cuda().to(torch.float32)
            model_input['extrinsics'] = model_input['extrinsics'].cuda().to(torch.float32)
            model_input['obj_rot'] = model_input['obj_rot'].cuda().to(torch.float32)
            model_input['obj_tran'] = model_input['obj_tran'].cuda().to(torch.float32)
            model_input['world_to_obj'] = model_input['world_to_obj'].cuda().to(torch.float32)
            model_input['centroid'] = model_input['centroid'].cuda().to(torch.float32)
            model_input['none_equal_scale'] = model_input['none_equal_scale'].cuda().to(torch.float32)
            model_input['scene_scale'] = model_input['scene_scale'].cuda().to(torch.float32)

            optimizer.zero_grad()

            # t1 = time.time()

            model_outputs = model(model_input, indices)

            loss_output = loss(model_outputs, ground_truth, e)
            total_loss = loss_output['total_loss']
            total_loss.backward()

            # t2 = time.time()
            # print(f'total time {t2 - t1}s')

            '''gradient clip'''
            torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=0.7, norm_type=2)
            total_norm = 0
            parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
            for p in parameters:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            optimizer.step()

            psnr = get_psnr(model_outputs['rgb_values'], ground_truth['rgb'].cuda().reshape(-1,3))
            msg = '{:0>8},[epoch {}] ({}/{}): total_loss = {}, rgb_loss = {}, eikonal_loss = {}, depth_loss = {}, normal_l1 = {}, normal_cos = {}, ray_mask_loss = {}, instance_mask_loss = {}, sdf_loss = {}, vis_sdf_loss = {}, psnr = {}, bete={}, alpha={}'.format(
                    str(datetime.timedelta(seconds=round(time.time() - start_t))),
                    e, 
                    batch_id + 1,
                    len(train_loader), 
                    total_loss.item(),
                    loss_output['rgb_loss'].item(),
                    loss_output['eikonal_loss'].item(),
                    loss_output['depth_loss'].item(),
                    loss_output['normal_l1'].item(),
                    loss_output['normal_cos'].item(),
                    loss_output['ray_mask_loss'].item(),
                    loss_output['instance_mask_loss'].item(),
                    loss_output['sdf_loss'].item(),
                    loss_output['vis_sdf_loss'].item(),
                    psnr.item(),
                    model.module.density.get_beta().item(),
                    1. / model.module.density.get_beta().item()
                )
            cfg.log_string(msg)

            tb_logger.add_scalar('Loss/total_loss', total_loss.item(), iter)
            tb_logger.add_scalar('Loss/color_loss', loss_output['rgb_loss'].item(), iter)
            tb_logger.add_scalar('Loss/eikonal_loss', loss_output['eikonal_loss'].item(), iter)
            tb_logger.add_scalar('Loss/smooth_loss', loss_output['smooth_loss'].item(), iter)
            tb_logger.add_scalar('Loss/depth_loss', loss_output['depth_loss'].item(), iter)
            tb_logger.add_scalar('Loss/normal_l1_loss', loss_output['normal_l1'].item(), iter)
            tb_logger.add_scalar('Loss/normal_cos_loss', loss_output['normal_cos'].item(), iter)
            tb_logger.add_scalar('Loss/ray_mask_loss', loss_output['ray_mask_loss'].item(), iter)
            tb_logger.add_scalar('Loss/instance_mask_loss', loss_output['instance_mask_loss'].item(), iter)
            tb_logger.add_scalar('Loss/sdf_loss', loss_output['sdf_loss'].item(), iter)
            tb_logger.add_scalar('Loss/vis_sdf_loss', loss_output['vis_sdf_loss'].item(), iter)
            tb_logger.add_scalar('Loss/grad_norm', total_norm, iter)
            
            tb_logger.add_scalar('Statistics/beta', model.module.density.get_beta().item(), iter)
            tb_logger.add_scalar('Statistics/alpha', 1. / model.module.density.get_beta().item(), iter)
            tb_logger.add_scalar('Statistics/psnr', psnr.item(), iter)
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            tb_logger.add_scalar("train/lr", current_lr, iter)
                
            iter += 1

        # after model_save_interval epoch, evaluate the model
        if e % config['other']['model_save_interval'] == 0:
            model.eval()
            eval_loss = 0
            eval_loss_info = {
            }
            cfg.log_string("Switch Phase to Test")
            for batch_id, (indices, model_input, ground_truth) in enumerate(test_loader):
                torch.cuda.empty_cache()
                model_input["image"] = model_input["image"].cuda().to(torch.float32)
                model_input["intrinsics"] = model_input["intrinsics"].cuda().to(torch.float32)        # cpu -> gpu
                model_input["uv"] = model_input["uv"].cuda().to(torch.float32)
                model_input['pose'] = model_input['pose'].cuda().to(torch.float32)
                model_input['extrinsics'] = model_input['extrinsics'].cuda().to(torch.float32)
                model_input['obj_rot'] = model_input['obj_rot'].cuda().to(torch.float32)
                model_input['obj_tran'] = model_input['obj_tran'].cuda().to(torch.float32)
                model_input['world_to_obj'] = model_input['world_to_obj'].cuda().to(torch.float32)
                model_input['centroid'] = model_input['centroid'].cuda().to(torch.float32)
                model_input['none_equal_scale'] = model_input['none_equal_scale'].cuda().to(torch.float32)
                model_input['scene_scale'] = model_input['scene_scale'].cuda().to(torch.float32)
                model_outputs = model(model_input, indices)

                loss_output = loss(model_outputs, ground_truth, e)
                total_loss = loss_output['total_loss']

                psnr = get_psnr(model_outputs['rgb_values'], ground_truth['rgb'].cuda().reshape(-1,3))
                msg = 'Validation {:0>8},[epoch {}] ({}/{}): total_loss = {}, rgb_loss = {}, eikonal_loss = {}, depth_loss = {}, normal_l1 = {}, normal_cos = {}, ray_mask_loss = {}, instance_mask_loss = {}, sdf_loss = {}, vis_sdf_loss = {}, psnr = {}, bete={}, alpha={}'.format(
                    str(datetime.timedelta(seconds=round(time.time() - start_t))),
                    e, 
                    batch_id + 1,
                    len(test_loader), 
                    total_loss.item(),
                    loss_output['rgb_loss'].item(),
                    loss_output['eikonal_loss'].item(),
                    loss_output['depth_loss'].item(),
                    loss_output['normal_l1'].item(),
                    loss_output['normal_cos'].item(),
                    loss_output['ray_mask_loss'].item(),
                    loss_output['instance_mask_loss'].item(),
                    loss_output['sdf_loss'].item(),
                    loss_output['vis_sdf_loss'].item(),
                    psnr.item(),
                    model.module.density.get_beta().item(),
                    1. / model.module.density.get_beta().item()
                )
                cfg.log_string(msg)

                for key in loss_output:
                    if "total" not in key:
                        if key not in eval_loss_info:
                            eval_loss_info[key] = 0
                        eval_loss_info[key] += torch.mean(loss_output[key]).item()

                eval_loss += total_loss.item()
            
            avg_eval_loss = eval_loss / (batch_id + 1)
            for key in eval_loss_info:
                eval_loss_info[key] = eval_loss_info[key] / (batch_id + 1)
            eval_loss_msg = f'avg_eval_loss is {avg_eval_loss}'
            cfg.log_string(eval_loss_msg)
            tb_logger.add_scalar('eval/eval_loss', avg_eval_loss, e)
            for key in eval_loss_info:
                tb_logger.add_scalar("eval/" + key, eval_loss_info[key], e)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_eval_loss)
            else:
                scheduler.step()

            checkpoint.register_modules(epoch=e, iter=iter, min_loss=avg_eval_loss)
            if avg_eval_loss < min_eval_loss:
                checkpoint.save('best')
                min_eval_loss = avg_eval_loss
            else:
                checkpoint.save("latest")
