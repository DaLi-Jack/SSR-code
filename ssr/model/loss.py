import torch
from torch import nn
import math
from scipy.optimize import leastsq
from utils.sdf_util import *

# copy from MiDaS
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def get_scale(depth_pred, depth_gt):
    def func(p,x):
        k,b=p
        return k*x+b
    def error(p,x,y):
        return func(p,x)-y
    depth_pred_np = torch.flatten(depth_pred).cpu().detach().numpy()
    depth_gt_np = torch.flatten(depth_gt).cpu().detach().numpy()
    p0 = [1, 0]
    res = leastsq(error,p0,args=(depth_gt_np, depth_pred_np))
    w,q = res[0]

    return w, q


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0.0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        # w,q = get_scale(prediction, target)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)
# end copy
    
class soft_L1(nn.Module):
    def __init__(self):
        super(soft_L1, self).__init__()

    def forward(self, input, target, eps=0.0):
        ret = torch.abs(input - target) - eps
        ret = torch.clamp(ret, min=0.0)          # avoid torch.abs(input - target) < eps
        return ret

    
class MonoSDFLoss(nn.Module):
    def __init__(self, cfg, mode):
        super().__init__()
        self.num_pixels = cfg['data']['num_pixels'][mode]
        self.use_depth = cfg['data']['use_depth']
        self.use_normal = cfg['data']['use_normal']
        self.use_sdf = cfg['data']['use_sdf']
        self.use_instance_mask = cfg['data']['use_instance_mask']
        self.color_weight = cfg['loss']['color_weight']
        self.eikonal_weight = cfg['loss']['eikonal_weight']
        self.smooth_weight = cfg['loss']['smooth_weight']
        self.depth_weight = cfg['loss']['depth_weight']
        self.normal_l1_weight = cfg['loss']['normal_l1_weight']
        self.normal_cos_weight = cfg['loss']['normal_cos_weight']
        self.instance_mask_weight = cfg['loss']['instance_mask_weight']
        self.sdf_weight = cfg['loss']['sdf_weight']
        self.ray_mask_weight = cfg['loss']['ray_mask_weight']
        self.vis_sdf = cfg['loss']['vis_sdf']
        self.add_bdb3d_points = cfg['model']['ray_sampler']['add_bdb3d_points']

        self.vis_mask_loss = cfg['loss']['vis_mask_loss']
        self.use_curriculum_color = cfg['loss']['use_curriculum_color']
        self.use_curriculum_depth = cfg['loss']['use_curriculum_depth']
        self.use_curriculum_normal = cfg['loss']['use_curriculum_normal']
        if self.use_curriculum_color:
            self.curri_color = cfg['loss']['curri_color']
        if self.use_curriculum_depth:
            self.curri_depth = cfg['loss']['curri_depth']
        if self.use_curriculum_normal:
            self.curri_normal = cfg['loss']['curri_normal']
        
        if cfg['loss']['rgb_loss'] == 'L1loss':
            self.rgb_loss = nn.L1Loss(reduction='none')
        elif cfg['loss']['rgb_loss'] == 'MSEloss':
            self.rgb_loss = nn.MSELoss(reduction='none')

        self.dataset = cfg['data']['dataset']

        if self.dataset == 'Pix3D':
            self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)       # use scale and shift
        else:
            self.depth_loss = nn.L1Loss(reduction='none')

        self.sdf_loss = nn.L1Loss(reduction='mean')     # mean loss, return a scalar

        self.ray_mask_loss = nn.MSELoss(reduction='mean')

        if self.use_instance_mask:
            self.instance_mask_loss = nn.MSELoss(reduction='mean')
        
        print(f"using weight for loss RGB_{self.color_weight} EK_{self.eikonal_weight} SM_{self.smooth_weight} Depth_{self.depth_weight} NormalL1_{self.normal_l1_weight} NormalCos_{self.normal_cos_weight}")
        
        self.step = 0
        self.end_step = cfg['loss']['end_step']

    def get_rgb_loss(self,rgb_values, rgb_gt, vis_mask):
        rgb_loss = self.rgb_loss(rgb_values, rgb_gt)
        rgb_loss = (rgb_loss * vis_mask.float()).sum()
        non_zero_elements = vis_mask.sum()
        rgb_loss = rgb_loss / non_zero_elements
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_smooth_loss(self,model_outputs):
        # smoothness loss as unisurf
        g1 = model_outputs['grad_theta']
        g2 = model_outputs['grad_theta_nei']
        
        normals_1 = g1 / (g1.norm(2, dim=1).unsqueeze(-1) + 1e-5)
        normals_2 = g2 / (g2.norm(2, dim=1).unsqueeze(-1) + 1e-5)
        smooth_loss =  torch.norm(normals_1 - normals_2, dim=-1).mean()
        return smooth_loss
    
    def get_depth_loss(self, depth_pred, depth_gt, vis_mask):
        # TODO remove hard-coded scaling for depth
        if self.dataset == 'Pix3D':

            return self.depth_loss(depth_pred, (depth_gt * 50 + 0.5), vis_mask)
        
        depth_loss = self.depth_loss(depth_pred, depth_gt)
        depth_loss = (depth_loss * vis_mask.float()).sum()
        non_zero_elements = vis_mask.sum()
        depth_loss = depth_loss / non_zero_elements
        return depth_loss
        
    def get_normal_loss(self, normal_pred, normal_gt, vis_mask):
        """
        attention, this vis_mask is [B, N], in other loss, vis_mask is [B, N, 1]
        """
        normal_gt = torch.nn.functional.normalize(normal_gt, p=2, dim=-1)
        normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1)
        l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1)                         # [B, N]
        l1 = (l1 * vis_mask.float()).sum()
        non_zero_elements = vis_mask.sum()
        l1 = l1 / non_zero_elements

        cos = (1. - torch.sum(normal_pred * normal_gt, dim = -1))
        cos = (cos * vis_mask.float()).sum()
        non_zero_elements = vis_mask.sum()
        cos = cos / non_zero_elements
        return l1, cos

    def get_sdf_loss(self, sdf_pred, sdf_gt):

        sdf_loss = self.sdf_loss(sdf_pred, sdf_gt)
        
        return sdf_loss
    
    def get_ray_mask_loss(self, ray_mask_pred, ray_mask_gt):
        ray_mask_loss = self.ray_mask_loss(ray_mask_pred, ray_mask_gt)

        return ray_mask_loss
    
    def get_loss_weight(self, epoch, begin_epoch, end_epoch, begin_weight, end_weight):
        if epoch < begin_epoch:
            return begin_weight
        elif epoch > end_epoch:
            return end_weight
        else:                           # linear change
            return begin_weight + (end_weight - begin_weight) * (epoch - begin_epoch) / (end_epoch - begin_epoch)

    def forward(self, model_outputs, ground_truth, epoch):
        ground_truth['world_to_obj'] = ground_truth['world_to_obj'].float().cuda()
        rgb_pred = model_outputs['rgb_values'].reshape(-1, self.num_pixels, 3)          # [B, Num_pixels, 3]
        rgb_gt = ground_truth['rgb'].cuda()

        # # only supervised the foreground normal
        if self.vis_mask_loss:
            vis_mask = ground_truth['vis_pixel'].cuda()                             # [B, Num_pixels]
        else:
            vis_mask = torch.ones(rgb_pred.shape[0], rgb_pred.shape[1], dtype=bool).cuda()              # [B, Num_pixels]
        vis_mask = vis_mask.reshape(-1, self.num_pixels, 1)                             # [B, Num_pixels, 1]

        rgb_loss = self.get_rgb_loss(rgb_pred, rgb_gt, vis_mask)
        if self.use_curriculum_color:
            self.color_weight = self.get_loss_weight(epoch, begin_epoch=self.curri_color[0], end_epoch=self.curri_color[1], begin_weight=self.curri_color[2], end_weight=self.curri_color[3])
        total_loss = self.color_weight * rgb_loss

        ray_mask_pred = model_outputs['ray_mask'].reshape(-1, self.num_pixels)
        ray_mask_gt = ground_truth['full_mask_pixel'].cuda().to(torch.float32)
        ray_mask_loss = self.get_ray_mask_loss(ray_mask_pred, ray_mask_gt)
        total_loss += self.ray_mask_weight * ray_mask_loss

        output = {
            'rgb_loss': rgb_loss,
            'eikonal_loss': torch.tensor(0.0).cuda().float(),           # for log restore
            'smooth_loss': torch.tensor(0.0).cuda().float(),
            'depth_loss': torch.tensor(0.0).cuda().float(),
            'normal_l1': torch.tensor(0.0).cuda().float(),
            'normal_cos': torch.tensor(0.0).cuda().float(),
            'sdf_loss': torch.tensor(0.0).cuda().float(),
            'ray_mask_loss': ray_mask_loss,
            'instance_mask_loss': torch.tensor(0.0).cuda().float(),
        }

        # compute decay weights 
        if self.end_step > 0:
            decay = math.exp(-self.step / self.end_step * 10.)
        else:
            decay = 1.0
            
        self.step += 1

        # monocular depth and normal
        if self.use_depth:
            depth_gt = ground_truth['depth'].cuda()
            depth_gt = torch.clamp(depth_gt, max=10)
            depth_pred = model_outputs['depth_values']
            depth_pred = depth_pred.reshape(-1, self.num_pixels, 1)                 # [B, Num_pixels, 1]
            depth_loss = self.get_depth_loss(depth_pred, depth_gt, vis_mask)
            output['depth_loss'] = depth_loss
            if self.use_curriculum_depth:
                self.depth_weight = self.get_loss_weight(epoch, begin_epoch=self.curri_depth[0], end_epoch=self.curri_depth[1], begin_weight=self.curri_depth[2], end_weight=self.curri_depth[3])
            total_loss += decay * self.depth_weight * depth_loss
        
        if self.use_normal:
            normal_gt = ground_truth['normal'].cuda()
            normal_pred = model_outputs['normal_map']
            normal_pred = normal_pred.reshape(-1, self.num_pixels, 3)               # [B, Num_pixels, 3]
            normal_l1, normal_cos = self.get_normal_loss(normal_pred, normal_gt, vis_mask.reshape(-1, self.num_pixels))
            output['normal_l1'] = normal_l1
            output['normal_cos'] = normal_cos
            if self.use_curriculum_normal:
                self.normal_l1_weight = self.get_loss_weight(epoch, begin_epoch=self.curri_normal[0], end_epoch=self.curri_normal[1], begin_weight=self.curri_normal[2], end_weight=self.curri_normal[3])
                self.normal_cos_weight = self.normal_l1_weight
            total_loss += decay * self.normal_l1_weight * normal_l1
            total_loss += decay * self.normal_cos_weight * normal_cos
        
        if self.use_sdf:
            # get sdf gt for sample points
            sample_points = model_outputs['sample_points']                                            # world coords, [B, Num_pixels, Num_points_a_ray, 3]
            sdf_gt = get_sdf_gt_worldcoords(sample_points, ground_truth)                                          # [B, C, 1, Num_pixels, Num_points_a_ray]  C=1
            sdf_pred = model_outputs['sdf']
            sdf_pred = sdf_pred.reshape(-1, 1)
            # sdf loss
            sdf_loss = self.get_sdf_loss(sdf_pred, sdf_gt)
            vis_sdf_loss = self.get_sdf_loss(sdf_pred, sdf_gt)       # for log restore
            
            if self.add_bdb3d_points:       # add sdf loss of add points
                add_points_world = model_outputs['add_points_world']
                add_sdf_gt = get_sdf_gt_worldcoords(add_points_world, ground_truth)
                add_sdf_pred = model_outputs['add_sdf']         # (B * N * N_add_points, 1)
                add_sdf_loss = self.get_sdf_loss(add_sdf_pred, add_sdf_gt)
                sdf_loss += add_sdf_loss

                vis_add_sdf_loss = self.get_sdf_loss(add_sdf_pred, add_sdf_gt)
                vis_sdf_loss += vis_add_sdf_loss

            output['sdf_loss'] = sdf_loss
            output['vis_sdf_loss'] = vis_sdf_loss
            total_loss += decay * self.sdf_weight * sdf_loss
            if self.vis_sdf:
                # vis sdf gt
                vis_sdf(sample_points.reshape(-1, 3), sdf_gt, mode='all points')
                # vis sdf pred
                vis_sdf(sample_points.reshape(-1, 3), sdf_pred, mode='all points')
        
        if 'grad_theta' in model_outputs:
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
            smooth_loss = self.get_smooth_loss(model_outputs)
            output['eikonal_loss'] = eikonal_loss
            output['smooth_loss'] = smooth_loss
            total_loss += self.eikonal_weight * eikonal_loss
            total_loss += self.smooth_weight * smooth_loss

        if self.use_instance_mask:
            pred_mask = model_outputs['pred_mask']
            gt_mask = ground_truth['instance_mask'].cuda().to(torch.float32)
            gt_mask = F.interpolate(gt_mask, size=(pred_mask.shape[2], pred_mask.shape[3]), mode="nearest")

            instance_mask_loss = self.instance_mask_loss(pred_mask, gt_mask)
            output['instance_mask_loss'] = instance_mask_loss

            total_loss += self.instance_mask_weight * instance_mask_loss

        output['total_loss'] = total_loss
        return output

