import os, sys
import numpy as np
from datetime import datetime
import trimesh
import argparse
import torch
import json
import time
import subprocess
from joblib import Parallel, delayed, parallel_backend
import yaml
import scipy

sys.path.append(os.getcwd())
from external.pyTorchChamferDistance.chamfer_distance import ChamferDistance
import multiprocessing as mp
dist_chamfer=ChamferDistance()

from ssr.ssr_utils.network_utils import fix_random_seed

OCCNET_FSCORE_EPS = 1e-09

def percent_below(dists, thresh):
  return np.mean((dists**2 <= thresh).astype(np.float32)) * 100.0

def f_score(a_to_b, b_to_a, thresh):
  precision = percent_below(a_to_b, thresh)
  recall = percent_below(b_to_a, thresh)

  return (2 * precision * recall) / (precision + recall + OCCNET_FSCORE_EPS)
def pointcloud_neighbor_distances_indices(source_points, target_points):
  target_kdtree = scipy.spatial.cKDTree(target_points)
  distances, indices = target_kdtree.query(source_points, workers=-1)
  return distances, indices
def fscore(points1,points2,tau=0.002):
  """Computes the F-Score at tau between two meshes."""
  dist12, _ = pointcloud_neighbor_distances_indices(points1, points2)
  dist21, _ = pointcloud_neighbor_distances_indices(points2, points1)
  f_score_tau = f_score(dist12, dist21, tau)
  return f_score_tau


class Evaluate:
    def __init__(self, config, render=None):
        self.cfg = config
        self.logfile = os.path.join(self.cfg['result_dir'], self.cfg['log'])
        self.cd_loss_dict = {}
        self.f_score_dict = {}
        self.debug = self.cfg['debug']
        self.classnames = self.cfg['class_name']
        self.cmd_app = './external/ldif/gaps/bin/x86_64/mshalign'
        self.loginfo = []
        self.get_files()

        self.render = render

    def show(self, data):
        if self.render is not None:
            self.render.show(data)
            self.render.clear()

    def get_files(self):
        self.split = []
        if isinstance(self.classnames, list):
            raise ValueError('not support list, must be all_subset!!!')

        else:
            self.split_path = os.path.join(self.cfg['split_path'], self.classnames + ".json")
            with open(self.split_path, 'rb') as f:
                split = json.load(f)
            if 'FRONT3D' in self.split_path:
                for idx in range(len(split)):
                    if idx > 2000 - 1:                  # only test 2000 items like InstPIFu
                        break
                    self.split.append(split[idx])
            else:
                self.split = split
        print('load {} test items'.format(len(self.split)))

    def calculate_cd(self, pred, label):
        pred_sample_points=pred.sample(10000)
        gt_sample_points=label.sample(10000)
        fst=fscore(pred_sample_points,gt_sample_points)

        pred_sample_gpu=torch.from_numpy(pred_sample_points).float().cuda().unsqueeze(0)
        gt_sample_gpu=torch.from_numpy(gt_sample_points).float().cuda().unsqueeze(0)
        dist1,dist2=dist_chamfer(gt_sample_gpu,pred_sample_gpu)[:2]
        cd_loss=torch.mean(dist1)+torch.mean(dist2)
        return cd_loss.item()*1000, fst

    def get_result(self):
        total_cd = 0
        total_number = 0
        total_score = 0
        for key in self.cd_loss_dict:
            total_cd += np.sum(np.array(self.cd_loss_dict[key]))
            total_score += np.sum(np.array(self.f_score_dict[key]))
            total_number += len(self.cd_loss_dict[key])
            self.cd_loss_dict[key]=np.mean(np.array(self.cd_loss_dict[key]))
            self.f_score_dict[key]=np.mean(np.array(self.f_score_dict[key]))

        mean_f_score = total_score/total_number
        mean_cd = total_cd/total_number
        for key in self.cd_loss_dict:
            msg="cd/fscore loss of category %s is %f/%f"%(key, self.cd_loss_dict[key], self.f_score_dict[key])
            print(msg)
            self.loginfo.append(msg)
        msg = "cd/fscore loss of mean %f/%f"%(mean_cd, mean_f_score)
        print(msg)
        self.loginfo.append(msg)

        with open(self.logfile, 'a') as f:
            currentDateAndTime = datetime.now()
            time_str = currentDateAndTime.strftime('%D--%H:%M:%S')
            f.write('*'*30)
            f.write(time_str + '\n')
            for info in self.loginfo:
                f.write(info + "\n")

    def run_in_one(self, index):
        data = self.split[index]
        img_id, obj_id, classname = data

        # load truth size
        img_path = os.path.join(self.cfg['data_path'], img_id)
        post_fix = img_path.split('.')[-1]      # avoid '.png' '.jpg' '.jpeg'
        if 'rgb' in img_path:
            anno_path = img_path.replace('rgb', 'annotation').replace(f'.{post_fix}', '.json')
        else:
            anno_path = img_path.replace('img', 'annotation').replace(f'.{post_fix}', '.json')
        if not os.path.exists(anno_path):
            print(f'anno_path {anno_path} not exists')
            return 
        with open(anno_path, 'r') as f:
            sequence = json.load(f)             # load annotation
        size = np.array(sequence['obj_dict'][obj_id]['half_length'])

        img_id = img_id.split('/')[-1].split('.')[0]
        output_folder = os.path.join(self.cfg['result_dir'], classname, f'{str(img_id)}_{str(obj_id)}')
        pred_cube_mesh_path = os.path.join(output_folder, 'pred_cube.ply')
        gt_cube_mesh_path = os.path.join(output_folder, 'label_cube.ply')

        output_folder = os.path.join(output_folder, f'object_resize')
        os.makedirs(output_folder, exist_ok=True)

        align_mesh_path = os.path.join(output_folder, 'align.ply')
        if not os.path.exists(pred_cube_mesh_path):
            print(pred_cube_mesh_path)
            self.loginfo.append(f'pred: {pred_cube_mesh_path} is not exist!')
            return
        if not os.path.exists(gt_cube_mesh_path):
            print(gt_cube_mesh_path)
            self.loginfo.append(f'pred: {gt_cube_mesh_path} is not exist!')
            return
        if classname not in self.cd_loss_dict.keys():
            self.cd_loss_dict[classname]=[]
        if classname not in self.f_score_dict.keys():
            self.f_score_dict[classname] = []
        if not(os.path.exists(os.path.join(output_folder, 'cd.txt')) or
            os.path.exists(os.path.join(output_folder, 'f_score.txt'))):
            # load mesh files
            pred_mesh = trimesh.load(pred_cube_mesh_path)
            gt_mesh = trimesh.load(gt_cube_mesh_path)

            pred_mesh.vertices=pred_mesh.vertices/2*size/np.max(size)*2
            gt_mesh.vertices=gt_mesh.vertices/2*size/np.max(size)*2
            pred_mesh_path = os.path.join(output_folder, 'pred.ply')
            gt_mesh_path = os.path.join(output_folder, 'gt.ply')
            pred_mesh.export(pred_mesh_path)
            gt_mesh.export(gt_mesh_path)

            cmd = f"{self.cmd_app} {pred_mesh_path} {gt_mesh_path} {align_mesh_path}"

            ## align mesh use icp
            if os.path.exists(align_mesh_path):
                subprocess.check_output(cmd, shell=True)
            try:
                align_mesh = trimesh.load(align_mesh_path)
            except:
                subprocess.check_output(cmd, shell=True)
                align_mesh = trimesh.load(align_mesh_path)
            ## calculate the cd
            cd_loss, fscore = self.calculate_cd(align_mesh, gt_mesh)
        else:
            with open(os.path.join(output_folder, 'cd.txt'), 'r') as f:
                cd_loss = float(f.readline())
            with open(os.path.join(output_folder, 'f_score.txt'), 'r') as f:
                fscore = float(f.readline())
        self.cd_loss_dict[classname].append(cd_loss)
        self.f_score_dict[classname].append(fscore)
        msg="processing %d/%d %s_%s ,class %s, cd loss: %f, f score %f" % (index,len(self.split), img_id, str(obj_id),classname,cd_loss, fscore)
        with open(os.path.join(output_folder, 'cd.txt'), 'w') as f:
            f.write(str(cd_loss))
        with open(os.path.join(output_folder, 'f_score.txt'), 'w') as f:
            f.write(str(fscore))
        print(msg)
        self.loginfo.append(msg)

    def run(self):
        for index, data in enumerate(self.split):
            self.run_in_one(index)
        self.get_result()


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('ssr evaluate')
    parser.add_argument('--config', type=str, required=True, help='configure file for training or testing.')
    return parser.parse_args()


if __name__ == '__main__':
    fix_random_seed(seed=1029)

    # render = Render(off_screen=True)
    args = parse_args()

    with open(args.config, 'r') as f:
        eval_cfg = yaml.load(f, Loader=yaml.FullLoader)

    dataset = eval_cfg['data']['dataset']
    exp_folder = os.path.join(eval_cfg['save_root_path'], eval_cfg['exp_name'])
    testset = eval_cfg['data']['test_class_name']

    mode ='test'
    config = {
        'result_dir':  os.path.join(exp_folder, 'out'),
        'split_path': f'./dataset/{dataset}/split/test',
        'data_path': eval_cfg['data']['data_path'],
        'log': 'EvaluateLog.txt',
        'debug': True,
        'class_name': testset
    }
    evaluate = Evaluate(config)

    t1 = time.time()
    mp.set_start_method('spawn')
    with parallel_backend('multiprocessing', n_jobs=4):
        Parallel()(delayed(evaluate.run_in_one)(index) for index in range(len(evaluate.split)))

    # check all object align
    evaluate.run()

    t2 = time.time()
    print(f'total time {t2 - t1}s')
