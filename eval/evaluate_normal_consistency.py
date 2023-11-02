# modify from Occupancy Network 
import logging
import numpy as np
import trimesh
from scipy.spatial import KDTree
import time
import os
import argparse
import yaml


def scale_mesh(pred_mesh, gt_mesh):
    gt_mesh_bbox = gt_mesh.bounding_box.extents                       # [lx, ly, lz]
    pred_mesh_bbox = pred_mesh.bounding_box.extents                   # [lx, ly, lz]
    scale = gt_mesh_bbox / pred_mesh_bbox

    pred_mesh.apply_scale(scale)
    pred_mesh.export(os.path.join('testcode/normal_consistency', 'pred_scale.ply'))

    return pred_mesh


def sample_points_normals(mesh, n_points):
    ''' Sample points and normals from mesh.

    Args:
        mesh (trimesh): mesh
        n_points (int): number of points to sample

    '''
    pointcloud, face_idx = mesh.sample(n_points, return_index=True)             # sample points on surface, use face_idx to get normals
    pointcloud = np.asarray(pointcloud)
    pointcloud = pointcloud.astype(np.float32)
    normals = mesh.face_normals[face_idx]

    return pointcloud, normals


def distance_p2m(points, mesh):
    ''' Compute minimal distances of each point in points to mesh.

    Args:
        points (numpy array): points array
        mesh (trimesh): mesh

    '''
    _, dist, _ = trimesh.proximity.closest_point(mesh, points)
    return dist


def normals_dot_product(normals_src, normals_tgt):
    ''' Compute dot product of normals_src and normals_tgt.

    Args:
        normals_src (numpy array): normals of source mesh
        normals_tgt (numpy array): normals of target mesh

    '''
    normals_src = normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
    normals_tgt = normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

    normals_dot_product = (normals_tgt * normals_src).sum(axis=-1)
    normals_dot_product = np.abs(normals_dot_product)

    return normals_dot_product


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def eval_normal_consistency(pred_mesh, gt_mesh, n_points=100000):
    ''' Evaluate normal consistency.

    Args:
        pred_mesh (trimesh): predicted mesh
        gt_mesh (trimesh): ground truth mesh

    '''
    # Sample points on surface
    pred_points, pred_normals = sample_points_normals(pred_mesh, n_points)
    gt_points, gt_normals = sample_points_normals(gt_mesh, n_points)

    # Completeness: how far are the points of the target point cloud
    # from thre predicted point cloud
    _, completeness_normals = distance_p2p(
        gt_points, gt_normals, pred_points, pred_normals
    )
    completeness_normals = completeness_normals.mean()

    # Accuracy: how far are th points of the predicted pointcloud
    # from the target pointcloud
    _, accuracy_normals = distance_p2p(
        pred_points, pred_normals, gt_points, gt_normals
    )
    accuracy_normals = accuracy_normals.mean()

    normals_correctness = (
        0.5 * completeness_normals + 0.5 * accuracy_normals
    )

    return normals_correctness


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('ssr inference')
    parser.add_argument('--config', type=str, required=True, help='configure file for training or testing.')
    return parser.parse_args()


if __name__ == '__main__':

    args=parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    root_path = cfg['save_root_path']
    exp_name = cfg['exp_name']
    exp_path = os.path.join(root_path, exp_name)
    dataset_type = cfg['data']['dataset']

    if dataset_type == 'FRONT3D':
        category_list = ['sofa', 'bed', 'bookshelf', 'cabinet', 'desk', 'chair', 'night_stand', 'table']
    else:           # Pix3D
        category_list = ['sofa', 'bed', 'bookcase', 'desk', 'chair', 'table', 'misc', 'tool', 'wardrobe']

    cal_dic = {}
    num_dic = {}

    normals_correctness_total = 0
    number_total = 0

    for category in category_list:
        if category not in cal_dic.keys():
            cal_dic[category] = 0
            num_dic[category] = 0

        obj_list = os.listdir(os.path.join(exp_path, 'out', category))

        for obj_name in obj_list:

            output_folder = os.path.join(exp_path, 'out', category, obj_name, 'object_resize')
            if os.path.exists(os.path.join(output_folder, 'normals_correctness.txt')):
                with open(os.path.join(output_folder, 'normals_correctness.txt'), 'r') as f:
                    normals_correctness = float(f.readline())
                cal_dic[category] += normals_correctness
                num_dic[category] += 1
                normals_correctness_total += normals_correctness
                number_total += 1
                print(f'{exp_name} {obj_name} has been evaluated')
                continue

            pred_path = os.path.join(exp_path, 'out', category, obj_name, 'object_resize', 'pred.ply')
            gt_path = os.path.join(exp_path, 'out', category, obj_name, 'object_resize', 'gt.ply')

            if not os.path.exists(pred_path) or not os.path.exists(gt_path):
                print(f'{exp_name} {obj_name} has not been evaluated')
                continue

            t1 = time.time()

            pred_mesh = trimesh.load(pred_path)
            gt_mesh = trimesh.load(gt_path)

            normals_correctness = eval_normal_consistency(pred_mesh, gt_mesh)

            with open(os.path.join(output_folder, 'normals_correctness.txt'), 'w') as f:
                f.write(str(normals_correctness))

            cal_dic[category] += normals_correctness
            num_dic[category] += 1

            normals_correctness_total += normals_correctness
            number_total += 1

            t2 = time.time()

            print(f'{exp_name} {obj_name}: {normals_correctness:.5f} {number_total}')
            # print(f'cost time: {t2-t1}')

        print(f'{exp_name} {category}: {cal_dic[category]/num_dic[category]:.5f}')

    save_folder = os.path.join(exp_path, 'out')
    with open(os.path.join(save_folder, 'normals_correctness.txt'), 'w') as f:
        for category in category_list:
            f.write(f'{category}: {cal_dic[category]/num_dic[category]:.5f}\n')

        f.write(f'mean: {normals_correctness_total/number_total:.5f}\n')

    print('done')
