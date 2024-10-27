from __future__ import absolute_import, division

import numpy as np
import torch

from .camera import world_to_camera, normalize_screen_coordinates, Camera

camera_dict = {
    '54138969': [2.2901, 2.2876, 0.0251, 0.0289],
    '55011271': [2.2994, 2.2952, 0.0177, 0.0161],
    '58860488': [2.2983, 2.2976, 0.0396, 0.0028],
    '60457274': [2.2910, 2.2895, 0.0299, 0.0018],
}

def read_3d_data(dataset):
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                # pos_3d[:, :] -= pos_3d[:, :1]  # Remove global offset
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

    return dataset


def read_3d_data_me(dataset):
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            positions_3d = []
            camerad_para = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                # pos_3d[:, :] -= pos_3d[:, :1]  # Remove global offset
                positions_3d.append(pos_3d)
                camerad_para.append(camera_dict[cam['id']])
    
            anim['positions_3d'] = positions_3d
            anim['camerad_para'] = camerad_para

    return dataset

def read_3d_data_2view(dataset):
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            positions_3d = []
            cameras_para = []
            for cam_idx, cam in enumerate(anim['cameras']):
                pos_3d = anim['positions']
                camera_paras = Camera(cam['R'], cam['T'],
                                   cam['t'], cam['K'],
                                   cam['dist'])
                positions_3d.append(pos_3d)
                cameras_para.append(camera_paras.projection)
                if True in np.isnan(camera_paras.projection):
                    print(f"camera_projection_matrix of subject-{subject}, action-{action}, cam-{cam_idx} is wrong!")
    
            anim['positions_3d'] = positions_3d
            anim['camerad_para'] = cameras_para

    return dataset


def read_3d_data_me_xyz(dataset):
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            positions_3d = []
            camerad_para = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, :] -= pos_3d[:, :1]  # Remove global offset
                positions_3d.append(pos_3d)
                camerad_para.append(camera_dict[cam['id']])
    
            anim['positions_3d'] = positions_3d
            anim['camerad_para'] = camerad_para

    return dataset

def create_2d_data(data_path, dataset):
    keypoints = np.load(data_path, allow_pickle=True)
    keypoints = keypoints['positions_2d'].item()

    ### GJ: adjust the length of 2d data ###
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            for cam_idx in range(len(keypoints[subject][action])):
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]


    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., 1:3] = normalize_screen_coordinates(kps[..., 1:3], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    return keypoints

def fetch(subjects, dataset, keypoints, action_filter=None, stride=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_actions = []

    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    # if action.startswith(a):
                    if action.split(' ')[0] == a:
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
                out_actions.append([action.split(' ')[0]] * poses_2d[i].shape[0])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_poses_3d) == 0:
        out_poses_3d = None

    if stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            out_actions[i] = out_actions[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_poses_3d, out_poses_2d, out_actions


def fetch_me(subjects, dataset, keypoints, action_filter=None, stride=1, parse_3d_poses=True, selected_cam=None):
    out_poses_3d = []
    out_poses_2d = []
    out_actions = []
    out_camera_para = []
    
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    # if action.startswith(a):
                    if action.split(' ')[0] == a:
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                if len(poses_2d) < 4:
                    print(i)
                if selected_cam is not None:
                    if i not in selected_cam:
                        continue
                out_poses_2d.append(poses_2d[i])
                out_actions.append([action.split(' ')[0]] * poses_2d[i].shape[0])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:               
                poses_3d = dataset[subject][action]['positions_3d']
                camera_para = dataset[subject][action]['camerad_para']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    if selected_cam is not None:
                        if i not in selected_cam:
                            continue
                    out_poses_3d.append(poses_3d[i])
                    out_camera_para.append([camera_para[i]]* poses_3d[i].shape[0])

    if len(out_poses_3d) == 0:
        out_poses_3d = None

    if stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            out_actions[i] = out_actions[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]
                out_camera_para[i] = out_poses_3d[i][::stride]
                
    return out_poses_3d, out_poses_2d, out_actions, out_camera_para

def fetch_me_2view(subjects, dataset, keypoints, action_filter=None, stride=1, parse_3d_poses=True, selected_cam=None):
    out_poses_3d = []
    out_poses_2d = []
    out_actions = []
    out_camera_para = []
    
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    # if action.startswith(a):
                    if action.split(' ')[0] == a:
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            out_poses_2d_views = []
            for i in range(len(poses_2d)):  # Iterate across cameras
                if len(poses_2d) < 4:
                    print(i)
                if selected_cam is not None:
                    if i not in selected_cam:
                        continue
                out_poses_2d_views.append(poses_2d[i])
            out_poses_2d.append(np.array(out_poses_2d_views).swapaxes(0,1))
            
            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:               
                poses_3d = dataset[subject][action]['positions_3d']
                camera_para = dataset[subject][action]['camerad_para']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                out_camera_para_views = []
                for i in range(len(poses_3d)):  # Iterate across cameras
                    if selected_cam is not None:
                        if i not in selected_cam:
                            continue
                    out_camera_para_views.append([camera_para[i]]* poses_3d[i].shape[0])
                out_camera_para.append(np.array(out_camera_para_views).swapaxes(0,1))
                out_poses_3d.append(poses_3d[0])
                out_actions.append([action.split(' ')[0]] * poses_3d[0].shape[0])

    if len(out_poses_3d) == 0:
        out_poses_3d = None

    if stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            out_actions[i] = out_actions[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]
                out_camera_para[i] = out_poses_3d[i][::stride]
                
    return out_poses_3d, out_poses_2d, out_actions, out_camera_para


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def make_collate_fn(num_views):

    def collate_fn(items):
        items = list(filter(lambda x: x is not None, items))
        if len(items) == 0:
            print("All items in batch are None")
            return None

        batch = dict()
        total_n_views = min(len(item['keypoints_2d']) for item in items)
        indexes = np.arange(total_n_views)
        
        batch['proj_matrices'] = [[item['proj_matrices'][i] for item in items] for i in indexes]
        batch['extrins_matrices'] = [[item['extrins_matrices'][i] for item in items] for i in indexes]
        batch['keypoints_3d'] = [item['keypoints_3d'] for item in items]
        batch['keypoints_2d'] = [[item['keypoints_2d'][i] for item in items] for i in indexes]
        batch['indexes'] = [item['indexes'] for item in items]
        batch['action'] = [item['action'] for item in items]
        batch['action_idx'] = [item['action_idx'] for item in items]
        try:
            batch['baseline_width'] = [item['baseline_width'] for item in items]
        except:
            pass
        try:
            batch['pred_keypoints_2d'] = [[item['pred_keypoints_2d'][i] for item in items] for i in indexes]
        except:
            pass
        try:
            batch['heatmaps'] = [item['heatmaps'] for item in items]
        except:
            pass
        try:
            batch['pred_2d_kernel'] = [[item['pred_2d_kernel'][i] for item in items] for i in indexes]
        except:
            pass
        try:
            batch['pred_keypoints_3d'] = np.array([item['pred_keypoints_3d'] for item in items])
        except:
            pass
        return batch

    return collate_fn


def prepare_batch(batch, device, is_train, align_to_mean=True):
    # 3D keypoints
    keypoints_3d_batch_gt = torch.from_numpy(np.stack(batch['keypoints_3d'], axis=0)[:, :, :3]).float().to(device)

    # projection matricies
    proj_matricies_batch =  torch.from_numpy(np.stack(batch['proj_matrices'], axis=0)).float().to(device).transpose(1, 0)
    extrins_matrices_batch =  torch.from_numpy(np.stack(batch['extrins_matrices'], axis=0)).float().to(device).transpose(1, 0)

    # action
    action_batch =  np.stack(batch['action'], axis=0)
    action_idx_batch = torch.from_numpy(np.stack(batch['action_idx'], axis=0)).float().to(device).view(-1,1)

    # baseline width
    baseline_width_batch = None
    try:
        baseline_width_batch = torch.from_numpy(np.stack(batch['baseline_width'], axis=0)).float().to(device).view(-1,1)
    except:
        pass

    # 2D keypoints
    keypoints_2d_batch_gt = torch.from_numpy(np.stack(batch['keypoints_2d'], axis=0))[:,:,:,:2].float().to(device).transpose(1, 0)
    pred_2d_kernel_batch = torch.from_numpy(np.stack(batch['pred_2d_kernel'], axis=0))[:,:,:,:4].float().to(device).transpose(1, 0)
    heatmaps_2d_pred=None
    keypoints_2d_batch_pred = None
    keypoints_3d_batch_pred = None
    try:
        keypoints_2d_batch_pred = torch.from_numpy(np.stack(batch['pred_keypoints_2d'], axis=0))[:,:,:,:2].float().to(device).transpose(1, 0)
    except:
        pass
    try:
        keypoints_3d_batch_pred =  torch.from_numpy(np.stack(batch['pred_keypoints_3d'], axis=0)[:, :, :3]).float().to(device)
    except:
        pass
    try:
        heatmaps_2d_pred = torch.from_numpy(np.stack(batch['heatmaps'], axis=0)).float().to(device)
        heatmaps_2d_pred = heatmaps_2d_pred.permute(0,2,1,3,4)
    except:
        pass

    batch_size = keypoints_3d_batch_gt.shape[0]
    n_view = keypoints_2d_batch_gt.shape[1]
    out_pose_3d = keypoints_3d_batch_gt
    out_camerapara = proj_matricies_batch
    out_cameraextrin = extrins_matrices_batch
    out_action = action_batch
    out_action_idx = action_idx_batch
    out_baseline_width = baseline_width_batch
    out_targets_2d = keypoints_2d_batch_gt
    if is_train:
        try:
            keypoints_2d_batch_gt = keypoints_2d_batch_gt.permute(0,2,1,3).reshape(batch_size, -1, n_view*2)
            out_pose_uvxyz = torch.concat((keypoints_2d_batch_gt, out_pose_3d), dim=-1).float().to(device)
            out_pose_2d = keypoints_2d_batch_gt
        except:
            pass
    else:
        # align to the distribution with center of gt
        try:
            if align_to_mean:
                keypoints_2d_batch_pred = keypoints_2d_batch_pred - pred_2d_kernel_batch[:,:,:,:2]
            keypoints_2d_batch_pred = keypoints_2d_batch_pred.permute(0,2,1,3).reshape(batch_size, -1, n_view*2)
            out_pose_uvxyz = torch.concat((keypoints_2d_batch_pred, out_pose_3d), dim=-1).float().to(device)
            out_pose_2d = keypoints_2d_batch_pred
        except:
            pass
    
    kernel_variance = pred_2d_kernel_batch[:,:,:,2:].permute(0,2,1,3).reshape(batch_size, -1, n_view*2)
    out_pose_noise_scale = torch.concat((kernel_variance,torch.ones_like(out_pose_3d)), dim=-1)
    
    return out_pose_uvxyz, out_pose_noise_scale, out_pose_2d, out_targets_2d, out_pose_3d, out_action, out_camerapara, out_cameraextrin, heatmaps_2d_pred, out_action_idx, out_baseline_width, keypoints_3d_batch_pred