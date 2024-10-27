import os
import logging
import time
import glob
import argparse
import json
import pickle

import os.path as path
import numpy as np
import tqdm
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from collections import defaultdict
from thop import profile


from models.gcnpose import GCNpose, adj_mx_from_edges
from models.gcndiff import GCNdiff, adj_mx_from_edges
from models.ema import EMAHelper

from common.utils import *
from common.utils_diff import get_beta_schedule, generalized_steps_2view
from common.data_utils import fetch_me, fetch_me_2view, read_3d_data_2view, create_2d_data, make_collate_fn, worker_init_fn, prepare_batch
from common.generators import PoseGenerator_gmm, PoseGenerator_gmm_2view
from common.loss import mpjpe, p_mpjpe
from common.multiview import unproj_2d_to_3d, proj_3d_to_2d
from common.mhad_stereo import MHADStereoViewDataset
from common.human36m import Human36MMultiViewDataset
from common.camera import unnormalize_screen_coordinates


class Diffpose(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        self.loss_type = config.training.loss_type if hasattr(config.training, "loss_type") else "2d"
        self.hypothesis_agg = config.testing.hypothesis_agg
        self.model_var_type = config.model.var_type
        self.depth_norm_by_bw = config.model.depth_norm_by_bw if hasattr(config.model, "depth_norm_by_bw") else False
        # GraFormer mask
        self.src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True,
                                True, True, True, True, True, True, True]]]).cuda()
        
        # Generate Diffusion sequence parameters
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

    # prepare 2D and 3D skeleton for model training and testing 
    def prepare_data(self):
        args, config = self.args, self.config
        print('==> Using settings {}'.format(args))
        print('==> Using configures {}'.format(config))
        
        # load dataset
        if config.data.dataset == "human36m":
            from common.h36m_dataset import Human36mDataset, TRAIN_SUBJECTS, TEST_SUBJECTS
            dataset = Human36mDataset(config.data.dataset_path)
            self.subjects_train = TRAIN_SUBJECTS
            self.subjects_test = TEST_SUBJECTS
            self.dataset = read_3d_data_me(dataset)
            self.keypoints_train = create_2d_data(config.data.dataset_path_train_2d, dataset)
            self.keypoints_test = create_2d_data(config.data.dataset_path_test_2d, dataset)

            self.action_filter = None if args.actions == '*' else args.actions.split(',')
            if self.action_filter is not None:
                self.action_filter = map(lambda x: dataset.define_actions(x)[0], self.action_filter)
                print('==> Selected actions: {}'.format(self.action_filter))
        elif config.data.dataset == "human36m_2view":
            from common.h36m_2view_dataset import Human36m2ViewDataset, TRAIN_SUBJECTS, TEST_SUBJECTS
            dataset = Human36m2ViewDataset(config.data.dataset_path)
            self.subjects_train = TRAIN_SUBJECTS
            self.subjects_test = TEST_SUBJECTS
            self.dataset = read_3d_data_2view(dataset)
            self.keypoints_train = create_2d_data(config.data.dataset_path_train_2d, dataset)
            self.keypoints_test = create_2d_data(config.data.dataset_path_test_2d, dataset)

            self.action_filter = None if args.actions == '*' else args.actions.split(',')
            if self.action_filter is not None:
                self.action_filter = map(lambda x: dataset.define_actions(x)[0], self.action_filter)
                print('==> Selected actions: {}'.format(self.action_filter))
        elif config.data.dataset == "mhad_2view":
            pass
        elif config.data.dataset == "h36m_2view":
            pass
        else:
            raise KeyError('Invalid dataset')

    # create diffusion model
    def create_diffusion_model(self, model_path = None):
        args, config = self.args, self.config
        edges = torch.tensor([[2, 6], [3, 6], [7, 6], 
                     [1, 2], [4, 3], [8, 7], [12, 7], [13, 7], 
                     [0, 1], [5, 4], [16, 8], [11, 12], [14, 13],
                     [9, 16], [10, 11], [15, 14]], dtype=torch.long)
        adj = adj_mx_from_edges(num_pts=17, edges=edges, sparse=False)
        self.model_diff = GCNdiff(adj.cuda(), config).cuda()
        self.model_diff = torch.nn.DataParallel(self.model_diff)
        
        # load pretrained model
        if model_path:
            states = torch.load(model_path)
            self.model_diff.load_state_dict(states[0], strict=False)
            
    def create_pose_model(self, model_path = None):
        args, config = self.args, self.config
        
        # [input dimension u v, output dimension x y z]
        config.model.coords_dim = [2,3]
        edges = torch.tensor([[2, 6], [3, 6], [7, 6], 
                     [1, 2], [4, 3], [8, 7], [12, 7], [13, 7], 
                     [0, 1], [5, 4], [16, 8], [11, 12], [14, 13],
                     [9, 16], [10, 11], [15, 14]], dtype=torch.long)
        adj = adj_mx_from_edges(num_pts=17, edges=edges, sparse=False)
        self.model_pose = GCNpose(adj.cuda(), config).cuda()
        self.model_pose = torch.nn.DataParallel(self.model_pose)
        
        # load pretrained model
        if model_path:
            logging.info('initialize model by:' + model_path)
            states = torch.load(model_path)
            self.model_pose.load_state_dict(states[0], strict=False)
        else:
            logging.info('initialize model randomly')
    
    def setup_mhad_dataloaders(self, is_train, flip=False):
        config = self.config
        mhad_dataloader = None
        if is_train:
            # train
            train_dataset = MHADStereoViewDataset(
                mhad_root=config.dataset.train.mhad_root,
                labels_path=config.dataset.train.labels_path,
                pred_results_path=config.dataset.train.pred_results_path if hasattr(
                    config.dataset.train, "pred_results_path") else None,
                pred_2d_results_path=config.dataset.train.pred_2d_results_path if hasattr(
                    config.dataset.train, "pred_2d_results_path") else None,
                pred_2d_error_dis_path=config.dataset.train.pred_2d_error_dis_path if hasattr(
                    config.dataset.train, 'pred_2d_error_dis_path') else None,
                train=True,
                test=False,
                image_shape=config.dataset.image_shape if hasattr(
                    config.dataset, "image_shape") else (256, 256),
                crop=config.dataset.train.crop if hasattr(
                    config.dataset.train, "crop") else True,
                train_subset=config.dataset.train.subset if hasattr(config.dataset.train, "subset") else None,
                rectificated=config.dataset.train.rectificated if hasattr(
                    config.dataset.train, "rectificated") else True,
                baseline_width=config.dataset.train.baseline_width if hasattr(
                    config.dataset.train, "baseline_width") else 'm',
                flip=flip
            )
            train_dataloader = data.DataLoader(
                train_dataset,
                batch_size=config.training.batch_size,
                shuffle=config.training.shuffle,
                collate_fn=make_collate_fn(num_views = config.dataset.num_views),
                num_workers=config.training.num_workers,
                worker_init_fn=worker_init_fn,
                pin_memory=False
            )
            mhad_dataloader = train_dataloader
        else:
            # val
            val_dataset = MHADStereoViewDataset(
                mhad_root=config.dataset.val.mhad_root,
                labels_path=config.dataset.val.labels_path,
                pred_results_path=config.dataset.val.pred_results_path if hasattr(
                    config.dataset.val, "pred_results_path") else None,
                pred_2d_results_path=config.dataset.val.pred_2d_results_path if hasattr(
                    config.dataset.val, "pred_2d_results_path") else None,
                pred_2d_error_dis_path=config.dataset.val.pred_2d_error_dis_path if hasattr(
                    config.dataset.val, 'pred_2d_error_dis_path') else None,
                train=False,
                test=True,
                image_shape=config.dataset.image_shape if hasattr(
                    config.dataset, "image_shape") else (256, 256),
                heatmap_shape=config.dataset.heatmap_shape if hasattr(
                    config.dataset, "heatmap_shape") else (64, 64),
                crop=config.dataset.val.crop if hasattr(
                    config.dataset.val, "crop") else True,
                rectificated=config.dataset.val.rectificated if hasattr(
                    config.dataset.val, "rectificated") else True,
                baseline_width=config.dataset.val.baseline_width if hasattr(
                    config.dataset.val, "baseline_width") else 'm',
                heatmaps_load = "prob" in config.testing.hypothesis_agg
            )
            val_dataloader = data.DataLoader(
                val_dataset,
                batch_size=config.testing.batch_size,
                shuffle=config.testing.shuffle,
                collate_fn=make_collate_fn(num_views=config.dataset.num_views),
                num_workers=config.testing.num_workers,
                worker_init_fn=worker_init_fn,
                pin_memory=False
            )
            mhad_dataloader = val_dataloader

        return mhad_dataloader


    def setup_h36m_dataloaders(self, is_train, flip=False):
        config = self.config
        h36m_dataloader = None
        if is_train:
            # train
            train_dataset = Human36MMultiViewDataset(
                h36m_root=config.dataset.train.h36m_root,
                labels_path=config.dataset.train.labels_path,
                pred_results_path=config.dataset.train.pred_results_path if hasattr(
                    config.dataset.train, "pred_results_path") else None,
                pred_2d_results_path=config.dataset.train.pred_2d_results_path if hasattr(
                    config.dataset.train, "pred_2d_results_path") else None,
                pred_2d_error_dis_path=config.dataset.train.pred_2d_error_dis_path if hasattr(
                    config.dataset.train, 'pred_2d_error_dis_path') else None,
                train=True,
                test=False,
                image_shape=config.dataset.image_shape if hasattr(
                    config.dataset, "image_shape") else (256, 256),
                crop=config.dataset.train.crop if hasattr(
                    config.dataset.train, "crop") else True,
                rectificated=config.dataset.train.rectificated if hasattr(
                    config.dataset.train, "rectificated") else True
            )
            train_dataloader = data.DataLoader(
                train_dataset,
                batch_size=config.training.batch_size,
                shuffle=config.training.shuffle,
                collate_fn=make_collate_fn(num_views = config.dataset.num_views),
                num_workers=config.training.num_workers,
                worker_init_fn=worker_init_fn,
                pin_memory=False
            )
            h36m_dataloader = train_dataloader
        else:
            # val
            val_dataset = Human36MMultiViewDataset(
                h36m_root=config.dataset.val.h36m_root,
                labels_path=config.dataset.val.labels_path,
                pred_results_path=config.dataset.val.pred_results_path if hasattr(
                    config.dataset.val, "pred_results_path") else None,
                pred_2d_results_path=config.dataset.val.pred_2d_results_path if hasattr(
                    config.dataset.val, "pred_2d_results_path") else None,
                pred_2d_error_dis_path=config.dataset.val.pred_2d_error_dis_path if hasattr(
                    config.dataset.val, 'pred_2d_error_dis_path') else None,
                train=False,
                test=True,
                image_shape=config.dataset.image_shape if hasattr(
                    config.dataset, "image_shape") else (256, 256),
                heatmap_shape=config.dataset.heatmap_shape if hasattr(
                    config.dataset, "heatmap_shape") else (64, 64),
                crop=config.dataset.val.crop if hasattr(
                    config.dataset.val, "crop") else True,
                rectificated=config.dataset.val.rectificated if hasattr(
                    config.dataset.val, "rectificated") else True,
                heatmaps_load = "prob" in config.testing.hypothesis_agg
            )
            val_dataloader = data.DataLoader(
                val_dataset,
                batch_size=config.testing.batch_size,
                shuffle=config.testing.shuffle,
                collate_fn=make_collate_fn(num_views=config.dataset.num_views),
                num_workers=config.testing.num_workers,
                worker_init_fn=worker_init_fn,
                pin_memory=False
            )
            h36m_dataloader = val_dataloader

        return h36m_dataloader
    
    def train(self):
        cudnn.benchmark = True

        args, config, src_mask = self.args, self.config, self.src_mask

        # initialize the recorded best performance
        best_abs, best_epoch_abs = 200, 0
        best_re, best_epoch_re = 200, 0
        
        # skip rate when sample skeletons from video
        stride = self.args.downsample
        
        # create dataloader
        if config.data.dataset == "human36m":
            poses_train, poses_train_2d, actions_train, camerapara_train\
                = fetch_me(self.subjects_train, self.dataset, self.keypoints_train, self.action_filter, stride)
            data_loader = train_loader = data.DataLoader(
                PoseGenerator_gmm(poses_train, poses_train_2d, actions_train, camerapara_train),
                batch_size=config.training.batch_size, shuffle=True,\
                    num_workers=config.training.num_workers, pin_memory=True)
        elif config.data.dataset == "human36m_2view":
            from common.h36m_2view_dataset import SELECTED_CAMS
            poses_train, poses_train_2d, actions_train, camerapara_train\
                = fetch_me_2view(self.subjects_train, self.dataset, self.keypoints_train, self.action_filter, stride, selected_cam=SELECTED_CAMS)
            data_loader = train_loader = data.DataLoader(
                PoseGenerator_gmm_2view(poses_train, poses_train_2d, actions_train, camerapara_train),
                batch_size=config.training.batch_size, shuffle=True,\
                    num_workers=config.training.num_workers, pin_memory=True)
        elif config.data.dataset == "mhad_2view":
            data_loader = train_loader = self.setup_mhad_dataloaders(is_train=True)
            # data_loader_f = train_loader_f = self.setup_mhad_dataloaders(is_train=True, flip=True)
        elif config.data.dataset == "h36m_2view":
            data_loader = train_loader = self.setup_h36m_dataloaders(is_train=True)
        else:
            raise KeyError('Invalid dataset')
        
        optimizer = get_optimizer(self.config, self.model_diff.parameters())
        
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(self.model_diff)
        else:
            ema_helper = None
        
        start_epoch, step = 0, 0
        
        lr_init, decay, gamma = self.config.optim.lr, self.config.optim.decay, self.config.optim.lr_gamma
      
        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0

            # Switch to train mode
            torch.set_grad_enabled(True)
            self.model_diff.train()
            
            epoch_loss_diff = AverageMeter()
            weight_3d = torch.Tensor([1.5,1,1.5,
                                     1.5,1,1,
                                     2,1,1,1.2,
                                     2,1.5,1.2,1.2,1.5,2,1]).to(self.device).view(-1,1)
            # for train_loader in [data_loader, data_loader_f]:
            for i, batch in enumerate(train_loader):
                if config.data.dataset == "mhad_2view" or config.data.dataset == "h36m_2view":
                    targets_uvxyz, targets_noise_scale, targets_2d, _, targets_3d, _, cameras_proj, cameras_extrins, _, actions_idx, baseline_width = prepare_batch(batch, self.device, is_train=True)
                elif config.data.dataset == "human36m_2view":
                    (targets_uvxyz, targets_noise_scale, targets_2d, targets_3d, _, cameras_proj) = batch
                data_time += time.time() - data_start
                step += 1

                # generate nosiy bio_2d_kps based on seleted time t and beta
                n = targets_3d.size(0)
                x = targets_2d
                e = torch.randn_like(x)
                b = self.betas            
                t = torch.randint(low=0, high=self.num_timesteps,
                                size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]    
                a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)
                # generate x_t (refer to DDIM equation)
                e = e*(targets_noise_scale[:,:,:x.shape[2]])
                x = x * a.sqrt() + e * (1.0 - a).sqrt()
                
                # predict x_0 from 3d space
                x_3d, _ = unproj_2d_to_3d(x, cameras_proj)
                
                if self.config.data.root_relative:
                    x_3d_root = x_3d[:,6:7]
                    x_3d = x_3d - x_3d_root

                # norm keypoints by baseline width
                if self.depth_norm_by_bw:
                    x_3d[:,:,2] = x_3d[:,:,2] * (baseline_width / 400)
                    
                output_3d = self.model_diff(x_3d, src_mask, t.float(), action=actions_idx, x_depth=x_3d_root[:,0,2])

                macs, params = profile(self.model_diff.module, inputs=(x_3d, src_mask, t.float(), actions_idx, x_3d_root[:,0,2]))
                print("MACs:{}, Params:{}".format(macs, params))
                # denorm keypoints by baseline width
                if self.depth_norm_by_bw:
                    output_3d[:,:,2] = output_3d[:,:,2] * (400 / baseline_width)
                # predict x_2 in 2d image plane
                if self.config.data.root_relative:
                    output_3d = output_3d + x_3d_root
                output_2d = proj_3d_to_2d(output_3d, cameras_proj)
                loss_diff = 0
                if "2d" in self.loss_type:
                    loss_diff += (targets_2d - output_2d).square().sum(dim=(1, 2)).mean(dim=0)
                if "3d" in self.loss_type:
                    # loss_diff += (weight_3d.unsqueeze(0)*((targets_3d - output_3d).square())).sum(dim=(1, 2)).mean(dim=0)
                    loss_diff += ((targets_3d - output_3d).square()).sum(dim=(1, 2)).mean(dim=0)
                
                optimizer.zero_grad()
                loss_diff.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model_diff.parameters(), config.optim.grad_clip)                
                optimizer.step()
            
                epoch_loss_diff.update(loss_diff.item(), n)
            
                if self.config.model.ema:
                    ema_helper.update(self.model_diff)
                
                if i%10 == 0:
                    logging.info('| Epoch{:0>4d}: {:0>4d}/{:0>4d} | Step {:0>6d} | Data: {:.6f} | Loss: {:.6f} |'\
                        .format(epoch, i+1, len(data_loader), step, data_time, epoch_loss_diff.avg))
            
            data_start = time.time()

            if epoch % decay == 0:
                lr_now = lr_decay(optimizer, epoch, lr_init, decay, gamma) 
                
            if epoch % 1 == 0:
                states = [
                    self.model_diff.state_dict(),
                    optimizer.state_dict(),
                    epoch,
                    step,
                ]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())
                torch.save(states, os.path.join(self.args.log_path, "ckpt_last.pth"))
                
                logging.info('test the performance of current model')
                mpjpe_re, mpjpe = self.test_hyber(is_train=True, epoch=epoch)
                if mpjpe_re < best_re:
                    best_re = mpjpe_re
                    best_epoch_re = epoch
                    torch.save(states,os.path.join(self.args.log_path, "ckpt_best_mpjpe_re.pth"))
                if mpjpe < best_abs:
                    best_abs = mpjpe
                    best_epoch_abs = epoch
                    torch.save(states,os.path.join(self.args.log_path, "ckpt_best_mpjpe.pth"))
                logging.info('| Best MPJPE_re Epoch: {:0>4d} MPJPE_re: {:.2f} | Best MPJPE Epoch: {:0>4d} MPJPE: {:.2f} |'\
                    .format(best_epoch_re, best_re, best_epoch_abs, best_abs))
    
    def test_hyber(self, is_train=False, epoch=-1):
        cudnn.benchmark = True

        args, config, src_mask = self.args, self.config, self.src_mask
        test_times, test_timesteps, test_num_diffusion_timesteps, stride = \
            args.test_times, args.test_timesteps, config.testing.test_num_diffusion_timesteps, args.downsample
                
        if config.data.dataset == "human36m":
            poses_valid, poses_valid_2d, actions_valid, camerapara_valid = \
                fetch_me(self.subjects_test, self.dataset, self.keypoints_test, self.action_filter, stride)
            data_loader = valid_loader = data.DataLoader(
                PoseGenerator_gmm(poses_valid, poses_valid_2d, actions_valid, camerapara_valid),
                batch_size=config.training.batch_size, shuffle=False, 
                num_workers=config.training.num_workers, pin_memory=True)
        elif config.data.dataset == "human36m_2view":
            from common.h36m_2view_dataset import SELECTED_CAMS
            poses_valid, poses_valid_2d, actions_valid, camerapara_valid\
                = fetch_me_2view(self.subjects_test, self.dataset, self.keypoints_test, self.action_filter, stride, selected_cam=SELECTED_CAMS)
            data_loader = valid_loader = data.DataLoader(
                PoseGenerator_gmm_2view(poses_valid, poses_valid_2d, actions_valid, camerapara_valid),
                batch_size=config.training.batch_size, shuffle=False, 
                num_workers=config.training.num_workers, pin_memory=True)
        elif config.data.dataset == "mhad_2view":
            data_loader = valid_loader = self.setup_mhad_dataloaders(is_train=is_train)
        elif config.data.dataset == "h36m_2view":
            data_loader = valid_loader = self.setup_h36m_dataloaders(is_train=is_train)
        else:
            raise KeyError('Invalid dataset') 

        data_start = time.time()
        data_time = 0

        # Switch to test mode
        torch.set_grad_enabled(False)
        self.model_diff.eval()
        
        try:
            skip = self.args.skip
        except Exception:
            skip = 1
        
        if self.args.skip_type == "uniform":
            skip = test_num_diffusion_timesteps // test_timesteps
            seq = range(0, test_num_diffusion_timesteps, skip)
        elif self.args.skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(test_num_diffusion_timesteps * 0.8), test_timesteps)** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
        
        epoch_loss_3d_pos = AverageMeter()
        epoch_loss_3d_pos_re = AverageMeter()
        epoch_loss_3d_pos_procrustes = AverageMeter()
        epoch_loss_3d_pos_original = AverageMeter()
        epoch_loss_3d_pos_re_original = AverageMeter()
        epoch_loss_3d_pos_procrustes_original = AverageMeter()
        self.test_action_list = ['Directions','Discussion','Eating','Greeting','Phoning','Photo','Posing','Purchases','Sitting',\
            'SittingDown','Smoking','Waiting','WalkDog','Walking','WalkTogether']
        action_error_sum = define_error_list(self.test_action_list)    
        results = defaultdict(list)    

        for i, batch in enumerate(data_loader):
            _, input_noise_scale, input_2d, targets_2d, targets_3d, input_action, cameras_proj, cameras_extrins, heatmaps_2d, actions_idx, baseline_width, _ = prepare_batch(batch, self.device, is_train=False, align_to_mean=True)
             
            data_time += time.time() - data_start
            batch_size = targets_3d.shape[0]

            # build uvxyz
            input_uvxyz = input_2d
                        
            # generate distribution
            input_uvxyz = input_uvxyz.repeat(test_times,1,1)
            cameras_proj_diff = cameras_proj.repeat(test_times,1,1,1)
            cameras_extrins_diff = cameras_extrins.repeat(test_times,1,1,1)
            actions_idx = actions_idx.repeat(test_times,1)
            baseline_width = baseline_width.repeat(test_times,1)
            input_noise_scale = input_noise_scale[:,:,:input_uvxyz.shape[2]].repeat(test_times,1,1)
            
            # select diffusion step
            t = torch.ones(input_uvxyz.size(0)).type(torch.LongTensor).to(self.device)*test_num_diffusion_timesteps
            
            # prepare the diffusion parameters
            x = input_uvxyz.clone()
            e = torch.randn_like(input_uvxyz)
            b = self.betas   
            e = e*input_noise_scale        
            a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)
            
            output_uvxyz = generalized_steps_2view(x, src_mask, seq, self.model_diff, self.betas, 
                                                   eta=torch.mean(input_noise_scale, dim=[1,2],keepdim=True),  
                                                   cameras_proj=cameras_proj_diff,
                                                   actions_idx = actions_idx, 
                                                   baseline_width = baseline_width,
                                                   cameras_extrins=cameras_extrins_diff,
                                                   root_relatvie=self.config.data.root_relative,
                                                   action_emd=self.config.model.action_emd,
                                                   depth_emd=self.config.model.depth_emd,
                                                   depth_norm_by_bw = self.depth_norm_by_bw)
            outputs_2d = output_uvxyz[0]
            output_uvxyz = output_uvxyz[0][-1]        
            output_uvxyz = output_uvxyz.reshape(test_times,-1,17,4)  
            if self.hypothesis_agg == "pose_reproj":
                reproj_errors = torch.sum(torch.sqrt(torch.sum((output_uvxyz.reshape(test_times, -1, 17, 2, 2) - input_2d.unsqueeze(0).reshape(1, -1, 17, 2, 2))**2, dim=-1)), dim=[-1,-2])
                select_ind = torch.min(reproj_errors, dim=0).indices
                output_uvxyz = torch.gather(output_uvxyz, 0, select_ind.reshape(1, batch_size, 1, 1).repeat(1, 1, 17, 4)).squeeze(0)
    
            elif self.hypothesis_agg == "joint_reproj":
                reproj_errors = torch.sum(torch.sqrt(torch.sum((output_uvxyz.reshape(test_times, -1, 17, 2, 2) - input_2d.unsqueeze(0).reshape(1, -1, 17, 2, 2))**2, dim=-1)), dim=[-1])
                select_ind = torch.min(reproj_errors, dim=0, keepdim=True).indices 
                output_uvxyz = torch.gather(output_uvxyz, 0, select_ind.reshape(1, batch_size, 17, 1).repeat(1, 1, 1, 4)).squeeze(0)
                
            elif self.hypothesis_agg == "pose_prob":
                output_prob_index = output_uvxyz.clone().reshape(test_times, -1, 17, 2, 2)
                output_prob_index = (output_prob_index + 1) / 2 * self.config.dataset.image_shape[0]
                output_prob_index = output_prob_index * (self.config.dataset.heatmap_shape[0]/self.config.dataset.image_shape[0])
                output_prob_index_lu = torch.floor(output_prob_index)
                output_index_prob_lu = 1-(output_prob_index - output_prob_index_lu)
                output_prob_index_rd = torch.ceil(output_prob_index)
                output_index_prob_rd = 1-(output_prob_index_rd - output_prob_index)
                output_prob_index_ld = torch.concat((output_prob_index_lu[:,:,:,:,0:1], output_prob_index_rd[:,:,:,:,1:]),dim=-1)
                output_index_prob_ld = torch.concat((output_index_prob_lu[:,:,:,:,0:1], output_index_prob_rd[:,:,:,:,1:]),dim=-1)
                output_prob_index_ru = torch.concat((output_prob_index_lu[:,:,:,:,1:], output_prob_index_rd[:,:,:,:,0:1]),dim=-1)
                output_index_prob_ru = torch.concat((output_index_prob_lu[:,:,:,:,1:], output_index_prob_rd[:,:,:,:,0:1]),dim=-1)
                reproj_prob_lu = [torch.gather(torch.gather(heatmaps_2d, dim=3,index=output_prob_index_lu[i,:,:,:,1].view(batch_size,17,2,1,1).repeat(1,1,1,1,64).long()) \
                                                     , dim=-1,index=output_prob_index_lu[i,:,:,:,0].view(batch_size,17,2,1,1).long())  for i in range(test_times)]
                reproj_prob_lu = torch.concat(reproj_prob_lu, dim=0).reshape(test_times,batch_size,17,2)
                reproj_prob_rd = [torch.gather(torch.gather(heatmaps_2d, dim=3,index=output_prob_index_rd[i,:,:,:,1].view(batch_size,17,2,1,1).repeat(1,1,1,1,64).long()) \
                                                     , dim=-1,index=output_prob_index_rd[i,:,:,:,0].view(batch_size,17,2,1,1).long())  for i in range(test_times)]
                reproj_prob_rd = torch.concat(reproj_prob_rd, dim=0).reshape(test_times,batch_size,17,2)
                reproj_prob_ld = [torch.gather(torch.gather(heatmaps_2d, dim=3,index=output_prob_index_ld[i,:,:,:,1].view(batch_size,17,2,1,1).repeat(1,1,1,1,64).long()) \
                                                     , dim=-1,index=output_prob_index_ld[i,:,:,:,0].view(batch_size,17,2,1,1).long())  for i in range(test_times)]
                reproj_prob_ld = torch.concat(reproj_prob_ld, dim=0).reshape(test_times,batch_size,17,2)
                reproj_prob_ru = [torch.gather(torch.gather(heatmaps_2d, dim=3,index=output_prob_index_ru[i,:,:,:,1].view(batch_size,17,2,1,1).repeat(1,1,1,1,64).long()) \
                                                     , dim=-1,index=output_prob_index_ru[i,:,:,:,0].view(batch_size,17,2,1,1).long())  for i in range(test_times)]
                reproj_prob_ru = torch.concat(reproj_prob_ru, dim=0).reshape(test_times,batch_size,17,2)
                reproj_prob = (output_index_prob_lu[:,:,:,:,0]*output_index_prob_lu[:,:,:,:,1])*reproj_prob_lu \
                        +(output_index_prob_ru[:,:,:,:,0]*output_index_prob_ru[:,:,:,:,1])*reproj_prob_ru \
                        +(output_index_prob_ld[:,:,:,:,0]*output_index_prob_ld[:,:,:,:,1])*reproj_prob_ld \
                        +(output_index_prob_rd[:,:,:,:,0]*output_index_prob_rd[:,:,:,:,1])*reproj_prob_rd
                reproj_prob = torch.sum(reproj_prob,dim=[2,3])
                select_ind = torch.max(reproj_prob, dim=0).indices
                output_uvxyz = torch.gather(output_uvxyz, 0, select_ind.reshape(1, batch_size, 1, 1).repeat(1, 1, 17, 4)).squeeze(0)
    
            elif self.hypothesis_agg == "joint_prob":
                output_prob_index = output_uvxyz.clone().reshape(test_times, -1, 17, 2, 2)
                output_prob_index = (output_prob_index + 1) / 2 * self.config.dataset.image_shape[0]
                output_prob_index = output_prob_index * (self.config.dataset.heatmap_shape[0]/self.config.dataset.image_shape[0])
                output_prob_index_lu = torch.floor(output_prob_index)
                output_index_prob_lu = 1-(output_prob_index - output_prob_index_lu)
                output_prob_index_rd = torch.ceil(output_prob_index)
                output_index_prob_rd = 1-(output_prob_index_rd - output_prob_index)
                output_prob_index_ld = torch.concat((output_prob_index_lu[:,:,:,:,0:1], output_prob_index_rd[:,:,:,:,1:]),dim=-1)
                output_index_prob_ld = torch.concat((output_index_prob_lu[:,:,:,:,0:1], output_index_prob_rd[:,:,:,:,1:]),dim=-1)
                output_prob_index_ru = torch.concat((output_prob_index_lu[:,:,:,:,1:], output_prob_index_rd[:,:,:,:,0:1]),dim=-1)
                output_index_prob_ru = torch.concat((output_index_prob_lu[:,:,:,:,1:], output_index_prob_rd[:,:,:,:,0:1]),dim=-1)
                reproj_prob_lu = [torch.gather(torch.gather(heatmaps_2d, dim=3,index=output_prob_index_lu[i,:,:,:,1].view(batch_size,17,2,1,1).repeat(1,1,1,1,64).long()) \
                                                     , dim=-1,index=output_prob_index_lu[i,:,:,:,0].view(batch_size,17,2,1,1).long())  for i in range(test_times)]
                reproj_prob_lu = torch.concat(reproj_prob_lu, dim=0).reshape(test_times,batch_size,17,2)
                reproj_prob_rd = [torch.gather(torch.gather(heatmaps_2d, dim=3,index=output_prob_index_rd[i,:,:,:,1].view(batch_size,17,2,1,1).repeat(1,1,1,1,64).long()) \
                                                     , dim=-1,index=output_prob_index_rd[i,:,:,:,0].view(batch_size,17,2,1,1).long())  for i in range(test_times)]
                reproj_prob_rd = torch.concat(reproj_prob_rd, dim=0).reshape(test_times,batch_size,17,2)
                reproj_prob_ld = [torch.gather(torch.gather(heatmaps_2d, dim=3,index=output_prob_index_ld[i,:,:,:,1].view(batch_size,17,2,1,1).repeat(1,1,1,1,64).long()) \
                                                     , dim=-1,index=output_prob_index_ld[i,:,:,:,0].view(batch_size,17,2,1,1).long())  for i in range(test_times)]
                reproj_prob_ld = torch.concat(reproj_prob_ld, dim=0).reshape(test_times,batch_size,17,2)
                reproj_prob_ru = [torch.gather(torch.gather(heatmaps_2d, dim=3,index=output_prob_index_ru[i,:,:,:,1].view(batch_size,17,2,1,1).repeat(1,1,1,1,64).long()) \
                                                     , dim=-1,index=output_prob_index_ru[i,:,:,:,0].view(batch_size,17,2,1,1).long())  for i in range(test_times)]
                reproj_prob_ru = torch.concat(reproj_prob_ru, dim=0).reshape(test_times,batch_size,17,2)
                reproj_prob = (output_index_prob_lu[:,:,:,:,0]*output_index_prob_lu[:,:,:,:,1])*reproj_prob_lu \
                        +(output_index_prob_ru[:,:,:,:,0]*output_index_prob_ru[:,:,:,:,1])*reproj_prob_ru \
                        +(output_index_prob_ld[:,:,:,:,0]*output_index_prob_ld[:,:,:,:,1])*reproj_prob_ld \
                        +(output_index_prob_rd[:,:,:,:,0]*output_index_prob_rd[:,:,:,:,1])*reproj_prob_rd
                reproj_prob = torch.sum(reproj_prob,dim=[3])
                select_ind = torch.max(reproj_prob, dim=0, keepdim=True).indices 
                output_uvxyz = torch.gather(output_uvxyz, 0, select_ind.reshape(1, batch_size, 17, 1).repeat(1, 1, 1, 4)).squeeze(0)
            
            else:
                output_uvxyz = torch.mean(output_uvxyz,0)
            output_xyz, output_2d = unproj_2d_to_3d(output_uvxyz, cameras_proj)

            results['keypoints_3d'].append(output_xyz.detach().cpu().numpy())
            for output_i in range(len(outputs_2d)):
                results['keypoints_2d_iters_{}'.format(output_i)].append(outputs_2d[output_i].detach().cpu().numpy())
            results['keypoints_3d_gt'].append(targets_3d.detach().cpu().numpy())
            results['keypoints_2d'].append(output_2d.detach().cpu().numpy())
            results['keypoints_2d_gt'].append(targets_2d.detach().cpu().numpy())
            results['indexes'].append(batch['indexes'])
            
            # diffusion method error
            epoch_loss_3d_pos.update(mpjpe(output_xyz, targets_3d).item() * 1000.0, targets_3d.size(0))
            epoch_loss_3d_pos_re.update(mpjpe(output_xyz[:, :, :]-output_xyz[:, 6:7, :], targets_3d[:, :, :]-targets_3d[:, 6:7, :]).item() * 1000.0, targets_3d.size(0))
            epoch_loss_3d_pos_procrustes.update(p_mpjpe((output_xyz[:, :, :]-output_xyz[:, 6:7, :]).cpu().numpy(), (targets_3d[:, :, :]-targets_3d[:,6:7, :]).cpu().numpy()).item() * 1000.0, targets_3d.size(0))\
            
            data_start = time.time()

            # original method error
            input_xyz, input_2d_ori = unproj_2d_to_3d(input_2d, cameras_proj)
            results['keypoints_3d_original'].append(input_xyz.detach().cpu().numpy())
            results['keypoints_2d_original'].append(input_2d_ori.detach().cpu().numpy())
            epoch_loss_3d_pos_original.update(mpjpe(input_xyz, targets_3d).item() * 1000.0, targets_3d.size(0))
            epoch_loss_3d_pos_re_original.update(mpjpe(input_xyz[:, :, :]-input_xyz[:, 6:7, :], targets_3d[:, :, :]-targets_3d[:, 6:7, :]).item() * 1000.0, targets_3d.size(0))
            epoch_loss_3d_pos_procrustes_original.update(p_mpjpe((input_xyz[:, :, :]-input_xyz[:, 6:7, :]).cpu().numpy(), (targets_3d[:, :, :]-targets_3d[:,6:7, :]).cpu().numpy()).item() * 1000.0, targets_3d.size(0))\
            
            if i%10 == 0: # and i != 0:
                logging.info('({batch}/{size}) Data: {data:.6f}s | MPJPE: {e1: .4f} MPJPE_re: {e2: .4f} P-MPJPE: {e3: .4f} (after diffusion) | MPJPE: {e4: .4f} MPJPE_re: {e5: .4f} P-MPJPE: {e6: .4f} (before diffusion)'\
                        .format(batch=i + 1, size=len(data_loader), data=data_time,\
                                e1=epoch_loss_3d_pos.avg, e2=epoch_loss_3d_pos_re.avg, e3=epoch_loss_3d_pos_procrustes.avg,\
                                e4=epoch_loss_3d_pos_original.avg, e5=epoch_loss_3d_pos_re_original.avg, e6=epoch_loss_3d_pos_procrustes_original.avg))
        logging.info('sum ({batch}/{size}) Data: {data:.6f}s | MPJPE: {e1: .4f} MPJPE_re: {e2: .4f} P-MPJPE: {e3: .4f} (after diffusion) | MPJPE: {e4: .4f} MPJPE_re: {e5: .4f} P-MPJPE: {e6: .4f} (before diffusion)'\
                    .format(batch=i + 1, size=len(data_loader), data=data_time,\
                            e1=epoch_loss_3d_pos.avg, e2=epoch_loss_3d_pos_re.avg, e3=epoch_loss_3d_pos_procrustes.avg,\
                            e4=epoch_loss_3d_pos_original.avg, e5=epoch_loss_3d_pos_re_original.avg, e6=epoch_loss_3d_pos_procrustes_original.avg))
        
        
        results['keypoints_3d'] = np.concatenate(
            results['keypoints_3d'], axis=0)
        for output_i in range(len(outputs_2d)):
            results['keypoints_2d_iters_{}'.format(output_i)] = np.concatenate(
                results['keypoints_2d_iters_{}'.format(output_i)], axis=0)
        results['keypoints_3d_original'] = np.concatenate(
            results['keypoints_3d_original'], axis=0)
        results['keypoints_3d_gt'] = np.concatenate(
            results['keypoints_3d_gt'], axis=0)
        results['keypoints_2d'] = np.concatenate(
            results['keypoints_2d'], axis=0)
        results['keypoints_2d_original'] = np.concatenate(
            results['keypoints_2d_original'], axis=0)
        results['keypoints_2d_gt'] = np.concatenate(
            results['keypoints_2d_gt'], axis=0)
        results['indexes'] = np.concatenate(
            results['indexes'], axis=0)
        try:
            scalar_metric_re, scalar_metric,full_metric = data_loader.dataset.evaluate(
                results['keypoints_3d'])
        except Exception as e:
            print("Failed to evaluate. Reason: ", e)
            scalar_metric_re, scalar_metric, full_metric = 0.0, 0.0, {}
        
        logging.info('Epoch {epoch} | MPJPE: {e1: .4f} MPJPE_re: {e2: .4f}'\
                    .format(epoch=epoch,\
                            e1=scalar_metric, e2=scalar_metric_re))
        
        os.makedirs(os.path.join(self.args.log_path, "{:04}".format(epoch)), exist_ok=True)
        np.save(os.path.join(self.args.log_path, "{:04}".format(epoch), "keypoints_2d_ori.npy"), results['keypoints_2d_original'])
        np.save(os.path.join(self.args.log_path, "{:04}".format(epoch), "keypoints_3d_dualdiff.npy"), results['keypoints_3d'])
        np.save(os.path.join(self.args.log_path, "{:04}".format(epoch), "keypoints_3d_tri.npy"), results['keypoints_3d_original'])
        np.save(os.path.join(self.args.log_path, "{:04}".format(epoch), "keypoints_3d_gt.npy"), results['keypoints_3d_gt'])
        np.save(os.path.join(self.args.log_path, "{:04}".format(epoch), "index.npy"), results['indexes'])
        
        # dump results
        with open(os.path.join(self.args.log_path, "{:04}".format(epoch), "results.pkl"), 'wb') as fout:
            pickle.dump(results, fout)
        # dump metrics
        with open(os.path.join(self.args.log_path, "{:04}".format(epoch), "metric.json"), 'w') as fout:
            json.dump(full_metric, fout, indent=4, sort_keys=True)

        logging.info('----------------------Epoch: {}, test done!-------------------'.format(epoch))
        return scalar_metric_re, scalar_metric
    

    def simu(self):
        cudnn.benchmark = True

        args, config, src_mask = self.args, self.config, self.src_mask
        test_times, test_timesteps, test_num_diffusion_timesteps, stride = \
            args.test_times, args.test_timesteps, config.testing.test_num_diffusion_timesteps, args.downsample

        # initialize the recorded best performance
        best_abs, best_epoch_abs = 200, 0
        best_re, best_epoch_re = 200, 0
        
        # skip rate when sample skeletons from video
        stride = self.args.downsample
        
        if self.args.skip_type == "uniform":
            skip = test_num_diffusion_timesteps // test_timesteps
            seq = range(0, test_num_diffusion_timesteps, skip)
        elif self.args.skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(test_num_diffusion_timesteps * 0.8), test_timesteps)** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
        
        # create dataloader
        if config.data.dataset == "human36m":
            poses_train, poses_train_2d, actions_train, camerapara_train\
                = fetch_me(self.subjects_train, self.dataset, self.keypoints_train, self.action_filter, stride)
            data_loader = train_loader = data.DataLoader(
                PoseGenerator_gmm(poses_train, poses_train_2d, actions_train, camerapara_train),
                batch_size=config.training.batch_size, shuffle=True,\
                    num_workers=config.training.num_workers, pin_memory=True)
        elif config.data.dataset == "human36m_2view":
            from common.h36m_2view_dataset import SELECTED_CAMS
            poses_train, poses_train_2d, actions_train, camerapara_train\
                = fetch_me_2view(self.subjects_train, self.dataset, self.keypoints_train, self.action_filter, stride, selected_cam=SELECTED_CAMS)
            data_loader = train_loader = data.DataLoader(
                PoseGenerator_gmm_2view(poses_train, poses_train_2d, actions_train, camerapara_train),
                batch_size=config.training.batch_size, shuffle=True,\
                    num_workers=config.training.num_workers, pin_memory=True)
        elif config.data.dataset == "mhad_2view":
            data_loader = train_loader = self.setup_mhad_dataloaders(is_train=True)
            # data_loader_f = train_loader_f = self.setup_mhad_dataloaders(is_train=True, flip=True)
        elif config.data.dataset == "h36m_2view":
            data_loader = train_loader = self.setup_h36m_dataloaders(is_train=True)
        else:
            raise KeyError('Invalid dataset')
    
        
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(self.model_diff)
        else:
            ema_helper = None
        
        start_epoch, step = 0, 0
        
        data_start = time.time()
        data_time = 0

        results = defaultdict(list)    
        # for train_loader in [data_loader, data_loader_f]:
        for i, batch in enumerate(train_loader):
            if i>1:
                break
            if config.data.dataset == "mhad_2view" or config.data.dataset == "h36m_2view":
                targets_uvxyz, targets_noise_scale, targets_2d, _, targets_3d, _, cameras_proj, cameras_extrins, _, actions_idx, baseline_width = prepare_batch(batch, self.device, is_train=True)
            elif config.data.dataset == "human36m_2view":
                (targets_uvxyz, targets_noise_scale, targets_2d, targets_3d, _, cameras_proj) = batch
            data_time += time.time() - data_start
            step += 1

            # diffusion
            # generate nosiy bio_2d_kps based on seleted time t and beta
            n = targets_3d.size(0)
            x = targets_2d
            
            b = self.betas            
            t = torch.randint(low=0, high=self.num_timesteps,
                            size=(n // 2 + 1,)).to(self.device)
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]    
            x_peturbed = []
            for times in range(self.num_timesteps):
                t = torch.full_like(t, times)
                a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)
                # generate x_t (refer to DDIM equation)
                e = torch.randn_like(x)
                e = e*(targets_noise_scale[:,:,:x.shape[2]])
                x_p = x * a.sqrt() + e * (1.0 - a).sqrt()
                x_peturbed.append(x_p)
            
            # denoise
            output_uvxyz = generalized_steps_2view(x_p, src_mask, seq, self.model_diff, self.betas, 
                                                eta=0,  
                                                cameras_proj=cameras_proj,
                                                actions_idx = actions_idx, 
                                                baseline_width = baseline_width,
                                                cameras_extrins=cameras_extrins,
                                                root_relatvie=self.config.data.root_relative,
                                                action_emd=self.config.model.action_emd,
                                                depth_emd=self.config.model.depth_emd,
                                                depth_norm_by_bw = self.depth_norm_by_bw)
            outputs_2d = output_uvxyz[0]
            for input_i in range(len(x_peturbed)):
                results['keypoints_2d_input_{}'.format(input_i)].append(x_peturbed[input_i].detach().cpu().numpy())
            for output_i in range(len(outputs_2d)):
                results['keypoints_2d_output_{}'.format(output_i)].append(outputs_2d[output_i].detach().cpu().numpy())
            results['keypoints_2d_gt'].append(targets_2d.detach().cpu().numpy())
        
        for input_i in range(len(x_peturbed)):
            results['keypoints_2d_input_{}'.format(input_i)] = np.concatenate(
                results['keypoints_2d_input_{}'.format(input_i)], axis=0)
        for output_i in range(len(outputs_2d)):
            results['keypoints_2d_output_{}'.format(output_i)] = np.concatenate(
                results['keypoints_2d_output_{}'.format(output_i)], axis=0)
        results['keypoints_2d_gt'] = np.concatenate(
            results['keypoints_2d_gt'], axis=0)
        
        os.makedirs(os.path.join(self.args.log_path), exist_ok=True)
        # dump results
        with open(os.path.join(self.args.log_path, "results.pkl"), 'wb') as fout:
            pickle.dump(results, fout)