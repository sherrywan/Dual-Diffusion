'''
Author: sherrywaan sherrywaan@outlook.com
Date: 2024-04-09 09:43:55
LastEditors: sherrywaan sherrywaan@outlook.com
LastEditTime: 2024-05-16 22:32:59
FilePath: /wxy/3d_pose/Diffpose/common/utils_diff.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from __future__ import absolute_import, division

import os
import torch
import numpy as np
from common.multiview import unproj_2d_to_3d, proj_3d_to_2d

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
    return a


def generalized_steps(x, src_mask, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).cuda()
            next_t = (torch.ones(n) * j).cuda()
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1]
            et = model(xt, src_mask, t.float(), 0)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t)
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next)

    return xs, x0_preds

def generalized_steps_2view_from3d(x, src_mask, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).cuda()
            next_t = (torch.ones(n) * j).cuda()
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1]
            x_3d_t = xt
            if kwargs['root_relatvie']:
                x_3d_t_root = x_3d_t[:,6:7]
                x_3d_t = x_3d_t - x_3d_t_root
            # norm keypoints by baseline width
            if kwargs['depth_norm_by_bw']:
                x_3d_t[:,:,2] = x_3d_t[:,:,2] * (kwargs['baseline_width'] / 400)

            if 'visible' in kwargs.keys():
                x_3d_0, attn = model(x_3d_t, src_mask, t.float(), action=kwargs['actions_idx'], x_depth=x_3d_t_root[:,0,2], visible=True)      
            else:
                x_3d_0 = model(x_3d_t, src_mask, t.float(), action=kwargs['actions_idx'], x_depth=x_3d_t_root[:,0,2])
            
            # denorm keypoints by baseline width
            if kwargs['depth_norm_by_bw']:
                x_3d_0[:,:,2] = x_3d_0[:,:,2] * (400 / kwargs['baseline_width'])
            if kwargs['root_relatvie']:
                x_3d_0 = x_3d_0 + x_3d_t_root
            x0_t = x_3d_0
            # x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            et = (xt - x0_t*at.sqrt()) / ((1 - at).sqrt())
            x0_preds.append(x0_t)
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next)
    if 'visible' in kwargs.keys():
        return xs, x0_preds, attn
    else:
        return xs, x0_preds

def generalized_steps_2view(x, src_mask, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).cuda()
            next_t = (torch.ones(n) * j).cuda()
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1]
            x_3d_t, _ = unproj_2d_to_3d(xt, kwargs['cameras_proj'])
            if kwargs['root_relatvie']:
                x_3d_t_root = x_3d_t[:,6:7]
                x_3d_t = x_3d_t - x_3d_t_root
                # # 3d keypoints root in camera coordiate
                # x_3d_root_h = torch.ones((x_3d_t_root.shape[0],1,4)).to(x_3d_t_root.device)
                # x_3d_root_h[:,:,:-1] = x_3d_t_root
                # x_3d_t_root_ca = (kwargs['cameras_extrins'][:, 0:1] @ x_3d_root_h.unsqueeze(-1)).squeeze(-1)
            # norm keypoints by baseline width
            if kwargs['depth_norm_by_bw']:
                x_3d_t[:,:,2] = x_3d_t[:,:,2] * (kwargs['baseline_width'] / 400)

            if 'visible' in kwargs.keys():
                x_3d_0, attn = model(x_3d_t, src_mask, t.float(), action=kwargs['actions_idx'], x_depth=x_3d_t_root[:,0,2], visible=True)      
            else:
                x_3d_0 = model(x_3d_t, src_mask, t.float(), action=kwargs['actions_idx'], x_depth=x_3d_t_root[:,0,2])
            
            # denorm keypoints by baseline width
            if kwargs['depth_norm_by_bw']:
                x_3d_0[:,:,2] = x_3d_0[:,:,2] * (400 / kwargs['baseline_width'])
            if kwargs['root_relatvie']:
                x_3d_0 = x_3d_0 + x_3d_t_root
            x0_t = proj_3d_to_2d(x_3d_0,  kwargs['cameras_proj'])
            # x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            et = (xt - x0_t*at.sqrt()) / ((1 - at).sqrt())
            x0_preds.append(x0_t)
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next)
    if 'visible' in kwargs.keys():
        return xs, x0_preds, attn
    else:
        return xs, x0_preds
    

def generalized_steps_2view_only2d(x, src_mask, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).cuda()
            next_t = (torch.ones(n) * j).cuda()
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1]
            x_3d_t = xt
            if kwargs['root_relatvie']:
                x_3d_t_root = x_3d_t[:,6:7]
                x_3d_t = x_3d_t - x_3d_t_root
                # # 3d keypoints root in camera coordiate
                # x_3d_root_h = torch.ones((x_3d_t_root.shape[0],1,4)).to(x_3d_t_root.device)
                # x_3d_root_h[:,:,:-1] = x_3d_t_root
                # x_3d_t_root_ca = (kwargs['cameras_extrins'][:, 0:1] @ x_3d_root_h.unsqueeze(-1)).squeeze(-1)
            # norm keypoints by baseline width
            if kwargs['depth_norm_by_bw']:
                x_3d_t[:,:,2] = x_3d_t[:,:,2] * (kwargs['baseline_width'] / 400)

            x_3d_0 = model(x_3d_t, src_mask, t.float(), action=kwargs['actions_idx'])
            
            # denorm keypoints by baseline width
            if kwargs['depth_norm_by_bw']:
                x_3d_0[:,:,2] = x_3d_0[:,:,2] * (400 / kwargs['baseline_width'])
            if kwargs['root_relatvie']:
                x_3d_0 = x_3d_0 + x_3d_t_root
            x0_t = x_3d_0
            # x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            et = (xt - x0_t*at.sqrt()) / ((1 - at).sqrt())
            x0_preds.append(x0_t)
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next)

    return xs, x0_preds

def generalized_steps_2view_2viewkpt_cond(x, src_mask, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).cuda()
            next_t = (torch.ones(n) * j).cuda()
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1]
            x_3d_t = xt
            if kwargs['root_relatvie']:
                x_3d_t_root = x_3d_t[:,6:7]
                x_3d_t = x_3d_t - x_3d_t_root
                # # 3d keypoints root in camera coordiate
                # x_3d_root_h = torch.ones((x_3d_t_root.shape[0],1,4)).to(x_3d_t_root.device)
                # x_3d_root_h[:,:,:-1] = x_3d_t_root
                # x_3d_t_root_ca = (kwargs['cameras_extrins'][:, 0:1] @ x_3d_root_h.unsqueeze(-1)).squeeze(-1)

            x_3d_0 = model(x_3d_t, src_mask, t.float(), kpt2ds=kwargs['kpt2ds'])
            
            if kwargs['root_relatvie']:
                x_3d_0 = x_3d_0 + x_3d_t_root
            x0_t = x_3d_0
            # x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            et = (xt - x0_t*at.sqrt()) / ((1 - at).sqrt())
            x0_preds.append(x0_t)
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next)

    return xs, x0_preds