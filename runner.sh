### train ###
###
 # @Author: sherrywaan sherrywaan@outlook.com
 # @Date: 2024-10-26 17:24:32
 # @LastEditors: sherrywaan sherrywaan@outlook.com
 # @LastEditTime: 2024-10-27 17:24:35
 # @FilePath: /wxy/3d_pose/Dual-Diffusion/runner.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

## human 3.6
# CUDA_VISIBLE_DEVICES=2 python main_diffpose_2view_frame.py --train \
# --config /data0/wxy/3d_pose/Diffpose/configs/h36m_2view_diffpose_uvxyz_rsb152.yml --batch_size 1024 \
# --model_diff_path /data0/wxy/3d_pose/Diffpose/exp/mhad_2view_diffpose_uvxyz_rsb152_75@10.05.2024-12:08:44/ckpt_best_mpjpe_re.pth \
# --doc h36m_2view_diffpose_uvxyz_rsb152_75 --exp exp --ni

## mhad
# CUDA_VISIBLE_DEVICES=0 python main_diffpose_2view_frame.py --train \
# --config mhad_2view_diffpose_uvxyz_resnet50.yml --batch_size 1024 \
# --model_diff_path /data0/wxy/3d_pose/Diffpose/exp/mhad_2view_diffpose_uvxyz_resnet50@25.04.2024-10:57:09/ckpt_best_mpjpe_re.pth \
# --doc mhad_2view_diffpose_uvxyz_resnet50 --exp exp --ni

# CUDA_VISIBLE_DEVICES=0 python main_diffpose_2view_frame.py --train \
# --config /data0/wxy/3d_pose/Diffpose/configs/mhad_2view_diffpose_uvxyz_rsb50.yml --batch_size 1024 \
# --model_diff_path /data0/wxy/3d_pose/Diffpose/exp/mhad_2view_diffpose_uvxyz_rsb50@05.05.2024-16:56:30/ckpt_best_mpjpe.pth \
# --doc mhad_2view_diffpose_uvxyz_rsb50 --exp exp --ni


### test ###
# # h36m
# CUDA_VISIBLE_DEVICES=0 python main_diffpose_2view_frame.py \
# --config h36m_2view_diffpose_uvxyz_rsb152.yml \
# --test_timesteps 1 \
# --model_diff_path checkpoints/ckpt_h36m_rsb152.pth \
# --doc t_h36m_2view_diffpose_uvxyz_rsb152 --exp exp --ni

# # mhad
CUDA_VISIBLE_DEVICES=0 python main_diffpose_2view_frame.py \
--config mhad_2view_diffpose_uvxyz_rsb152.yml \
--test_timesteps 1 \
--model_diff_path checkpoints/ckpt_mhad_rsb152.pth \
--doc t_mhad_2view_diffpose_uvxyz_rsb152 --exp exp --ni
