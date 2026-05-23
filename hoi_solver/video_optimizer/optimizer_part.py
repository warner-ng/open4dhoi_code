import numpy as np  
import matplotlib.pyplot as plt  
import math
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import torch  
import torch.optim as optim  
import torch.nn.functional as F   
import smplx  
import json  
import cv2  
import open3d as o3d  
from torch import nn  
import tqdm
from .utils.loss_utils import (
    HOCollisionLoss,
    compute_contact_loss,
    compute_collision_loss,  
    compute_mask_loss,
    joint_mask_parameters,
    visualize_vertex_gradients,
)  
from PIL import Image    
from tqdm import tqdm  
import torchvision.transforms.functional as TF
import time
import shutil
from .utils.camera_utils import transform_to_global, inverse_transform_to_incam
from .utils.hoi_utils import get_all_body_joints
from copy import deepcopy
from .utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from typing import Optional
from .config import load_optimizer_config
from .utils.smoothing_utils import (
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    quaternion_slerp,
    ema_smooth_series,
    box_smooth_series,
    gaussian_smooth_series,
    smooth_quaternion_sequence,
    box_smooth_quaternion_sequence,
    gaussian_smooth_quaternion_sequence,
)
from .kp_common import resource_path

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_j_regressor() -> torch.Tensor:
    candidates = [
        resource_path("video_optimizer/J_regressor.pt"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "shared_data", "J_regressor.pt"),
    ]

    for p in candidates:
        if os.path.exists(p):
            return torch.load(p, map_location="cpu", weights_only=False).float().to(_DEVICE)

    # Fallback: derive from SMPLX_NEUTRAL.npz
    smplx_npz_candidates = [
        os.environ.get("SMPLX_MODEL", ""),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "shared_data", "SMPLX_NEUTRAL.npz"),
    ]
    for p in smplx_npz_candidates:
        if p and os.path.exists(p):
            smplx_data = np.load(p, allow_pickle=True)
            J = smplx_data["J_regressor"]
            try:
                import scipy.sparse
                if scipy.sparse.issparse(J):
                    J = J.toarray()
            except Exception:
                pass
            J = np.asarray(J)
            if J.ndim == 2 and J.shape[0] >= 22:
                J = J[:22, :]
            return torch.from_numpy(J).float().to(_DEVICE)

    raise FileNotFoundError(
        "J_regressor not found. Tried video_optimizer/J_regressor.pt and shared_data/J_regressor.pt, "
        "and could not derive from SMPLX_NEUTRAL.npz."
    )


J_regressor = _load_j_regressor()
with open(resource_path("video_optimizer/data/joint_sim.json"), "r", encoding="utf-8") as _f:
    joint_sim = json.load(_f)

def load_downsampling_mapping(filepath):
    data = np.load(filepath)
    from scipy.sparse import csr_matrix
    D = csr_matrix((data['D_data'], data['D_indices'], data['D_indptr']), 
                   shape=data['D_shape'])
    faces_ds = data['faces_ds']
    print(f"Downsampling mapping loaded from {filepath}")
    return D, faces_ds

downsampling_file_path = resource_path("video_optimizer/smplx_downsampling_1000.npz")
if os.path.exists(downsampling_file_path):
    D, faces_ds = load_downsampling_mapping(downsampling_file_path)
    D_torch = torch.tensor(D.toarray(), dtype=torch.float32, device=_DEVICE)
else:
    D = None
    faces_ds = None
    D_torch = None
    print(f"[WARN] Downsampling mapping not found: {downsampling_file_path}. Using full SMPL-X mesh.")
class VideoBodyObjectOptimizer:  
    def __init__(self,   
                 body_params,
                 global_body_params,
                 hand_params,  
                 object_points_idx,   
                 body_points_idx, 
                 body_kp_name,
                 pairs_2d, 
                 object_meshes, 
                 sampled_obj_meshes,
                 icp_transform_matrix,
                 smpl_model,
                 start_frame,
                 end_frame,
                 video_dir,  
                 lr=0.1,
                 is_static_object=False,
                 best_frame=None):  
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.seq_length = end_frame - start_frame
        self.smpl_model = smpl_model  
        self.pairs_2d = pairs_2d
        self.body_params_sequence = body_params  
        self.global_body_params = global_body_params
        self.hand_poses = hand_params
        self.object_meshes = object_meshes
        self.sampled_obj_meshes = sampled_obj_meshes
        self.icp_transform_matrix = icp_transform_matrix
        self.video_dir = video_dir
        self.is_static_object = is_static_object
        cap = cv2.VideoCapture(os.path.join(video_dir, "video.mp4"))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = width
        self.height = height
        self.image_size = max(width, height)
        self.lr = lr
        
                                
        self.body_pose_params = []  
        self.shape_params = []
        self.left_hand_params = []
        self.right_hand_params = []
        self.global_orient = []
        self.transl = []
        self.obj_x_params = []  
        self.obj_y_params = []  
        self.obj_z_params = []  
        self.obj_transl_params = []  
        self.obj_transl_limit = torch.tensor([0.1, 0.1, 0.1]).cuda()

        self.icp_obj_R = []
        self.icp_obj_transl = []

        self.R_incam_static = []
        self.T_incam_static = []
        self.R_total_frames = None
        self.T_total_frames = None
        self.frames_optimized = []
        self.pre_contact_best_frame = None
        self.R_incam_pre_contact = {}
        self.T_incam_pre_contact = {}
        self.incam_pelvis, self.global_pelvis, self.incam_orient_o, self.global_orient_o, self.incam_transl_o, self.global_transl_o = get_all_body_joints(
            self.body_params_sequence, 
            self.global_body_params, 
            self.smpl_model, 
            self.start_frame, 
            self.end_frame
        )

        for i in range(self.start_frame, self.end_frame):
            self.body_pose_params.append(nn.Parameter(self.body_params_sequence["body_pose"][i].cuda(), requires_grad=True)) 
            self.shape_params.append(nn.Parameter(self.body_params_sequence['betas'][i].cuda(), requires_grad=True))  
            handpose=self.hand_poses[str(i)]
                           
            left_hand_pose = torch.from_numpy(np.asarray(handpose['left_hand']).reshape(-1,3)).float().cuda()
            right_hand_pose = torch.from_numpy(np.asarray(handpose['right_hand']).reshape(-1,3)).float().cuda()
            self.left_hand_params.append(nn.Parameter(left_hand_pose, requires_grad=True))
            self.right_hand_params.append(nn.Parameter(right_hand_pose, requires_grad=True))
            self.global_orient.append(self.body_params_sequence['global_orient'][i].cuda())
            self.transl.append(self.body_params_sequence['transl'][i].cuda())

            trans_mat = self.icp_transform_matrix[i - self.start_frame]
            R_mat = trans_mat[:3, :3]
            transl_vec = trans_mat[:3, 3]
            
            self.icp_obj_R.append(torch.from_numpy(R_mat).float().cuda())
            self.icp_obj_transl.append(torch.from_numpy(transl_vec).float().cuda())

            self.obj_x_params.append(nn.Parameter(torch.tensor(0.0, dtype=torch.float32).cuda(), requires_grad=True))
            self.obj_y_params.append(nn.Parameter(torch.tensor(0.0, dtype=torch.float32).cuda(), requires_grad=True))
            self.obj_z_params.append(nn.Parameter(torch.tensor(0.0, dtype=torch.float32).cuda(), requires_grad=True))
            self.obj_transl_params.append(nn.Parameter(torch.zeros(3, dtype=torch.float32).cuda(), requires_grad=True))

        self.body_points_idx = body_points_idx
        self.object_points_idx = object_points_idx  
        self.body_kp_name = body_kp_name
        if is_static_object:
            if best_frame is None:
                raise ValueError("best_frame is required when is_static_object=True")
            self.best_frame = best_frame
        else:
            self.best_frame = None
        self.mask = None  
        self.optimizer = None  
        self.current_frame = 0  
    
    def training_setup(self):
        params_list = []
        for i in range(self.seq_length):
            frame_params = [
                {'params': [self.body_pose_params[i]], 'lr': 0.001, 'name': f'pose_{i}'},
                {'params': [self.shape_params[i]], 'lr': 0.001, 'name': f'shape_{i}'},
                {'params': [self.left_hand_params[i]], 'lr': 0.003, 'name': f'left_hand_{i}'},
                {'params': [self.right_hand_params[i]], 'lr': 0.003, 'name': f'right_hand_{i}'},
                {'params': [self.obj_x_params[i]], 'lr': 0.002, 'name': f'x_angle_{i}'},
                {'params': [self.obj_y_params[i]], 'lr': 0.002, 'name': f'y_angle_{i}'},
                {'params': [self.obj_z_params[i]], 'lr': 0.002, 'name': f'z_angle_{i}'},
                {'params': [self.obj_transl_params[i]], 'lr': 0.001, 'name': f'transl_{i}'},
            ]
            
            params_list.extend(frame_params)                                                                
        self.optimizer = optim.Adam(params_list, lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

    def get_body_points(self, frame_idx=None, sampled=False):  
        if frame_idx is None:  
            frame_idx = self.current_frame  
        body_pose = self.body_pose_params[frame_idx].reshape(1, -1).cuda()  
        body_pose_save = body_pose.clone().detach().cpu().numpy()
        shape = self.shape_params[frame_idx].reshape(1, -1).cuda()  
        global_orient = self.global_orient[frame_idx].reshape(1, 3).cuda()
        left_hand_pose = self.left_hand_params[frame_idx].reshape(1, -1).cuda()
        right_hand_pose = self.right_hand_params[frame_idx].reshape(1, -1).cuda()
        zero_pose = torch.zeros((1, 3)).float().repeat(1, 1).cuda()
        transl = self.transl[frame_idx].reshape(1, -1).cuda()
        output = self.smpl_model(betas=shape,   
                                body_pose=body_pose,
                                left_hand_pose=left_hand_pose,   
                                right_hand_pose=right_hand_pose,   
                                jaw_pose=zero_pose,   
                                leye_pose=zero_pose,
                                reye_pose=zero_pose,
                                global_orient=global_orient,
                                expression=torch.zeros((1, 10)).float().cuda(),
                                transl=transl)
                                                        
        xyz = output.vertices[0]
        if sampled and D_torch is not None:
            xyz = torch.einsum('vw,wc->vc', D_torch, xyz)
        return xyz
    def get_body_faces(self, sampled=False):  
        body_faces = self.smpl_model.faces  
        if sampled and faces_ds is not None:
            body_faces = faces_ds
        return body_faces  
    
    def get_object_faces(self, frame_idx=None, sampled=False):
        if frame_idx is None:
            frame_idx = self.current_frame
        if sampled:
            object_mesh = self.sampled_obj_meshes[frame_idx]
        else:
            object_mesh = self.object_meshes[frame_idx]
        object_faces = object_mesh.triangles
        return np.asarray(object_faces).astype(np.int64)
    def get_object_points(self, frame_idx=None, sampled=False):
        if frame_idx is None:
            frame_idx = self.current_frame
        if sampled:
            object_mesh = self.sampled_obj_meshes[frame_idx]
        else:
            object_mesh = self.object_meshes[frame_idx]
        R_final, t_final = self.get_object_params(frame_idx)
        object_points = torch.tensor(np.asarray(object_mesh.vertices), 
                                   dtype=torch.float32, device=R_final.device)
        object_points = torch.mm(object_points, R_final.T) + t_final
        return object_points    
    def get_object_transform(self, frame_idx=None):  
        if frame_idx is None:  
            frame_idx = self.current_frame  
        
        x_angle = torch.deg2rad(self.obj_x_params[frame_idx])  
        y_angle = torch.deg2rad(self.obj_y_params[frame_idx])  
        z_angle = torch.deg2rad(self.obj_z_params[frame_idx]) 
        
        RX = torch.stack([  
            torch.tensor([1.0, 0.0, 0.0], device=x_angle.device, dtype=torch.float32),  
            torch.stack([torch.tensor(0.0, device=x_angle.device, dtype=torch.float32), torch.cos(x_angle), -torch.sin(x_angle)]),  
            torch.stack([torch.tensor(0.0, device=x_angle.device, dtype=torch.float32), torch.sin(x_angle), torch.cos(x_angle)])  
        ])  
        
        RY = torch.stack([  
            torch.stack([torch.cos(y_angle), torch.tensor(0.0, device=y_angle.device, dtype=torch.float32), torch.sin(y_angle)]),  
            torch.tensor([0.0, 1.0, 0.0], device=y_angle.device, dtype=torch.float32),  
            torch.stack([-torch.sin(y_angle), torch.tensor(0.0, device=y_angle.device, dtype=torch.float32), torch.cos(y_angle)])  
        ])  
        
        RZ = torch.stack([  
            torch.stack([torch.cos(z_angle), -torch.sin(z_angle), torch.tensor(0.0, device=z_angle.device, dtype=torch.float32)]),  
            torch.stack([torch.sin(z_angle), torch.cos(z_angle), torch.tensor(0.0, device=z_angle.device, dtype=torch.float32)]),  
            torch.tensor([0.0, 0.0, 1.0], device=z_angle.device, dtype=torch.float32)  
        ])    
        R = torch.mm(torch.mm(RZ, RY), RX)  
        return R  
    
    def get_optimized_parameters(self):
        human_params = {
            'body_pose': [],
            'betas': [],
            'global_orient': [],
            'transl': [],
            'left_hand_pose': [],
            'right_hand_pose': [],
        }
        
        object_params = {
            'poses': [], 
            'centers': [] 
        }
        for frame_idx in range(self.seq_length):
                  
            body_pose = self.body_pose_params[frame_idx].reshape(1, -1).cpu().detach().numpy()
            shape = self.shape_params[frame_idx].reshape(1, -1).cpu().detach().numpy()
            global_orient = self.global_orient[frame_idx].reshape(1, 3).cpu().detach().numpy()
            left_hand_pose = self.left_hand_params[frame_idx].reshape(1, -1).cpu().detach().numpy()
            right_hand_pose = self.right_hand_params[frame_idx].reshape(1, -1).cpu().detach().numpy()
            transl = self.transl[frame_idx].reshape(1, -1).cpu().detach().numpy()
            
            human_params['body_pose'].append(body_pose.tolist())
            human_params['betas'].append(shape.tolist())
            human_params['global_orient'].append(global_orient.tolist())
            human_params['transl'].append(transl.tolist())
            human_params['left_hand_pose'].append(left_hand_pose.tolist())
            human_params['right_hand_pose'].append(right_hand_pose.tolist())
            
                                      
            if self.is_static_object:
                obj_frame_idx = self.best_frame
            else:
                obj_frame_idx = frame_idx
                
            R_final, t_final = self.get_object_params(obj_frame_idx)
            
            object_params['poses'].append(R_final.cpu().detach().numpy().tolist())
            object_params['centers'].append(t_final.cpu().detach().numpy().tolist())
        return {
            'human_params': human_params,
            'object_params': object_params,
            'frame_range': {
                'start_frame': self.start_frame,
                'end_frame': self.end_frame
            }
        }
    def get_corresponding_point(self, frame_idx=None):  
        if frame_idx is None:  
            frame_idx = self.current_frame  
        object_points_idx =  self.object_points_idx[frame_idx]
        body_points_idx = np.asarray(self.body_points_idx[frame_idx])
        interacting_indices = object_points_idx[:, 1] != 0  
        interacting_body_indices = body_points_idx[interacting_indices]  
        body_points = self.get_body_points(frame_idx)[interacting_body_indices]  
        object_points = self.get_object_points(frame_idx)
        obj_index = object_points_idx[interacting_indices][:, 0]  
        interactiong_obj = object_points[obj_index]
        corresponding_points = {  
            'body_points': body_points,  
            'object_points': interactiong_obj 
        }  
        return corresponding_points  
    def _has_contact(self, frame_idx):
        object_points_idx = self.object_points_idx[frame_idx]
        if object_points_idx is None:
            return False
        interacting_indices = object_points_idx[:, 1] != 0
        return bool(np.any(interacting_indices))
    def optimize(self,   
                config_path: Optional[str] = None,
                steps: Optional[int] = None,
                print_every: Optional[int] = None,
                contact_weight: Optional[float] = None,
                collision_weight: Optional[float] = None,
                mask_weight: Optional[float] = None,
                optimize_interval: Optional[int] = None,
                smoothing_alpha: Optional[float] = None,
                smoothing_beta: Optional[float] = None,
                smoothing_window: Optional[int] = None,
                smoothing_passes: Optional[int] = None,
                smoothing_method: Optional[str] = None,
                **kwargs):
        _ = kwargs  # allow deprecated/unknown kwargs without breaking older callers
        cfg = load_optimizer_config(config_path)
        if steps is None:
            steps = int(cfg.get('optimize', {}).get('steps', 100))
        if print_every is None:
            print_every = int(cfg.get('optimize', {}).get('print_every', 10))
        if optimize_interval is None:
            optimize_interval = int(cfg.get('optimize', {}).get('optimize_interval', 3))

        if contact_weight is None:
            contact_weight = float(cfg.get('loss_weights', {}).get('contact', 50.0))
        if collision_weight is None:
            collision_weight = float(cfg.get('loss_weights', {}).get('collision', 8.0))
        if mask_weight is None:
            mask_weight = float(cfg.get('loss_weights', {}).get('mask', 0.05))

        if smoothing_alpha is None:
            smoothing_alpha = float(cfg.get('smoothing', {}).get('alpha', 0.25))
        if smoothing_beta is None:
            smoothing_beta = float(cfg.get('smoothing', {}).get('beta', 0.25))
        if smoothing_window is None:
            smoothing_window = int(cfg.get('smoothing', {}).get('window', 7))
        if smoothing_passes is None:
            smoothing_passes = int(cfg.get('smoothing', {}).get('passes', 2))
        if smoothing_method is None:
            smoothing_method = str(cfg.get('smoothing', {}).get('method', 'ema_box'))

        # Load smoothing enable flag (default: False)
        smoothing_enable = cfg.get('smoothing', {}).get('enable', False)

        self.training_setup()   
                  
        self.leave_hand = False
        self.leave_hand_begin, self.leave_hand_end = None, None
        self.leave_hand_pairs = []

                               
        pre_contact_mode = False
        self.pre_contact_best_frame = None
        if not self.is_static_object and not self._has_contact(0):
            for idx in range(self.seq_length):
                if self._has_contact(idx):
                    self.pre_contact_best_frame = idx
                    pre_contact_mode = True
                    break
        if self.is_static_object and getattr(self, 'best_frame', None) is not None:
            frames_to_optimize = [self.best_frame]
        else:
            frames_to_optimize = list(range(0, self.seq_length, optimize_interval))
            if frames_to_optimize[-1] != self.seq_length:
                frames_to_optimize.append(self.seq_length-1)             

        if pre_contact_mode:
            if self.pre_contact_best_frame is None:
                frames_to_optimize = []
            else:
                frames_to_optimize = [f for f in frames_to_optimize if f >= self.pre_contact_best_frame]
                if self.pre_contact_best_frame not in frames_to_optimize:
                    frames_to_optimize.append(self.pre_contact_best_frame)
                frames_to_optimize = sorted(set(frames_to_optimize))
                                                                   
        self.frames_optimized = frames_to_optimize
        for frame_idx in tqdm(frames_to_optimize):
            self.current_frame = frame_idx
            corresponding_points= self.get_corresponding_point(frame_idx)
            if corresponding_points['body_points'].numel() == 0 and not self.leave_hand:
                self.leave_hand = True
                self.leave_hand_begin = max(frame_idx - optimize_interval, 0)
                         
            if corresponding_points['body_points'].numel() > 0 and self.leave_hand:
                self.leave_hand_end = frame_idx
                self.leave_hand_pairs.append((self.leave_hand_begin, self.leave_hand_end))
                self.leave_hand = False
                self.leave_hand_begin, self.leave_hand_end = None, None
            for step in range(steps):
                self.optimizer.zero_grad() if self.optimizer is not None else None
                corresponding_points= self.get_corresponding_point(frame_idx)
                contact_loss = compute_contact_loss(corresponding_points)   
                hverts = self.get_body_points(frame_idx, sampled=True).unsqueeze(0)
                overts = self.get_object_points(frame_idx, sampled=True).unsqueeze(0)
                hfaces = self.get_body_faces(sampled=True)
                ofaces = self.get_object_faces(frame_idx, sampled=True)
                collision_loss = compute_collision_loss(hverts, overts, hfaces, ofaces, h_weight=10.0, threshold=0) + 1e-5
                mask_loss= compute_mask_loss(self.width, self.height, self.video_dir, hverts, overts, hfaces, ofaces, mask_weight=1.5, edge_weight=1e-3, frame_idx=frame_idx + self.start_frame)
        
                         
        
                loss = ( contact_weight * contact_loss
                        + collision_weight * collision_loss
                        + mask_weight * mask_loss
                        )

                param_idx = self.best_frame if self.is_static_object else frame_idx
                if torch.any(torch.abs(self.obj_transl_params[param_idx]) > self.obj_transl_limit):
                    limit_mask = torch.abs(self.obj_transl_params[param_idx]) > self.obj_transl_limit
                    total_loss = loss + 1e6*F.mse_loss(self.obj_transl_params[param_idx][limit_mask], self.obj_transl_limit[limit_mask])
                else:
                    total_loss = loss
                if total_loss.requires_grad and total_loss.grad_fn is not None:
                    total_loss.backward()
                joint_mask_parameters(self.smpl_model, self.optimizer, frame_idx, self.body_kp_name, joint_sim)
                self.optimizer.step()     
                if step % print_every == 0:
                    tqdm.write(f"Frame {frame_idx}, Step {step}: Loss = {loss.item():.4f}, "
                            f"Contact = {contact_loss.item():.4f}, "
                                                                                
                            f"Collision = {collision_loss.item():.4f}, "
                            f"Mask = {mask_loss.item():.4f}"
                                )

        if self.is_static_object:
            with torch.no_grad():
                R_best_incam, T_best_incam  = self.get_object_params(self.best_frame)
                R_best_global, T_best_global = transform_to_global(
                    R_best_incam, 
                    T_best_incam, 
                    self.incam_orient_o[self.best_frame], 
                    self.global_orient_o[self.best_frame], 
                    self.incam_pelvis[self.best_frame], 
                    self.global_pelvis[self.best_frame],
                )
                for frame_idx in range(self.seq_length):
                    if frame_idx == self.best_frame:
                        self.R_incam_static.append(R_best_incam)
                        self.T_incam_static.append(T_best_incam)
                    else:
                        R_incam_final, T_incam_final = inverse_transform_to_incam(
                            R_best_global, 
                            T_best_global,
                            self.incam_orient_o[frame_idx], 
                            self.global_orient_o[frame_idx], 
                            self.incam_pelvis[frame_idx], 
                            self.global_pelvis[frame_idx],
                        ) 
                        self.R_incam_static.append(R_incam_final)
                        self.T_incam_static.append(T_incam_final)            
                                                                                                
        if pre_contact_mode and self.pre_contact_best_frame is not None:
            with torch.no_grad():
                                                                        
                R_best_incam, T_best_incam  = self.get_object_params(self.pre_contact_best_frame)
                R_best_global, T_best_global = transform_to_global(
                    R_best_incam,
                    T_best_incam,
                    self.incam_orient_o[self.pre_contact_best_frame],
                    self.global_orient_o[self.pre_contact_best_frame],
                    self.incam_pelvis[self.pre_contact_best_frame],
                    self.global_pelvis[self.pre_contact_best_frame],
                )
                for frame_idx in range(self.pre_contact_best_frame):
                    R_incam_final, T_incam_final = inverse_transform_to_incam(
                        R_best_global,
                        T_best_global,
                        self.incam_orient_o[frame_idx],
                        self.global_orient_o[frame_idx],
                        self.incam_pelvis[frame_idx],
                        self.global_pelvis[frame_idx],
                    )
                    self.R_incam_pre_contact[frame_idx] = R_incam_final
                    self.T_incam_pre_contact[frame_idx] = T_incam_final
        
        if optimize_interval > 1 :

            self._interpolate_frames(optimize_interval)

        self._interpolate_object_pose_slerp(optimize_interval)
        for begin_frame, end_frame in self.leave_hand_pairs:
            self._interpolate_depths(begin_frame, end_frame)

        # Apply smoothing only if enabled
        if smoothing_enable:
            self._lowpass_smooth_all(
                alpha=smoothing_alpha,
                beta_quat=smoothing_beta,
                bidirectional=True,
                ema_passes=smoothing_passes,
                window_size=smoothing_window,
                method=smoothing_method,
            )
        else:
            print("[INFO] Smoothing disabled (smoothing.enable=False)")
    def _interpolate_depths(self, begin_frame, end_frame):
        with torch.no_grad():
            obj_center_begin = self.get_object_points(begin_frame, sampled=True).mean(dim=0)
            obj_center_end = self.get_object_points(end_frame, sampled=True).mean(dim=0)
            depth_begin = obj_center_begin[2]
            depth_end = obj_center_end[2]
            for frame_idx in range(begin_frame, end_frame):
                alpha = (frame_idx - begin_frame) / (end_frame - begin_frame)
                interpolate_depth = (1 - alpha) * depth_begin + alpha * depth_end
                frame_obj_center = self.get_object_points(frame_idx, sampled=True).mean(dim=0)
                oringin_depth = frame_obj_center[2]
                if self.T_total_frames is not None and frame_idx < len(self.T_total_frames) and self.T_total_frames[frame_idx] is not None:
                    self.T_total_frames[frame_idx][2] = self.T_total_frames[frame_idx][2] + interpolate_depth - oringin_depth
                else:
                    self.obj_transl_params[frame_idx][2] = self.obj_transl_params[frame_idx][2] + interpolate_depth - oringin_depth

    def _interpolate_frames(self, interval):
        with torch.no_grad():
            for i in range(0, self.seq_length, interval):
                start_frame = i
                end_frame = min(i + interval, self.seq_length - 1)
                if start_frame >= end_frame:
                    continue
                for mid_frame in range(start_frame, end_frame):
                    alpha = (mid_frame - start_frame) / (end_frame - start_frame)
                    
                    self.body_pose_params[mid_frame].copy_(
                        (1 - alpha) * self.body_pose_params[start_frame] + 
                        alpha * self.body_pose_params[end_frame]
                    )
                    self.shape_params[mid_frame].copy_(
                        (1 - alpha) * self.shape_params[start_frame] + 
                        alpha * self.shape_params[end_frame]
                    )
                    self.left_hand_params[mid_frame].copy_(
                        (1 - alpha) * self.left_hand_params[start_frame] + 
                        alpha * self.left_hand_params[end_frame]
                    )
                    self.right_hand_params[mid_frame].copy_(
                        (1 - alpha) * self.right_hand_params[start_frame] + 
                        alpha * self.right_hand_params[end_frame]
                    )
    def _interpolate_object_pose_slerp(self, interval):
        with torch.no_grad():
                        
            self.R_total_frames = [None for _ in range(self.seq_length)]
            self.T_total_frames = [None for _ in range(self.seq_length)]

                                                                                                    
            if self.is_static_object and len(self.R_incam_static) == self.seq_length and len(self.T_incam_static) == self.seq_length:
                for f in range(self.seq_length):
                    self.R_total_frames[f] = self.R_incam_static[f]
                    self.T_total_frames[f] = self.T_incam_static[f]
                return

            for f in self.R_incam_pre_contact.keys():
                if 0 <= f < self.seq_length:
                    self.R_total_frames[f] = self.R_incam_pre_contact[f]
                    self.T_total_frames[f] = self.T_incam_pre_contact[f]

                               
            anchors = [] if self.frames_optimized is None else list(self.frames_optimized)
            anchors = sorted(set(int(x) for x in anchors))
            if 0 not in anchors:
                anchors = [0] + anchors
            if (self.seq_length - 1) not in anchors:
                anchors = anchors + [self.seq_length - 1]

                                                                   
            anchor_quats = {}
            anchor_trans = {}
            for idx in anchors:
                Rk, Tk = self._compute_object_params_no_cache(idx)
                anchor_quats[idx] = rotation_matrix_to_quaternion(Rk)
                anchor_trans[idx] = Tk

                                                                                             
            for a, b in zip(anchors[:-1], anchors[1:]):
                qa, qb = anchor_quats[a], anchor_quats[b]
                Ta, Tb = anchor_trans[a], anchor_trans[b]
                length = max(b - a, 1)
                for f in range(a, b + 1):
                                                                   
                    if self.R_total_frames[f] is not None and self.T_total_frames[f] is not None:
                        continue
                    if f == a:
                        alpha = 0.0
                    elif f == b:
                        alpha = 1.0
                    else:
                        alpha = (f - a) / float(length)
                    qf = quaternion_slerp(qa, qb, torch.tensor(alpha, dtype=torch.float32, device=qa.device))
                    Rf = quaternion_to_rotation_matrix(qf)
                    Tf = (1 - alpha) * Ta + alpha * Tb
                    self.R_total_frames[f] = Rf
                    self.T_total_frames[f] = Tf

                                                     
            for f in range(self.seq_length):
                if self.R_total_frames[f] is None or self.T_total_frames[f] is None:
                    Rf, Tf = self._compute_object_params_no_cache(f)
                    self.R_total_frames[f] = Rf
                    self.T_total_frames[f] = Tf

    def _compute_object_params_no_cache(self, frame_idx):
                
        if self.is_static_object and len(self.R_incam_static) > 0 and len(self.T_incam_static) > 0:
            return self.R_incam_static[frame_idx], self.T_incam_static[frame_idx]
                     
        if frame_idx in self.R_incam_pre_contact and frame_idx in self.T_incam_pre_contact:
            return self.R_incam_pre_contact[frame_idx], self.T_incam_pre_contact[frame_idx]
        R_residual = self.get_object_transform(frame_idx)
        R_icp = self.icp_obj_R[frame_idx]
        R_final = torch.mm(R_residual, R_icp)
        t_residual = self.obj_transl_params[frame_idx]
        t_icp = self.icp_obj_transl[frame_idx]
        depth_centers = torch.zeros(3, dtype=torch.float32, device=R_final.device)
        t_final = torch.mv(R_final, depth_centers) + torch.mv(R_residual, t_icp) + t_residual
        return R_final, t_final

    def _lowpass_smooth_all(self, alpha=0.5, beta_quat=0.5, bidirectional=True, ema_passes=1, window_size=1, method='ema_box', cutoff=0.08, butter_order=4, fs=1.0):
        with torch.no_grad():
                                     
            if len(self.body_pose_params) > 0:
                series = [p.data for p in self.body_pose_params]
                if method in ('ema', 'ema_box', 'gaussian'):
                    for _ in range(max(1, int(ema_passes))):
                        series = ema_smooth_series(series, alpha=alpha, bidirectional=bidirectional)
                    if method == 'ema_box' and window_size and window_size > 1:
                        series = box_smooth_series(series, window_size)
                    if method == 'gaussian' and window_size and window_size > 1:
                        series = gaussian_smooth_series(series, window_size)
                smoothed = series
                for i in range(len(self.body_pose_params)):
                    self.body_pose_params[i].copy_(smoothed[i])
            if len(self.shape_params) > 0:
                series = [p.data for p in self.shape_params]
                if method in ('ema', 'ema_box', 'gaussian'):
                    for _ in range(max(1, int(ema_passes))):
                        series = ema_smooth_series(series, alpha=alpha, bidirectional=bidirectional)
                    if method == 'ema_box' and window_size and window_size > 1:
                        series = box_smooth_series(series, window_size)
                    if method == 'gaussian' and window_size and window_size > 1:
                        series = gaussian_smooth_series(series, window_size)
                smoothed = series
                for i in range(len(self.shape_params)):
                    self.shape_params[i].copy_(smoothed[i])
            if len(self.left_hand_params) > 0:
                series = [p.data for p in self.left_hand_params]
                if method in ('ema', 'ema_box', 'gaussian'):
                    for _ in range(max(1, int(ema_passes))):
                        series = ema_smooth_series(series, alpha=alpha, bidirectional=bidirectional)
                    if method == 'ema_box' and window_size and window_size > 1:
                        series = box_smooth_series(series, window_size)
                    if method == 'gaussian' and window_size and window_size > 1:
                        series = gaussian_smooth_series(series, window_size)
                smoothed = series
                for i in range(len(self.left_hand_params)):
                    self.left_hand_params[i].copy_(smoothed[i])
            if len(self.right_hand_params) > 0:
                series = [p.data for p in self.right_hand_params]
                if method in ('ema', 'ema_box', 'gaussian'):
                    for _ in range(max(1, int(ema_passes))):
                        series = ema_smooth_series(series, alpha=alpha, bidirectional=bidirectional)
                    if method == 'ema_box' and window_size and window_size > 1:
                        series = box_smooth_series(series, window_size)
                    if method == 'gaussian' and window_size and window_size > 1:
                        series = gaussian_smooth_series(series, window_size)
                smoothed = series
                for i in range(len(self.right_hand_params)):
                    self.right_hand_params[i].copy_(smoothed[i])                         
            if self.R_total_frames is not None and self.T_total_frames is not None:
                                    
                q_list = []
                idx_map = []
                for i, R in enumerate(self.R_total_frames):
                    if R is None:
                        q_list.append(None)
                    else:
                        q_list.append(rotation_matrix_to_quaternion(R))
                    idx_map.append(i)
                                                                  
                for i in range(len(q_list)):
                    if q_list[i] is None:
                        R_fallback, _ = self._compute_object_params_no_cache(i)
                        q_list[i] = rotation_matrix_to_quaternion(R_fallback)
                qs = q_list
                if method in ('ema', 'ema_box', 'gaussian'):
                    for _ in range(max(1, int(ema_passes))):
                        qs = smooth_quaternion_sequence(qs, beta=beta_quat, bidirectional=bidirectional)
                    if method == 'ema_box' and window_size and window_size > 1:
                        qs = box_smooth_quaternion_sequence(qs, window_size)
                    if method == 'gaussian' and window_size and window_size > 1:
                        qs = gaussian_smooth_quaternion_sequence(qs, window_size)
                q_sm = qs
                for i in range(len(q_sm)):
                    self.R_total_frames[i] = quaternion_to_rotation_matrix(q_sm[i])

                                            
                t_list = [t if t is not None else self._compute_object_params_no_cache(i)[1] for i, t in enumerate(self.T_total_frames)]
                ts = t_list
                if method in ('ema', 'ema_box', 'gaussian'):
                    for _ in range(max(1, int(ema_passes))):
                        ts = ema_smooth_series(ts, alpha=alpha, bidirectional=bidirectional)
                    if method == 'ema_box' and window_size and window_size > 1:
                        ts = box_smooth_series(ts, window_size)
                    if method == 'gaussian' and window_size and window_size > 1:
                        ts = gaussian_smooth_series(ts, window_size)
                t_sm = ts
                for i in range(len(t_sm)):
                    self.T_total_frames[i] = t_sm[i]
    def save_sequence(self, output_dir):  
        os.makedirs(output_dir, exist_ok=True)  
        
        for i in range(self.seq_length):
            frame_dir = os.path.join(output_dir, f'frame_{i + self.start_frame:04d}')  
            os.makedirs(frame_dir, exist_ok=True)   
            self.current_frame = i 
            human_faces = self.get_body_faces(sampled=False)  
            human_verts = self.get_body_points(i, sampled=False).detach().cpu().numpy()
            object_vertices = self.get_object_points(i, sampled=False).detach().cpu().numpy()
            object_vertices_ = self.get_object_points(i, sampled=True).detach().cpu().numpy()
            incam_params = (self.global_orient[i], self.transl[i])
            global_params = (self.global_body_params["global_orient"][i], self.global_body_params["transl"][i])
            h_mesh = o3d.geometry.TriangleMesh()  
            h_mesh.vertices = o3d.utility.Vector3dVector(human_verts)  
            h_mesh.triangles = o3d.utility.Vector3iVector(human_faces)  
            o3d.io.write_triangle_mesh(os.path.join(frame_dir, 'human.obj'), h_mesh)  
            obj_mesh = o3d.geometry.TriangleMesh()  
            obj_mesh.vertices = o3d.utility.Vector3dVector(object_vertices)  
            obj_mesh.triangles = o3d.utility.Vector3iVector(self.get_object_faces(i, sampled=False))
            obj_mesh_ = o3d.geometry.TriangleMesh()
            obj_mesh_.vertices = o3d.utility.Vector3dVector(object_vertices_)
            obj_mesh_.triangles = o3d.utility.Vector3iVector(self.get_object_faces(i, sampled=True))
                                                                                                      
            o3d.io.write_triangle_mesh(os.path.join(frame_dir, 'object.obj'), obj_mesh)
            o3d.io.write_triangle_mesh(os.path.join(frame_dir, 'object_sampled.obj'), obj_mesh_)
            
                     
            corresponding_points = self.get_corresponding_point(i)  
            body_points = corresponding_points['body_points'].detach().cpu().numpy()  
            object_points = corresponding_points['object_points'].detach().cpu().numpy()  
            
            lines = [[i, i + len(body_points)] for i in range(len(body_points))]  
            points = np.vstack((body_points, object_points))  
            colors = [[0, 1, 0] for _ in range(len(lines))]  
            
            line_set = o3d.geometry.LineSet(  
                points=o3d.utility.Vector3dVector(points),  
                lines=o3d.utility.Vector2iVector(lines),  
            )  
            line_set.colors = o3d.utility.Vector3dVector(colors)  
            o3d.io.write_line_set(os.path.join(frame_dir, 'contact_points.ply'), line_set)  
    

    def create_visualization_video(self, output_dir, K, video_path=None, fps=3, clear=True):
        if os.path.exists(output_dir) and clear:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        human_faces = np.array(self.get_body_faces(sampled=True), dtype=np.int32)
        obj_faces= np.array(self.sampled_obj_meshes[0].triangles, dtype=np.int32)
                                                                              
        renderer = Renderer(self.width, self.height, device="cuda",faces_human=human_faces,faces_obj=obj_faces,K=K)
        for i in tqdm(range(0, self.seq_length, 2), desc="rendering frames"):
            human_verts = self.get_body_points(i, sampled=True)
            object_mesh = self.sampled_obj_meshes[i]
            transform = self.get_object_transform(i).detach().cpu().numpy()
            object_vertices = self.get_object_points(i,sampled=True)
            img_raw = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255                                     
            if hasattr(object_mesh, 'vertex_colors'):
                object_color = np.asarray(object_mesh.vertex_colors)
            else:
                object_color = [0.3, 0.5, 0.7]
            if len(object_color) == 0:
                object_color = [0.3, 0.5, 0.7]

            img = renderer.render_mesh_hoi(human_verts, object_vertices, img_raw, [0.8, 0.8, 0.8],object_color)
            start_frame = (self.start_frame // 2) * 2
            frame_path = os.path.join(output_dir, f"frame_{i + start_frame:04d}.png")

            img = Image.fromarray(img)
            img.save(frame_path)


    def get_object_params(self, frame_idx=None):
        if frame_idx is None:
            frame_idx = self.current_frame
                                                           
        if self.R_total_frames is not None and self.T_total_frames is not None:
            if frame_idx < len(self.R_total_frames):
                Rt = self.R_total_frames[frame_idx]
                Tt = self.T_total_frames[frame_idx] if frame_idx < len(self.T_total_frames) else None
                if Rt is not None and Tt is not None:
                    return Rt, Tt
        if self.is_static_object and len(self.R_incam_static) > 0 and len(self.T_incam_static) > 0:
            R_final = self.R_incam_static[frame_idx]
            t_final = self.T_incam_static[frame_idx]
            return R_final, t_final
                                                                             
        if frame_idx in self.R_incam_pre_contact and frame_idx in self.T_incam_pre_contact:
            return self.R_incam_pre_contact[frame_idx], self.T_incam_pre_contact[frame_idx]
        R_residual = self.get_object_transform(frame_idx)
        R_icp = self.icp_obj_R[frame_idx]
        R_final = torch.mm(R_residual, R_icp)
        t_residual = self.obj_transl_params[frame_idx]
        t_icp = self.icp_obj_transl[frame_idx]
        depth_centers = torch.zeros(3, dtype=torch.float32, device=R_final.device)
        t_final = torch.mv(R_final, depth_centers) + torch.mv(R_residual, t_icp) + t_residual
        return R_final, t_final               
    def get_optimize_result(self):
        R_finals = []
        t_finals = []
                                                        
        print(self.start_frame,self.end_frame)
        body_pose_finals=[]
        betas_finals=[]
        global_orient_finals=[]
        transl_finals=[]
        hand_poses_finals=[]
        for i in range(self.seq_length):
            R_final, t_final = self.get_object_params(i)
            R_finals.append(R_final.detach().cpu().numpy())
            t_finals.append(t_final.detach().cpu().numpy())
            body_pose_finals.append(self.body_pose_params[i].reshape(1, -1).detach().cpu())
            betas_finals.append(self.shape_params[i].reshape(1, -1).detach().cpu())
            global_orient_finals.append(self.global_orient[i].reshape(1, 3).detach().cpu())
            transl_finals.append(self.transl[i].reshape(1, -1).detach().cpu())
            hand_poses_finals.append({
                "left_hand": self.left_hand_params[i].reshape(1, -1).detach().cpu().numpy(),
                "right_hand": self.right_hand_params[i].reshape(1, -1).detach().cpu().numpy()
            })                
        body_params_all={
            "body_pose": torch.cat(body_pose_finals, dim=0).numpy(),
            "betas": torch.cat(betas_finals, dim=0).numpy(),
            "global_orient": torch.cat(global_orient_finals, dim=0).numpy(),
            "transl": torch.cat(transl_finals, dim=0).numpy()
        }
        return body_params_all, hand_poses_finals, R_finals, t_finals
