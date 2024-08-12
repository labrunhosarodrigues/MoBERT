# -*- coding: utf-8 -*-
"""
dataset.primary_evaluator_dataset
----------------------------------

Dataset definitions for using with primary evaluator.

"""
# Imports

# built-in
from multiprocessing import Pool
from pathlib import Path
import random
import os
import pickle

# local
from mobert.human_body_prior.body_model.body_model import BodyModel
from mobert.metrics import *
from mobert.common.skeleton import Skeleton
from mobert.common.quaternion import *
from mobert.paramUtil import *
from mobert.utils.stat_tracking import *  
import mobert.sample_data as sample_data

# 3rd-party
import torch
from torch.utils.data import Dataset
import numpy as np
import tqdm
import pandas as pd


def convert_vel_acel(pos_seq):
    vel_seq = pos_seq[1:, ...] - pos_seq[:-1, ...]
    acel_seq = vel_seq[1:, ...] - vel_seq[:-1, ...]
    return vel_seq, acel_seq


def uniform_skeleton(
        positions, target_offset,
        n_raw_offsets, kinematic_chain,
        l_idx1, l_idx2, face_joint_indx
):
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()

    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)

    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints


def foot_detect(positions, thres, fid_l, fid_r):
    velfactor, _ = np.array([thres, thres]), np.array([3.0, 2.0])

    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
    return feet_l, feet_r


def get_rifke(positions, r_rot):
    '''Local pose'''
    positions[..., 0] -= positions[:, 0:1, 0]
    positions[..., 2] -= positions[:, 0:1, 2]
    '''All pose face Z+'''
    positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
    return positions


def get_cont6d_params(positions, n_raw_offsets, kinematic_chain, face_joint_indx):
    skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    # (seq_len, joints_num, 4)
    quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

    '''Quaternion to continuous 6D'''
    cont_6d_params = quaternion_to_cont6d_np(quat_params)
    # (seq_len, 4)
    r_rot = quat_params[:, 0].copy()
    #     print(r_rot[0])
    '''Root Linear Velocity'''
    # (seq_len - 1, 3)
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
    #     print(r_rot.shape, velocity.shape)
    velocity = qrot_np(r_rot[1:], velocity)
    '''Root Angular Velocity'''
    # (seq_len - 1, 4)
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
    # (seq_len, joints_num, 4)
    return cont_6d_params, r_velocity, velocity, r_rot


def process_file(
        positions, feet_thre, tgt_offsets,
        face_joint_indx, n_raw_offsets,
        kinematic_chain,
        fid_l, fid_r,
        l_idx1, l_idx2
):
    # (seq_len, joints_num, 3)

    '''Uniform Skeleton'''
    positions = uniform_skeleton(
        positions, tgt_offsets,
        n_raw_offsets, kinematic_chain,
        l_idx1, l_idx2, face_joint_indx
    )

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions = qrot_np(root_quat_init, positions)

    '''New ground truth positions'''
    global_positions = positions.copy()

    """ Get Foot Contacts """
    feet_l, feet_r = foot_detect(positions, feet_thre, fid_l, fid_r)

    '''Quaternion and Cartesian representation'''
    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions, n_raw_offsets, kinematic_chain, face_joint_indx)
    positions = get_rifke(positions, r_rot)

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    # (seq_len-1, 2) linear velovity on xz plane
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(
        np.repeat(
            r_rot[:-1, None], global_positions.shape[1], axis=1
        ),
        global_positions[1:] - global_positions[:-1]
    )
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, global_positions, positions, l_velocity


def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos

def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]

    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


class JudgementDataset(Dataset):
    """
    Dataset definition to load prepared AMASS dataset with evaluation data.
    """
    
    def load_motion_data(self, eval_df, evaluation_path, humanML3D_path, device):
        bm_path = str(Path('./human_body_prior/smplh/female/model.npz'))
        dmpl_path = str(Path('./human_body_prior/dmpls/female/model.npz'))

        num_betas = 10 # number of body parameters
        num_dmpls = 8 # number of DMPL parameters
        
        bm = BodyModel(bm_fname=bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_path).to(device)

        feature_data = {}
        position_data = {}
        vel_data = {}
        acel_data = {}
        pos_vel_data = {}
        pos_vel_acel_data = {}
        for index, row in tqdm.tqdm(eval_df.iterrows(), total=len(eval_df), desc="Loading Evaluation Data"):
            if (row.Model not in feature_data):
                feature_data[row.Model] = {}
                position_data[row.Model] = {}
                vel_data[row.Model] = {}
                acel_data[row.Model] = {}
                pos_vel_data[row.Model] = {}
                pos_vel_acel_data[row.Model] = {}

            data_path = Path(evaluation_path) / "motions" / f"AMASS_motion_{row.Model}_{row.OriginalSample}.npz"
            bdata = np.load(data_path)
            with torch.no_grad():
                root_orient = torch.Tensor(bdata['poses'][:, :3]).to("cuda") # controls the global root orientation
                pose_body = torch.Tensor(bdata['poses'][:, 3:66]).to("cuda") # controls the body
                pose_hand = torch.Tensor(bdata['poses'][:, 66:66+90]).to("cuda") # controls the finger articulation
                betas = torch.Tensor(bdata['betas'][:10][np.newaxis]).repeat(len(bdata['poses']), 1).to("cuda") # controls the body shape
                trans = torch.Tensor(bdata['trans']).to("cuda") 
                body = bm(pose_body=pose_body, pose_hand=pose_hand, betas=betas, root_orient=root_orient)
                joint_loc = body.Jtr + trans[:, None, :]
                source_data = joint_loc.cpu().numpy()[:, :22, :]

            formated_data, _, _, _ = self.process_file(source_data, 0.002)

            feature_data[row.Model][row.OriginalSample] = formated_data
        return feature_data

    def __init__(self, cache_path, path, norm_path, chunk_size, overlap, pad_len=17, device="cpu"):
        os.makedirs(cache_path, exist_ok=True)
        self.pad_len = pad_len
        self.all_data = []
        self.all_masks = []
        self.all_files = []
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.text_sample_map = []
        evaluation_data_df = pd.read_csv(path / "ratings_and_captions.csv", header=0)
        if (not os.path.exists(cache_path / 'feat_data.obj')):
            humanML3D_path = Path("HumanML3D")
            # Lower legs
            self.l_idx1, self.l_idx2 = 5, 8
            # Right/Left foot
            self.fid_r, self.fid_l = [8, 11], [7, 10]
            # Face direction, r_hip, l_hip, sdr_r, sdr_l
            self.face_joint_indx = [2, 1, 17, 16]
            # l_hip, r_hip
            self.r_hip, self.l_hip = 2, 1
            self.joints_num = 22
            # ds_num = 8

            self.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
            self.kinematic_chain = t2m_kinematic_chain

            # Get offsets of target skeleton
            example_data = np.load(sample_data.SAMPLE000000)
            example_data = example_data.reshape(len(example_data), -1, 3)
            example_data = torch.from_numpy(example_data)
            self.tgt_skel = Skeleton(self.n_raw_offsets, self.kinematic_chain, 'cpu')
            # (joints_num, 3)
            self.tgt_offsets = self.tgt_skel.get_offsets_joints(example_data[0])
            self.feat_data = self.load_motion_data(evaluation_data_df, path, humanML3D_path, device)
            with open(cache_path / 'feat_data.obj', 'wb') as fileobj:
                pickle.dump(self.feat_data, fileobj)
        else:
            with open(cache_path / 'feat_data.obj', 'rb') as fileobj:
                self.feat_data = pickle.load(fileobj)
        self.evaluation_data_df = evaluation_data_df

        if norm_path is None:
            self.mean = np.load(sample_data.MEAN).reshape(1, -1)
            self.std = np.load(sample_data.STD).reshape(1, -1)
        else:
            self.mean = np.load(norm_path / "Mean.npy").reshape(1, -1)
            self.std = np.load(norm_path / "Std.npy").reshape(1, -1)
    
    def __len__(self):
        return len(self.evaluation_data_df)
    
    def __getitem__(self, idx):
        row = self.evaluation_data_df.iloc[idx]
        data = self.feat_data[row.Model][row.OriginalSample]
        data_len = len(data)
        motion_chunks = []
        for i in range(0, data_len, self.chunk_size):
            if (len(motion_chunks) == self.pad_len):
                continue
            motion_chunk = data[i: i+self.chunk_size + self.overlap].flatten()
            if (i+self.chunk_size + self.overlap >= len(data)):
                continue
            motion_chunks.append(motion_chunk.reshape(1, self.chunk_size + self.overlap, -1))
        
        motion_chunks = np.concatenate(motion_chunks, axis=0)
        motion_chunks = (motion_chunks - self.mean) / self.std

        motion_masks = np.zeros(self.pad_len)
        motion_masks[:len(motion_chunks)] = 1
        return {
            "motion_chunks": np.concatenate([motion_chunks, np.zeros((self.pad_len - len(motion_chunks), self.chunk_size + self.overlap, motion_chunks.shape[2]))], axis=0),
            "motion_masks": motion_masks, 
            "texts": row.Caption,
            "faithfulness": row.Faithfulness,
            "naturalness": row.Naturalness,
            "model": row.Model
        }

class TMRefDataset(Dataset):

    def _filter_files(self, args):
        files, chunk_size = args
        admits = []
        for file in files:
            data = np.load(file)
            if (len(data) <= 200 and len(data) >= 4 * chunk_size):
                admits.append(True)
            else:
                admits.append(False)
        return admits, files
    
    def __init__(self, exp_name, split, datapath, limit_size, chunk_size, overlap, pad_len=17, n_workers = 1, transform=None, augment=True):
        self.base_path = Path(f"../MotionDataset/ref_cache{exp_name}")
        os.makedirs(self.base_path, exist_ok=True)
        self.pad_len = pad_len
        self.all_data = []
        self.augment = True
        self.all_masks = []
        self.all_files = []
        self.exp_name = exp_name
        self.split = split
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.text_sample_map = []
        self.limit_size = limit_size
        print("Gathering Chunked Motion Dataset Files")
        filei = 0
        if (os.path.exists(self.base_path / f"{split}_temp_files.bin")):
            with open(self.base_path / f"{split}_temp_files.bin", "rb") as fileref:
                self.all_files = pickle.load(fileref)
            with open(self.base_path / f"{split}_temp_samples.bin", "rb") as fileref:
                self.text_sample_map = pickle.load(fileref)
        else:
            candidate_files = [list() for i in range(n_workers * 20)]
            for root, dir, files in os.walk(datapath / f"{split}_data/"):
                for file in tqdm.tqdm(files, total=len(files), desc="Gathering Candidate Files"):
                    if (Path(file).suffix != ".npy" or "HumanML3D" not in Path(file).stem):
                        continue
                    candidate_files[filei % (n_workers * 20)].append(Path(root) / file)
                    filei += 1
            for key in range(len(candidate_files)):
                candidate_files[key] = (candidate_files[key], chunk_size)
            with Pool(n_workers) as pool:
                for admits, files in tqdm.tqdm(pool.imap_unordered(self._filter_files, candidate_files), total=len(candidate_files), desc = "Filtering Files"):
                    
                    for i in range(len(files)):
                        admit = admits[i]
                        file = files[i]
                        
                        if (not admit):
                            print(admit, file)
                            continue
                        self.all_files.append(file)
                        self.text_sample_map.append(int(file.name.split("vec")[1].split("_")[1]))
            with open(self.base_path / f"{split}_temp_files.bin", "wb") as fileref:
                pickle.dump(self.all_files, fileref, -1)
            with open(self.base_path / f"{split}_temp_samples.bin", "wb") as fileref:
                pickle.dump(self.text_sample_map, fileref, -1)
        with open(Path(datapath) / f"{split}_texts.txt") as fileref:
            self.texts = list(map(lambda line: line.strip(), fileref.readlines()))
        self.mean = np.load(datapath / "Mean.npy").reshape(1, -1)
        self.std = np.load(datapath / "Std.npy").reshape(1, -1)
    
    def __len__(self):
        if (self.limit_size > 0):
            return min(len(self.all_files), self.limit_size)
        else:
            return len(self.all_files)
    
    def __getitem__(self, idx):
        data = np.load(self.all_files[idx])
        data_len = len(data)
        motion_chunks = []
        for i in range(0, data_len, self.chunk_size):
            if (len(motion_chunks) == self.pad_len):
                continue
            motion_chunk = data[i: i+self.chunk_size + self.overlap].flatten()
            if (i+self.chunk_size + self.overlap >= len(data)):
                continue
            motion_chunks.append(motion_chunk.reshape(1, self.chunk_size + self.overlap, -1))
        
        motion_chunks = np.concatenate(motion_chunks, axis=0)
        motion_chunks = (motion_chunks - self.mean) / self.std
        motion_masks = np.zeros(self.pad_len)
        motion_masks[:len(motion_chunks)] = 1
        rand_ind = random.randint(0, len(self.texts) - 1)
        if (rand_ind == self.text_sample_map[idx]):
            rand_ind = (rand_ind + random.randint(0, 100)) % len(self.texts)
        return {
            "motion_chunks": np.concatenate([motion_chunks, np.zeros((self.pad_len - len(motion_chunks), self.chunk_size + self.overlap, motion_chunks.shape[2]))], axis=0),
            "motion_masks": motion_masks, 
            "texts": self.texts[self.text_sample_map[idx]],
            "random_texts": self.texts[rand_ind]
        }


class HumanML3DDataset(Dataset):
    """
    Base implementation to create a dataset of MoBERT compatible chunks
    from the .npy files containg joint position sequences.
    """

    def _set_model_params(self, cache_path, path, device):
        if (not os.path.exists(cache_path / 'feat_data.obj')):

            # Get offsets of target skeleton
            example_data = np.load(sample_data.SAMPLE000000)
            example_data = example_data.reshape(len(example_data), -1, 3)
            example_data = torch.from_numpy(example_data)
            n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
            kinematic_chain = t2m_kinematic_chain
            tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, device)

            params = {
                "l_idx1": 5,
                "l_idx2": 8,
                "fid_r": [8, 11],
                "fid_l": [7, 10],
                "face_joint_indx": [2, 1, 17, 16],
                "n_raw_offsets": n_raw_offsets,
                "kinematic_chain": kinematic_chain,
                "tgt_offsets": tgt_skel.get_offsets_joints(example_data[0])
            }

            self.feat_data = self.load_motion_data(path, params)

            with open(cache_path / 'feat_data.obj', 'wb') as fileobj:
                pickle.dump(self.feat_data, fileobj)
        else:
            with open(cache_path / 'feat_data.obj', 'rb') as fileobj:
                self.feat_data = pickle.load(fileobj)

    def __init__(self, cache_path, path, chunk_size, overlap, norm_path=None, pad_len=17, device="cpu"):
        os.makedirs(cache_path, exist_ok=True)
        self.pad_len = pad_len
        self.all_data = []
        self.all_masks = []
        self.all_files = []
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.text_sample_map = []

        self._set_model_params(cache_path, path, device)

        if norm_path is None:
            self.mean = np.load(sample_data.MEAN).reshape(1, -1)
            self.std = np.load(sample_data.STD).reshape(1, -1)
        else:
            self.mean = np.load(norm_path / "Mean.npy").reshape(1, -1)
            self.std = np.load(norm_path / "Std.npy").reshape(1, -1)

    def __len__(self):
        raise NotImplementedError("__len__ method must be implemented.")
    
    def __getitem__(self, idx):
        raise NotImplementedError("__getitem__ method must be implemented.")

    def load_motion_data(self, path, device, params):
        """
        Follow structure:

        feature_data = {}
        for file in path:
            parse_file_data()
            for motion in file:
                positions = format_motion_data(motion)
                feature_data[file][motion_id] = process_file(positions, feet_thre, *params)
        
        self.feat_data = feature_data
        """
        raise NotImplementedError("load_motion_data method must be implemented.")
