# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# Modified: Chengchang Tian <chengchang_tian@seu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import random
import math
import copy
from collections import OrderedDict

import numpy as np
import torch
import open3d as o3d
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

from opencood.utils import box_utils as box_utils
from opencood.utils.box_utils import corner_to_center
from opencood.utils.camera_utils import (
    sample_augmentation,
    img_transform,
    normalize_img,
    img_to_tensor,
)
from opencood.utils.common_utils import (
    merge_features_to_dict,
    compute_iou,
    convert_format,
    read_json,
)
from opencood.utils.transformation_utils import (
    x1_to_x2,
    x_to_world,
    get_pairwise_transformation,
)
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.utils.pcd_utils import (
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum,
)
from opencood.utils.heter_utils import Adaptor
from opencood.data_utils.pre_processor import build_preprocessor


def farthest_point_sampling(points, num_samples):
    """FPS via Open3D, keeps intensity channel by reindexing back to source."""
    xyz = points[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    sampled_pcd = pcd.farthest_point_down_sample(num_samples)
    sampled_xyz = np.asarray(sampled_pcd.points)

    indices = []
    for point in sampled_xyz:
        idx = np.where(np.all(xyz == point, axis=1))[0][0]
        indices.append(idx)

    return points[indices]


def project_world_objects_dairv2x(object_list, lidar_pose, order='lwh'):
    """
    Project the objects under world coordinates into LiDAR coordinates.

    Parameters
    ----------
    object_list : list
        The list contains all objects surrounding a certain cav.
    lidar_pose : list
        (6, ), lidar pose under world coordinate, [x, y, z, roll, yaw, pitch].
    order : str
        'lwh' or 'hwl'
    """
    boxes_lidar = []
    corners_lidar_list = []
    for object_content in object_list:
        lidar_to_world = x_to_world(lidar_pose)  # T_world_lidar
        world_to_lidar = np.linalg.inv(lidar_to_world)

        corners_world = np.array(object_content['world_8_points'])  # [8, 3]
        corners_world_homo = np.pad(corners_world, ((0, 0), (0, 1)), constant_values=1)  # [8, 4]
        corners_lidar = (world_to_lidar @ corners_world_homo.T).T
        corners_lidar_list.append(corners_lidar)

        bbx_lidar = np.expand_dims(corners_lidar[:, :3], 0)  # [1, 8, 3]
        bbx_lidar = corner_to_center(bbx_lidar, order=order)
        boxes_lidar.append(bbx_lidar[0])
    return boxes_lidar, corners_lidar_list


def getIntermediateheterFusionDataset(cls):
    """
    cls: the Basedataset.
    """
    class IntermediateheterFusionDataset(cls):
        def __init__(self, params, visualize, train=True):
            super().__init__(params, visualize, train)
            # intermediate and supervise single
            self.supervise_single = (
                'supervise_single' in params['model']['args']
                and params['model']['args']['supervise_single']
            )
            self.proj_first = (
                False if 'proj_first' not in params['fusion']['args']
                else params['fusion']['args']['proj_first']
            )

            self.anchor_box = self.post_processor.generate_anchor_box()
            self.anchor_box_torch = torch.from_numpy(self.anchor_box)

            self.heterogeneous = True
            self.modality_assignment = (
                None if (
                    'assignment_path' not in params['heter']
                    or params['heter']['assignment_path'] is None
                    or params['heter']['assignment_path'] == 'None'
                )
                else read_json(params['heter']['assignment_path'])
            )

            self.ego_modality = params['heter']['ego_modality']  # e.g. "m1", "m1&m2", "m3"

            self.modality_name_list = list(params['heter']['modality_setting'].keys())
            self.sensor_type_dict = OrderedDict()

            lidar_channels_dict = params['heter'].get('lidar_channels_dict', OrderedDict())
            mapping_dict = params['heter']['mapping_dict']
            cav_preference = params['heter'].get('cav_preference', None)

            self.adaptor = Adaptor(
                self.ego_modality,
                self.modality_name_list,
                self.modality_assignment,
                lidar_channels_dict,
                mapping_dict,
                cav_preference,
                train,
            )

            for modality_name, modal_setting in params['heter']['modality_setting'].items():
                self.sensor_type_dict[modality_name] = modal_setting['sensor_type']
                if modal_setting['sensor_type'] == 'lidar':
                    setattr(
                        self,
                        f"pre_processor_{modality_name}",
                        build_preprocessor(modal_setting['preprocess'], train),
                    )
                elif modal_setting['sensor_type'] == 'camera':
                    setattr(self, f"data_aug_conf_{modality_name}", modal_setting['data_aug_conf'])
                else:
                    raise ValueError("Not support this type of sensor")

            self.reinitialize()

            self.kd_flag = params.get('kd_flag', False)

            self.box_align = False
            if "box_align" in params:
                self.box_align = True
                self.stage1_result_path = (
                    params['box_align']['train_result'] if train
                    else params['box_align']['val_result']
                )
                self.stage1_result = read_json(self.stage1_result_path)
                self.box_align_args = params['box_align']['args']

            # ---- PHD (CDAM § 3.2.1) hyperparameters ----
            # If 'phd' is absent from YAML, defaults match the settings used in the paper.
            phd_cfg = params.get('phd', {})
            self.dth                = phd_cfg.get('dth', 50.0)            # paper § 4.2: d_th = 50 m
            self.Nmax               = phd_cfg.get('Nmax', 2)              # paper § 4.2: N_max = 2
            self.alpha              = phd_cfg.get('alpha', 0.8)           # inner-box scaling factor α
            self.beta_in            = phd_cfg.get('beta_in', 0.6)         # paper § 4.2: β_in  = 0.6
            self.beta_out           = phd_cfg.get('beta_out', 0.8)        # paper § 4.2: β_out = 0.8
            self.inner_cap          = phd_cfg.get('inner_cap', 30)        # impl. cap on inner samples
            self.outer_cap          = phd_cfg.get('outer_cap', 50)        # impl. cap on outer samples
            self.min_points_per_obj = phd_cfg.get('min_points_per_obj', 100)  # min #points to consider an object

        def get_item_single_car(self, selected_cav_base, ego_cav_base):
            """
            Process a single CAV's information for the train/test pipeline.

            Parameters
            ----------
            selected_cav_base : dict
                The dictionary contains a single CAV's raw information,
                including 'params', 'camera_data'.
            ego_cav_base : dict
                The ego CAV's base data dict (used for pose reference).

            Returns
            -------
            selected_cav_processed : dict
                The dictionary contains the cav's processed information.
            """
            selected_cav_processed = {}
            ego_pose = ego_cav_base['params']['lidar_pose']
            ego_pose_clean = ego_cav_base['params']['lidar_pose_clean']

            # calculate the transformation matrix
            transformation_matrix = x1_to_x2(
                selected_cav_base['params']['lidar_pose'], ego_pose
            )  # T_ego_cav
            transformation_matrix_clean = x1_to_x2(
                selected_cav_base['params']['lidar_pose_clean'], ego_pose_clean
            )

            modality_name = selected_cav_base['modality_name']
            sensor_type = self.sensor_type_dict[modality_name]

            # ---------- lidar ----------
            if sensor_type == "lidar" or self.visualize:
                lidar_np = selected_cav_base['lidar_np']
                lidar_np = shuffle_points(lidar_np)
                # remove points that hit itself
                lidar_np = mask_ego_points(lidar_np)
                # project the lidar to ego space (x,y,z in ego space)
                projected_lidar = box_utils.project_points_by_matrix_torch(
                    lidar_np[:, :3], transformation_matrix
                )
                if self.proj_first:
                    lidar_np[:, :3] = projected_lidar

                if self.visualize:
                    selected_cav_processed.update({'projected_lidar': projected_lidar})

                if self.kd_flag:
                    lidar_proj_np = copy.deepcopy(lidar_np)
                    lidar_proj_np[:, :3] = projected_lidar
                    selected_cav_processed.update({'projected_lidar': lidar_proj_np})

                    # 2023.8.31, to correct discretization errors. Replace one point to avoid empty voxels.
                    lidar_proj_np[np.random.randint(0, lidar_proj_np.shape[0]), :3] = np.array([0, 0, 0])
                    processed_lidar_proj = eval(f"self.pre_processor_{modality_name}").preprocess(lidar_proj_np)
                    selected_cav_processed.update(
                        {f'processed_features_{modality_name}_proj': processed_lidar_proj}
                    )

                if sensor_type == "lidar":
                    processed_lidar = eval(f"self.pre_processor_{modality_name}").preprocess(lidar_np)
                    selected_cav_processed.update(
                        {f'processed_features_{modality_name}': processed_lidar}
                    )

            # ---------- single-view GT (reference pose is itself) ----------
            object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center(
                [selected_cav_base], selected_cav_base['params']['lidar_pose']
            )
            label_dict = self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=self.anchor_box,
                mask=object_bbx_mask,
            )
            selected_cav_processed.update({
                "single_label_dict": label_dict,
                "single_object_bbx_center": object_bbx_center,
                "single_object_bbx_mask": object_bbx_mask,
            })

            # ---------- camera ----------
            if sensor_type == "camera":
                camera_data_list = selected_cav_base["camera_data"]
                params = selected_cav_base["params"]
                imgs, rots, trans, intrins = [], [], [], []
                extrinsics, post_rots, post_trans = [], [], []

                for idx, img in enumerate(camera_data_list):
                    camera_to_lidar, camera_intrinsic = self.get_ext_int(params, idx)

                    intrin = torch.from_numpy(camera_intrinsic)
                    rot = torch.from_numpy(camera_to_lidar[:3, :3])  # R_wc (world-coord == lidar-coord)
                    tran = torch.from_numpy(camera_to_lidar[:3, 3])  # T_wc

                    post_rot = torch.eye(2)
                    post_tran = torch.zeros(2)

                    img_src = [img]

                    # depth
                    if self.load_depth_file:
                        depth_img = selected_cav_base["depth_data"][idx]
                        img_src.append(depth_img)

                    # data augmentation
                    resize, resize_dims, crop, flip, rotate = sample_augmentation(
                        eval(f"self.data_aug_conf_{modality_name}"), self.train
                    )
                    img_src, post_rot2, post_tran2 = img_transform(
                        img_src, post_rot, post_tran,
                        resize=resize, resize_dims=resize_dims,
                        crop=crop, flip=flip, rotate=rotate,
                    )
                    # promote augmentation matrices to 3x3
                    post_tran = torch.zeros(3)
                    post_rot = torch.eye(3)
                    post_tran[:2] = post_tran2
                    post_rot[:2, :2] = post_rot2

                    # decouple RGB and Depth
                    img_src[0] = normalize_img(img_src[0])
                    if self.load_depth_file:
                        img_src[1] = img_to_tensor(img_src[1]) * 255

                    imgs.append(torch.cat(img_src, dim=0))
                    intrins.append(intrin)
                    extrinsics.append(torch.from_numpy(camera_to_lidar))
                    rots.append(rot)
                    trans.append(tran)
                    post_rots.append(post_rot)
                    post_trans.append(post_tran)

                selected_cav_processed.update({
                    f"image_inputs_{modality_name}": {
                        "imgs": torch.stack(imgs),  # [Ncam, 3or4, H, W]
                        "intrins": torch.stack(intrins),
                        "extrinsics": torch.stack(extrinsics),
                        "rots": torch.stack(rots),
                        "trans": torch.stack(trans),
                        "post_rots": torch.stack(post_rots),
                        "post_trans": torch.stack(post_trans),
                    }
                })

            # ---------- anchor box ----------
            selected_cav_processed.update({"anchor_box": self.anchor_box})

            # ---------- object center in ego coord ----------
            object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center(
                [selected_cav_base], ego_pose_clean
            )

            selected_cav_processed.update({
                "object_bbx_center": object_bbx_center[object_bbx_mask == 1],
                "object_bbx_mask": object_bbx_mask,
                "object_ids": object_ids,
                'transformation_matrix': transformation_matrix,
                'transformation_matrix_clean': transformation_matrix_clean,
            })

            return selected_cav_processed

        def proximal_region_hierarchical_downsampling(self, base_data_dict):
            """
            Proximal-region Hierarchical Downsampling (PHD), CDAM § 3.2.1, Fig. 3.

            Balances point density across distance ranges on the ego agent while
            preserving contour points and physical occlusion relationships.

            Step 1 — Proximal Object Selection : pick up to N_max objects with
                                                 d_i(o_k) <= d_th
            Step 2 — Region Partition          : split each object's points into
                                                 inner (B^in) and outer (B^out \\ B^in)
                                                 regions via concentric boxes scaled by α
            Step 3 — Density Balancing         : FPS with ratios β_in (inner) /
                                                 β_out (outer); inner is sampled more
                                                 aggressively, outer is preserved for contour

            Notation maps to the paper as:
                outer_boxes          -> B^out_{i,k}
                inner_boxes          -> B^in_{i,k}      (α-scaled concentric box)
                outer_region_indices -> points ∈ B^out
                inner_region_indices -> points ∈ B^in
                outer_only_*         -> points ∈ B^out \\ B^in   (= R^out)
                inner_region_*       -> points ∈ B^in            (= R^in)
                *_sampled            -> R̃^in / R̃^out (after FPS)
                object_points_balanced -> P̃_{i,k} = R̃^in ∪ R̃^out
            """
            ego_lidar_pose = base_data_dict[0]['params']['lidar_pose']

            transformation_matrix = x1_to_x2(
                base_data_dict[0]['params']['lidar_pose'], ego_lidar_pose
            )
            lidar_np = base_data_dict[0]['lidar_np']
            # project the lidar to ego space
            lidar_np[:, :3] = box_utils.project_points_by_matrix_torch(
                lidar_np[:, :3], transformation_matrix
            )
            ego_center = box_utils.project_points_by_matrix_torch(
                np.array([[0, 0, -0.3]]).astype('float32'), transformation_matrix
            )
            points = lidar_np
            annos = base_data_dict[0]['params']['vehicles_all']
            if len(annos) == 0:
                return base_data_dict

            # B^out and B^in (α-scaled concentric)
            outer_boxes_list, _ = project_world_objects_dairv2x(annos, ego_lidar_pose)
            outer_boxes = np.array(outer_boxes_list)
            inner_boxes = np.concatenate(
                (outer_boxes[:, 0:3], outer_boxes[:, 3:6] * self.alpha, outer_boxes[:, 6:7]),
                axis=1,
            )
            num_obj = outer_boxes.shape[0]

            # points within outer / inner boxes
            outer_region_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, :3]), torch.from_numpy(outer_boxes)
            ).numpy()  # (nboxes, npoints)
            inner_region_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, :3]), torch.from_numpy(inner_boxes)
            ).numpy()  # (nboxes, npoints)

            # ---- Step 1: Proximal Object Selection (d_i(o_k) <= d_th, dense enough) ----
            ego_position = ego_center[0]
            distances = np.linalg.norm(outer_boxes[:, 0:3] - ego_position, axis=1)
            within_proximal_region = distances <= self.dth

            point_counts = np.array(
                [np.sum(outer_region_indices[i] > 0) for i in range(num_obj)]
            )
            dense_enough = point_counts > self.min_points_per_obj

            proximal_pool = np.where(within_proximal_region & dense_enough)[0]
            num_to_select = min(self.Nmax, len(proximal_pool))
            if num_to_select > 0:
                selected_proximal_objects = np.random.choice(
                    proximal_pool, size=num_to_select, replace=False
                )
            else:
                selected_proximal_objects = np.array([], dtype=int)

            # ---- Steps 2 & 3: Region Partition + Density Balancing ----
            balanced_points_list = []
            for i in range(num_obj):
                outer_mask = outer_region_indices[i] > 0
                object_points = points[outer_mask]
                if object_points.shape[0] == 0:
                    continue  # Skip if no points in GT box

                if i in selected_proximal_objects:
                    # inner region (R^in): aggressive FPS with β_in
                    inner_mask = inner_region_indices[i] > 0
                    inner_region_points = points[inner_mask]
                    num_inner_points = inner_region_points.shape[0]
                    if num_inner_points > 0:
                        inner_sample_size = min(
                            self.inner_cap, int(self.beta_in * num_inner_points)
                        )
                        inner_points_sampled = farthest_point_sampling(
                            inner_region_points, inner_sample_size
                        )
                    else:
                        inner_points_sampled = inner_region_points

                    # outer region (B^out \ B^in = R^out): conservative FPS with β_out, keep contour
                    outer_only_mask = outer_mask & (~inner_mask)
                    outer_region_points = points[outer_only_mask]
                    num_outer_points = outer_region_points.shape[0]
                    if num_outer_points > 0:
                        outer_sample_size = min(
                            self.outer_cap, int(self.beta_out * num_outer_points)
                        )
                        outer_points_sampled = farthest_point_sampling(
                            outer_region_points, outer_sample_size
                        )
                    else:
                        outer_points_sampled = outer_region_points

                    # P̃_{i,k} = R̃^in ∪ R̃^out
                    object_points_balanced = np.vstack(
                        (inner_points_sampled, outer_points_sampled)
                    )
                    balanced_points_list.append(object_points_balanced)
                else:
                    # not selected → keep original points
                    balanced_points_list.append(object_points)

            # rebuild scene point cloud: background + balanced foreground, then project back
            if len(balanced_points_list) != 0:
                background_mask = np.ones(points.shape[0], dtype=bool)
                for i in range(num_obj):
                    background_mask[outer_region_indices[i] > 0] = False
                background_points = points[background_mask]
                new_point_cloud = np.vstack(
                    (background_points, np.vstack(balanced_points_list))
                )

                # inverse of the rigid transform
                rotation_matrix = transformation_matrix[:3, :3]
                translation_vector = transformation_matrix[:3, 3]
                inverse_rotation_matrix = rotation_matrix.T
                inverse_translation_vector = -np.dot(inverse_rotation_matrix, translation_vector)
                inverse_transform_matrix = np.eye(4)
                inverse_transform_matrix[:3, :3] = inverse_rotation_matrix
                inverse_transform_matrix[:3, 3] = inverse_translation_vector

                new_point_cloud[:, :3] = box_utils.project_points_by_matrix_torch(
                    new_point_cloud[:, :3], inverse_transform_matrix
                )

                base_data_dict[0]['lidar_np'] = new_point_cloud

            return base_data_dict

        # Short alias matching the paper acronym.
        phd = proximal_region_hierarchical_downsampling

        def __getitem__(self, idx):
            base_data_dict = self.retrieve_base_data(idx)
            if base_data_dict is None:
                return self.__getitem__(random.randint(0, len(self) - 1))

            # V2XSET: if the current ego is an RSU, move it to the back and pick a vehicle ego.
            if 'V2XSET' in self.params['root_dir']:
                for cav_id, cav_content in base_data_dict.items():
                    if cav_content['ego'] and cav_content['params']['RSU']:
                        cav_content['ego'] = False
                        base_data_dict.move_to_end(cav_id)
                        if len(base_data_dict) > 1:
                            second_key = list(base_data_dict.keys())[0]
                            base_data_dict[second_key]['ego'] = True
                        break

            if self.train:
                base_data_dict = self.phd(base_data_dict)

            base_data_dict = add_noise_data_dict(base_data_dict, self.params['noise_setting'])

            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = {}

            ego_id = -1
            ego_lidar_pose = []
            ego_cav_base = None

            # find ego
            for cav_id, cav_content in base_data_dict.items():
                if cav_content['ego']:
                    ego_id = cav_id
                    ego_lidar_pose = cav_content['params']['lidar_pose']
                    ego_cav_base = cav_content
                    break

            assert cav_id == list(base_data_dict.keys())[0], \
                "The first element in the OrderedDict must be ego"
            assert ego_id != -1
            assert len(ego_lidar_pose) > 0

            input_list_m1 = []  # can contain lidar or camera
            input_list_m2 = []
            input_list_m3 = []
            input_list_m4 = []

            agent_modality_list = []
            object_stack = []
            object_id_stack = []
            single_label_list = []
            single_object_bbx_center_list = []
            single_object_bbx_mask_list = []
            exclude_agent = []
            lidar_pose_list = []
            lidar_pose_clean_list = []
            cav_id_list = []

            source_stack = []
            source_v2xset_stack = []

            if self.visualize or self.kd_flag:
                projected_lidar_stack = []
                input_list_m1_proj = []  # 2023.8.31 to correct discretization errors with kd flag
                input_list_m2_proj = []
                input_list_m3_proj = []
                input_list_m4_proj = []

            # loop over all CAVs to process information
            for cav_id, selected_cav_base in base_data_dict.items():
                # check communication range
                distance = math.sqrt(
                    (selected_cav_base['params']['lidar_pose'][0] - ego_lidar_pose[0]) ** 2
                    + (selected_cav_base['params']['lidar_pose'][1] - ego_lidar_pose[1]) ** 2
                )
                if distance > self.params['comm_range']:
                    exclude_agent.append(cav_id)
                    continue

                # modality must match
                if self.adaptor.unmatched_modality(selected_cav_base['modality_name']):
                    exclude_agent.append(cav_id)
                    continue

                lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
                lidar_pose_list.append(selected_cav_base['params']['lidar_pose'])  # 6dof pose
                cav_id_list.append(cav_id)

            if len(cav_id_list) == 0:
                return None

            for cav_id in exclude_agent:
                base_data_dict.pop(cav_id)

            ########## Updated by Yifan Lu 2022.1.26 ############
            # box align to correct pose.
            # stage1_content contains all agents, including those out of comm range.
            if self.box_align and str(idx) in self.stage1_result.keys():
                from opencood.models.sub_modules.box_align_v2 import (
                    box_alignment_relative_sample_np,
                )
                stage1_content = self.stage1_result[str(idx)]
                if stage1_content is not None:
                    all_agent_id_list = stage1_content['cav_id_list']
                    all_agent_corners_list = stage1_content['pred_corner3d_np_list']
                    all_agent_uncertainty_list = stage1_content['uncertainty_np_list']

                    cur_agent_id_list = cav_id_list
                    cur_agent_pose = [
                        base_data_dict[cav_id]['params']['lidar_pose'] for cav_id in cav_id_list
                    ]
                    cur_agnet_pose = np.array(cur_agent_pose)
                    cur_agent_in_all_agent = [
                        all_agent_id_list.index(cur_agent) for cur_agent in cur_agent_id_list
                    ]

                    pred_corners_list = [
                        np.array(all_agent_corners_list[i], dtype=np.float64)
                        for i in cur_agent_in_all_agent
                    ]
                    uncertainty_list = [
                        np.array(all_agent_uncertainty_list[i], dtype=np.float64)
                        for i in cur_agent_in_all_agent
                    ]

                    if sum([len(pc) for pc in pred_corners_list]) != 0:
                        refined_pose = box_alignment_relative_sample_np(
                            pred_corners_list,
                            cur_agnet_pose,
                            uncertainty_list=uncertainty_list,
                            **self.box_align_args,
                        )
                        cur_agnet_pose[:, [0, 1, 4]] = refined_pose

                        for i, cav_id in enumerate(cav_id_list):
                            lidar_pose_list[i] = cur_agnet_pose[i].tolist()
                            base_data_dict[cav_id]['params']['lidar_pose'] = cur_agnet_pose[i].tolist()

            pairwise_t_matrix = get_pairwise_transformation(
                base_data_dict, self.max_cav, self.proj_first
            )

            lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)  # [N_cav, 6]
            lidar_poses_clean = np.array(lidar_pose_clean_list).reshape(-1, 6)

            # merge preprocessed features from different cavs into the same dict
            cav_num = len(cav_id_list)
            if (cav_num < 4) & (hasattr(self, "time_delay")):
                return self.__getitem__(random.randint(0, len(self) - 1))

            for _i, cav_id in enumerate(cav_id_list):
                selected_cav_base = base_data_dict[cav_id]
                modality_name = selected_cav_base['modality_name']
                sensor_type = self.sensor_type_dict[modality_name]

                # dynamic object center generator (for heterogeneous input)
                if not self.visualize:
                    self.generate_object_center = eval(f"self.generate_object_center_{sensor_type}")
                else:
                    # In test/visualize phase, use lidar label.
                    self.generate_object_center = self.generate_object_center_lidar

                selected_cav_processed = self.get_item_single_car(selected_cav_base, ego_cav_base)

                object_stack.append(selected_cav_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']

                if 'source' in selected_cav_base:
                    source_stack.append(selected_cav_base['source'])
                if 'params' in selected_cav_base and 'RSU' in selected_cav_base['params']:
                    source_v2xset_stack.append(selected_cav_base['params']['RSU'])

                if sensor_type == "lidar":
                    eval(f"input_list_{modality_name}").append(
                        selected_cav_processed[f"processed_features_{modality_name}"]
                    )
                elif sensor_type == "camera":
                    eval(f"input_list_{modality_name}").append(
                        selected_cav_processed[f"image_inputs_{modality_name}"]
                    )
                else:
                    raise

                agent_modality_list.append(modality_name)

                if self.visualize or self.kd_flag:
                    if (
                        'V2XSET' in self.params['root_dir']
                        or 'v2xsim' in self.params['root_dir']
                        or 'OPV2V' in self.params['root_dir']
                    ):
                        # V2XSET / v2xsim / OPV2V — no cav_id restriction
                        projected_lidar_stack.append(selected_cav_processed['projected_lidar'])
                        if sensor_type == "lidar" and self.kd_flag:
                            eval(f"input_list_{modality_name}_proj").append(
                                selected_cav_processed[f"processed_features_{modality_name}_proj"]
                            )
                    else:
                        # DAIR-V2X / V2X-SIM — only cav_id <= 1
                        if cav_id <= 1:
                            projected_lidar_stack.append(selected_cav_processed['projected_lidar'])
                            if sensor_type == "lidar" and self.kd_flag:
                                eval(f"input_list_{modality_name}_proj").append(
                                    selected_cav_processed[f"processed_features_{modality_name}_proj"]
                                )

                if self.supervise_single or self.heterogeneous:
                    single_label_list.append(selected_cav_processed['single_label_dict'])
                    single_object_bbx_center_list.append(selected_cav_processed['single_object_bbx_center'])
                    single_object_bbx_mask_list.append(selected_cav_processed['single_object_bbx_mask'])

            # generate single-view GT label
            if self.supervise_single or self.heterogeneous:
                single_label_dicts = self.post_processor.collate_batch(single_label_list)
                single_object_bbx_center = torch.from_numpy(np.array(single_object_bbx_center_list))
                single_object_bbx_mask = torch.from_numpy(np.array(single_object_bbx_mask_list))
                processed_data_dict['ego'].update({
                    "single_label_dict_torch": single_label_dicts,
                    "single_object_bbx_center_torch": single_object_bbx_center,
                    "single_object_bbx_mask_torch": single_object_bbx_mask,
                })

            # exclude repetitive objects — DAIR-V2X vs OPV2V-H paths
            if self.params['fusion']['dataset'] == 'dairv2x':
                if len(object_stack) == 1:
                    object_stack = object_stack[0]
                else:
                    ego_boxes_np = object_stack[0]
                    cav_boxes_np = object_stack[1]
                    order = self.params['postprocess']['order']
                    ego_corners_np = box_utils.boxes_to_corners_3d(ego_boxes_np, order)
                    cav_corners_np = box_utils.boxes_to_corners_3d(cav_boxes_np, order)
                    ego_polygon_list = list(convert_format(ego_corners_np))
                    cav_polygon_list = list(convert_format(cav_corners_np))
                    iou_thresh = 0.05

                    gt_boxes_from_cav = []
                    for i in range(len(cav_polygon_list)):
                        cav_polygon = cav_polygon_list[i]
                        ious = compute_iou(cav_polygon, ego_polygon_list)
                        if (ious > iou_thresh).any():
                            continue
                        gt_boxes_from_cav.append(cav_boxes_np[i])

                    if len(gt_boxes_from_cav):
                        object_stack_from_cav = np.stack(gt_boxes_from_cav)
                        object_stack = np.vstack([ego_boxes_np, object_stack_from_cav])
                    else:
                        object_stack = ego_boxes_np

                unique_indices = np.arange(object_stack.shape[0])
                object_id_stack = np.arange(object_stack.shape[0])
            else:
                # OPV2V-H
                unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
                object_stack = np.vstack(object_stack)
                object_stack = object_stack[unique_indices]

            # pad bbox center to fixed length
            object_bbx_center = np.zeros((self.params['postprocess']['max_num'], 7))
            mask = np.zeros(self.params['postprocess']['max_num'])
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1

            for modality_name in self.modality_name_list:
                if self.sensor_type_dict[modality_name] == "lidar":
                    merged_feature_dict = merge_features_to_dict(eval(f"input_list_{modality_name}"))
                    processed_data_dict['ego'].update({f'input_{modality_name}': merged_feature_dict})
                elif self.sensor_type_dict[modality_name] == "camera":
                    merged_image_inputs_dict = merge_features_to_dict(
                        eval(f"input_list_{modality_name}"), merge='stack'
                    )
                    processed_data_dict['ego'].update({f'input_{modality_name}': merged_image_inputs_dict})

            if self.kd_flag:
                # heterogeneous setting does not support DiscoNet's kd; we only emit per-modality proj features
                for modality_name in self.modality_name_list:
                    processed_data_dict['ego'].update({
                        f'input_{modality_name}_proj':
                            merge_features_to_dict(eval(f"input_list_{modality_name}_proj"))
                    })

            processed_data_dict['ego'].update({'agent_modality_list': agent_modality_list})

            # final targets
            label_dict = self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=self.anchor_box,
                mask=mask,
            )

            processed_data_dict['ego'].update({
                'object_bbx_center': object_bbx_center,
                'object_bbx_mask': mask,
                'object_ids': [object_id_stack[i] for i in unique_indices],
                'anchor_box': self.anchor_box,
                'label_dict': label_dict,
                'cav_num': cav_num,
                'pairwise_t_matrix': pairwise_t_matrix,
                'lidar_poses_clean': lidar_poses_clean,
                'lidar_poses': lidar_poses,
            })

            if self.visualize:
                processed_data_dict['ego'].update({'origin_lidar': np.vstack(projected_lidar_stack)})

            processed_data_dict['ego'].update({
                'sample_idx': idx,
                'cav_id_list': cav_id_list,
            })

            if hasattr(self, 'time_delay'):
                processed_data_dict['ego'].update({
                    'frame_id': [
                        base_data_dict[0]['frame_id'],
                        base_data_dict[1]['frame_id'],
                        base_data_dict[2]['frame_id'],
                        base_data_dict[3]['frame_id'],
                    ],
                    'time_delay': self.time_delay,
                })

            if 'source' in selected_cav_base:
                processed_data_dict['ego'].update({'source': source_stack})

            if 'params' in selected_cav_base and 'RSU' in selected_cav_base['params']:
                processed_data_dict['ego'].update({'source_v2xset': source_v2xset_stack})

            return processed_data_dict

        def collate_batch_train(self, batch):
            # Intermediate fusion is different from the other two
            output_dict = {'ego': {}}

            object_bbx_center = []
            object_bbx_mask = []
            object_ids = []
            inputs_list_m1 = []
            inputs_list_m2 = []
            inputs_list_m3 = []
            inputs_list_m4 = []

            inputs_list_m1_proj = []
            inputs_list_m2_proj = []
            inputs_list_m3_proj = []
            inputs_list_m4_proj = []

            agent_modality_list = []
            record_len = []
            label_dict_list = []
            lidar_pose_list = []
            origin_lidar = []
            lidar_pose_clean_list = []
            pairwise_t_matrix_list = []

            ### 2022.10.10 single gt ####
            if self.supervise_single or self.heterogeneous:
                pos_equal_one_single = []
                neg_equal_one_single = []
                targets_single = []
                object_bbx_center_single = []
                object_bbx_mask_single = []

            for i in range(len(batch)):
                ego_dict = batch[i]['ego']
                object_bbx_center.append(ego_dict['object_bbx_center'])
                object_bbx_mask.append(ego_dict['object_bbx_mask'])
                object_ids.append(ego_dict['object_ids'])
                lidar_pose_list.append(ego_dict['lidar_poses'])  # [N, 6]
                lidar_pose_clean_list.append(ego_dict['lidar_poses_clean'])

                for modality_name in self.modality_name_list:
                    if ego_dict[f'input_{modality_name}'] is not None:
                        eval(f"inputs_list_{modality_name}").append(ego_dict[f'input_{modality_name}'])

                agent_modality_list.extend(ego_dict['agent_modality_list'])

                record_len.append(ego_dict['cav_num'])
                label_dict_list.append(ego_dict['label_dict'])
                pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])

                if self.visualize:
                    origin_lidar.append(ego_dict['origin_lidar'])

                if self.kd_flag:
                    for modality_name in self.modality_name_list:
                        if ego_dict[f'input_{modality_name}_proj'] is not None:
                            eval(f"inputs_list_{modality_name}_proj").append(
                                ego_dict[f"input_{modality_name}_proj"]
                            )

                if self.supervise_single or self.heterogeneous:
                    pos_equal_one_single.append(ego_dict['single_label_dict_torch']['pos_equal_one'])
                    neg_equal_one_single.append(ego_dict['single_label_dict_torch']['neg_equal_one'])
                    targets_single.append(ego_dict['single_label_dict_torch']['targets'])
                    object_bbx_center_single.append(ego_dict['single_object_bbx_center_torch'])
                    object_bbx_mask_single.append(ego_dict['single_object_bbx_mask_torch'])

            # (B, max_num, 7)
            object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
            object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

            # 2023.2.5 per-modality merge
            for modality_name in self.modality_name_list:
                if len(eval(f"inputs_list_{modality_name}")) != 0:
                    if self.sensor_type_dict[modality_name] == "lidar":
                        merged_feature_dict = merge_features_to_dict(eval(f"inputs_list_{modality_name}"))
                        processed_lidar_torch_dict = eval(
                            f"self.pre_processor_{modality_name}"
                        ).collate_batch(merged_feature_dict)
                        output_dict['ego'].update({f'inputs_{modality_name}': processed_lidar_torch_dict})
                    elif self.sensor_type_dict[modality_name] == "camera":
                        merged_image_inputs_dict = merge_features_to_dict(
                            eval(f"inputs_list_{modality_name}"), merge='cat'
                        )
                        output_dict['ego'].update({f'inputs_{modality_name}': merged_image_inputs_dict})

            output_dict['ego'].update({"agent_modality_list": agent_modality_list})

            record_len = torch.from_numpy(np.array(record_len, dtype=int))
            lidar_pose = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0))
            lidar_pose_clean = torch.from_numpy(np.concatenate(lidar_pose_clean_list, axis=0))
            label_torch_dict = self.post_processor.collate_batch(label_dict_list)

            # for centerpoint
            label_torch_dict.update({
                'object_bbx_center': object_bbx_center,
                'object_bbx_mask': object_bbx_mask,
            })

            # (B, max_cav)
            pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))
            label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
            label_torch_dict['record_len'] = record_len

            # object id is only used during inference (batch size == 1) — take first
            output_dict['ego'].update({
                'object_bbx_center': object_bbx_center,
                'object_bbx_mask': object_bbx_mask,
                'record_len': record_len,
                'label_dict': label_torch_dict,
                'object_ids': object_ids[0],
                'pairwise_t_matrix': pairwise_t_matrix,
                'lidar_pose_clean': lidar_pose_clean,
                'lidar_pose': lidar_pose,
                'anchor_box': self.anchor_box_torch,
            })

            if hasattr(self, 'time_delay'):
                frame_id_batch = []
                time_delay = []
                for fd in range(len(batch)):
                    frame_id_batch.append(batch[fd]['ego']['frame_id'])
                    time_delay.append(batch[fd]['ego']['time_delay'])
                output_dict['ego'].update({'frame_id': frame_id_batch})
                output_dict['ego'].update({'time_delay': time_delay})

            if 'source' in batch[0]['ego']:
                source_batch = []
                for fd in range(len(batch)):
                    source_batch.append(batch[fd]['ego']['source'])
                output_dict['ego'].update({'source': source_batch})

            if 'source_v2xset' in batch[0]['ego']:
                source_v2xset_batch = []
                for fd in range(len(batch)):
                    source_v2xset_batch.append(batch[fd]['ego']['source_v2xset'])
                output_dict['ego'].update({'source_v2xset': source_v2xset_batch})

            if self.visualize:
                origin_lidar = np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict['ego'].update({'origin_lidar': origin_lidar})

            if self.kd_flag:
                for modality_name in self.modality_name_list:
                    if (
                        len(eval(f"inputs_list_{modality_name}_proj")) != 0
                        and self.sensor_type_dict[modality_name] == "lidar"
                    ):
                        merged_feature_proj_dict = merge_features_to_dict(
                            eval(f"inputs_list_{modality_name}_proj")
                        )
                        processed_lidar_torch_proj_dict = eval(
                            f"self.pre_processor_{modality_name}"
                        ).collate_batch(merged_feature_proj_dict)
                        output_dict['ego'].update(
                            {f'inputs_{modality_name}_proj': processed_lidar_torch_proj_dict}
                        )

            if self.supervise_single or self.heterogeneous:
                output_dict['ego'].update({
                    "label_dict_single": {
                        "pos_equal_one": torch.cat(pos_equal_one_single, dim=0),
                        "neg_equal_one": torch.cat(neg_equal_one_single, dim=0),
                        "targets": torch.cat(targets_single, dim=0),
                        "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
                        "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0),
                    },
                    "object_bbx_center_single": torch.cat(object_bbx_center_single, dim=0),
                    "object_bbx_mask_single": torch.cat(object_bbx_mask_single, dim=0),
                })

            return output_dict

        def collate_batch_test(self, batch):
            assert len(batch) <= 1, "Batch size 1 is required during testing!"
            if batch[0] is None:
                return None
            output_dict = self.collate_batch_train(batch)
            if output_dict is None:
                return None

            if batch[0]['ego']['anchor_box'] is not None:
                output_dict['ego'].update({'anchor_box': self.anchor_box_torch})

            # transformation matrix (4, 4) to ego — only used in post-process (no-op since we predict in ego coord)
            transformation_matrix_torch = torch.from_numpy(np.identity(4)).float()
            transformation_matrix_clean_torch = torch.from_numpy(np.identity(4)).float()

            output_dict['ego'].update({
                'transformation_matrix': transformation_matrix_torch,
                'transformation_matrix_clean': transformation_matrix_clean_torch,
            })

            output_dict['ego'].update({
                "sample_idx": batch[0]['ego']['sample_idx'],
                "cav_id_list": batch[0]['ego']['cav_id_list'],
                "agent_modality_list": batch[0]['ego']['agent_modality_list'],
            })
            return output_dict

        def post_process(self, data_dict, output_dict):
            """
            Process the outputs of the model to 2D/3D bounding box.

            Parameters
            ----------
            data_dict : dict
                The dictionary containing the origin input data of model.
            output_dict : dict
                The dictionary containing the output of the model.

            Returns
            -------
            pred_box_tensor : torch.Tensor
                The tensor of prediction bounding box after NMS.
            pred_score : torch.Tensor
                The tensor of prediction confidence scores.
            gt_box_tensor : torch.Tensor
                The tensor of gt bounding box.
            """
            pred_box_tensor, pred_score = self.post_processor.post_process(data_dict, output_dict)
            gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)
            return pred_box_tensor, pred_score, gt_box_tensor

    return IntermediateheterFusionDataset