# Borrowed from OmniSeg3D-GS (https://github.com/OceanYing/OmniSeg3D-GS)
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_contrastive_feature
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
# from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import colorsys
import cv2
from sklearn.decomposition import PCA

# from scene.gaussian_model import GaussianModel
from scene import Scene, GaussianModel, FeatureGaussianModel
import dearpygui.dearpygui as dpg
import math
from scene.cameras import Camera
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

from scipy.spatial.transform import Rotation as R

# from cuml.cluster.hdbscan import HDBSCAN
from hdbscan import HDBSCAN

# PLY 저장을 위한 추가 import
try:
    from plyfile import PlyData, PlyElement
    PLY_AVAILABLE = True
except ImportError:
    print("Warning: plyfile not available. PLY export will be disabled.")
    PLY_AVAILABLE = False

def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min() + 1e-7)
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)
    return depth_img

class CONFIG:
    r = 2   # scale ratio
    window_width = int(2160/r)
    window_height = int(1200/r)

    width = int(2160/r)
    height = int(1200/r)

    radius = 2

    debug = False
    dt_gamma = 0.2

    # gaussian model
    sh_degree = 3

    convert_SHs_python = False
    compute_cov3D_python = False

    white_background = False

    FEATURE_DIM = 32
    MODEL_PATH = './output/figurines' # 30000

    FEATURE_GAUSSIAN_ITERATION = 10000
    SCENE_GAUSSIAN_ITERATION = 30000

    SCALE_GATE_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/scale_gate.pt')

    FEATURE_PCD_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/contrastive_feature_point_cloud.ply')
    SCENE_PCD_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(SCENE_GAUSSIAN_ITERATION)}/scene_point_cloud.ply')


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_quat(
            [0, 0, 0, 1]
        )  # init camera matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!
        self.right = np.array([1, 0, 0], dtype=np.float32)  # need to be normalized!
        self.fovy = fovy
        self.translate = np.array([0, 0, self.radius])
        self.scale_f = 1.0


        self.rot_mode = 1   # rotation mode (1: self.pose_movecenter (movable rotation center), 0: self.pose_objcenter (fixed scene center))
        # self.rot_mode = 0


    @property
    def pose_movecenter(self):
        # --- first move camera to radius : in world coordinate--- #
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        
        # --- rotate: Rc --- #
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        # --- translate: tc --- #
        res[:3, 3] -= self.center
        
        # --- Convention Transform --- #
        # now we have got matrix res=c2w=[Rc|tc], but gaussian-splatting requires convention as [Rc|-Rc.T@tc]
        res[:3, 3] = -rot[:3, :3].transpose() @ res[:3, 3]
        
        return res
    
    @property
    def pose_objcenter(self):
        res = np.eye(4, dtype=np.float32)
        
        # --- rotate: Rw --- #
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        # --- translate: tw --- #
        res[2, 3] += self.radius    # camera coordinate z-axis
        res[:3, 3] -= self.center   # camera coordinate x,y-axis
        
        # --- Convention Transform --- #
        # now we have got matrix res=w2c=[Rw|tw], but gaussian-splatting requires convention as [Rc|-Rc.T@tc]=[Rw.T|tw]
        res[:3, :3] = rot[:3, :3].transpose()
        
        return res

    @property
    def opt_pose(self):
        # --- deprecated ! Not intuitive implementation --- #
        res = np.eye(4, dtype=np.float32)

        res[:3, :3] = self.rot.as_matrix()

        scale_mat = np.eye(4)
        scale_mat[0, 0] = self.scale_f      # why apply scale ratio to rotation matrix? It's confusing.
        scale_mat[1, 1] = self.scale_f
        scale_mat[2, 2] = self.scale_f

        transl = self.translate - self.center
        transl_mat = np.eye(4)
        transl_mat[:3, 3] = transl

        return transl_mat @ scale_mat @ res

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])

    def orbit(self, dx, dy):
        if self.rot_mode == 1:    # rotate the camera axis, in world coordinate system
            up = self.rot.as_matrix()[:3, 1]
            side = self.rot.as_matrix()[:3, 0]
        elif self.rot_mode == 0:    # rotate in camera coordinate system
            up = -self.up
            side = -self.right
        rotvec_x = up * np.radians(0.01 * dx)
        rotvec_y = side * np.radians(0.01 * dy)

        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        # self.radius *= 1.1 ** (-delta)    # non-linear version
        self.radius -= 0.1 * delta      # linear version

    def pan(self, dx, dy, dz=0):
        
        if self.rot_mode == 1:
            # pan in camera coordinate system: project from [Coord_c] to [Coord_w]
            self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])
        elif self.rot_mode == 0:
            # pan in world coordinate system: at [Coord_w]
            self.center += 0.0005 * np.array([-dx, dy, dz])


class GaussianSplattingGUI:
    def __init__(self, opt, gaussian_model:GaussianModel, feature_gaussian_model:FeatureGaussianModel, scale_gate: torch.nn.modules.container.Sequential) -> None:
        self.opt = opt

        self.width = opt.width
        self.height = opt.height
        self.window_width = opt.window_width
        self.window_height = opt.window_height
        self.camera = OrbitCamera(opt.width, opt.height, r=opt.radius)

        bg_color = [1, 1, 1] if opt.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        bg_feature = [0 for i in range(opt.FEATURE_DIM)]
        bg_feature = torch.tensor(bg_feature, dtype=torch.float32, device="cuda")

        self.bg_color = background
        self.bg_feature = bg_feature
        self.render_buffer = np.zeros((self.width, self.height, 3), dtype=np.float32)
        self.update_camera = True
        self.dynamic_resolution = True
        self.debug = opt.debug
        self.engine = {
            'scene': gaussian_model,
            'feature': feature_gaussian_model,
            'scale_gate': scale_gate
        }

        self.cluster_point_colors = None
        self.label_to_color = np.random.rand(1000, 3)
        self.seg_score = None

        self.proj_mat = None

        self.load_model = False
        print("loading model file...")
        self.engine['scene'].load_ply(self.opt.SCENE_PCD_PATH)
        self.engine['feature'].load_ply(self.opt.FEATURE_PCD_PATH)
        self.engine['scale_gate'].load_state_dict(torch.load(self.opt.SCALE_GATE_PATH))
        self.do_pca()   # calculate self.proj_mat
        self.load_model = True

        print("loading model file done.")

        self.mode = "image"  # choose from ['image', 'depth']

        dpg.create_context()
        self.register_dpg()

        self.frame_id = 0

        # --- for better operation --- #
        self.moving = False
        self.moving_middle = False
        self.mouse_pos = (0, 0)

        # --- for interactive segmentation --- #
        self.img_mode = 0
        self.clickmode_button = False
        self.clickmode_multi_button = False     # choose multiple object 
        self.new_click = False
        self.prompt_num = 0
        self.new_click_xy = []
        self.clear_edit = False                 # clear all the click prompts
        self.roll_back = False
        self.preview = False    # binary segmentation mode
        self.segment3d_flag = False
        self.reload_flag = False        # reload the whole scene / point cloud
        self.object_seg_id = 0          # to store the segmented object with increasing index order (path at: ./)
        self.cluster_in_3D_flag = False

        self.render_mode_rgb = False
        self.render_mode_similarity = False
        self.render_mode_pca = False
        self.render_mode_cluster = False

        self.save_flag = False
        
        # PLY 저장 관련 플래그들 추가
        self.save_ply_flag = False
        self.save_all_objects_flag = False

    def __del__(self):
        dpg.destroy_context()

    def save_segmented_gaussians_as_ply(self, save_path):
        """
        세그멘테이션된 가우시안들만 PLY 파일로 저장 (로컬 좌표계로 변환)
        """
        if not PLY_AVAILABLE:
            print("Error: plyfile library not available. Please install with: pip install plyfile")
            return False
            
        try:
            # 현재 세그멘테이션 마스크 가져오기
            mask = self.engine['scene']._mask == self.engine['scene'].segment_times + 1
            
            if mask.sum() == 0:
                print("No segmented gaussians to save!")
                return False
                
            print(f"Saving {mask.sum()} gaussians out of {mask.shape[0]} total gaussians...")
            
            # 디버깅: 실제 데이터 크기 확인
            print(f"Debug info:")
            print(f"  - Mask shape: {mask.shape}, True count: {mask.sum()}")
            print(f"  - _xyz shape: {self.engine['scene']._xyz.shape}")
            print(f"  - _features_dc shape: {self.engine['scene']._features_dc.shape}")
            print(f"  - _opacity shape: {self.engine['scene']._opacity.shape}")
            print(f"  - _scaling shape: {self.engine['scene']._scaling.shape}")
            print(f"  - _rotation shape: {self.engine['scene']._rotation.shape}")
            
            # 선택된 가우시안들의 파라미터 추출 (마스크 적용)
            xyz_full = self.engine['scene']._xyz.detach().cpu().numpy()
            xyz_selected = xyz_full[mask.cpu().numpy()]
            
            # 세그멘테이션된 객체의 중심점 계산
            center = np.mean(xyz_selected, axis=0)
            print(f"Object center: {center}")
            
            # 로컬 좌표계로 변환 (중심점을 원점으로 이동)
            xyz = xyz_selected - center
            print(f"Extracted xyz (centered): {xyz.shape}")
            
            # features_dc 차원 체크 및 안전한 추출 (마스크 적용)
            features_dc_raw = self.engine['scene']._features_dc.detach().cpu().numpy()
            if len(features_dc_raw.shape) == 3:  # (N, 1, 3) 형태
                features_dc = features_dc_raw[mask.cpu().numpy()].reshape(-1, 3)
            else:  # (N, 3) 형태
                features_dc = features_dc_raw[mask.cpu().numpy()]
            print(f"Extracted features_dc: {features_dc.shape}")
                
            # features_rest 차원 체크 및 안전한 추출 (마스크 적용)
            features_rest_raw = self.engine['scene']._features_rest.detach().cpu().numpy()
            if len(features_rest_raw.shape) == 3:  # (N, 15, 3) 형태
                features_rest = features_rest_raw[mask.cpu().numpy()].reshape(-1, 45)  # 15*3=45
            else:  # 이미 (N, 45) 형태
                features_rest = features_rest_raw[mask.cpu().numpy()]
            print(f"Extracted features_rest: {features_rest.shape}")
                
            # 나머지 파라미터들도 마스크 적용
            opacity = self.engine['scene']._opacity.detach().cpu().numpy()[mask.cpu().numpy()]
            scaling = self.engine['scene']._scaling.detach().cpu().numpy()[mask.cpu().numpy()]
            rotation = self.engine['scene']._rotation.detach().cpu().numpy()[mask.cpu().numpy()]
            print(f"Extracted opacity: {opacity.shape}, scaling: {scaling.shape}, rotation: {rotation.shape}")
            
            # 피처 가우시안도 함께 저장하고 싶다면
            # point_features = None
            # if hasattr(self.engine['feature'], 'get_point_features'):
            #     try:
            #         point_features_raw = self.engine['feature'].get_point_features
            #         if point_features_raw is not None:
            #             # 마스크 크기가 일치하는지 확인
            #             feature_data_size = point_features_raw.shape[0]
            #             if feature_data_size == mask.shape[0]:
            #                 point_features = point_features_raw[mask].detach().cpu().numpy()
            #                 print(f"Extracted point_features: {point_features.shape}")
            #             else:
            #                 print(f"Warning: Feature size mismatch. Scene: {mask.shape[0]}, Features: {feature_data_size}")
            #                 # 크기 조정
            #                 if feature_data_size < mask.shape[0]:
            #                     feature_mask = mask[:feature_data_size]
            #                 else:
            #                     feature_mask = torch.zeros(feature_data_size, dtype=mask.dtype, device=mask.device)
            #                     feature_mask[:mask.shape[0]] = mask
                            
            #                 if feature_mask.sum() > 0:
            #                     point_features = point_features_raw[feature_mask].detach().cpu().numpy()
            #                     print(f"Adjusted and extracted point_features: {point_features.shape}")
            #     except Exception as e:
            #         print(f"Warning: Could not extract point features: {e}")
            #         point_features = None
            
            # PLY 형식으로 저장할 데이터 준비
            dtype_full = [(attribute, 'f4') for attribute in ['x', 'y', 'z']]  # xyz
            dtype_full += [(f'f_dc_{i}', 'f4') for i in range(3)]  # RGB features
            dtype_full += [(f'f_rest_{i}', 'f4') for i in range(features_rest.shape[1])]  # SH features (동적 크기)
            dtype_full += [('opacity', 'f4')]  # opacity
            dtype_full += [(f'scale_{i}', 'f4') for i in range(3)]  # scaling
            dtype_full += [(f'rot_{i}', 'f4') for i in range(4)]  # rotation
            
            # 피처도 함께 저장
            # if point_features is not None:
            #     dtype_full += [(f'feature_{i}', 'f4') for i in range(point_features.shape[1])]
            
            # 데이터 배열 생성
            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            
            # xyz
            elements['x'] = xyz[:, 0]
            elements['y'] = xyz[:, 1]
            elements['z'] = xyz[:, 2]
            
            # RGB features (DC component)
            for i in range(3):
                elements[f'f_dc_{i}'] = features_dc[:, i]
                
            # SH features (Rest components) - 동적 크기 처리
            for i in range(features_rest.shape[1]):
                elements[f'f_rest_{i}'] = features_rest[:, i]
                
            # Opacity
            if len(opacity.shape) > 1:
                elements['opacity'] = opacity.flatten()
            else:
                elements['opacity'] = opacity
            
            # Scaling
            for i in range(3):
                elements[f'scale_{i}'] = scaling[:, i]
                
            # Rotation
            for i in range(4):
                elements[f'rot_{i}'] = rotation[:, i]
                
            # Features (선택적)
            # if point_features is not None:
            #     for i in range(point_features.shape[1]):
            #         elements[f'feature_{i}'] = point_features[:, i]
            
            # PLY 파일로 저장
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(save_path)
            
            print(f"Successfully saved segmented gaussians to {save_path}")
            print(f"  - Points: {xyz.shape}")
            print(f"  - Features DC: {features_dc.shape}")
            print(f"  - Features Rest: {features_rest.shape}")
            # if point_features is not None:
            #     print(f"  - Point Features: {point_features.shape}")
            return True
            
        except Exception as e:
            import traceback
            print(f"Error saving PLY file: {e}")
            print("Full traceback:")
            print(traceback.format_exc())
            return False

    def save_all_segmented_objects(self, base_dir="./segmented_objects"):
        """
        현재까지 세그멘테이션된 모든 객체들을 개별 PLY 파일로 저장
        """
        if not PLY_AVAILABLE:
            print("Error: plyfile library not available. Please install with: pip install plyfile")
            return 0
            
        os.makedirs(base_dir, exist_ok=True)
        
        try:
            # 세그멘테이션 히스토리에서 모든 객체 추출
            unique_masks = torch.unique(self.engine['scene']._mask)
            saved_count = 0
            
            for mask_id in unique_masks:
                if mask_id == 0:  # 배경은 제외
                    continue
                    
                # 각 객체별로 임시 마스크 설정
                temp_segment_times = self.engine['scene'].segment_times
                self.engine['scene'].segment_times = mask_id - 1
                
                save_path = os.path.join(base_dir, f"object_{mask_id:03d}.ply")
                
                if self.save_segmented_gaussians_as_ply(save_path):
                    saved_count += 1
                    
            # 원래 상태 복구
            self.engine['scene'].segment_times = temp_segment_times
            
            print(f"Saved {saved_count} segmented objects to {base_dir}")
            return saved_count
            
        except Exception as e:
            print(f"Error saving all objects: {e}")
            return 0

    def prepare_buffer(self, outputs):
        if self.model == "images":
            return outputs["render"]
        else:
            return np.expand_dims(outputs["depth"], -1).repeat(3, -1)
    
    def grayscale_to_colormap(self, gray):
        """Convert a grayscale value to Jet colormap RGB values."""
        # Ensure the grayscale values are in the range [0, 1]
        # gray = np.clip(gray, 0, 1)

        # Jet colormap ranges (these are normalized to [0, 1])
        jet_colormap = np.array([
            [0, 0, 0.5],
            [0, 0, 1],
            [0, 0.5, 1],
            [0, 1, 1],
            [0.5, 1, 0.5],
            [1, 1, 0],
            [1, 0.5, 0],
            [1, 0, 0],
            [0.5, 0, 0]
        ])

        # Corresponding positions for the colors in the colormap
        positions = np.linspace(0, 1, jet_colormap.shape[0])

        # Interpolate the RGB values based on the grayscale value
        r = np.interp(gray, positions, jet_colormap[:, 0])
        g = np.interp(gray, positions, jet_colormap[:, 1])
        b = np.interp(gray, positions, jet_colormap[:, 2])

        return np.stack((r, g, b), axis=-1)

    def register_dpg(self):
        
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.width, self.height, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window
        with dpg.window(tag="_primary_window", width=self.window_width+300, height=self.window_height):
            dpg.add_image("_texture")   # add the texture

        dpg.set_primary_window("_primary_window", True)

        # def callback_depth(sender, app_data):
            # self.img_mode = (self.img_mode + 1) % 4
            
        # --- interactive mode switch --- #
        def clickmode_callback(sender):
            self.clickmode_button = 1 - self.clickmode_button
        def clickmode_multi_callback(sender):
            self.clickmode_multi_button = dpg.get_value(sender)
            print("clickmode_multi_button = ", self.clickmode_multi_button)
        def preview_callback(sender):
            self.preview = dpg.get_value(sender)
            # print("binary_threshold_button = ", self.binary_threshold_button)
        def clear_edit():
            self.clear_edit = True
        def roll_back():
            self.roll_back = True
        def callback_segment3d():
            self.segment3d_flag = True
        def callback_save():
            self.save_flag = True
        def callback_save_ply():
            """PLY로 저장하는 콜백"""
            self.save_ply_flag = True
        def callback_save_all_objects():
            """모든 객체를 PLY로 저장하는 콜백"""
            self.save_all_objects_flag = True
        def callback_reload():
            self.reload_flag = True
        def callback_cluster():
            self.cluster_in_3D_flag =True
        def callback_reshuffle_color():
            self.label_to_color = np.random.rand(1000, 3)
            try:
                self.cluster_point_colors = self.label_to_color[self.seg_score.argmax(dim = -1).cpu().numpy()]
                self.cluster_point_colors[self.seg_score.max(dim = -1)[0].detach().cpu().numpy() < 0.5] = (0,0,0)
            except:
                pass

        def render_mode_rgb_callback(sender):
            self.render_mode_rgb = not self.render_mode_rgb
        def render_mode_similarity_callback(sender):
            self.render_mode_similarity = not self.render_mode_similarity
        def render_mode_pca_callback(sender):
            self.render_mode_pca = not self.render_mode_pca
        def render_mode_cluster_callback(sender):
            self.render_mode_cluster = not self.render_mode_cluster
        # control window
        with dpg.window(label="Control", tag="_control_window", width=300, height=650, pos=[self.window_width+10, 0]):

            dpg.add_text("Mouse position: click anywhere to start. ", tag="pos_item")
            dpg.add_slider_float(label="Scale", default_value=0.5,
                                 min_value=0.0, max_value=1.0, tag="_Scale")
            dpg.add_slider_float(label="ScoreThres", default_value=0.0,
                                 min_value=0.0, max_value=1.0, tag="_ScoreThres")
            # dpg.add_button(label="render_option", tag="_button_depth",
                            # callback=callback_depth)
            dpg.add_text("\nRender option: ", tag="render")
            dpg.add_checkbox(label="RGB", callback=render_mode_rgb_callback, user_data="Some Data")
            dpg.add_checkbox(label="PCA", callback=render_mode_pca_callback, user_data="Some Data")
            dpg.add_checkbox(label="SIMILARITY", callback=render_mode_similarity_callback, user_data="Some Data")
            dpg.add_checkbox(label="3D CLUSTER", callback=render_mode_cluster_callback, user_data="Some Data")
            

            dpg.add_text("\nSegment option: ", tag="seg")
            dpg.add_checkbox(label="clickmode", callback=clickmode_callback, user_data="Some Data")
            dpg.add_checkbox(label="multi-clickmode", callback=clickmode_multi_callback, user_data="Some Data")
            dpg.add_checkbox(label="preview_segmentation_in_2d", callback=preview_callback, user_data="Some Data")
            
            dpg.add_text("\n")
            dpg.add_button(label="segment3d", callback=callback_segment3d, user_data="Some Data")
            dpg.add_button(label="roll_back", callback=roll_back, user_data="Some Data")
            dpg.add_button(label="clear", callback=clear_edit, user_data="Some Data")
            
            dpg.add_text("\nSave options: ", tag="save_options")
            dpg.add_button(label="save as .pt", callback=callback_save, user_data="Some Data")
            dpg.add_input_text(label="PT filename", default_value="precomputed_mask", tag="save_name")
            
            if PLY_AVAILABLE:
                dpg.add_button(label="save as PLY", callback=callback_save_ply, user_data="Some Data")
                dpg.add_input_text(label="PLY filename", default_value="segmented_object", tag="ply_save_name")
                dpg.add_button(label="save all objects PLY", callback=callback_save_all_objects, user_data="Some Data")
            else:
                dpg.add_text("PLY export disabled (install plyfile)", color=(255, 100, 100))
            
            dpg.add_text("\n")

            dpg.add_button(label="cluster3d", callback=callback_cluster, user_data="Some Data")
            dpg.add_button(label="reshuffle_cluster_color", callback=callback_reshuffle_color, user_data="Some Data")
            dpg.add_button(label="reload_data", callback=callback_reload, user_data="Some Data")

            def callback(sender, app_data, user_data):
                self.load_model = False
                file_data = app_data["selections"]
                file_names = []
                for key in file_data.keys():
                    file_names.append(key)

                self.opt.ply_file = file_data[file_names[0]]

                # if not self.load_model:
                print("loading model file...")
                self.engine.load_ply(self.opt.ply_file)
                self.do_pca()   # calculate new self.proj_mat after loading new .ply file
                print("loading model file done.")
                self.load_model = True

        if self.debug:
            with dpg.collapsing_header(label="Debug"):
                dpg.add_separator()
                dpg.add_text("Camera Pose:")
                dpg.add_text(str(self.camera.pose), tag="_log_pose")


        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            delta = app_data
            self.camera.scale(delta)
            self.update_camera = True
            if self.debug:
                dpg.set_value("_log_pose", str(self.camera.pose))
        

        def toggle_moving_left():
            self.moving = not self.moving


        def toggle_moving_middle():
            self.moving_middle = not self.moving_middle


        def move_handler(sender, pos, user):
            if self.moving and dpg.is_item_focused("_primary_window"):
                dx = self.mouse_pos[0] - pos[0]
                dy = self.mouse_pos[1] - pos[1]
                if dx != 0.0 or dy != 0.0:
                    self.camera.orbit(-dx*30, dy*30)
                    self.update_camera = True

            if self.moving_middle and dpg.is_item_focused("_primary_window"):
                dx = self.mouse_pos[0] - pos[0]
                dy = self.mouse_pos[1] - pos[1]
                if dx != 0.0 or dy != 0.0:
                    self.camera.pan(-dx*20, dy*20)
                    self.update_camera = True
            
            self.mouse_pos = pos


        def change_pos(sender, app_data):
            # if not dpg.is_item_focused("_primary_window"):
            #     return
            xy = dpg.get_mouse_pos(local=False)
            dpg.set_value("pos_item", f"Mouse position = ({xy[0]}, {xy[1]})")
            if self.clickmode_button and app_data == 1:     # in the click mode and right click
                print(xy)
                self.new_click_xy = np.array(xy)
                self.new_click = True


        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Left, callback=lambda:toggle_moving_left())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Left, callback=lambda:toggle_moving_left())
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Middle, callback=lambda:toggle_moving_middle())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Middle, callback=lambda:toggle_moving_middle())
            dpg.add_mouse_move_handler(callback=lambda s, a, u:move_handler(s, a, u))
            
            dpg.add_mouse_click_handler(callback=change_pos)
            
        dpg.create_viewport(title="Gaussian-Splatting-Viewer", width=self.window_width+320, height=self.window_height, resizable=False)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        dpg.show_viewport()


    def render(self):
        while dpg.is_dearpygui_running():
            # update texture every frame
            # TODO : fetch rgb and depth
            if self.load_model:
                cam = self.construct_camera()
                self.fetch_data(cam)
            dpg.render_dearpygui_frame()


    def construct_camera(
        self,
    ) -> Camera:
        if self.camera.rot_mode == 1:
            pose = self.camera.pose_movecenter
        elif self.camera.rot_mode == 0:
            pose = self.camera.pose_objcenter

        R = pose[:3, :3]
        t = pose[:3, 3]

        ss = math.pi / 180.0
        fovy = self.camera.fovy * ss

        fy = fov2focal(fovy, self.height)
        fovx = focal2fov(fy, self.width)

        cam = Camera(
            colmap_id=0,
            R=R,
            T=t,
            FoVx=fovx,
            FoVy=fovy,
            image=torch.zeros([3, self.height, self.width]),
            gt_alpha_mask=None,
            image_name=None,
            uid=0,
        )
        cam.feature_height, cam.feature_width = self.height, self.width
        return cam
    
    def cluster_in_3D(self):
        # try:
        #     self.engine['scene'].roll_back()
        #     self.engine['feature'].roll_back()
        # except:
        #     pass
        point_features = self.engine['feature'].get_point_features

        scale_conditioned_point_features = torch.nn.functional.normalize(point_features, dim = -1, p = 2) * self.gates.unsqueeze(0)

        normed_point_features = torch.nn.functional.normalize(scale_conditioned_point_features, dim = -1, p = 2)

        sampled_point_features = scale_conditioned_point_features[torch.rand(scale_conditioned_point_features.shape[0]) > 0.98]

        normed_sampled_point_features = sampled_point_features / torch.norm(sampled_point_features, dim = -1, keepdim = True)

        clusterer = HDBSCAN(min_cluster_size=10, cluster_selection_epsilon=0.01, allow_single_cluster = False)

        cluster_labels = clusterer.fit_predict(normed_sampled_point_features.detach().cpu().numpy())

        cluster_centers = torch.zeros(len(np.unique(cluster_labels)), normed_sampled_point_features.shape[-1])
        for i in range(0, len(np.unique(cluster_labels))):
            cluster_centers[i] = torch.nn.functional.normalize(normed_sampled_point_features[cluster_labels == i-1].mean(dim = 0), dim = -1)

        self.seg_score = torch.einsum('nc,bc->bn', cluster_centers.cpu(), normed_point_features.cpu())
        self.cluster_point_colors = self.label_to_color[self.seg_score.argmax(dim = -1).cpu().numpy()]
        # self.cluster_point_colors[self.seg_score.max(dim = -1)[0].detach().cpu().numpy() < 0.5] = (0,0,0)


    def pca(self, X, n_components=3):
        n = X.shape[0]
        mean = torch.mean(X, dim=0)
        X = X - mean
        covariance_matrix = (1 / n) * torch.matmul(X.T, X).float()  # An old torch bug: matmul float32->float16, 
        # Hermitian (대칭) 행렬이므로 eigh 사용
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

        # 절댓값 취해서 안정화 (음수 아주 작게 나올 수 있음)
        eigenvalues = torch.abs(eigenvalues)

        # 큰 고유값부터 정렬
        idx = torch.argsort(-eigenvalues)

        # 정렬 적용
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # eigenvalues = torch.norm(eigenvalues, dim=1)
        idx = torch.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        proj_mat = eigenvectors[:, 0:n_components]
        
        return proj_mat
    

    def do_pca(self):
        sems = self.engine['feature'].get_point_features.clone().squeeze()
        N, C = sems.shape
        torch.manual_seed(0)
        randint = torch.randint(0, N, [200_000])
        sems /= (torch.norm(sems, dim=1, keepdim=True) + 1e-6)
        sem_chosen = sems[randint, :]
        self.proj_mat = self.pca(sem_chosen, n_components=3)
        print("project mat initialized !")


    @torch.no_grad()
    def fetch_data(self, view_camera):
        
        scene_outputs = render(view_camera, self.engine['scene'], self.opt, self.bg_color)
        feature_outputs = render_contrastive_feature(view_camera, self.engine['feature'], self.opt, self.bg_feature)
        if self.cluster_in_3D_flag:
            self.cluster_in_3D_flag = False
            print("Clustering in 3D...")
            self.cluster_in_3D()
            print("Clustering finished.")
        self.rendered_cluster = None if self.cluster_point_colors is None else render(view_camera, self.engine['scene'], self.opt, self.bg_color, override_color=torch.from_numpy(self.cluster_point_colors).cuda().float())["render"].permute(1, 2, 0)
        # --- RGB image --- #
        img = scene_outputs["render"].permute(1, 2, 0)  #

        rgb_score = img.clone()
        depth_score = rgb_score.cpu().numpy().reshape(-1)

        # --- semantic image --- #
        sems = feature_outputs["render"].permute(1, 2, 0)
        H, W, C = sems.shape
        sems /= (torch.norm(sems, dim=-1, keepdim=True) + 1e-6)
        sem_transed = sems @ self.proj_mat
        sem_transed_rgb = torch.clip(sem_transed*0.5+0.5, 0, 1)

        scale = dpg.get_value('_Scale')
        self.gates = self.engine['scale_gate'](torch.tensor([scale]).cuda())
        scale_gated_feat = sems * self.gates.unsqueeze(0).unsqueeze(0)
        scale_gated_feat = torch.nn.functional.normalize(scale_gated_feat, dim = -1, p = 2)
        
        if self.clear_edit:
            self.new_click_xy = []
            self.clear_edit = False
            self.prompt_num = 0
            try:
                self.engine['scene'].clear_segment()
                self.engine['feature'].clear_segment()
            except:
                pass

        if self.roll_back:
            self.new_click_xy = []
            self.roll_back = False
            self.prompt_num = 0
            # try:
            self.engine['scene'].roll_back()
            self.engine['feature'].roll_back()
            # except:
                # pass
        
        if self.reload_flag:
            self.reload_flag = False
            print("loading model file...")
            self.engine['scene'].load_ply(self.opt.SCENE_PCD_PATH)
            self.engine['feature'].load_ply(self.opt.FEATURE_PCD_PATH)
            self.engine['scale_gate'].load_state_dict(torch.load(self.opt.SCALE_GATE_PATH))
            self.do_pca()   # calculate self.proj_mat
            self.load_model = True

        score_map = None
        if len(self.new_click_xy) > 0:

            featmap = scale_gated_feat.reshape(H, W, -1)
            
            if self.new_click:
                xy = self.new_click_xy
                new_feat = featmap[int(xy[1])%H, int(xy[0])%W, :].reshape(featmap.shape[-1], -1)
                if (self.prompt_num == 0) or (self.clickmode_multi_button == False):
                    self.chosen_feature = new_feat
                else:
                    self.chosen_feature = torch.cat([self.chosen_feature, new_feat], dim=-1)    # extend to get more prompt features
                self.prompt_num += 1
                self.new_click = False
            
            score_map = featmap @ self.chosen_feature
            # print(score_map.shape, score_map.min(), score_map.max(), "score_map_shape")

            score_map = (score_map + 1.0) / 2
            score_binary = score_map > dpg.get_value('_ScoreThres')
            
            score_map[~score_binary] = 0.0
            score_map = torch.max(score_map, dim=-1).values
            score_norm = (score_map - dpg.get_value('_ScoreThres')) / (1 - dpg.get_value('_ScoreThres'))

            if self.preview:
                rgb_score = img * torch.max(score_binary, dim=-1, keepdim=True).values    # option: binary
            else:
                rgb_score = img
            depth_score = 1 - torch.clip(score_norm, 0, 1)
            depth_score = depth2img(depth_score.cpu().numpy()).astype(np.float32)/255.0

            if self.segment3d_flag:
                """ gaussian point cloud core params
                self.engine._xyz            # (N, 3)
                self.engine._features_dc    # (N, 1, 3)
                self.engine._features_rest  # (N, 15, 3)
                self.engine._opacity        # (N, 1)
                self.engine._scaling        # (N, 3)
                self.engine._rotation       # (N, 4)
                self.engine._objects_dc     # (N, 1, 16)
                """
                self.segment3d_flag = False
                feat_pts = self.engine['feature'].get_point_features.squeeze()
                scale_gated_feat_pts = feat_pts * self.gates.unsqueeze(0)
                scale_gated_feat_pts = torch.nn.functional.normalize(scale_gated_feat_pts, dim = -1, p = 2)

                score_pts = scale_gated_feat_pts @ self.chosen_feature
                score_pts = (score_pts + 1.0) / 2
                self.score_pts_binary = (score_pts > dpg.get_value('_ScoreThres')).sum(1) > 0

                # save_path = "./debug_robot_{:0>3d}.ply".format(self.object_seg_id)
                # try:
                #     self.engine['scene'].roll_back()
                #     self.engine['feature'].roll_back()
                # except:
                #     pass
                self.engine['scene'].segment(self.score_pts_binary)
                self.engine['feature'].segment(self.score_pts_binary)

        if self.save_flag:
            print("Saving ...")
            self.save_flag = False
            try:
                os.makedirs("./segmentation_res", exist_ok=True)
                save_mask = self.engine['scene']._mask == self.engine['scene'].segment_times + 1
                torch.save(save_mask, f"./segmentation_res/{dpg.get_value('save_name')}.pt")
            except:
                with dpg.window(label="Tips"):
                    dpg.add_text('You should segment the 3D object before save it (click segment3d first).')

        if self.save_ply_flag:
            print("Saving segmented gaussians as PLY...")
            self.save_ply_flag = False
            seg_output_folder = args.model_path + "/segmentation_objs" 
            try:
                os.makedirs(seg_output_folder, exist_ok=True)
                ply_filename = dpg.get_value('ply_save_name')
                
                if not ply_filename.endswith('.ply'):
                    ply_filename += '.ply'
                
                import re
                ply_filename = re.sub(r'[<>:"/\\|?*]', '_', ply_filename)
                
                save_path = os.path.join(seg_output_folder, ply_filename)
                self.save_segmented_gaussians_as_ply(save_path)
                print(f"PLY file saved as: {save_path}")
            except Exception as e:
                print(f"Error: {e}")
                with dpg.window(label="Tips"):
                    dpg.add_text('You should segment the 3D object before save it (click segment3d first).')

        if self.save_all_objects_flag:
            print("Saving all segmented objects as PLY...")
            self.save_all_objects_flag = False
            self.save_all_segmented_objects()

        self.render_buffer = None
        render_num = 0
        if self.render_mode_rgb or (not self.render_mode_pca and not self.render_mode_cluster and not self.render_mode_similarity):
            self.render_buffer = rgb_score.cpu().numpy().reshape(-1)
            render_num += 1
        
        if self.render_mode_pca:
            self.render_buffer = sem_transed_rgb.cpu().numpy().reshape(-1) if self.render_buffer is None else self.render_buffer + sem_transed_rgb.cpu().numpy().reshape(-1)
            render_num += 1
        if self.render_mode_cluster:
            if self.rendered_cluster is None:
                self.render_buffer = rgb_score.cpu().numpy().reshape(-1) if self.render_buffer is None else self.render_buffer + rgb_score.cpu().numpy().reshape(-1)
            else:
                self.render_buffer = self.rendered_cluster.cpu().numpy().reshape(-1) if self.render_buffer is None else self.render_buffer + self.rendered_cluster.cpu().numpy().reshape(-1)
            
            render_num += 1
        if self.render_mode_similarity:
            if score_map is not None:
                self.render_buffer = self.grayscale_to_colormap(score_map.squeeze().cpu().numpy()).reshape(-1).astype(np.float32) if self.render_buffer is None else self.render_buffer + self.grayscale_to_colormap(score_map.squeeze().cpu().numpy()).reshape(-1).astype(np.float32)
            else:
                self.render_buffer = rgb_score.cpu().numpy().reshape(-1) if self.render_buffer is None else self.render_buffer + rgb_score.cpu().numpy().reshape(-1)

            render_num += 1
        self.render_buffer /= render_num

        dpg.set_value("_texture", self.render_buffer)


if __name__ == "__main__":
    parser = ArgumentParser(description="GUI option")

    parser.add_argument('-m', '--model_path', type=str, default="./output/figurines")
    parser.add_argument('-f', '--feature_iteration', type=int, default=10000)
    parser.add_argument('-s', '--scene_iteration', type=int, default=30000)

    args = parser.parse_args()

    opt = CONFIG()

    opt.MODEL_PATH = args.model_path
    opt.FEATURE_GAUSSIAN_ITERATION = args.feature_iteration
    opt.SCENE_GAUSSIAN_ITERATION = args.scene_iteration

    opt.SCALE_GATE_PATH = os.path.join(opt.MODEL_PATH, f'point_cloud/iteration_{str(opt.FEATURE_GAUSSIAN_ITERATION)}/scale_gate.pt')
    opt.FEATURE_PCD_PATH = os.path.join(opt.MODEL_PATH, f'point_cloud/iteration_{str(opt.FEATURE_GAUSSIAN_ITERATION)}/contrastive_feature_point_cloud.ply')
    opt.SCENE_PCD_PATH = os.path.join(opt.MODEL_PATH, f'point_cloud/iteration_{str(opt.SCENE_GAUSSIAN_ITERATION)}/scene_point_cloud.ply')

    gs_model = GaussianModel(opt.sh_degree)
    feat_gs_model = FeatureGaussianModel(opt.FEATURE_DIM)
    scale_gate = torch.nn.Sequential(
        torch.nn.Linear(1, opt.FEATURE_DIM, bias=True),
        torch.nn.Sigmoid()
    ).cuda()
    gui = GaussianSplattingGUI(opt, gs_model, feat_gs_model, scale_gate)

    gui.render()