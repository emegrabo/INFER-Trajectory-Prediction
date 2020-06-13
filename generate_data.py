from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
import numpy as np
import copy
import PIL
import os
from skimage import color
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
import cv2
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.input_representation.utils import get_crops, get_rotation_matrix, convert_to_pixel_coords
#nusc = NuScenes(version='v1.0-mini', dataroot='data/sets/nuscenes', verbose=True)
nusc = NuScenes(version='v1.0-trainval', dataroot='full_data/sets/nuscenes', verbose=True)

import os.path as osp

palette = {}

palette['Bird'] = [165, 42, 42]
palette['Ground Animal'] = [0, 192, 0]
palette['Curb'] = [196, 196, 196]
palette['Fence'] = [190, 153, 153]
palette['Guard Rail'] = [180, 165, 180]
palette['Barrier'] = [90, 120, 150]
palette['Wall'] = [102, 102, 156]
palette['Bile Lane'] = [128, 64, 255]
palette['Crosswalk - Plain'] = [140, 140, 200]
palette['Curb Cut'] = [170, 170, 170]
palette['Parking'] = [250, 170, 160]
palette['Pedestrian Area'] = [96, 96, 96]
palette['Rail Track'] = [230, 150, 140]
palette['Road'] = [128, 64, 128]
palette['Service Lane'] = [110, 110, 110]
palette['Sidewalk'] = [244, 35, 232]
palette['Bridge'] = [150, 100, 100]
palette['Building'] = [70, 70, 70]
palette['Tunnel'] = [150, 120, 90]
palette['Person'] = [220, 20, 60]
palette['Bicyclist'] = [255, 0, 0]
palette['Motorcyclist'] = [255, 0, 100]
palette['Other Rider'] = [255, 0, 200]
palette['Lane Marking - Crosswalk'] = [200, 128, 128]
palette['Lane Marking - General'] = [255, 255, 255]
palette['Mountain'] = [64, 170, 64]
palette['Sand'] = [230, 160, 50]
palette['Sky'] = [70, 130, 180]
palette['Snow'] = [190, 255, 255]
palette['Terrain'] = [152, 251, 152]
palette['Vegetation'] = [107, 142, 35]
palette['Water'] = [0, 170, 30]
palette['Banner'] = [255, 255, 128]
palette['Bench'] = [250, 0, 30]
palette['Bike Rack'] = [100, 140, 180]
palette['Billboard'] = [220, 220, 220]
palette['Catch Basin'] = [220, 128, 128]
palette['CCTV Camera'] = [222, 40, 40]
palette['Fire Hydrant'] = [100, 170, 30]
palette['Junction Box'] = [40, 40, 40]
palette['Mailbox'] = [33, 33, 33]
palette['Manhole'] = [100, 128, 160]
palette['Phone Booth'] = [142, 0, 0]
palette['Pothole'] = [70, 100, 150]
palette['Street Light'] = [210, 170, 100]
palette['Pole'] = [153, 153, 153]
palette['Traffic Sign Frame'] = [128, 128, 128]
palette['Utility Pole'] = [0, 0, 80]
palette['Traffic Light'] = [250, 170, 30]
palette['Traffic Sign (Back)'] = [192, 192, 192]
palette['Traffic Sign (Front)'] = [220, 220, 0]
palette['Trash Can'] = [140, 140, 20]
palette['Bicycle'] = [119, 11, 32]
palette['Boat'] = [150, 0, 255]
palette['Bus'] = [0, 60, 100]
palette['Car'] = [0, 0, 142]
palette['Caravan'] = [0, 0, 90]
palette['Motorcycle'] = [0, 0, 230]
palette['On Rails'] = [0, 80, 100]
palette['Other Vehicle'] = [128, 64, 64]
palette['Trailer'] = [0, 0, 110]
palette['Truck'] = [0, 0, 70]
palette['Wheeled Slow'] = [0, 0, 192]
palette['Car Mount'] = [32, 32, 32]
palette['Ego Vehicle'] = [120, 10, 10]

for k in palette.keys():
    palette[k].append(255)
    
pixel_to_classidx = {}
class_to_idx = {}
count = 0 
for k in palette: 
    pixel_to_classidx[tuple(palette[k])] = (k, count)
    class_to_idx[k] = count
    count+=1

idx_to_class = {class_to_idx[k] : k for k in class_to_idx}

def get_seg(og_seg, plot_images=False, colortype='rgb'):
    '''
    og_seg - image from which intermediate representations will be extracted from (array)
    plot_images - plots images if True, does not plot if False
    colortype - 'binary': plots 1 or 0 (used for occupancy grid), 'grayscale': plots in grayscale for lidar mapping,
                'rgb' or any other string: plots in original palette colors
    returns ret - list of intermediate representations(road, lane, and obstacle in that order so far)
    '''
    if(plot_images):
        plt.figure()
        plt.imshow(og_seg)
    ret = []
    #road segmentation
    inter_seg_road = copy.deepcopy(og_seg)
    inter_seg_road[(og_seg != palette['Road']).any(axis=2)] = [0,0,0,255]  
    ret.append(inter_seg_road)
    
    #lane segmentation
    inter_seg_lane = copy.deepcopy(og_seg)
    crosswalk = (og_seg != palette['Lane Marking - Crosswalk']).any(axis=2)
    general = (og_seg != palette['Lane Marking - General']).any(axis=2) 
    inter_seg_lane[np.logical_and(crosswalk, general)] = [0,0,0,255] 
    ret.append(inter_seg_lane)
    
    #obstacle segmentation (did not include Curb Cut as obstacle but did include Curb)
    inter_seg_obstacle = copy.deepcopy(og_seg)
    building = (og_seg != palette['Building']).any(axis=2)
    curb = (og_seg != palette['Curb']).any(axis=2)
    vegetation = (og_seg != palette['Vegetation']).any(axis=2)
    inter_seg_obstacle[np.logical_and(np.logical_and(building,curb),vegetation)] = [0,0,0,255]
    ret.append(inter_seg_obstacle)
    
    if(colortype == 'grayscale'):
        for i in range(0, len(ret)):
            temp = color.rgb2gray(ret[i])
            ret[i] = temp
            
    elif(colortype == 'binary'):
        for i in range(0, len(ret)):
            temp = color.rgb2gray(ret[i])
            temp[temp > 0] = 1
            ret[i] = temp
    
    
    if(plot_images):
        for i in ret:
            plt.figure()
            if(colortype =='grayscale'):
                plt.imshow(i, cmap='gray')
            else:
                plt.imshow(i)
    return ret
    
    
def get_semantic_class(points, cam_img):
    num_points = points.shape[1]
    
    class_vec = [-1] * num_points
    num_points_classified = 0
    for i in range(num_points):
        current_point = np.round(points[:,i][:2])
        r = int(current_point[0])
        c = int(current_point[1])
        object_type, class_id = pixel_to_classidx[(tuple(cam_img[c,r]))]
        class_vec[i] = class_id
        num_points_classified+=1
#     print(str(num_points_classified) + '/' + str(len(class_vec)))
            
    return class_vec

def project_points_to_image(current_pc, pointsensor, cam):
    
    pc = copy.deepcopy(current_pc)
    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))
    
#     import pdb; pdb.set_trace()
    # Second step: transform to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))
    
    t2 = transform_matrix(translation=poserecord['translation'],rotation=Quaternion(poserecord['rotation']),inverse=True)
#     import pdb; pdb.set_trace()
    global_frame_pc = copy.deepcopy(pc)

    # Third step: transform into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)
    
    return pc, global_frame_pc
    
    
def mask_img_points(points, depths, im, min_dist=1.0):
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    return mask


import math

def print_stats(s_c_p):
    
    objects_found_front = {}

    for point in s_c_p[0]:
        if idx_to_class[point] not in objects_found_front:
            objects_found_front[idx_to_class[point]] = 1
        else:
             objects_found_front[idx_to_class[point]] += 1

    objects_found_front_left = {}

    for point in s_c_p[1]:
        if idx_to_class[point] not in objects_found_front_left:
            objects_found_front_left[idx_to_class[point]] = 1
        else:
             objects_found_front_left[idx_to_class[point]] += 1

    objects_found_front_right = {} 

    for point in s_c_p[2]:
        if idx_to_class[point] not in objects_found_front_right:
            objects_found_front_right[idx_to_class[point]] = 1
        else:
             objects_found_front_right[idx_to_class[point]] += 1

    objects_found_back = {} 

    for point in s_c_p[3]:
        if idx_to_class[point] not in objects_found_back:
            objects_found_back[idx_to_class[point]] = 1
        else:
             objects_found_back[idx_to_class[point]] += 1
                
    print(" Found in front camera")
    
    for class_point in objects_found_front: 
        print("{} found {} times in front camera".format(class_point, objects_found_front[class_point]))
        
    print(" Found in front left camera")
    
    for class_point in objects_found_front_left: 
        print("{} found {} times in front camera".format(class_point, objects_found_front_left[class_point]))  
    
    print(" Found in front right camera")
    
    for class_point in objects_found_front_right: 
        print("{} found {} times in front camera".format(class_point, objects_found_front_right[class_point]))  

    print(" Found in back camera")
    
    for class_point in objects_found_back: 
        print("{} found {} times in front camera".format(class_point, objects_found_back[class_point]))  
        
        
from typing import Any, Dict, List, Tuple, Callable
import numpy as np
History = Dict[str, List[Dict[str, Any]]]

def reverse_history(history: History) -> History:
    """
    Reverse history so that most distant observations are first.
    We do this because we want to draw more recent bounding boxes on top of older ones.
    :param history: result of get_past_for_sample PredictHelper method.
    :return: History with the values reversed.
    """
    return {token: anns[::-1] for token, anns in history.items()}


def add_present_time_to_history(current_time: List[Dict[str, Any]],
                                history: History) -> History:
    """
    Adds the sample annotation records from the current time to the
    history object.
    :param current_time: List of sample annotation records from the
        current time. Result of get_annotations_for_sample method of
        PredictHelper.
    :param history: Result of get_past_for_sample method of PredictHelper.
    :return: History with values from current_time appended.
    """

    for annotation in current_time:
        token = annotation['instance_token']

        if token in history:

            # We append because we've reversed the history
            history[token].append(annotation)

        else:
            history[token] = [annotation]

    return history

def pixels_to_box_corners(row_pixel: int,
                          column_pixel: int,
                          length_in_pixels: float,
                          width_in_pixels: float,
                          yaw_in_radians: float) -> np.ndarray:
    """
    Computes four corners of 2d bounding box for agent.
    The coordinates of the box are in pixels.
    :param row_pixel: Row pixel of the agent.
    :param column_pixel: Column pixel of the agent.
    :param length_in_pixels: Length of the agent.
    :param width_in_pixels: Width of the agent.
    :param yaw_in_radians: Yaw of the agent (global coordinates).
    :return: numpy array representing the four corners of the agent.
    """

    # cv2 has the convention where they flip rows and columns so it matches
    # the convention of x and y on a coordinate plane
    # Also, a positive angle is a clockwise rotation as opposed to counterclockwise
    # so that is why we negate the rotation angle
    coord_tuple = ((column_pixel, row_pixel), (length_in_pixels, width_in_pixels), -yaw_in_radians * 180 / np.pi)

    box = cv2.boxPoints(coord_tuple)

    return box


def get_track_box(annotation: Dict[str, Any],
                  center_coordinates: Tuple[float, float],
                  center_pixels: Tuple[float, float],
                  resolution: float = 0.1) -> np.ndarray:
    """
    Get four corners of bounding box for agent in pixels.
    :param annotation: The annotation record of the agent.
    :param center_coordinates: (x, y) coordinates in global frame
        of the center of the image.
    :param center_pixels: (row_index, column_index) location of the center
        of the image in pixel coordinates.
    :param resolution: Resolution pixels/meter of the image.
    """

    assert resolution > 0

    location = annotation['translation'][:2]
    yaw_in_radians = quaternion_yaw(Quaternion(annotation['rotation']))

    #print('yaw_in_radians', yaw_in_radians)
    row_pixel, column_pixel = convert_to_pixel_coords(location,
                                                      center_coordinates,
                                                      center_pixels, resolution)
    #print('row_pixel, column_pixel', row_pixel, column_pixel)
    #print('center_pixels', center_pixels)

    width = annotation['size'][0] / resolution
    length = annotation['size'][1] / resolution

    # Width and length are switched here so that we can draw them along the x-axis as
    # opposed to the y. This makes rotation easier.
    return pixels_to_box_corners(row_pixel, column_pixel, length, width, yaw_in_radians)


def draw_other_vehicle_boxes(ref_ego_pose: Dict[str, Any],
                        center_agent_pixels: Tuple[float, float],
                        vehicle_history: History,
                        base_image: np.ndarray,
                        resolution: float = 0.1) -> None:
    color = (1, 1, 1)
    ref_loc_x, ref_loc_y = ref_ego_pose['translation'][:2]

    for instance_token, annotations in vehicle_history.items():

        num_points = len(annotations)

        for i, annotation in enumerate(annotations):

            box = get_track_box(annotation, (ref_loc_x, ref_loc_y), center_agent_pixels, resolution)

            if 'object' in annotation['category_name']:
                continue
            
            cv2.fillPoly(base_image, pts=[np.int0(box)], color=color)

def get_intermediate_rep(BEV_points, mask, semantic_class_points, center_coordinates):
    corresponding_BEV_points= BEV_points[:,mask] 
    
    num_points = corresponding_BEV_points.shape[1]
    lane_rep =  np.zeros((256,256))
    road_rep = np.zeros((256,256))
    obstacle_rep = np.zeros((256,256))
    res = 0.5
    for i in range(num_points):        
        row_pixel, column_pixel = convert_to_pixel_coords(corresponding_BEV_points[:,i][:2], \
                                                          center_coordinates, \
                                                          (128, 128), res)
        if row_pixel < 0 or column_pixel < 0 or row_pixel >= 256 or column_pixel >= 256: 
            import pdb; pdb.set_trace()
        semantic_class = semantic_class_points[i]
        if semantic_class == class_to_idx['Road']:
            road_rep[row_pixel][column_pixel] += 1
            
        if semantic_class == class_to_idx['Lane Marking - General'] or semantic_class == class_to_idx['Lane Marking - Crosswalk']:
            lane_rep[row_pixel][column_pixel] += 1
            
        if semantic_class == class_to_idx['Building'] or semantic_class == class_to_idx['Curb'] or semantic_class == class_to_idx['Vegetation']:
            obstacle_rep[row_pixel][column_pixel] += 1
        
        
    
    return road_rep, lane_rep, obstacle_rep
    
    
            
            
past_seconds = 2
future_seconds = 4
helper = PredictHelper(nusc)
starting_scene = 0

for j in range(starting_scene, len(nusc.scene)//10):

    #every scene
    os.makedirs("nuScenes_project_dataset/scene_" + str(j))
    
    print(j)
    my_scene_token = nusc.scene[j]['token']
    scene_rec = nusc.get('scene', my_scene_token)
    #print(len(scene_rec))

    current_token = scene_rec['first_sample_token']
    current_seq = []
    
    sequence_count = 0
    
    for i in range(scene_rec['nbr_samples']): 
        #get current sample data
        sample_rec = nusc.get('sample', current_token)
        
        annotation_tokens = sample_rec['anns']

        #get camera tokens
        camera_token = sample_rec['data']['CAM_FRONT']
        camera_token_FRONT_LEFT = sample_rec['data']['CAM_FRONT_LEFT']
        camera_token_FRONT_RIGHT = sample_rec['data']['CAM_FRONT_RIGHT']
        camera_token_BACK = sample_rec['data']['CAM_BACK']

        pointsensor_token = sample_rec['data']['LIDAR_TOP']
        pointsensor = nusc.get('sample_data', pointsensor_token)

        #get camera info
        cam = nusc.get('sample_data', camera_token)
        cam_front_left = nusc.get('sample_data', camera_token_FRONT_LEFT)
        cam_front_right = nusc.get('sample_data', camera_token_FRONT_RIGHT)
        cam_back = nusc.get('sample_data', camera_token_BACK)

        orig_pc, times = LidarPointCloud.from_file_multisweep(nusc, sample_rec, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=10)

        #get ego vehicle pose
        pose_record = nusc.get('ego_pose', pointsensor['ego_pose_token'])

        #get point cloud in camera frame prior to putting inside image plane
        pc, global_frame_pc = project_points_to_image(orig_pc, pointsensor, cam) 
        pc_front_left, _ = project_points_to_image(orig_pc, pointsensor, cam_front_left)
        pc_front_right, _ = project_points_to_image(orig_pc, pointsensor, cam_front_right)
        pc_back, _ = project_points_to_image(orig_pc, pointsensor, cam_back)
        

        #get image representation of camera data
        im = PIL.Image.open(osp.join(nusc.dataroot, cam['filename']))
        im_fl = PIL.Image.open(osp.join(nusc.dataroot, cam_front_left['filename']))
        im_fr = PIL.Image.open(osp.join(nusc.dataroot, cam_front_right['filename']))
        im_b = PIL.Image.open(osp.join(nusc.dataroot, cam_back['filename']))
        
         # Grab the depths (camera frame z axis points away from the camera).
        depths = pc.points[2, :]
        depths_front_left = pc_front_left.points[2,:]
        depths_front_right = pc_front_right.points[2,:]
        depths_back = pc_back.points[2,:]

        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        cs_record_front_left = nusc.get('calibrated_sensor', cam_front_left['calibrated_sensor_token'])
        cs_record_front_right = nusc.get('calibrated_sensor', cam_front_right['calibrated_sensor_token'])
        cs_record_back = nusc.get('calibrated_sensor', cam_back['calibrated_sensor_token'])

        #get matrix representation of camera image
        cam_data_arr_front = plt.imread(osp.join(nusc.dataroot, cam['filename']))
        cam_data_arr_front_left = plt.imread(osp.join(nusc.dataroot, cam_front_left['filename']))
        cam_data_arr_front_right = plt.imread(osp.join(nusc.dataroot, cam_front_right['filename']))
        cam_data_arr_back = plt.imread(osp.join(nusc.dataroot, cam_back['filename']))

        #get point cloud data in the image plane across all cameras
        points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
        points_front_left = view_points(pc_front_left.points[:3, :], np.array(cs_record_front_left['camera_intrinsic']), normalize=True)
        points_front_right = view_points(pc_front_right.points[:3, :], np.array(cs_record_front_right['camera_intrinsic']), normalize=True)
        points_back = view_points(pc_back.points[:3, :], np.array(cs_record_back['camera_intrinsic']), normalize=True)

        #get points that are actually inside the image plane
        mask_front = mask_img_points(points, depths, im)
        mask_front_left = mask_img_points(points_front_left, depths_front_left, im_fl)
        mask_front_right = mask_img_points(points_front_right, depths_front_right, im_fr)
        mask_back = mask_img_points(points_back, depths_back, im_b)

        # get points inside the image
        valid_img_points_front = points[:, mask_front]
        valid_img_points_front_left = points_front_left[:, mask_front_left]
        valid_img_points_front_right = points_front_right[:, mask_front_right]
        valid_img_points_back = points_back[:, mask_back]

        
        #perform point painting and get class per associated valid lidar point
        semantic_class_points_front = get_semantic_class(valid_img_points_front, cam_data_arr_front)   

        semantic_class_points_front_left = get_semantic_class(valid_img_points_front_left, cam_data_arr_front_left)   

        semantic_class_points_front_right = get_semantic_class(valid_img_points_front_right, cam_data_arr_front_right)   
        semantic_class_points_back = get_semantic_class(valid_img_points_back, cam_data_arr_back)   

        masks = [mask_front, mask_front_left, mask_front_right, mask_back]
        semantic_class_points = [semantic_class_points_front,\
                                 semantic_class_points_front_left,\
                                 semantic_class_points_front_right,\
                                 semantic_class_points_back]
        images = [im, im_fl, im_fr, im_b]

        current_seq.append((pc, global_frame_pc, pointsensor, pose_record, masks, semantic_class_points, images, current_token))
        
        #create sequences
        if len(current_seq) >= 13:
            #every sequence
            os.makedirs("nuScenes_project_dataset/scene_" + str(j) + "/sequence_" + str(sequence_count))
            
            intermediate_reps = []
            
            current_frame = i - 2*future_seconds
            
            time_frame = current_seq[current_frame-2*past_seconds:-1]
            needed_ego_pose = current_seq[current_frame][3]

            res = 0.5
            offset = 64

            yaw = quaternion_yaw(Quaternion(needed_ego_pose['rotation']))
            rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)],[np.sin(yaw), np.cos(yaw)]]).T
            ref_ego_x, ref_ego_y = needed_ego_pose['translation'][:2]
            
            

        
            for pc_h, gpc_orig, ps_h, ep_h, m, s_c_p, ims, token in time_frame:
            
                gpc_h = copy.deepcopy(gpc_orig)
                
                #get target vehicle representation
                location = ep_h['translation'][:2]
                print(ref_ego_x == location[0] and ref_ego_y == location[1])
                
                row_pixel, column_pixel = convert_to_pixel_coords(location, \
                                                                  (ref_ego_x, ref_ego_y), \
                                                                  (128, 128), res)
                target_rep = np.zeros((256,256,3))
                width_target = 1
                length_target = 1
                yaw_in_radians = quaternion_yaw(Quaternion(ep_h['rotation']))
                box = pixels_to_box_corners(row_pixel, column_pixel, length_target, width_target, yaw_in_radians)
                cv2.fillPoly(target_rep, pts=[np.int0(box)], color=1)
                rotation_mat = get_rotation_matrix(target_rep.shape, yaw+np.pi/2)
                target_rep = cv2.warpAffine(target_rep, rotation_mat, (target_rep.shape[1], target_rep.shape[0]))
                row_crop, col_crop = get_crops(offset, offset, offset, offset, res,target_rep.shape[0])
                target_rep = target_rep[row_crop, col_crop]
                #only get location where car is at 
                target_rep[target_rep != 1] = 0
                target_rep = cv2.GaussianBlur(target_rep[:,:,0], (5,5),0)

                
                #get other vehicle representation 
                history = helper.get_past_for_sample(token,
                                      0,
                                      in_agent_frame=False,
                                      just_xy=False)
                history = reverse_history(history)
                present_time = helper.get_annotations_for_sample(token)

                history = add_present_time_to_history(present_time, history)
                vehicle_rep = np.zeros((256, 256))
                draw_other_vehicle_boxes(needed_ego_pose, (128,128),
                         history, vehicle_rep, resolution=res)
                vehicle_rep = cv2.warpAffine(vehicle_rep, rotation_mat, vehicle_rep.shape)
                row_crop, col_crop = get_crops(offset, offset, offset, offset, res,target_rep.shape[0])
                vehicle_rep = vehicle_rep[row_crop, col_crop]
                                
                min_coor_x = ref_ego_x - offset
                min_coor_y = ref_ego_y - offset
                
                max_coor_x = ref_ego_x + offset
                max_coor_y = ref_ego_y + offset
                
                #ensures we don't get lidar points that are too far away
                point_filter_1_x = gpc_h.points[0,:] > min_coor_x
                point_filter_1_y = gpc_h.points[1,:] > min_coor_y
                point_filter_1 = np.logical_and(point_filter_1_x, point_filter_1_y)

                point_filter_2_x = gpc_h.points[0,:] < max_coor_x
                point_filter_2_y = gpc_h.points[1,:] < max_coor_y
                point_filter_2 = np.logical_and(point_filter_2_x, point_filter_2_y)

                point_filter = np.logical_and(point_filter_1, point_filter_2)
                
                m_f, m_f_l, m_f_r, m_b = m[0], m[1], m[2], m[3]
                s_c_p_f, s_c_p_f_l, s_c_p_f_r, s_c_p_b = s_c_p[0], s_c_p[1], s_c_p[2], s_c_p[3]


                all_colored = np.array([-1] * gpc_h.points.shape[1])
                all_colored[m_f] = s_c_p[0]
                all_colored[m_f_l] =  s_c_p[1]
                all_colored[m_f_r] = s_c_p[2]
                all_colored[m_b] = s_c_p[3]

                m_f_l, m_f_r = np.logical_and(point_filter, m_f_l),np.logical_and(point_filter, m_f_r)
                m_f, m_b = np.logical_and(point_filter, m_f),np.logical_and(point_filter, m_b)

                s_c_p_f = all_colored[m_f]
                s_c_p_f_l = all_colored[m_f_l]
                s_c_p_f_r = all_colored[m_f_r]
                s_c_p_b = all_colored[m_b]
                
                road_rep_fl, lane_rep_fl, obstacle_rep_fl = get_intermediate_rep(gpc_h.points, m_f_l, s_c_p_f_l, (ref_ego_x, ref_ego_y))
                road_rep_fr, lane_rep_fr, obstacle_rep_fr = get_intermediate_rep(gpc_h.points, m_f_r, s_c_p_f_r, (ref_ego_x, ref_ego_y))
                road_rep_f, lane_rep_f, obstacle_rep_f = get_intermediate_rep(gpc_h.points, m_f, s_c_p_f, (ref_ego_x, ref_ego_y))
                road_rep_b, lane_rep_b, obstacle_rep_b = get_intermediate_rep(gpc_h.points, m_b, s_c_p_b, (ref_ego_x, ref_ego_y))

                road_rep = road_rep_fl + road_rep_fr + road_rep_f + road_rep_b
                lane_rep = lane_rep_fl + lane_rep_fr + lane_rep_f + lane_rep_b

                obstacle_rep = obstacle_rep_fl + obstacle_rep_fr + obstacle_rep_f + obstacle_rep_b
                
                #rotate image and perform the dilation for the road, lane, and obstacle representations
                road_rep = cv2.warpAffine(road_rep, rotation_mat, road_rep.shape)
                kernel = np.ones((5,5), np.uint8)
                road_rep = cv2.dilate(road_rep,kernel,iterations = 1)
                
                lane_rep = cv2.warpAffine(lane_rep, rotation_mat, lane_rep.shape)
                kernel = np.ones((2,2), np.uint8)
                lane_rep = cv2.dilate(lane_rep,kernel,iterations = 1)
                
                obstacle_rep = cv2.warpAffine(obstacle_rep, rotation_mat, obstacle_rep.shape)
                kernel = np.ones((5,5), np.uint8)
                obstacle_rep = cv2.dilate(obstacle_rep,kernel,iterations = 1)                
                
                intermediate_reps.append([target_rep, lane_rep, obstacle_rep, road_rep, vehicle_rep]) 

            frame_count = 0
            for k,data in enumerate(intermediate_reps):
                if k < len(intermediate_reps) - 1:
                    data.append(intermediate_reps[k+1][4])
                else:
                    ego_pose_current = copy.deepcopy(current_seq[-1][3])
                    #generate target rep
                    target_rep = np.zeros((256,256,3))
                    width_target = 1
                    length_target = 1
                    yaw_in_radians = quaternion_yaw(Quaternion(ego_pose_current['rotation']))
                    box = pixels_to_box_corners(row_pixel, column_pixel, length_target, width_target, yaw_in_radians)
                    cv2.fillPoly(target_rep, pts=[np.int0(box)], color=1)
                    rotation_mat = get_rotation_matrix(target_rep.shape, yaw+np.pi/2)
                    target_rep = cv2.warpAffine(target_rep, rotation_mat, (target_rep.shape[1], target_rep.shape[0]))
                    row_crop, col_crop = get_crops(offset, offset, offset, offset, res,target_rep.shape[0])
                    target_rep = target_rep[row_crop, col_crop]
                    #only get location where car is at 
                    target_rep[target_rep != 1] = 0
                    target_rep = cv2.GaussianBlur(target_rep[:,:,0], (5,5),0)
                    
                    data.append(target_rep)
                #every frame
                np.save("nuScenes_project_dataset/scene_" + str(j) + "/sequence_" + str(sequence_count) + "/frame_" + str(frame_count), np.array(data))
                print("Scene: " + str(j) + ", Sequence: " + str(sequence_count) + ", Frame: " + str(frame_count))
                frame_count += 1
            
            sequence_count += 1
            
        
        current_token = sample_rec['next']
            



