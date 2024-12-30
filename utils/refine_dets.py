
import os
import json
import cv2
import numpy as np
from utils.savings import *
from copy import deepcopy
import ipdb

    
def is_bbox_inside(detect, detection):
    x1, y1, x2, y2 = detect[0], detect[1], detect[2], detect[3]
    x3, y3, x4, y4 = detection[0], detection[1], detection[2], detection[3]
    
    if x1 <= x3 < x4 <= x2 and y1 <= y3 < y4 <= y2:
        return True
    return False

# check if a bounding box is inside another bounding box
def check_detections(bboxes_dets):
    bboxes_to_remove = []
    for i in range(len(bboxes_dets)):
        for j in range(i+1, len(bboxes_dets)):
            if is_bbox_inside(bboxes_dets[i], bboxes_dets[j]):
                bboxes_to_remove.append(bboxes_dets[j])
                # frame_dets.remove(bboxes_dets[j])
                break
            elif is_bbox_inside(bboxes_dets[j], bboxes_dets[i]):
                bboxes_to_remove.append(bboxes_dets[i])
                # frame_dets.remove(bboxes_dets[i])
                break
    return bboxes_to_remove
    
def calculate_box_ious(bboxes1, bboxes2, box_format='x0y0x1y1', do_ioa=False):
        """ Calculates the IOU (intersection over union) between two arrays of boxes.
        Allows variable box formats ('xywh' and 'x0y0x1y1').
        If do_ioa (intersection over area) , then calculates the intersection over the area of boxes1 - this is commonly
        used to determine if detections are within crowd ignore region.
        """
        if box_format in 'xywh':
            # layout: (x0, y0, w, h)
            bboxes1 = deepcopy(bboxes1)
            bboxes2 = deepcopy(bboxes2)

            bboxes1[:, 2] = bboxes1[:, 0] + bboxes1[:, 2]
            bboxes1[:, 3] = bboxes1[:, 1] + bboxes1[:, 3]
            bboxes2[:, 2] = bboxes2[:, 0] + bboxes2[:, 2]
            bboxes2[:, 3] = bboxes2[:, 1] + bboxes2[:, 3]
        elif box_format not in 'x0y0x1y1':
            # raise (TrackEvalException('box_format %s is not implemented' % box_format))
            print('box_format %s is not implemented' % box_format)

        # layout: (x0, y0, x1, y1)
        min_ = np.minimum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
        max_ = np.maximum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
        intersection = np.maximum(min_[..., 2] - max_[..., 0], 0) * np.maximum(min_[..., 3] - max_[..., 1], 0)
        # Calculate the area of first boxes
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])

        if do_ioa:
            ioas = np.zeros_like(intersection)
            valid_mask = area1 > 0 + np.finfo('float').eps
            ioas[valid_mask, :] = intersection[valid_mask, :] / area1[valid_mask][:, np.newaxis]
            return ioas
        else:
            # Calculate the area of second boxes
            area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
            union = area1[:, np.newaxis] + area2[np.newaxis, :] - intersection
            intersection[area1 <= 0 + np.finfo('float').eps, :] = 0
            intersection[:, area2 <= 0 + np.finfo('float').eps] = 0
            intersection[union <= 0 + np.finfo('float').eps] = 0
            union[union <= 0 + np.finfo('float').eps] = 1
            ious = intersection / union
            return ious


def select_data(data1, data2, confidence=0.5):
    """Select values from data2 to add to data1 if the iou bwtween data 1 and dataa 2 values 
    is smaller than confidence.
    """
    corrected_data = {}    
    best_id = []
    dets = []
    
    
    for frame, frame_dets in data1.items():
        # There is no detections in both data dictionaries
        if frame_dets == [] and data2[frame] == []:
            corrected_data[int(frame)] = []
            continue
        
        # There is no detections in data1 but there are detections in data2
        if frame_dets == [] and data2[frame] != []:
            corrected_data[int(frame)] = data2[frame]
            
        # There is no detections in data2 but there are detections in data1
        elif frame_dets != [] and data2[frame] == []:
            corrected_data[int(frame)] = frame_dets
        
        # There are detections in both data dictionaries
        else:
            dets = frame_dets
            for det2 in data2[frame]:  
                best_iou = 0.0           
                for det1 in frame_dets:
                    iou = calculate_box_ious(np.array(det2['BboxP']).reshape(1, 4), np.array(det1['BboxP']).reshape(1, 4))[0][0]
                    # print("iou: ", iou)
                    #Check overlap condition
                    if (iou > confidence and iou > best_iou):
                        best_iou = iou
                        # best_id = det1['id']
                
                # print("best_iou: ", best_iou)
                if best_iou != 0.0:
                    continue
                else:
                    dets.append(det2)                    
                    
            corrected_data[int(frame)] = dets
            
    return corrected_data
    
def calc_iou(det1, dets, confidence=0.5):
    best_iou = 0.0     
    best_det = {}                    
    for det2 in dets:
        iou = calculate_box_ious(np.array(det1['BboxP']).reshape(1, 4), np.array(det2['BboxP']).reshape(1, 4))[0][0]
        # print("iou: ", iou)
        #Check overlap condition
        if (iou > confidence and iou > best_iou):
            best_iou = iou
            best_det = det2    
                
    return best_iou, best_det

def correct_id_between_frames(data, confidence=0.5, post_processing=False, max_age=8):
    """Check iou between detections in consecutive frames and asign the previos id to 
    the current matched frame detection if it exists.
    """
    prev_dets = []
    # act_det = []
    corrected_data = {}

    for frame, frame_dets in data.items():
        act_dets = frame_dets    
            
        if (int(frame) == 1 and prev_dets == []):
            # ipdb.set_trace()
            corrected_data[int(frame)] = act_dets
            prev_dets = act_dets
            # print("Frame: ", frame)
            # print(act_dets)
            # print(corrected_data[int(frame)])
            continue       
                    
        if prev_dets != [] and act_dets != []:
            #check iou between detections in consecutive frames
            #check if there is a detection in the current frame that was already
            #detected in the previous frame and asign the id of the previous detection
            for p, p_det in enumerate(prev_dets):   
                best_iou, best_det = calc_iou(p_det, act_dets, confidence)
                
                # print("best_iou: ", best_iou)
                if best_iou != 0.0:
                    if not post_processing:
                        frame_dets[frame_dets.index(best_det)]['id'] = p_det['id']  
                    else:
                        if p_det['id'] <= 0:
                            fut_iou = False
                            # Check if there is a future id with non 0 or negative value within a max_age of frames
                            for fut_frame in range(int(frame)+1, int(frame)+max_age):
                                best_fut_det = {}
                                # future_dets = data.get(str(fut_frame))
                                # print ("fut_frame: ", fut_frame)
                                # print("data: ", data[fut_frame])
                                future_dets = data[fut_frame]
                                if future_dets is None or future_dets == []:
                                    continue
                                
                                best_fut_iou, best_fut_det = calc_iou(p_det, future_dets, confidence)
                                
                                if best_fut_iou != 0.0 and best_fut_det['id'] > 0:
                                    frame_dets[frame_dets.index(best_det)]['id'] = best_fut_det['id'] 
                                    data[int(frame)-1][p]['id'] = best_fut_det['id'] 
                                    fut_iou = True
                                    break
                                
                            # Change yolo dets id in actual and previous frame for a previous id deepsort if it 
                            # exist in a max_age frame range, in case that there is no future id with non 0 or 
                            # negative value:
                            # si hay id <= 0 y no tiene coincidencia mediante iou con otro id > 0, 
                            # en el frame actual ni en los futuros (dentro de un max_age), mirar si 
                            # tiene coincidencia con un id > 0 en el pasado (dentro de un max_age)                            
                            if not fut_iou and (int(frame)-max_age) > 0:
                                print(frame, fut_iou)
                                for prev_frame in range(int(frame)-max_age, int(frame)-1):
                                    best_prev_det = {}                                    
                                    previous_dets = data[prev_frame]
                                    
                                    print ("prev_frame: ", prev_frame)
                                    print("data: ", previous_dets)
                                    
                                    if previous_dets is None or previous_dets == []:
                                        continue
                                    
                                    best_prev_iou, best_prev_det = calc_iou(p_det, previous_dets, confidence)
                                
                                    if best_prev_iou != 0.0 and best_prev_det['id'] > 0:
                                        frame_dets[frame_dets.index(best_det)]['id'] = best_prev_det['id'] 
                                        data[int(frame)-1][p]['id'] = best_prev_det['id'] 
                                        break
                                                         
                        else: 
                            if p_det['id'] == best_det['id']: continue
                            frame_dets[frame_dets.index(best_det)]['id'] = p_det['id']                      
                            
        corrected_data[int(frame)] = act_dets
        prev_dets = act_dets
        
    return corrected_data
    

def delete_bbox_inside(data):
    corrected_data = {}
    for frame, frame_dets in data.items():
        # act_dets = frame_dets
             
        if not(len(frame_dets) == 0):   
            Bboxes = [det['BboxP'] for det in frame_dets if 'BboxP' in det]
            # print("Bboxes: ", Bboxes)
            bboxes_to_remove = check_detections(Bboxes)
            
            if not(len(bboxes_to_remove) == 0):
                print("bboxes_to_remove: ", bboxes_to_remove)                   
                # Remove the bboxes_to_remove from the frame_dets
                for det in frame_dets:
                    if det['BboxP'] in bboxes_to_remove:
                        print("frame_dets: ", frame_dets)
                        frame_dets.remove(det)  
                        print("frame_dets: ", frame_dets)   
        
        corrected_data[int(frame)] = frame_dets   
        
    return corrected_data


def correct_flickering(data, max_age=10, from_frame=None, to_frame=None):
    """
    Correct flickering detections among a maximum of max_age frames.
    """
    corrected_data = deepcopy(data)
    # Select the frames in which to correct the flickering detections
    # If initial and final frames are not provided, correct flickering  
    # detections in all frames
    data_to_process = data if from_frame is None or to_frame is None else {
        k: v for k, v in data.items() if int(k) >= from_frame and int(k) <= to_frame}

    prev_dets = []
    init_frame = True

    for frame, frame_dets in data_to_process.items():
        act_dets = frame_dets    
        
        if init_frame:
            corrected_data[frame] = act_dets
            prev_dets = act_dets
            init_frame = False
            continue       
        
        if prev_dets:
            id_values = [det['id'] for det in frame_dets if 'id' in det]
            for n, p_det in enumerate(prev_dets): 
                # Check if the id of the previous detection is in the current frame
                # If it is, continue with the next frame detection
                # If it is not, check if the id of the previous detection is in the future frames  
                if p_det["id"] not in id_values:
                    # Check if the id of the previous detection is in the future frames within the max_age
                    for fut_frame in range(int(frame)+1, int(frame)+max_age):
                        # Check if the future frame exists in the data dictionary
                        if str(fut_frame) not in data:
                            break                                                                 
                        future_dets = data[str(fut_frame)]
                        id_fut_values = [fut_det['id'] for fut_det in future_dets if 'id' in fut_det]
                        if p_det["id"] in id_fut_values:
                            act_dets.append(p_det)
                            break
                            
        corrected_data[frame] = act_dets
        prev_dets = act_dets

    return corrected_data


# def correct_flickering_barrido(data, max_age=10):
#     """
#     # Correct flickering detections among a maximum of max_age frames.
#     """
#     prev_dets = []
#     # act_det = []
#     corrected_data = {}

#     for frame, frame_dets in data.items():
#         act_dets = frame_dets    
            
#         if (int(frame) == 1 and prev_dets == []) or int(frame) + max_age >= len(data):
#             # ipdb.set_trace()
#             corrected_data[int(frame)] = act_dets
#             prev_dets = act_dets
#             # print("Frame: ", frame)
#             # print(act_dets)
#             # print(corrected_data[int(frame)])
#             continue       
        
#         if not (len(prev_dets) == 0):
#             for n, p_det in enumerate(prev_dets):     
#                 if not(len(act_dets) == 0):                       
#                     id_values = [det['id'] for det in frame_dets if 'id' in det] 
#                     # Check if the id of the previous detection is in the current frame
#                     # If it is, continue with the next frame detection
#                     # If it is not, check if the id of the previous detection is in the future frames 
#                     # print(frame, p_det, id_values)                  
#                     if p_det["id"] in id_values:
#                         continue                     
                
#                 # Check if the id of the previous detection is in the future frames within the max_age
#                 for fut_frame in range(int(frame)+1, int(frame)+max_age):
#                     # future_dets = data.get(str(fut_frame))
#                     # print ("fut_frame: ", fut_frame)
#                     # print("data: ", data[fut_frame])
#                     future_dets = data[fut_frame]
#                     if future_dets is None:
#                         continue
#                     id_fut_values = [fut_det['id'] for fut_det in future_dets if 'id' in fut_det]
#                     if p_det["id"] in id_fut_values:
#                         bbox_x0, bbox_y0 = p_det["BboxP"][0], p_det["BboxP"][1] 
#                         bbox_x1, bbox_y1 = p_det["BboxP"][2], p_det["BboxP"][3] 
                        
#                         dx0 = future_dets[id_fut_values.index(p_det["id"])]["BboxP"][0] - p_det["BboxP"][0]
#                         dy0 = future_dets[id_fut_values.index(p_det["id"])]["BboxP"][1] - p_det["BboxP"][1]
#                         dx1 = future_dets[id_fut_values.index(p_det["id"])]["BboxP"][2] - p_det["BboxP"][2]
#                         dy1 = future_dets[id_fut_values.index(p_det["id"])]["BboxP"][3] - p_det["BboxP"][3]
                        
#                         for n in range(int(frame), fut_frame):
#                             dist = fut_frame - int(frame)
#                             bbox_x0 += dx0/dist
#                             bbox_y0 += dy0/dist
#                             bbox_x1 += dx1/dist
#                             bbox_y1 += dy1/dist                                
#                             bbox = list(map(round,[bbox_x0, bbox_y0, bbox_x1, bbox_y1]))             
                        
#                             new_det = {"id": p_det["id"], "BboxP": bbox, "BboxF": []}                        
#                             data[n].append(new_det)
#                         break 
                            
#         corrected_data[int(frame)] = act_dets
#         prev_dets = act_dets

#     return corrected_data


def correct_flickering_barrido(data, max_age=10, from_frame=None, to_frame=None):
    """
    Correct flickering detections among a maximum of max_age frames.
    Adjusts bounding boxes based on detection continuity over future frames.
    """
    corrected_data = deepcopy(data)
    # Select the frames in which to correct the flickering detections
    # If initial and final frames are not provided, correct flickering  
    # detections in all frames
    data_to_process = data if from_frame is None or to_frame is None else {
        k: v for k, v in data.items() if int(k) >= from_frame and int(k) <= to_frame}

    prev_dets = []
    init_frame = True

    for frame, frame_dets in data_to_process.items():
        act_dets = frame_dets    
        
        if init_frame:
            corrected_data[frame] = act_dets
            prev_dets = act_dets
            init_frame = False
            continue       
        
        if prev_dets:
            id_values = [det['id'] for det in frame_dets if 'id' in det]
            for n, p_det in enumerate(prev_dets): 
                # Check if the id of the previous detection is in the current frame
                # If it is, continue with the next frame detection
                # If it is not, check if the id of the previous detection is in the future frames  
                if p_det["id"] not in id_values:
                    # Check if the id of the previous detection is in the future frames within the max_age
                    for fut_frame in range(int(frame)+1, int(frame)+max_age+1):
                        # Check if the future frame exists in the data dictionary
                        if str(fut_frame) not in data:
                            break                                                                 
                        future_dets = data[str(fut_frame)]
                        id_fut_values = [fut_det['id'] for fut_det in future_dets if 'id' in fut_det]
                        if p_det["id"] in id_fut_values:
                            index = id_fut_values.index(p_det["id"])
                            bbox_x0, bbox_y0 = p_det["BboxP"][0], p_det["BboxP"][1] 
                            bbox_x1, bbox_y1 = p_det["BboxP"][2], p_det["BboxP"][3] 
                            
                            dx0 = future_dets[index]["BboxP"][0] - bbox_x0
                            dy0 = future_dets[index]["BboxP"][1] - bbox_y0
                            dx1 = future_dets[index]["BboxP"][2] - bbox_x1
                            dy1 = future_dets[index]["BboxP"][3] - bbox_y1
                            
                            for n in range(int(frame), fut_frame):
                                dist = fut_frame - int(frame)
                                bbox_x0 += dx0 / dist
                                bbox_y0 += dy0 / dist
                                bbox_x1 += dx1 / dist
                                bbox_y1 += dy1 / dist                                
                                bbox = list(map(round,[bbox_x0, bbox_y0, bbox_x1, bbox_y1]))             
                            
                                new_det = {"id": p_det["id"], "BboxP": bbox, "BboxF": []}                        
                                data[str(n)].append(new_det)
                            break
                            
        corrected_data[frame] = act_dets
        prev_dets = act_dets

    return corrected_data


def correct_det(data, max_age):
    """
    Correct detections in the data dictionary.
    """
    # Delete bbox inside another bbox
    print("Deleting bbox inside another bbox")
    corrected_bbox_inside = delete_bbox_inside(data)
    
    # Correct flickering detections
    print("Correcting flickering detections")
    corrected_flickering = correct_flickering(corrected_bbox_inside, max_age)    
    # corrected_flickering = correct_flickering(data, max_age)


    return corrected_flickering







