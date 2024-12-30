import os
import json
import cv2
import numpy as np


def read_txt_file(file_path):
    """Read txt file in MOT16 format and return a dictionary with our dataset format"""
    data_dict = {}
    with open(file_path, "r") as txt_file:
        for line in txt_file:
            elements = line.strip().split(
                ","
            )  # strip() removes the newline character at the end of each line
            key = elements[0]
            bbox_mot16 = list(map(int, elements[2:6]))
            bbox = [
                bbox_mot16[0],
                bbox_mot16[1],
                bbox_mot16[0] + bbox_mot16[2],
                bbox_mot16[1] + bbox_mot16[3],
            ]
            value = {"id": int(elements[1]), "BboxP": bbox, "BboxF": [-1]}
            if key in data_dict:
                data_dict[key].append(value)
            else:
                data_dict[key] = [value]
    return data_dict


def read_json_file(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data


def saveJson(corrected_data, output):
    with open(output, "w") as json_file:
        json.dump(corrected_data, json_file, indent=2)


# Create video
def createVideo(video, output_video):
    w, h = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    fps = int(video.get(cv2.CAP_PROP_FPS))
    # print(w,h)
    # Below VideoWriter object will create a frame of above defined The output is stored in filename file.
    fourcc = cv2.VideoWriter_fourcc("D", "I", "V", "X")
    return cv2.VideoWriter(output_video, fourcc, fps, (w, h))


def draw_det(bbox, track, img_out, colors):
    # draw detections and labels
    color = colors[int(track) % len(colors)].tolist()

    if int(bbox[1]) < 10:
        # id rectangle bottom
        cv2.rectangle(
            img_out,
            (int(bbox[0]), int(bbox[3] + 15)),
            (int(bbox[0]) + len(str(track)) * 11, int(bbox[3])),
            color,
            -1,
        )
        # id text bottom
        cv2.putText(
            img_out,
            str(track),
            (int(bbox[0]), int(bbox[3] + 12)),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 0, 0),
            2,
        )
    else:
        # id rectangle top
        cv2.rectangle(
            img_out,
            (int(bbox[0]), int(bbox[1] - 15)),
            (int(bbox[0]) + len(str(track)) * 11, int(bbox[1])),
            color,
            -1,
        )
        # id text top
        cv2.putText(
            img_out,
            str(track),
            (int(bbox[0]), int(bbox[1] - 2)),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 0, 0),
            2,
        )
    # bbox body rectangle
    cv2.rectangle(
        img_out, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2
    )

    # det_i = {"id": track.track_id, "BboxP":[int(bbox[0]), int(bbox[1]),int(bbox[2]), int(bbox[3])] , "BboxF":[]}
    # det.append(det_i)


def saveVideo(video_input, video_output, corrected_data):
    # Read the video
    video = cv2.VideoCapture(video_input)
    vid_writer = createVideo(video, video_output)

    # Random colors for the bounding boxes
    # Define a seed for having always the same colors
    np.random.seed(0)
    colors = (np.random.rand(32, 3) * 255).astype(int)

    # Read the video frame by frame
    while True:
        ret, img = video.read()

        if not ret:
            break

        img_out = img.copy()

        frame_pos = int(video.get(cv2.CAP_PROP_POS_FRAMES))

        if corrected_data[str(frame_pos)] is not []:
            for det in corrected_data[str(frame_pos)]:
                id_label = det["id"]
                bboxP = det["BboxP"]
                if "BboxF" in det:
                    bboxF = det["BboxF"]
                else:
                    bboxF = None
                draw_det(bboxP, id_label, img_out, colors)
                if bboxF is not None:
                    if bboxF == [-1]:
                        bboxF = []
                    if bboxF != []:
                        # print("bboxF: ", bboxF)
                        draw_det(bboxF, id_label, img_out, colors)

        # frame number text in the video
        cv2.putText(
            img_out, str(frame_pos), (10, 12), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2
        )

        vid_writer.write(img_out)

    vid_writer.release()
