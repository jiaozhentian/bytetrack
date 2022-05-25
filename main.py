from unittest import result
from black import right_hand_split
import cv2
import base64
from cv2 import log
import numpy as np
import json
import argparse
import requests
import time

from sqlalchemy import false
from utils.visualize import plot_tracking
from src.byte_tracker import BYTETracker
from utils import logs

def make_parser():
    parser = argparse.ArgumentParser(description='Bytes Track demo')
    parser.add_argument("-f", "--fps", type=int, default=30, required=False, help="FPS of the video")
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks, usually as same with FPS")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )

    return parser

def track_main(tracker, detection_results, frame_id, image_height, image_width, test_size):
    '''
    main function for tracking
    :param args: the input arguments, mainly about track_thresh, track_buffer, match_thresh
    :param detection_results: the detection bounds results, a list of [x1, y1, x2, y2, score]
    :param frame_id: the current frame id
    :param image_height: the height of the image
    :param image_width: the width of the image
    :param test_size: the size of the inference model
    '''
    online_targets = tracker.update(detection_results, [image_height, image_width], test_size)
    online_tlwhs = []
    online_ids = []
    online_scores = []
    results = []

    for target in online_targets:
        tlwh = target.tlwh
        tid = target.track_id
        vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
        if tlwh[2] * tlwh[3] > args.min_box_area or vertical:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(target.score)
            # save results
            results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{target.score:.2f},-1,-1,-1\n"
                    )

    return online_tlwhs, online_ids

def cv2_base64(image_path):
    img = cv2.imread(image_path)
    binary_str = cv2.imencode('.jpg', img)[1].tobytes()#编码
    base64_str = base64.b64encode(binary_str)#解码
    base64_str = base64_str.decode('utf-8')
    return base64_str
    
def inference_from_tfserving(image_path):
    Src_Img = cv2.imread(image_path)
    instance = [{"b64": cv2_base64(image_path)}]
    predict_request = json.dumps({"instances": instance})
    headers = {"Content-Type": "application/json"}
    response = requests.post("http://127.0.0.1:8501/v1/models/object_detection:predict", data=predict_request, headers=headers)
    response.raise_for_status()
    prediction = np.array(response.json()["predictions"])
    prediction = prediction[0]
    if prediction != []:
        results = []
        boxes = prediction[:, 1:5]
        boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]
        boxes[:, 2:4] += boxes[:, 0:2] # 老模型要这一操作，新模型不需要
        classes = prediction[:, 6].astype(int)
        scores = prediction[:, 5]
        # 对检测结果数据进行循环分析
        count_num = 0
        for score, classe in zip(scores, classes):
            # 如果置信度大于0.4
            if score > 0.4 and classe == 5:
                # warning_status = 1
                left = int(boxes[count_num][0])
                top = int(boxes[count_num][1])
                right = int(boxes[count_num][2])
                bottom = int(boxes[count_num][3])
                # 进行渲染
                cv2.rectangle(Src_Img, (left, top), (right, bottom), (255, 0, 0), 1)
            count_num += 1
            results.append([left, top, right, bottom, score])
        # cv2.imshow("result", Src_Img)
        # cv2.waitKey(0)
        return results

if __name__ == '__main__':
    logger = logs.Log("bytetrack test").logs_setup()
    logger.info("bytetrack is starting...")
    args = make_parser().parse_args(args=[])
    # traker have to be initialized out of the track_main function, cause the trak is occurred in BYTETracker.update()
    tracker = BYTETracker(args)
    video_path = "./data/video/test.mp4"
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    image_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    image_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    test_size = (1080, 1920)

    video_writer = cv2.VideoWriter('./data/video/result.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), video_fps, (int(image_width), int(image_height)))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        inference_start_time = 1000 * time.time()
        cv2.imwrite('./temp/temp.jpg', frame)
        inference_results = np.array(inference_from_tfserving('./temp/temp.jpg'))
        inference_end_time = 1000 * time.time()
        inference_time = inference_end_time - inference_start_time
        # bytetrack requires the input inference results are (left, top, right, bottom, score) format
        # Herer is the main function about byte track
        online_tlwhs, online_ids = track_main(tracker, inference_results, frame_id, image_height, image_width, test_size)
        # Here is plot the result
        online_im = plot_tracking(frame, online_tlwhs, online_ids, frame_id=frame_id, fps=0.0)
        track_end_time = 1000 * time.time()
        track_time = track_end_time - inference_end_time
        video_writer.write(online_im)
        frame_id += 1
        logger.debug('inference time is {} ms'.format(inference_time))
        logger.debug('track time is {} ms'.format(track_time))
        cv2.imshow("online_im", online_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
        