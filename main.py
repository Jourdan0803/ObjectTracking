import time
import numpy as np
from absl import app, logging
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from yolo_model.models import Yolo
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

class_path = 'yolo_model/label'  # categories
weight_path = './data/weight/yolov3.tf'
video_path = './data/video/vlc-record.mp4'
output_path = './data/video/result.avi'
size = 416                                  # resize image to this size
num = 80                                    # number of classes in the model


def transform_images(img, size):
    img = tf.image.resize(img, (size, size))
    img = img / 255
    return img


def convert_boxes(image, boxes):
    returned_boxes = []
    for box in boxes:
        box[0] = (box[0] * image.shape[1]).astype(int)
        box[1] = (box[1] * image.shape[0]).astype(int)
        box[2] = (box[2] * image.shape[1]).astype(int)
        box[3] = (box[3] * image.shape[0]).astype(int)
        box[2] = int(box[2] - box[0])
        box[3] = int(box[3] - box[1])
        box = box.astype(int)
        box = box.tolist()
        if box != [0, 0, 0, 0]:
            returned_boxes.append(box)
    return returned_boxes


def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0

    # initialize deep sort
    model_filename = 'data/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    # create the CNN and load the weight
    yolo = Yolo(classes=num)
    yolo.load_weights(weight_path)
    obj_names = [c.strip() for c in open(class_path).readlines()]

    out = None
    # read the video
    try:
        video = cv2.VideoCapture(int(video_path))
    except:
        video = cv2.VideoCapture(video_path)
    # set the parameters for output video, (default VideoCapture returns float)
    if output_path:
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, codec, fps, (width, height))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    frame_count = 0
    while True:
        _, img = video.read()
        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            frame_count += 1
            if frame_count < 3:
                continue
            else:
                break

        # pre-process the image
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(obj_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(converted_boxes, scores[0], names, features)]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1] - 30)),
                          (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color, -1)
            cv2.putText(img, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                        (255, 255, 255), 2)

        cv2.imshow('output', img)
        frame_index = 0
        if output_path:
            out.write(img)
            frame_index = frame_index + 1
            list_file.write(str(frame_index) + ' ')
            if len(converted_boxes) != 0:
                for i in range(0, len(converted_boxes)):
                    list_file.write(str(converted_boxes[i][0]) + ' ' + str(converted_boxes[i][1]) + ' ' + str(
                        converted_boxes[i][2]) + ' ' + str(converted_boxes[i][3]) + ' ')
            list_file.write('\n')

        # press q to quit
        if cv2.waitKey(1) == ord('q'):
            break
    video.release()
    if output_path:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
