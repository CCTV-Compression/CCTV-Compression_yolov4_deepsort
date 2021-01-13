import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
##################################################
# HBSH
# Add 'tracks_info' parameter
# tracks_info: deleted track's info
flags.DEFINE_string('tracks_info_dir', './outputs/tracks_info/tracks_info.txt', 'path to tracks info')
flags.DEFINE_string('track_img_dir', './outputs/track_img', 'path to tracked object img')
##################################################
flags.DEFINE_string('output', './outputs/demo.mp4', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.5, 'iou threshold')
flags.DEFINE_float('score', 0.5, 'score threshold')
flags.DEFINE_boolean('dont_show', True, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

######################################################
# tracker > tracks > track
######################################################

def main(_argv):
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # if len(physical_devices) > 0:
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    
    ##################################################
    # HBSH
    # Add 'tracks_info' parameter
    # initialize tracker
    # tracker = Tracker(metric, tracks_info_dir=FLAGS.tracks_info_dir)
    ##################################################
    tracker = Tracker(metric)
    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    print('model load completed')
    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
    start_time = time.time()    
    frame_id = 0
    # while video is running
    while True:
        ###############################################
        # HBSH
        isTracked = False # to store only when there is an object in motion
        ###############################################
        return_value, frame = vid.read()
        
        if return_value:
            origin_frame = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
    
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        #start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        # allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       
        
        ######################################################
        # HBSH
        # track.center = {22: (70, 424), 23: (84, 427), 25: (109, 425)}
        # {frame_id: (center_x, center_y)}
        # Call the tracker
        tracker.predict()
        tracker.update(detections, frame_id) # HBSH: Add frame_id
        ######################################################
        
        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class().upper()
            #print(str(track.track_id) +': ' +str(len(track.center)))
            if len(track.center) == 2:
                # When it first appears, it saves image
                track_img = origin_frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]                
                cv2.imwrite(FLAGS.track_img_dir + '/' + str(track.track_id) + '.jpg', track_img)
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            ######################################################
            # HBSH
            if class_name == 'PERSON':
                isTracked = True
                if frame_id in track.center:
                    track.is_moving[frame_id] = True
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*19, int(bbox[1])), color, -1)
                    cv2.line(frame, track.center[frame_id],  track.center[frame_id], color, 10) # HBSH: Draw red point to the center.
                    cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                    ######################################################
                    # HBSH: Add line of moving
                    before_center = None
                    for i, center in enumerate(reversed(list(track.center.values()))):
                        if before_center != None and i < 10:
                            cv2.line(frame, center,  before_center, color, 2)
                        before_center = center
                    ######################################################
            else:
                if frame_id in track.center:
                    moving_15 = [dist for dist in track.moving[-15:] if dist[0] > 4 and dist[1] > 3] # Save as a list that both distance variables are larger than a certain size for 30 frames
                    if len(moving_15) < 5:
                        # When NOT Moving!
                        track.is_moving[frame_id] = False
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 0), 2)                        
                        sub_img = frame.copy()[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                        black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
                        res = cv2.addWeighted(sub_img, 0.6, black_rect, 0.4, 1.0)
                        frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = res
                        
                        # Not moving time
                        total_fps = 0
                        for tmp_frame_id in reversed(list(track.is_moving.keys())):
                            if track.is_moving[tmp_frame_id] == False:
                                total_fps = total_fps+1
                            else:
                                break
                        cv2.putText(frame, str(int(total_fps/fps)) + ' sec',(int((bbox[0]+bbox[2])/2-10), int((bbox[1]+bbox[3])/2)),0, 0.75, (255,255,255),2)
                    else:
                        # When Moving
                        isTracked = True
                        track.is_moving[frame_id] = True
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*19, int(bbox[1])), color, -1)
                        cv2.line(frame, track.center[frame_id],  track.center[frame_id], (255,0,0), 10) # HBSH: Draw red point to the center.
                        cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                    
                        ######################################################
                        # HBSH: Add line of moving
                        before_center = None
                        for i, center in enumerate(reversed(list(track.center.values()))):
                            if before_center != None and i < 10:
                                cv2.line(frame, center,  before_center, color, 2)
                            before_center = center
                        ######################################################
            ######################################################
                
        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                print("Center: ", str(track.center))
                print("Moving: ", str(track.moving))
                moving_30 = [dist for dist in track.moving[-30:] if dist[0] > 7 and dist[1] > 3] # Save as a list that both distance variables are larger than a certain size for 60 frames
                if len(moving_30) < 5:
                    print('Not moving!!!!!!')
                else:
                    print('Move, Move!')
                print()

        ######################################################
        # HBSH: only exist object
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #if tracker.tracks:            
        if isTracked:
            if not FLAGS.dont_show:
                cv2.imshow("Output Video", result)            
            print('Sombody here!')
            # if output flag is set, save video file
            if FLAGS.output: 
                out.write(result)
        else:
            if not FLAGS.dont_show:
                cv2.imshow("Output Video", result)
            print('Nobody here...')
        
        ######################################################      
        '''
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output: 
            out.write(result)
        '''
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break        
        frame_id += 1
    print(frame_id)
    print(time.time() - start_time)
    print("FPS: %.2f" % (frame_id / (time.time() - start_time)))
    ######################################################
    # HBSH
    # Write all track's info
    for track in tracker.tracks:
        if os.path.isfile(FLAGS.tracks_info_dir):
            moving_time_list, is_exist = track.moving_time(fps) # If there is nothing in the moving_time_list, do not write.
            if is_exist:
                f = open(FLAGS.tracks_info_dir, 'a')
                f.write('[ID: '+str(track.track_id) + '/ Class: ' + track.class_name + ']\n')
                for i in moving_time_list:
                    f.write(i + '\n')
                f.close()
                
        else:
            moving_time_list, is_exist = track.moving_time(fps)
            if is_exist:
                f = open(FLAGS.tracks_info_dir, 'w')
                f.write('[ID: '+str(track.track_id) + '/ Class: ' + track.class_name + ']\n')
                for i in moving_time_list:
                    f.write(i + '\n')
                f.close()
    ######################################################
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
