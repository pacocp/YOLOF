import numpy as np
import argparse
import cv2 as cv
import time
from yolo_utils import infer_image, fill_image, get_patch
from imutils.video import WebcamVideoStream

FLAGS = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model-path',
                        type=str,
                        default='./yolov3-coco/',
                        help='The directory where the model weights and \
			  configuration files are.')

    parser.add_argument('-w', '--weights',
                        type=str,
                        default='./yolov3-coco/yolo-lite.weights',
                        help='Path to the file which contains the weights \
			 	for YOLOv3.')
    
    parser.add_argument('-v', '--video-path',
		                type=str,
		                help='The path to the video file')
    
    parser.add_argument('-vo', '--video-output-path',
		                type=str,
                        default='./output.avi',
                        help='The path of the output video file')

    parser.add_argument('-cfg', '--config',
                        type=str,
                        default='./yolov3-coco/yolo-lite.cfg',
                        help='Path to the configuration file for the YOLOv3 model.')

    parser.add_argument('-l', '--labels',
                        type=str,
                        default='./yolov3-coco/coco-labels',
                        help='Path to the file having the \
					labels in a new-line seperated way.')

    parser.add_argument('-c', '--confidence',
                        type=float,
                        default=0.5,
                        help='The model will reject boundaries which has a \
				probabiity less than the confidence value. \
				default: 0.5')

    parser.add_argument('-th', '--threshold',
                        type=float,
                        default=0.2,
                        help='The threshold to use when applying the \
				Non-Max Suppresion')

    parser.add_argument('-t', '--show-time',
                        type=bool,
                        default=False,
                        help='Show the time taken to infer each image.')

    parser.add_argument('-ver', '--verbose',
                        type=bool,
                        default=False,
                        help='If you want timing information.')

    FLAGS, unparsed = parser.parse_known_args()

    # Get the labels
    labels = open(FLAGS.labels).read().strip().split('\n')

    # Intializing colors to represent each label uniquely
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Load the weights and configutation to form the pretrained YOLOv3 model
    net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

    # Get the output layer names of the model
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1]
                   for i in net.getUnconnectedOutLayers()]

    # Infer real-time on webcam
    count = 0

    if(FLAGS.video_path):
        # Read the video
        try:
            vid = cv.VideoCapture(FLAGS.video_path)
            height, width = None, None
            writer = None
        except:
            raise 'Video cannot be loaded!\n\
                            Please check the path provided!'

        finally:
            grabbed, frame_prev = vid.read()
            while True:
                grabbed, frame = vid.read()

                # Checking if the complete video is read
                if not grabbed:
                    break

                if width is None or height is None:
                    height, width = frame.shape[:2]

                matrix = np.zeros_like(frame, dtype=np.uint8)
                
                start = time.time()
                frame, boxes, confidences, classids, idxs = infer_image(net, 
                                                                        layer_names,
                                                                        height, width, 
                                                                        frame, 
                                                                        colors, 
                                                                        labels, FLAGS)
                end = time.time()
                if(FLAGS.verbose):
                    print("[INFO] Inference time: {}".format(end-start))
                if(boxes != []):
                    patch_prev = get_patch(frame_prev, boxes, idxs)
                    patch = get_patch(frame, boxes, idxs)
                    if(patch_prev != [] and patch != []):
                        for prev, actual in zip(patch_prev, patch):
                            try:
                                start = time.time()
                                next_ = cv.cvtColor(actual, cv.COLOR_BGR2GRAY)
                                prvs = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
                                flow = cv.calcOpticalFlowFarneback(prvs, next_, None, 0.5, 2, 15, 3, 5, 1.2, 0)
                                hsv = np.zeros_like(actual)
                                hsv[..., 1] = 255
                                mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
                                hsv[..., 0] = ang*180/np.pi/2
                                hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
                                matrix_patch = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
                                matrix = fill_image(matrix, matrix_patch, boxes, idxs)
                                end = time.time()
                                if(FLAGS.verbose):
                                    print("[INFO] Timing computing OF: {}".format(end-start))
                                # change previous frames value
                            except:
                                if(FLAGS.verbose):
                                    print("[INFO] not able to perform")
                    frame_prev = frame
                if writer is None:
                    # Initialize the video writer
                    fourcc = cv.VideoWriter_fourcc(*"MJPG")
                    writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30, 
                                    (matrix.shape[1], matrix.shape[0]), True)
                writer.write(matrix)
            print ("[INFO] Cleaning up...")
            writer.release()
    else:
        # vid = cv.VideoCapture(0)¡
        stream = WebcamVideoStream(src=0).start()  # default camera
        frame_prev = stream.read()
        frame_prev = cv.resize(frame_prev, None, fx=0.5, fy=0.5, interpolation = cv.INTER_LINEAR)
        previous_of = None
        while True:
            # _, frame = vid.read()
            frame = stream.read()
            frame = cv.resize(frame, None, fx=0.5, fy=0.5, interpolation = cv.INTER_LINEAR)
            height, width = frame.shape[:2]
            matrix = np.zeros_like(frame, dtype=np.uint8)
            if count == 0:
                start = time.time()
                frame, boxes, confidences, classids, idxs = infer_image(net, layer_names,
                                                                        height, width, frame, colors, labels, FLAGS)
                end = time.time()
                count += 1
                if(FLAGS.verbose):
                    print("[INFO] Inference time: {}".format(end-start))
                if(boxes != []):
                    patch_prev = get_patch(frame_prev, boxes, idxs)
                    patch = get_patch(frame, boxes, idxs)
                    if(patch_prev != [] and patch != []):
                        for prev, actual in zip(patch_prev, patch):
                            try:
                                start = time.time()
                                next_ = cv.cvtColor(actual, cv.COLOR_BGR2GRAY)
                                prvs = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
                                flow = cv.calcOpticalFlowFarneback(prvs, next_, None, 0.5, 2, 15, 3, 5, 1.2, 0)
                                hsv = np.zeros_like(actual)
                                hsv[..., 1] = 255
                                mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
                                hsv[..., 0] = ang*180/np.pi/2
                                hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
                                matrix_patch = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
                                matrix = fill_image(matrix, matrix_patch, boxes, idxs)
                                previous_of = matrix
                                end = time.time()
                                if(FLAGS.verbose):
                                    print("[INFO] Timing computing OF: {}".format(end-start))
                                # change previous frames value
                                frame_prev = frame
                            except:
                                if(FLAGS.verbose):
                                    print("[INFO] not able to perform")
            else:
                start = time.time()
                frame, boxes, confidences, classids, idxs = infer_image(net, layer_names,
                                                                        height, width, frame, colors, labels, FLAGS, boxes, confidences, classids, idxs, infer=False)
                count = (count + 1) % 3
                end = time.time()
                if(FLAGS.verbose):
                    print("[INFO] Inference time: {}".format(end-start))
                if(boxes != [] and previous_of.any()):
                    matrix = previous_of

            # show matrix
            matrix = cv.resize(matrix, None, fx=2, fy=2, interpolation = cv.INTER_LINEAR)
            cv.imshow('webcam', matrix)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        # vid.release()
        cv.destroyAllWindows()
