from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse
import csv
import os
import random
import skvideo.io

def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def save_csv(csv_path, csv_content):
    with open(csv_path, 'w') as csvfile:
        wr = csv.writer(csvfile)
        for i in range(len(csv_content)):
            wr.writerow(csv_content[i])
# cap = cv2.VideoCapture(0)
detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-sth', '--scorethreshold', dest='score_thresh', type=float,
                        default=0.85, help='Score threshold for displaying bounding boxes')
    parser.add_argument('-fps', '--fps', dest='fps', type=int,
                        default=1, help='Show FPS on detection/display visualization')
    parser.add_argument('-src', '--source', dest='video_source',
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=1280, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=720, help='Height of the frames in the video stream.')
    parser.add_argument('-ds', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=4, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    parser.add_argument('-path', '--path', dest='datapath', type=str,
                        default="/home/zhc/projects/netease/FreashBird/GestureRecognition/dataset", help='record object person.')

    args = parser.parse_args()

    # new csv file
    header = ['filename', 'width', 'height',
              'class', 'xmin', 'ymin', 'xmax', 'ymax']
    csvholdertrain = []
    csvholdertrain.append(header)

    csvholdertest=[]
    csvholdertest.append(header)

    image_dir = "../../GestureRecognition/images"
    train_dir = image_dir + "/train"
    test_dir = image_dir + "/test"
    create_directory(image_dir)
    create_directory(train_dir)
    create_directory(test_dir)
    random.seed()

    valid_num = 0

    for f in os.listdir(args.datapath):
        feature=f
        for filename in os.listdir(args.datapath+"/"+f):
            cap = cv2.VideoCapture(args.datapath+"/"+f+"/"+filename)
            # cap = skvideo.io.VideoCapture(args.datapath + "/" + f + "/" + filename)
            print(args.datapath+"/"+f+"/"+filename)

            if not cap.isOpened():
                print("camera or video open failed")
                break

            # cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

            start_time = datetime.datetime.now()
            num_frames = 0
            errnum = 0
            im_width, im_height = (cap.get(3), cap.get(4))

            # max number of hands we want to detect/track
            num_hands_detect = 1
            centroidX=-1
            centroidY=-1

            while True:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                ret, image_np_back = cap.read()

                # image_np = cv2.flip(image_np, 1)
                if not ret:
                    break

                if im_width<im_height:
                    cv2.transpose(image_np_back,image_np_back)
                    # image_np_back.transpose()
                size = (1280, 720)
                image_np_back = cv2.resize(image_np_back, size, interpolation=cv2.INTER_AREA)
                im_width=1280
                im_height=720
                try:
                    image_np = cv2.cvtColor(image_np_back, cv2.COLOR_BGR2RGB)
                except:
                    print("Error converting to RGB")



                # actual detection
                boxes, scores = detector_utils.detect_objects(
                    image_np, detection_graph, sess)


                # Calculate Frames per second (FPS)
                num_frames += 1
                elapsed_time = (datetime.datetime.now() -
                                start_time).total_seconds()
                fps = num_frames / elapsed_time

                # record data

                for i in range(num_hands_detect):
                    if (scores[i] > args.score_thresh):
                        rnd=random.randint(1,10)


                        (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                                      boxes[i][0] * im_height, boxes[i][2] * im_height)
                        left -= 10
                        right += 10
                        top -= 10
                        bottom += 10
                        if (left < 0): left = 0
                        if (right >= im_width): right = im_width - 1
                        if top < 0: top = 0
                        if bottom >= im_height: bottom = im_height - 1

                        if centroidX!=-1 and centroidY!=-1 and (abs(centroidX - (left + right) / 2) > 100 or abs(centroidY - (top + bottom) / 2) > 100):
                            scores[0] = 0
                            errnum+=1
                            if(errnum>20):
                                centroidX = -1
                                centroidY = -1
                            break
                        else:
                            errnum=0
                            centroidX = (left + right) / 2
                            centroidY = (top + bottom) / 2
                            # print centroidX, centroidY
                            valid_num+=1
                            filename = str(valid_num) + ".jpg"
                            if valid_num % 5 == 0:
                                row = [filename, int(im_width), int(im_height), feature, int(left), int(top), int(right),
                                       int(bottom)]
                                if rnd>=9:#test
                                    csvholdertest.append(row)
                                    cv2.imwrite(test_dir+"/"+filename,image_np_back)
                                else:#train
                                    csvholdertrain.append(row)
                                    cv2.imwrite( train_dir+ "/" + filename,image_np_back)

                # # draw bounding boxes
                # detector_utils.draw_box_on_image(
                #     num_hands_detect, args.score_thresh, scores, boxes, im_width, im_height, image_np)
                # if (args.display > 0):
                #     # Display FPS on frame
                #     if (args.fps > 0):
                #         detector_utils.draw_fps_on_image(
                #             "FPS : " + str(int(fps)), image_np)
                #
                #     cv2.imshow('Single Threaded Detection', cv2.cvtColor(
                #         image_np, cv2.COLOR_RGB2BGR))
                #
                #     if cv2.waitKey(1) & 0xFF == ord('q'):
                #         cap.release()
                #         cv2.destroyAllWindows()
                #         break
                # else:
                #     print("frames processed: ",  num_frames,
                #           "elapsed time: ", elapsed_time, "fps: ", str(int(fps)))

    save_csv(train_dir + "/train_labels.csv", csvholdertrain)
    save_csv(test_dir + "/test_labels.csv", csvholdertest)
