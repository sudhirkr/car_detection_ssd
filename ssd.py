import numpy as np
import argparse
import time
import cv2
import yaml
import logging

# config file path
CONFIG_FILE="config.yml"

# JSON extract of configs
config = None

# VOC0712 classes. SSD model is trained on VOC
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
           "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", 
           "motorbike", "person", "pottedplant", "sheep", "sofa", "train", 
           "tvmonitor"]

"""[summary]
    Configurations are in yaml file. Load the config file.
    Returns:
        [JSON] -- []
"""
def open_yaml():
    global CONFIG_FILE 
    with open(CONFIG_FILE, "r") as data:
        try:
            config = yaml.load(data, Loader=yaml.FullLoader)
            return(config)
        except yaml.YAMLError as exc:
            print(exc)
            os.exit(1)
        except Exception as e:
            print(e)
            os.exit(1)


def ssd_detector(args):
     
    # Colour of the bounding box
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    logging.info("loading model…")
     
    '''
    load the model
    '''
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
     
    image = cv2.imread(args["image"])
    (h, w) = image.shape[:2]
     
    '''
    load the input image and construct an input blob (it is collection of 
    single binary data stored in some database system) for the image and then 
    resize it to a fixed 300*300 pixels and after that, normalize the images 
    (note: normalization is done via the authors of MobileNet SSD implementation)
    '''
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, 
                                  (300, 300), 127.5)
     
    logging.info("computing object detections…")
    start = time.time()
     
    '''
    pass the blob through the network and compute the forward pass to detect 
    the objects and predictions
    '''
    net.setInput(blob)
    detections = net.forward()
     
    end = time.time() - start
    logging.info("SSD took: {:.6f}".format(end))
         
    '''
    loop over all the detection and extract the confidence score for each 
    detection. Filter out all the weak detections whose probability is less 
    than 20%. Print the detected object and their confidence score 
    (it tells us that how confident the model is that box contains an object 
    and also how accurate it is). It is calculated as 
     
    confident scores= probability of an object * IOU 
    IOU stands for Intersection over union.IOU= area of overlap / area of union
    '''
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence >= args["confidence"]:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)     
            print("[INFO] {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)     
            y = startY - 15 if startY - 15 > 15 else startY + 15     
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, COLORS[idx], 2)
            
    result_dir = config["result_dir"]
    img_path = result_dir + "frame.jpg"
    cv2.imwrite(img_path, image)
#cv2.waitKey(0)

if __name__ == "__main__":
    # arguments passed in command line
    logging.basicConfig(level=logging.INFO, 
                         format='%(asctime)s - %(levelname)s - %(message)s')
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,help="path to input image")
    ap.add_argument("-p", "--prototxt", required=True, 
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True, 
                    help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.3, 
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # fetch the configs from config file and populate inmory structure
    config = open_yaml()
    ssd_detector(args)
    cv2.destroyAllWindows()
