# car_detection_ssd
Car detection using SSD model through opencv.

# run
python ssd.py --image /home2/sudhir/carparking_final/images/test11.jpg --prototxt model/MobileNetSSD_deploy.prototxt.txt --model model/MobileNetSSD_deploy.caffemodel

# results
Output will be stored in carpark/result directory.

# Links referred
Calculate Intersection over Union: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
SSD code: https://honingds.com/blog/ssd-single-shot-object-detection-mobilenet-opencv/

Data augmentation and Image Classifier:

https://www.pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/
https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/
