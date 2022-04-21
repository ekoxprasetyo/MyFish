import numpy as np
import argparse
import time
import cv2
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import PIL
import numpy

confthres=0.3
nmsthres=0.5
yolo_path="./"

dic = {0 : 'Sangat Segar',1 : 'Segar',2 : 'Tidak Segar'}

model_segar_kepala = load_model('ekperimen2-kesegaran-kepala-45-0.7038-0.7320.h5')
model_segar_ikan_utuh = load_model('ekperimen2-kesegaran-ikan_utuh-56-0.7374-0.7517.h5')
model_segar_ekor = load_model('ekperimen2-kesegaran-ekor-54-0.6871-0.7063.h5')

model_segar_kepala.make_predict_function()

def predict_segar(img_path, objek):
    #i = image.load_img(img_path, target_size=(180,180))
    i = image.load_img(img_path, target_size=(112,112))
    i = image.img_to_array(i)/255.0
    #i = i.reshape(1, 180,180,3)
    i = i.reshape(1, 112,112,3)
    #p = model.predict_classes(i)
    if objek == 0 :
        ohe = model_segar_kepala.predict(i)
    elif objek == 1 :
        ohe = model_segar_ekor.predict(i)
    else :
        ohe = model_segar_ikan_utuh.predict(i)
    print("kelas = ")
    print(ohe[0])
    p = numpy.argmax(ohe, axis=1)
    return dic[p[0]], ohe[0]

def get_labels(labels_path):
    # load the COCO class labels our YOLO model was trained on
    #labelsPath = os.path.sep.join([yolo_path, "yolo_v3/coco.names"])
    lpath=os.path.sep.join([yolo_path, labels_path])
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS

def get_colors(LABELS):
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
    return COLORS

def get_weights(weights_path):
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath

def get_config(config_path):
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath

def load_model(configpath,weightspath):
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    print(weightspath)
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net


def get_predection(image,net,LABELS,COLORS, nama_file, image_awal):
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    #blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (608, 608),
    #                             swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    print(layerOutputs)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            # print(scores)
            classID = np.argmax(scores)
            # print(classID)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confthres:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)

    # Buat folder
    dirbuat = "static/"+nama_file[:-4]
    #if os.path.isdir(dirbuat):
    #    print("Direktori "+dirbuat+" sudah ada.")
    #else:
    #    os.mkdir("static/"+nama_file[:-4])
	# ensure at least one detection exists
    #image_awal = image
    #cv2.imwrite("static/abcd.jpg", image_awal)
    box_hasil = []
    if len(idxs) > 0 and os.path.isdir(dirbuat) == False:
		# Buat folder
        os.mkdir(dirbuat)
        cv2.imwrite("static/abcd0.jpg", image_awal)

        
        j = 0;
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            j += 1
            print("i = "+str(i))
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])


			
			# CROP BOX
            y_min, y_max, x_min, x_max = y, y+h, x, x+w
            crop_image = image_awal[y_min:y_max, x_min:x_max]
            #cv2.imwrite("static/abcd"+str(i)+".jpg", image_awal)
            nama_file_box = "static/"+nama_file[:-4]+"/hasil_"+nama_file[:-4]+"_"+str(i)+".jpg"
            print("nama file:"+nama_file_box)
            cv2.imwrite(nama_file_box, crop_image)

			# prediksi kesegaran
            p, ohe = predict_segar(nama_file_box, classIDs[i])

			# tentukan tebal garis BB dan teks kelas
            tinggi = image_awal.shape[0]
            lebar = image_awal.shape[1]
            if lebar < tinggi:
                basis = lebar
            else:
                basis = tinggi
            tebal = round(basis/300)
            tebal2 = round(basis/1300,2)

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, tebal)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            print(boxes)
            print(classIDs)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,tebal2, color, tebal)

            data_box = [j, classIDs[i], confidences[i], y_min, y_max, x_min, x_max, nama_file_box, p, ohe]
            box_hasil.append(data_box)
    #print(box_hasil)
    return image, box_hasil

def runModel(image, nama_file, image_awal):
    # load our input image and grab its spatial dimensions
    # image = cv2.imread(img)
    
	#labelsPath="./coco.names"
    #cfgpath="cfg/yolov4-tiny.cfg"
    #wpath="yolov4-tiny.weights"
    
    #labelsPath="./obj.names"
    #cfgpath="cfg/anchor_yolov4-tiny_new_5_wfm.cfg"
    #wpath="anchor_yolov4-tiny_new_5_wfm_best.weights"
    
    labelsPath="./obj.names"
    cfgpath="cfg/anchor_yolov4-tiny_new_5_wfm.cfg"
    #wpath="anchor_yolov4-tiny_new_5_wfm_best_from_pc.weights"
    wpath="anchor_yolov4-tiny_new_5_wfm_best_from_colab.weights"
    
    Lables=get_labels(labelsPath)
    CFG=get_config(cfgpath)
    Weights=get_weights(wpath)
    nets=load_model(CFG,Weights)
    Colors=get_colors(Lables)
    res, box_hasil=get_predection(image,nets,Lables,Colors, nama_file, image_awal)
    return res, box_hasil