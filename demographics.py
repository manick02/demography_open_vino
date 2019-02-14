#!/usr/bin/env python
"""
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import cv2
import numpy as np
import logging as log
from time import time
from openvino.inference_engine import IENetwork, IEPlugin
from pprint import pprint
from centroid_tracker import CentroidTracker
from pubsub import pub
import datetime


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
    parser.add_argument("-mg", "--agegender", help="Path to an .xml file with a trained model.", required=True, type=str)
    # parser.add_argument("-i", "--input", help="Path to a folder with images or path to an image files", required=True,
    #                     type=str, nargs="+")
    parser.add_argument("-l", "--cpu_extension",
                        help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels "
                             "impl.", type=str, default=None)
    parser.add_argument("-pp", "--plugin_dir", help="Path to a plugin folder", type=str, default=None)
    parser.add_argument("-d", "--device",
                        help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device specified (CPU by default)", default="CPU",
                        type=str)
    parser.add_argument("-s","--referenceid",help="reference id for reporting",default="ns",type=str)
    parser.add_argument("-c","--subreferenceid",help="source reference id for reporting",default="subns",type=str)

    # parser.add_argument("--labels", help="Labels mapping file", default=None, type=str)
    # parser.add_argument("-nt", "--number_top", help="Number of top results", default=10, type=int)
    # parser.add_argument("-ni", "--number_iter", help="Number of inference iterations", default=1, type=int)
    # parser.add_argument("-pc", "--perf_counts", help="Report performance counters", default=False, action="store_true")

    return parser


def main():
    ct = CentroidTracker(10)   
    pub.subscribe(face_out_of_frame,'face_out_of_frame')
    pub.subscribe(face_in_frame,'face_in_frame')
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    stream = cv2.VideoCapture(0)
    model_bin, model_xml = get_face_detection_model(args)
    model_age_gender_xml, model_age_gender_bin = get_age_gender_detection_model(args)
    # Plugin initialization for specified device and load extensions library if specified
    plugin = get_plugin(args)
    print('running on device',args.device)
    add_extension_to_plugin(args, plugin)
    face_detection_net = IENetwork(model=model_xml, weights=model_bin)
    check_for_unsupported_layers(plugin, face_detection_net)    
    age_gender_net = IENetwork(model=model_age_gender_xml,weights=model_age_gender_bin)
    check_for_unsupported_layers(plugin, age_gender_net)
    # /opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.xml
   
    log.info("Preparing inputs")
    face_net_input_blob = next(iter(face_detection_net.inputs))
    face_net_output_blob = next(iter(face_detection_net.outputs))
    face_detection_net.batch_size = 1

    # Read and pre-process input images
    n, c, h, w = face_detection_net.inputs[face_net_input_blob].shape
    print('number of images :', n, 'number of channels:',c,'height of image:',h,'width of image:',w)
    # Loading model to the plugin
    log.info("Loading model ton the plugin")
    exec_net = plugin.load(network=face_detection_net)
    age_gender_input_blob = next(iter(age_gender_net.inputs))
    # print(face_detection_net.inputs,face_detection_net.outputs,age_gender_net.inputs,age_gender_net.outputs)
    # print(age_gender_net.outputs,len(age_gender_net.outputs))
    age_blob='age_conv3'
    gender_blob='prob'
    
    age_output = age_gender_net.outputs[age_blob]
    gender_output = age_gender_net.outputs[gender_blob]
    print("age,gender,model input specs",age_gender_net.inputs[age_gender_input_blob].shape)
    agen,agec,ageh,agew = age_gender_net.inputs[age_gender_input_blob].shape
    print("loading page gender model to the plugin")
    exec_age_gender_net = plugin.load(network=age_gender_net)
    
    while(True):
        status, image = stream.read()
        res, initialw, initialh = infer_face(n, c, h, w, image, exec_net, face_net_input_blob)        
        out = res[face_net_output_blob]
        count = 0
        
        tfaces = np.ndarray(shape=(agen,agec,ageh,agew))
        rects = [] 
        for obj in out[0][0]:
            threshold = obj[2]
            class_id = int(obj[1])
            if threshold > 0.9:
                count = count + 1
                xmin = int(obj[3] * initialw)
                ymin = int(obj[4] * initialh)
                xmax = int(obj[5] * initialw)
                ymax = int(obj[6] * initialh)
                color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
                face = image[ymin:ymax,xmin:xmax]
                (fh,fw)=face.shape[:-1]
                if fh < ageh or fw < agew:
                    continue
                tface = cv2.resize(face, (agew, ageh))
                tface = tface.transpose((2,0,1))
                tfaces[0]=tface
                t0 = time()
                out_age_gender=exec_age_gender_net.infer(inputs={face_net_input_blob:tfaces})
                print('inferencetime age,gender detection',(time() - t0) * 1000)
                age, gender, checkedin = get_age_gender(out_age_gender, age_blob, gender_blob)
                
                rects.append((xmin,ymin,xmax,ymax,age,gender,checkedin))

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        
        print("number of faces in frame",count)
        if count > 0:
            x=ct.update(rects)
            print(list(x.items()))
            cv2.imshow("face",face)

        cv2.imshow("Display", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def get_age_gender(out_age_gender, age_blob, gender_blob):
    ageprob =  out_age_gender[age_blob][0][0][0][0]
    genderprob = out_age_gender[gender_blob][0][0][0][0]
    age = int(ageprob * 100)
    gender ='M'
    checkedin = datetime.datetime.now()
    if genderprob >= 0.5 :
        gender ='F'
    return age, gender, checkedin


def check_for_unsupported_layers(plugin, network):
    if plugin.device == "CPU":
        supported_layers = plugin.get_supported_layers(network)
        not_supported_layers = [l for l in network.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                        format(plugin.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                        "or --cpu_extension command line argument")

            sys.exit(1)
    

def add_extension_to_plugin(args, plugin):
    if args.cpu_extension and 'CPU' in args.device:
        plugin.add_cpu_extension(args.cpu_extension)

def get_age_gender_detection_model(args):
    model_age_gender_xml = args.agegender
    model_age_gender_bin=os.path.splitext(model_age_gender_xml)[0] + ".bin"
    return model_age_gender_xml, model_age_gender_bin

def get_face_detection_model(args):
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    return model_bin, model_xml

def get_plugin(args):
    plugin = IEPlugin(device=args.device, plugin_dirs=args.plugin_dir)
    return plugin

def infer_face(n, c, h, w, image, exec_net, input_blob):
    images = np.ndarray(shape=(n, c, h, w))
    for i in range(n):
        #image = cv2.imread(args.input[i])
        initialh,initialw = image.shape[:-1]
        if image.shape[:-1] != (h, w):
            #log.warning("Image {} is resized from {} to {}".format(args.input[i], image.shape[:-1], (h, w)))
            timage = cv2.resize(image, (w, h))
        timage = timage.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        images[i] = timage
    t0 = time()
    res = exec_net.infer(inputs={input_blob: images})
    print('inferencetime face detection',(time() - t0) * 1000)
    return res, initialw, initialh

def process_single_classification(res,out_blob,limit,labels):
    out = res[out_blob]
    log.info("Top {} results: ".format(limit))
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
    else:
        labels_map = None
    for i, probs in enumerate(out):
        probs = np.squeeze(probs)
        top_ind = np.argsort(probs)[-limit:][::-1]
        print("Image {}\n".format(args.input[i]))
        for id in top_ind:
            det_label = labels_map[id] if labels_map else "#{}".format(id)
            print("{:.7f} label {}".format(probs[id], det_label))
        print("\n")

def face_out_of_frame(arg1,arg2=None):
    (xmin,ymin,xmax,ymax,age,gender,checkedin)= arg1
    timedelta = int((datetime.datetime.now() - checkedin).seconds)
    print(timedelta,"seconds")
    print("OUT")
    print(arg1)

def face_in_frame(arg1,arg2=None):
    print("+++")
    #print(arg1)

def dump(obj):
  for attr in dir(obj):
    print("obj.%s = %r" % (attr, getattr(obj, attr)))

if __name__ == '__main__':
    sys.exit(main() or 0)
