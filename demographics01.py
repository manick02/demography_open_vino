import cv2 as cv
def main():
    net = cv.dnn.readNet('/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.bin',
                        '/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.xml')
    
    # net = cv.dnn.readNet('/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-retail-0004/FP32/face-detection-retail-0004.bin',
    #                     '/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-retail-0004/FP32/face-detection-retail-0004.xml')
    
    cap = cv.VideoCapture(0)

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        blob = cv.dnn.blobFromImage(frame, size=(672, 384))
        net.setInput(blob)
        out = net.forward()

        for detection in out.reshape(-1, 7):
            confidence = float(detection[2])
            xmin = int(detection[3] * frame.shape[1])
            ymin = int(detection[4] * frame.shape[0])
            xmax = int(detection[5] * frame.shape[1])
            ymax = int(detection[6] * frame.shape[0])

            if confidence > 0.5:
                cv.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))

        cv.imshow('OpenVINO face detection', frame)
if __name__ == '__main__':
    print('running openvino face detection')
    main() 