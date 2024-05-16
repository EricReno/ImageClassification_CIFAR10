import cv2
import time
import numpy
import onnxruntime

cap = cv2.VideoCapture('video.mp4')

ort_session = onnxruntime.InferenceSession("resnet18.onnx")

classnames = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

frame_count = 0  
last_time = time.time() 

while True:
    ret, frame = cap.read()
    if ret:
        input = frame.astype(numpy.float32)
        input = input.transpose([2, 0, 1])
        input = numpy.expand_dims(input, 0)

        ort_inputs = {'input': input}
        ort_output = ort_session.run(['output'], ort_inputs)[0]

        max_index = numpy.argmax(ort_output[0]) 

        frame = cv2.resize(frame, (320, 320))
        text2 = "pred:%s"%(classnames[max_index])
        (text_width, text_height), baseline = cv2.getTextSize(text2, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
        cv2.rectangle(frame, (10, 10),  (10 + text_width, 10 + text_height), (0, 128, 255), -1) 
        cv2.putText(frame, text2, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        frame_count += 1  
        elapsed_time = time.time() - last_time  
        fps = frame_count / elapsed_time  
        cv2.putText(frame, 'FPS: '+str(round(fps,2)), (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

        cv2.imshow('image', frame)

        if cv2.waitKey(1) == ord('q'):  
            break
    else:
        break

cv2.destroyAllWindows()
