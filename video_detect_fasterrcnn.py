import time

import cv2
import gluoncv as gcv
import mxnet as mx

# Load the model
net = gcv.model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True, ctx=mx.gpu(0))

# Load the webcam handler
video = cv2.VideoCapture('./townCenter.mp4')
time.sleep(1) ### letting the camera autofocus

try:
    frameID = 0
    ret = True
    while(video.isOpened()):
        ret, frame = video.read()
        frameID += 1
        print(frameID)

        if ret == False:
            break

        ## Preprocess the img
        frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
        rgb_nd, frame = gcv.data.transforms.presets.ssd.transform_test(frame, short=512, max_size=700)

        class_ids, scores, bboxes = net(rgb_nd.as_in_context(mx.gpu(0)))

        for i in range(len(scores[0])):
            cid = int(class_ids[0][i].asnumpy())
            cname = net.classes[cid]
            score = float(scores[0][i].asnumpy())
            if score < 0.5:
                break
            tag = "{}: {:.2f}".format(cname, score)
            print(tag)

except Exception as err:
    print(err)
finally:
    video.release()
    cv2.destroyAllWindows()
    print("DONE")

