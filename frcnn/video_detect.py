import time
import cv2
import gluoncv as gcv
import mxnet as mx
from utils import return_tags

def tagger(net, video_path, frames_process=1): 
    # Load the video
    video = cv2.VideoCapture(video_path)
    time.sleep(1) ### letting the camera autofocus
    found_tags = []
    print("STARTING")
    try:
        frame_id = -1
        ret = True
        while(video.isOpened()):
            ret, frame = video.read()
            print("GOT FRAME")
            print(ret)
            frame_id += 1
            if frame_id % frames_process != 0:
                print(frame_id)
                continue 
            else:
                if ret == False:
                    break

                ## Preprocess the img
                frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
                rgb_nd, frame = gcv.data.transforms.presets.rcnn.transform_test(frame, short=512, max_size=700)

                class_ids, scores, bboxes = net(rgb_nd.as_in_context(mx.gpu(0)))

                found_tags.append(return_tags(rgb_nd, bboxes[0], scores[0], class_ids[0], class_names=net.classes))

            if ret == False:
                break

        print("RETURNING FOUND TAGS")
        return found_tags

    except Exception as err:
        print(err)
    finally:
        #print(found_tags)
        video.release()
        cv2.destroyAllWindows()
        print("DONE")

# Load the model
#net = gcv.model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True, ctx=mx.gpu(0))
#net = gcv.model_zoo.get_model('ssd_512_resnet50_v1_coco', pretrained=True, ctx=mx.gpu(0))
net = gcv.model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True, ctx=mx.gpu(0))
#net = gcv.model_zoo.get_model('faster_rcnn_resnet101_v1d_coco', pretrained=True, ctx=mx.gpu(0))

for i in range(1,5):
    print(tagger(net, './20-26-20.mp4', 20))
#print(tagger(net, './rickgarage.mp4', 20))

