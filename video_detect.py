import time
import cv2
import gluoncv as gcv
import mxnet as mx

def extract_tags(net, video_path, frames_process=1): 
    # Load the video
    video = cv2.VideoCapture(video_path)
    time.sleep(1) ### letting the camera autofocus
    found_tags = []
    try:
        frame_id = -1
        ret = True
        while(video.isOpened()):
            ret, frame = video.read()
            frame_id += 1
            if frame_id % frames_process != 0:
                print(frame_id)
                continue 
            else:
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
                    found_tags.append(tag)

            if ret == False:
                break

    except Exception as err:
        print(err)
    finally:
        print(found_tags)
        video.release()
        cv2.destroyAllWindows()
        print("DONE")
        return found_tags


# Load the model
#net = gcv.model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True, ctx=mx.gpu(0))
net = gcv.model_zoo.get_model('ssd_512_resnet50_v1_coco', pretrained=True, ctx=mx.gpu(0))

for i in range(1,20):
    extract_tags(net, './rickgarage.mp4', 20)

    
    import time
import cv2
import gluoncv as gcv
import mxnet as mx

def extract_tags(net, video_path, frames_process=1):
    # Load the video
    video = cv2.VideoCapture(video_path)
    time.sleep(1) ### letting the camera autofocus
    found_tags = []
    try:
        frame_id = -1
        ret = True
        while(video.isOpened()):
            ret, frame = video.read()
            frame_id += 1
            if frame_id % frames_process != 0:
                print(frame_id)
                continue
            else:
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
                    found_tags.append(tag)

            if ret == False:
                break

    except Exception as err:
        print(err)
    finally:
        print(found_tags)
        video.release()
        cv2.destroyAllWindows()
        print("DONE", video_path)
        return found_tags


# Load the model
#net = gcv.model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True, ctx=mx.gpu(0))
net = gcv.model_zoo.get_model('ssd_512_resnet50_v1_coco', pretrained=True, ctx=mx.gpu(0))

videos = ['./peopleVid.mp4', './rickgarage.mp4']
#for i in range(1,20):
#extract_tags(net, './rickgarage.mp4', 20)

for vid in videos:
    extract_tags(net, vid, 20)
#extract_tags(net, 'frontgateperson1.mp4', 20)
