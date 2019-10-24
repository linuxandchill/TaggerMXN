import mxnet as mx
from gluoncv import model_zoo, data, utils
import gluoncv as gcv
import numpy as np

net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True, ctx=mx.gpu(0))
im_fname = './biking.jpg'
x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)
class_IDs, scores, bounding_boxes = net(x.as_in_context(mx.gpu(0)))

## (4) Display 
for i in range(len(scores[0])):
    cid = int(class_IDs[0][i].asnumpy())
    cname = net.classes[cid]
    score = float(scores[0][i].asnumpy())
    if score < 0.5:
        break
    tag = "{}: {:.2f}".format(cname, score)
    print(tag)

