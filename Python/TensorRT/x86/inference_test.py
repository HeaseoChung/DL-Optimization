import cv2
import numpy as np
from inference_engine import TRTEngine


trt_engine = TRTEngine("model_zoo/test_pth.trt",256,256,dynamic=True)

image = cv2.imread("example/Lenna.png")
height, width, _ = image.shape

trt_engine.inputs, trt_engine.outputs, trt_engine.bindings = trt_engine.dynamicBindingProcess(height,width, 2)

pre = np.transpose(image,[2,0,1])
pre = np.expand_dims(pre,axis=0)/255.

sr = trt_engine.do_inference(pre)

post = sr*255
post = np.clip(post,0,255)
post = np.reshape(post, (1,3,height*2, width*2)).astype(np.uint8)
post = post.squeeze(0)
post = np.transpose(post,[1,2,0])

cv2.imwrite("example/Lenna_x2.png", post)

