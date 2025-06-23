# Detections
Simple implementation of detection bounding box, to be used in multiple projects.

# Installation
Project requires Numpy to be imported and PyTorch, PILLOW for full functionality. Pytorch should be installed manually, the rest can be installed as follows:
```
pip install git+https://github.com/AzethMeron/Detections.git
```

If there's aproblem with dependencies, you can also try disregarding versions. Shouldn't matter anyway, not guaranteed tho.
```
pip install numpy
pip install --no-dependencies git+https://github.com/AzethMeron/Detections.git
```

Recommanded import
```py
from Detections import Detection
```

# Representation
Detections are represented using single class, ```Detection```. It is equivalent to single bounding box. Attributes are as follows
```py
detection.cx # Position of the center of bounding box, in pixels. X coord ("column" in the image)
detection.cy # Position of the center of bounding box, in pixels. Y coord ("row" in the image)
detection.w # Total width of the bounding box (length alongside X axis for rotation=0)
detection.h # Total height of the bounding box (length alongside Y axis for rotation=0)
detection.rotation # Counter-clockwise rotation, given in degrees of rotation.
detection.class_id # Class id, integer.
detection.confidence # Confidence, float number (0..1)
```
As you can see, internally class uses CXCYWH representation + rotation. 

You can easily create detections from standard representations using following static methods. Note all of them work "one detection (bbox) at a time".
```
det = Detection.from_xyxy(x1, y1, x2, y2, class_id, confidence)
det = Detection.from_xywh(x, y, w, h, class_id, confidence)
det = Detection(cx, cy, w, h, rotation, class_id, confidence)
```

Useful methods:
```py
det = det.Rescale( (orig_W, orig_H), (new_W, new_H) )
det = det.Rotate( angle, (rot_center_x, rot_center_y) )
det = det.HorizontalFlip(image_width)
det = det.VerticalFlip(image_height)
det = det.TranslateW(dx)
det = det.TranslateH(dy)
det.Draw( pillow_image, class_id,to_name) # class_id_to_name is a function(class_id: int) -> string
[ (x1,y1), (x2,y2), (x3,y3), (x4,y4) ] = det.RotatedCorners() # Gets corners of the rotated bounding box
x1,y1, x2,y2 = det.ToXYXY() # Converts to XYXY format, creating bbox that encapsulates entire rotated bbox
sv_detections = Detection.ToSupervision(list_of_Detection) # Converts list of Detections to Roboflow's Supervision format. REQUIRES SUPERVISION
list_of_detections = Detection.NaiveNMS(list_of_detections, iou_threshold=0.5) # Applies non maximum supression to detections, disregarding classes. REQUIRES PYTORCH
list_of_detections = Detection.NMS(list_of_detections, iou_threshold=0.5) # Applies non maximum supression to detections of the same class. REQUIRES PYTORCH
iou = det.IOU(another_det) # REQUIRES PYTORCH
ap = Detection.ComputeAP(pred_dets, true_dets, iou_threshold = 0.5) # Computes AP for lists of detections passed. Disregards classes. REQUIRES PYTORCH
ap_per_class, mAP = Detection.ComputeMAP(pred_dets, true_dets, iou_threshold = 0.5) # Computes AP per class and mean AP. REQUIRES PYTORCH
```
