import math
import numpy as np

def PlotPillowImage(image):
  from matplotlib import pyplot as plt
  plt.imshow(image)
  plt.axis('off')
  plt.show()

# Used for loading, plotting and manipulating detection boxes.
# Uses CXCYWH
class Detection:
    def __init__(self, cx, cy, w, h, rotation, class_id, confidence):
      self.cx = cx
      self.cy = cy
      self.w = w
      self.h = h
      self.rotation = rotation
      self.class_id = class_id
      self.confidence = confidence
    def Center(self):
      return self.cx, self.cy
    def Width(self):
      return self.w
    def Height(self):
      return self.h
    def ClassId(self):
      return self.class_id
    def Confidence(self):
      return self.confidence
    def Rotation(self):
      return self.rotation
    def __str__(self):
      return f"Detection(x={self.cx}, y={self.cy}, w={self.w}, h={self.h}, class_id={self.class_id}, confidence={self.confidence})"
    def __repr__(self):
      return str(self)
    @staticmethod
    def from_xyxy(x1, y1, x2, y2, class_id, confidence):
      w = x2 - x1
      h = y2 - y1
      x = x1 + w / 2
      y = y1 + h / 2
      return Detection(x, y, w, h, 0, class_id, confidence)
    @staticmethod
    def from_topleft_xywh(x, y, w, h, class_id, confidence):
      x = x + w / 2
      y = y + h / 2
      return Detection(x, y, w, h, 0, class_id, confidence)
    @staticmethod
    def from_xywh(x, y, w, h, class_id, confidence):
      return Detection.from_topleft_xywh(x,y,w,h,class_id,confidence)
    @staticmethod
    def from_cxcywh(cx, cy, w, h, class_id, confidence):
      return Detection(cx, cy, w, h, 0, class_id, confidence)
    def Explode(self):
      return (self.cx, self.cy, self.w, self.h, self.rotation), self.class_id, self.confidence
    def Rescale(self, orig_size, new_size):
      orig_W, orig_H = orig_size
      new_W, new_H = new_size
      cx = self.cx * (new_W / orig_W)
      cy = self.cy * (new_H / orig_H)
      w = self.w * (new_W / orig_W)
      h = self.h * (new_H / orig_H)
      return Detection.from_cxcywh(cx, cy, w, h, self.class_id, self.confidence)
    def Rotate(self, angle, rotation_center):
          """Rotate the detection box around a given center by `angle` degrees."""
          angle_rad = math.radians(angle)
          cx, cy = rotation_center

          dx = self.cx - cx
          dy = self.cy - cy

          cos_a = math.cos(angle_rad)
          sin_a = math.sin(angle_rad)

          new_x = cx + cos_a * dx - sin_a * dy
          new_y = cy + sin_a * dx + cos_a * dy

          new_rotation = (self.rotation + angle) % 360
          return Detection(new_x, new_y, self.w, self.h, new_rotation, self.class_id, self.confidence)

    def RotatedCorners(self):
          """Return list of four (x, y) corner points after applying rotation."""
          angle_rad = math.radians(self.rotation)
          cos_a = math.cos(angle_rad)
          sin_a = math.sin(angle_rad)

          hw = self.w / 2
          hh = self.h / 2

          # Define corners relative to center
          local_corners = [
              (-hw, -hh),
              ( hw, -hh),
              ( hw,  hh),
              (-hw,  hh)
          ]

          # Apply rotation and translate to global coords
          return [
              (
                  self.cx + dx * cos_a - dy * sin_a,
                  self.cy + dx * sin_a + dy * cos_a
              )
              for dx, dy in local_corners
          ]
    def HorizontalFlip(self, image_width):
      flipped_x = image_width - self.cx
      flipped_rotation = (-self.rotation) % 360
      return Detection(flipped_x, self.cy, self.w, self.h, flipped_rotation, self.class_id, self.confidence)

    def VerticalFlip(self, image_height):
      flipped_y = image_height - self.cy
      flipped_rotation = (180 - self.rotation) % 360
      return Detection(self.cx, flipped_y, self.w, self.h, flipped_rotation, self.class_id, self.confidence)

    def TranslateW(self, dx):
      return Detection(self.cx + dx, self.cy, self.w, self.h, self.rotation, self.class_id, self.confidence)

    def TranslateH(self, dy):
      return Detection(self.cx, self.cy + dy, self.w, self.h, self.rotation, self.class_id, self.confidence)

    def Draw(self, pil_image, class_id_to_name = None):
      from PIL import ImageDraw
      draw = ImageDraw.Draw(pil_image)
      corners = self.RotatedCorners()
      draw.polygon(corners, outline="red", width=2)
      if class_id_to_name:
        draw.text((self.cx, self.cy), f"{class_id_to_name(self.class_id)}", fill="red")
      else:
        draw.text((self.cx, self.cy), f"{self.class_id}", fill="red")
      return pil_image

    def ToXYXY(self):
      # Convert to axis-aligned bounding box in xyxy format
      corners = self.RotatedCorners()
      xs = [p[0] for p in corners]
      ys = [p[1] for p in corners]
      return [min(xs), min(ys), max(xs), max(ys)]

    @staticmethod
    def ToSupervision(detections):
      import supervision as sv
      return sv.Detections(
          xyxy=np.array([det.ToXYXY() for det in detections]),
          confidence=np.array([det.confidence for det in detections]),
          class_id=np.array([det.class_id for det in detections])
          )

    @staticmethod
    def NaiveNMS(detections, iou_threshold=0.5): # Ignores classes
      import torch
      from torchvision.ops import nms
      if not detections:
        return []
      boxes = torch.tensor([det.ToXYXY() for det in detections], dtype=torch.float32)
      scores = torch.tensor([det.confidence for det in detections], dtype=torch.float32)
      keep_indices = nms(boxes, scores, iou_threshold).tolist()
      return [detections[i] for i in keep_indices]

    @staticmethod
    def NMS(detections, iou_threshold=0.5): # Performs NMS on every class separately
      classful_nms = dict()
      for det in detections:
        if det.class_id not in classful_nms: classful_nms[det.class_id] = []
        classful_nms[det.class_id].append(det)
      output = []
      for class_id in classful_nms:
        output.extend( Detection.NaiveNMS(classful_nms[class_id], iou_threshold) )
      return output

    def IOU(self, other):
        import torch
        box1 = torch.tensor([self.ToXYXY()])
        box2 = torch.tensor([other.ToXYXY()])
        iou = box_iou(box1, box2).item()
        return iou

    @staticmethod
    def ComputeAP(pred_boxes, gt_boxes, iou_threshold=0.5): # WARNING! THIS ASSUMES ALL BOXES TO BE OF THE SAME CLASS
        import torch
        from torchvision.ops import box_iou
        pred_boxes = sorted(pred_boxes, key=lambda x: x.confidence, reverse=True)

        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        matched_gt = set()

        gt_xyxy = torch.tensor([gt.ToXYXY() for gt in gt_boxes])

        for idx, pred in enumerate(pred_boxes):
            pred_xyxy = torch.tensor([pred.ToXYXY()])
            ious = box_iou(pred_xyxy, gt_xyxy).numpy().flatten()

            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]

            if max_iou >= iou_threshold and max_iou_idx not in matched_gt:
                tp[idx] = 1
                matched_gt.add(max_iou_idx)
            else:
                fp[idx] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        recalls = tp_cum / len(gt_boxes)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

        ap = 0
        for t in np.linspace(0, 1, 11):
            precisions_above_t = precisions[recalls >= t]
            precision_value = precisions_above_t.max() if precisions_above_t.size > 0 else 0
            ap += precision_value / 11

        return ap

    @staticmethod
    def ComputeMAP(predictions, ground_truths, iou_threshold=0.5):
        from collections import defaultdict

        gt_boxes_by_class = defaultdict(list)
        pred_boxes_by_class = defaultdict(list)

        for gt in ground_truths:
            gt_boxes_by_class[gt.class_id].append(gt)

        for pred in predictions:
            pred_boxes_by_class[pred.class_id].append(pred)

        ap_per_class = {}

        for class_id in gt_boxes_by_class.keys():
            gt_boxes = gt_boxes_by_class[class_id]
            pred_boxes = pred_boxes_by_class.get(class_id, [])
            ap_per_class[class_id] = Detection.ComputeAP(pred_boxes, gt_boxes, iou_threshold)

        mAP = np.mean(list(ap_per_class.values())) if ap_per_class else 0

        return ap_per_class, mAP
