from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from ultralytics import YOLO

from src.utils.geometry import BBox, Point


# Pose keypoint indexes (COCO-ish):
# 5: left shoulder, 6: right shoulder, 7: left elbow, 8: right elbow,
# 9: left wrist, 10: right wrist, 11: left hip, 12: right hip,
# 13: left knee, 14: right knee


@dataclass
class Pose:
    keypoints: Dict[int, Point]  # index -> Point
    score: float = 1.0

    def get(self, idx: int) -> Optional[Point]:
        return self.keypoints.get(idx)


class PersonDetector:
    def __init__(self, conf_threshold: float = 0.4, model_path: Optional[str] = None, device: str = "cpu"):
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        self.device = device
        # TODO: load your model here using model_path/device

    def predict(self, frame) -> List[BBox]:
        # TODO: Replace with real detector
        return []


class HandDetector:
    def __init__(self, conf_threshold: float = 0.4, model_path: Optional[str] = None, device: str = "cpu"):
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        self.device = device
        # TODO: load your model here using model_path/device

    def predict(self, frame) -> List[BBox]:
        # TODO: Replace with real detector
        return []


class ObjectDetector:
    def __init__(self, conf_threshold: float = 0.4, model_path: Optional[str] = None, device: str = "cpu"):
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        self.device = device
        # TODO: load your model here using model_path/device

    def predict(self, frame) -> List[BBox]:
        # TODO: Replace with real detector
        return []


class PoseEstimator:
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu", conf_threshold: float = 0.3):
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.model = YOLO(model_path) if model_path else None
        if self.model is not None:
            print(f"Loading pose model on device: {device}")
            self.model.to(device)
            # Enable CUDA optimizations if using GPU
            if 'cuda' in device.lower():
                print("Enabling CUDA optimizations for pose estimation")

    def predict(self, frame) -> List[Pose]:
        if self.model is None:
            return []
        results = self.model(frame, verbose=False)
        poses: List[Pose] = []
        frame_height, frame_width = frame.shape[:2]
        
        for r in results:
            if not hasattr(r, "keypoints") or r.keypoints is None:
                continue
            kpts_xy: np.ndarray = r.keypoints.xy.cpu().numpy()  # (N, K, 2)
            kpts_conf: Optional[np.ndarray] = None
            if hasattr(r.keypoints, "conf") and r.keypoints.conf is not None:
                kpts_conf = r.keypoints.conf.cpu().numpy()  # (N, K)
            
            for i in range(kpts_xy.shape[0]):
                keypoints: Dict[int, Point] = {}
                
                for k in range(kpts_xy.shape[1]):
                    x, y = float(kpts_xy[i, k, 0]), float(kpts_xy[i, k, 1])
                    sc = float(kpts_conf[i, k]) if kpts_conf is not None else 1.0
                    
                    # Filter out low confidence keypoints
                    if sc < self.conf_threshold:
                        continue
                    
                    # Filter out invalid coordinates (0,0) or out of bounds
                    if (x <= 1 and y <= 1) or x < 0 or y < 0 or x >= frame_width or y >= frame_height:
                        continue
                    
                    # Use COCO keypoint indices - the model should already provide them in COCO format
                    # If your model uses different indices, you may need to map them here
                    keypoints[k] = Point(x=x, y=y, score=sc)
                
                # Only add pose if we have at least some valid keypoints
                if len(keypoints) > 0:
                    poses.append(Pose(keypoints=keypoints, score=1.0))
        return poses


class HOIDetector:
    """Unified detector that returns persons, hands, and objects together.

    Replace the predict() stub with your HOI model inference and split outputs
    into three lists of BBox: persons, hands, objects.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu", conf_threshold: float = 0.4):
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.model = YOLO(model_path) if model_path else None
        if self.model is not None:
            print(f"Loading HOI model on device: {device}")
            self.model.to(device)
            # Enable CUDA optimizations if using GPU
            if 'cuda' in device.lower():
                print("Enabling CUDA optimizations for HOI detection")
        # Build class name to id mapping for routing
        self.person_ids: List[int] = []
        self.hand_ids: List[int] = []
        self.object_ids: List[int] = []
        if self.model is not None and hasattr(self.model, "names"):
            names = self.model.names
            for cid, name in names.items():
                lname = str(name).lower()
                if "person" in lname:
                    self.person_ids.append(cid)
                elif "hand" in lname:
                    self.hand_ids.append(cid)
                else:
                    self.object_ids.append(cid)

    def _route_det(self, cls_id: int) -> str:
        if cls_id in self.person_ids:
            return "person"
        if cls_id in self.hand_ids:
            return "hand"
        return "object"

    def predict(self, frame) -> Tuple[List[BBox], List[BBox], List[BBox]]:
        if self.model is None:
            return [], [], []
        results = self.model(frame, verbose=False)
        persons: List[BBox] = []
        hands: List[BBox] = []
        objects: List[BBox] = []
        for r in results:
            if r.boxes is None:
                continue
            # xyxy, conf, cls
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else np.ones((xyxy.shape[0],), dtype=float)
            cls = r.boxes.cls.cpu().numpy().astype(int) if r.boxes.cls is not None else np.zeros((xyxy.shape[0],), dtype=int)
            for i in range(xyxy.shape[0]):
                if float(conf[i]) < self.conf_threshold:
                    continue
                x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
                label = self._route_det(int(cls[i]))
                bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2, score=float(conf[i]), label=label)
                if label == "person":
                    persons.append(bbox)
                elif label == "hand":
                    hands.append(bbox)
                else:
                    objects.append(bbox)
        return persons, hands, objects
