from typing import List, Dict, Optional, Tuple
from src.utils.geometry import BBox


class SimpleIOUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: Dict[int, Tuple[BBox, int]] = {}  # id -> (last_bbox, age)

    @staticmethod
    def iou(a: BBox, b: BBox) -> float:
        inter_x1 = max(a.x1, b.x1)
        inter_y1 = max(a.y1, b.y1)
        inter_x2 = min(a.x2, b.x2)
        inter_y2 = min(a.y2, b.y2)
        iw = max(0, inter_x2 - inter_x1)
        ih = max(0, inter_y2 - inter_y1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        ua = a.area() + b.area() - inter
        return inter / ua if ua > 0 else 0.0

    def update(self, detections: List[BBox]) -> List[BBox]:
        assigned: Dict[int, int] = {}  # det_idx -> track_id
        used_tracks = set()

        # Greedy match by IoU
        for d_idx, det in enumerate(detections):
            best_iou, best_tid = 0.0, None
            for tid, (tb, age) in self.tracks.items():
                if tid in used_tracks:
                    continue
                i = self.iou(tb, det)
                if i > best_iou:
                    best_iou, best_tid = i, tid
            if best_tid is not None and best_iou >= self.iou_threshold:
                assigned[d_idx] = best_tid
                used_tracks.add(best_tid)

        # Age and remove stale tracks
        for tid in list(self.tracks.keys()):
            if tid not in used_tracks:
                bbox, age = self.tracks[tid]
                self.tracks[tid] = (bbox, age + 1)
                if self.tracks[tid][1] > self.max_age:
                    del self.tracks[tid]

        # Create tracks for unmatched dets; update matched
        output: List[BBox] = []
        for d_idx, det in enumerate(detections):
            if d_idx in assigned:
                tid = assigned[d_idx]
                self.tracks[tid] = (det, 0)
                det.track_id = tid
                output.append(det)
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = (det, 0)
                det.track_id = tid
                output.append(det)
        return output
