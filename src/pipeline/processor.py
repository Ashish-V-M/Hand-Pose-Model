from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np

from src.utils.geometry import BBox, Point, angle_between
from src.models.adapters import Pose


@dataclass
class FrameResult:
    interactions_in_frame: int
    hands_in_frame: int
    sus1: bool
    sus2: bool
    sus3: bool
    suspicious_now: bool


@dataclass
class PersonResult:
    track_id: int
    interactions: int
    hands_in_frame: int
    sus1: bool
    sus2: bool
    sus3: bool
    suspicious_now: bool


class SuspicionTracker:
    def __init__(
        self,
        interaction_threshold: int = 3,
        no_interaction_reset_frames: int = 3,
        hands_presence_needed: int = 5,
        hands_absent_trigger_frames: int = 3,
        elbow_angle_threshold_deg: float = 160.0,
        elbow_low_min_frames: int = 3,
        elbow_reset_frames: int = 2,
        suspicious_buffer_size: int = 30,
    ):
        self.interaction_threshold = interaction_threshold
        self.no_interaction_reset_frames = no_interaction_reset_frames
        self.hands_presence_needed = hands_presence_needed
        self.hands_absent_trigger_frames = hands_absent_trigger_frames
        self.elbow_angle_threshold_deg = elbow_angle_threshold_deg
        self.elbow_low_min_frames = elbow_low_min_frames
        self.elbow_reset_frames = elbow_reset_frames

        # States
        self.total_interactions = 0
        self.no_interaction_streak = 0
        self.sus1 = False

        self.hand_presence_streak = 0
        self.hand_absent_streak = 0
        self.hands_presence_phase_complete = False
        self.sus2 = False

        self.elbow_low_streak = 0
        self.elbow_high_reset_streak = 0
        self.sus3 = False

        self.frame_buffer = deque(maxlen=suspicious_buffer_size)

    def update_interactions(self, interactions_in_frame: int):
        if interactions_in_frame > 0:
            self.total_interactions += interactions_in_frame
            self.no_interaction_streak = 0
        else:
            self.no_interaction_streak += 1

        if self.total_interactions >= self.interaction_threshold:
            self.sus1 = True
        if self.no_interaction_streak >= self.no_interaction_reset_frames:
            self.sus1 = False
            self.total_interactions = 0

    def update_hands_presence(self, hands_in_frame: int):
        if hands_in_frame > 0:
            self.hand_presence_streak += 1
            self.hand_absent_streak = 0
            if self.hand_presence_streak >= self.hands_presence_needed:
                self.hands_presence_phase_complete = True
        else:
            if self.hands_presence_phase_complete:
                self.hand_absent_streak += 1
                if self.hand_absent_streak >= self.hands_absent_trigger_frames:
                    self.sus2 = True
            else:
                self.hand_absent_streak += 1

        if hands_in_frame > 0:
            # Once hands visible again, reset the absent streak trigger phase
            self.hands_presence_phase_complete = False
            self.hand_absent_streak = 0

    def update_elbow_angle(self, low_angle_in_frame: bool):
        if low_angle_in_frame:
            self.elbow_low_streak += 1
            self.elbow_high_reset_streak = 0
            if self.elbow_low_streak >= self.elbow_low_min_frames:
                self.sus3 = True
        else:
            if self.sus3:
                self.elbow_high_reset_streak += 1
                if self.elbow_high_reset_streak >= self.elbow_reset_frames:
                    self.sus3 = False
                    self.elbow_low_streak = 0
            else:
                self.elbow_low_streak = 0
                self.elbow_high_reset_streak = 0

    def add_frame(self, frame):
        self.frame_buffer.append(frame.copy())

    def dump_suspicious(self, out_dir: str, frame_idx: int) -> List[str]:
        import os
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for i, f in enumerate(list(self.frame_buffer)):
            p = os.path.join(out_dir, f"sus_{frame_idx:06d}_{i:02d}.jpg")
            cv2.imwrite(p, f)
            paths.append(p)
        return paths


class MultiPersonSuspicion:
    def __init__(self, **tracker_kwargs):
        self.trackers: Dict[int, SuspicionTracker] = {}
        self.tracker_kwargs = tracker_kwargs

    def get(self, tid: int) -> SuspicionTracker:
        if tid not in self.trackers:
            self.trackers[tid] = SuspicionTracker(**self.tracker_kwargs)
        return self.trackers[tid]

    def prune(self, active_ids: List[int]):
        # Remove trackers for IDs not seen recently if desired
        for tid in list(self.trackers.keys()):
            if tid not in active_ids:
                del self.trackers[tid]


def find_pocket_rois(pose: Pose, padding: int = 25) -> List[BBox]:
    # Pocket ROI: region between hip and knee points. Compute for left and right.
    # Added padding to make pocket regions slightly larger for better detection
    rois: List[BBox] = []
    left_hip = pose.get(11)
    right_hip = pose.get(12)
    left_knee = pose.get(13)
    right_knee = pose.get(14)
    
    if left_hip and left_knee:
        # Calculate base coordinates
        x1 = int(min(left_hip.x, left_knee.x))
        y1 = int(min(left_hip.y, left_knee.y))
        x2 = int(max(left_hip.x, left_knee.x))
        y2 = int(max(left_hip.y, left_knee.y))
        
        # Add padding to expand the pocket region
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = x2 + padding
        y2 = y2 + padding
        
        rois.append(BBox(x1, y1, x2, y2, 1.0, "left_pocket"))
        
    if right_hip and right_knee:
        # Calculate base coordinates
        x1 = int(min(right_hip.x, right_knee.x))
        y1 = int(min(right_hip.y, right_knee.y))
        x2 = int(max(right_hip.x, right_knee.x))
        y2 = int(max(right_hip.y, right_knee.y))
        
        # Add padding to expand the pocket region
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = x2 + padding
        y2 = y2 + padding
        
        rois.append(BBox(x1, y1, x2, y2, 1.0, "right_pocket"))
        
    return rois


def check_interactions(
    hand_keypoints: List[Point],
    hand_bboxes: List[BBox],
    object_bboxes: List[BBox],
    pocket_rois: List[BBox],
    dist_thresh_px: float = 50.0,
) -> int:
    # Count interactions when hand is close to object OR hand enters pocket ROI
    # Smart filtering to reduce false positives from normal objects
    interactions = 0

    # Define suspicious vs normal object types
    normal_objects = {'phone', 'cell phone', 'book', 'bottle', 'cup', 'laptop', 'bag', 'backpack'}
    large_objects = {'suitcase', 'chair', 'table', 'tv', 'monitor', 'refrigerator'}

    # hand-object proximity via closest hand keypoint to object center
    for obj in object_bboxes:
        obj_label = (obj.label or '').lower()
        
        # Skip interactions with normal/large objects that are unlikely to be stolen
        if any(normal_item in obj_label for normal_item in normal_objects):
            continue
        if any(large_item in obj_label for large_item in large_objects):
            continue
            
        # Skip very large objects (likely not theft targets)
        obj_width = obj.x2 - obj.x1
        obj_height = obj.y2 - obj.y1
        if obj_width > 200 or obj_height > 200:  # Large objects unlikely to be stolen
            continue
            
        obj_center = obj.center()
        min_dist = None
        for hp in hand_keypoints:
            d = (hp.x - obj_center.x) ** 2 + (hp.y - obj_center.y) ** 2
            if min_dist is None or d < min_dist:
                min_dist = d
        if min_dist is not None and min_dist ** 0.5 <= dist_thresh_px:
            interactions += 1

    # hand entering pocket ROIs
    for hb in hand_bboxes:
        for roi in pocket_rois:
            if hb.intersects(roi):
                interactions += 1
                break

    for hp in hand_keypoints:
        for roi in pocket_rois:
            if roi.contains_point(hp):
                interactions += 1
                break

    return interactions


def any_elbow_low_angle(pose: Pose, angle_thresh: float) -> bool:
    ls, rs = pose.get(5), pose.get(6)
    le, re = pose.get(7), pose.get(8)
    lw, rw = pose.get(9), pose.get(10)
    angles = []
    if ls and le and lw:
        angles.append(angle_between(ls, le, lw))
    if rs and re and rw:
        angles.append(angle_between(rs, re, rw))
    return any(a < angle_thresh for a in angles)


def assign_pose_to_person(person: BBox, poses: List[Pose]) -> Optional[Pose]:
    # Pick pose whose bbox (min/max of keypoints) has highest IoU with person
    best_pose, best_iou = None, 0.0
    person_center = person.center()
    
    for p in poses:
        # Filter out poses with very few keypoints
        if len(p.keypoints) < 3:
            continue
            
        xs = [pt.x for pt in p.keypoints.values() if pt.x > 0 and pt.y > 0]
        ys = [pt.y for pt in p.keypoints.values() if pt.x > 0 and pt.y > 0]
        if not xs or not ys:
            continue
            
        # Create bounding box from valid keypoints only
        pb = BBox(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        
        # Additional check: pose center should be reasonably close to person center
        pose_center = pb.center()
        distance = ((pose_center.x - person_center.x)**2 + (pose_center.y - person_center.y)**2)**0.5
        
        # If pose is too far from person center, skip it
        person_diagonal = ((person.x2 - person.x1)**2 + (person.y2 - person.y1)**2)**0.5
        if distance > person_diagonal * 2:  # Allow some flexibility
            continue
        
        # reuse SimpleIOUTracker.iou logic or reimplement here
        from src.pipeline.tracking import SimpleIOUTracker
        iou_val = SimpleIOUTracker.iou(pb, person)
        if iou_val > best_iou:
            best_iou, best_pose = iou_val, p
    return best_pose


def assign_hands_to_person(person: BBox, hands: List[BBox], dist_thresh: float = 100.0) -> List[BBox]:
    cx, cy = person.center().x, person.center().y
    owned = []
    for h in hands:
        hx, hy = h.center().x, h.center().y
        if (hx - cx) ** 2 + (hy - cy) ** 2 <= dist_thresh ** 2:
            owned.append(h)
    return owned


def _draw_box(frame, b: BBox, color, label: Optional[str] = None, thickness: int = 2):
    cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), color, thickness)
    if label:
        # Calculate text size for background rectangle
        font_scale = 0.5
        font_thickness = 2
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_x = b.x1
        text_y = max(text_size[1] + 5, b.y1 - 5)
        
        # Draw background rectangle for better visibility
        cv2.rectangle(frame,
                     (text_x - 2, text_y - text_size[1] - 2),
                     (text_x + text_size[0] + 2, text_y + baseline + 2),
                     (0, 0, 0), -1)  # Black background
        
        # Draw white text with thicker font
        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)


def _draw_pose(frame, pose: Pose, pocket_rois: List[BBox] = None):
    # Draw keypoints and simple limbs for shoulders->elbows->wrists and hips->knees
    k = pose.keypoints
    frame_height, frame_width = frame.shape[:2]
    
    def kp(i):
        pt = k.get(i)
        # Validate keypoint coordinates
        if pt and 0 <= pt.x < frame_width and 0 <= pt.y < frame_height:
            return pt
        return None

    pairs = [
        (5, 7), (7, 9),   # left shoulder->elbow->wrist
        (6, 8), (8, 10),  # right shoulder->elbow->wrist
        (11, 13),         # left hip->knee
        (12, 14),         # right hip->knee
        (5, 6),           # shoulders
        (11, 12),         # hips
    ]
    
    # Check if hand keypoints are in pocket regions
    hand_in_pocket = False
    if pocket_rois:
        lw, rw = pose.get(9), pose.get(10)  # left and right wrists
        for roi in pocket_rois:
            if (lw and roi.contains_point(lw)) or (rw and roi.contains_point(rw)):
                hand_in_pocket = True
                break
    
    # Draw keypoints with validation
    for idx, pt in k.items():
        # Validate coordinates before drawing
        if not (0 <= pt.x < frame_width and 0 <= pt.y < frame_height):
            continue
            
        # Also check confidence if available
        if hasattr(pt, 'score') and pt.score < 0.3:
            continue
            
        if idx in [9, 10] and hand_in_pocket:  # left and right wrists
            cv2.circle(frame, (int(pt.x), int(pt.y)), 3, (0, 0, 255), -1)  # red
        else:
            cv2.circle(frame, (int(pt.x), int(pt.y)), 3, (0, 255, 255), -1)  # yellow
    
    # Draw lines between keypoints with validation
    for a, b in pairs:
        pa, pb = kp(a), kp(b)
        if pa and pb:
            # Additional check for reasonable distance between keypoints
            distance = ((pa.x - pb.x)**2 + (pa.y - pb.y)**2)**0.5
            if distance < min(frame_width, frame_height):  # Reasonable distance check
                cv2.line(frame, (int(pa.x), int(pa.y)), (int(pb.x), int(pb.y)), (0, 200, 200), 2)


def draw_overlays(
    frame,
    persons: List[BBox],
    hands: List[BBox],
    objects: List[BBox],
    rois: List[BBox],
    poses: List[Pose],
    interactions_in_frame: int,
    hands_in_frame: int,
    sus1: bool,
    sus2: bool,
    sus3: bool,
    suspicious_now: bool,
):
    # Persons in blue
    for b in persons:
        _draw_box(frame, b, (255, 0, 0), b.label or "person")
    # Hands in green
    for b in hands:
        _draw_box(frame, b, (0, 255, 0), b.label or "hand")
    # Objects in cyan/yellow
    for b in objects:
        _draw_box(frame, b, (0, 255, 255), b.label or "object")
    # Pocket ROIs in magenta
    for r in rois:
        _draw_box(frame, r, (255, 0, 255), r.label or "pocket")
    # Poses
    for p in poses:
        _draw_pose(frame, p, rois)

    # Enhanced HUD with better visibility
    status = f"sus1:{int(sus1)} sus2:{int(sus2)} sus3:{int(sus3)} {'🚨 SUSPICIOUS 🚨' if suspicious_now else 'NORMAL'}"
    counts = f"interactions:{interactions_in_frame} hands:{hands_in_frame}"
    
    # Draw background rectangles for better text visibility
    status_color = (0, 0, 255) if suspicious_now else (255, 255, 255)
    status_font_scale = 0.8
    status_thickness = 3 if suspicious_now else 2
    
    # Status text background
    status_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, status_font_scale, status_thickness)[0]
    cv2.rectangle(frame, (5, 5), (15 + status_size[0], 35), (0, 0, 0), -1)  # Black background
    if suspicious_now:
        cv2.rectangle(frame, (5, 5), (15 + status_size[0], 35), (0, 0, 255), 3)  # Red border for suspicious
    
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, status_font_scale, status_color, status_thickness)
    
    # Counts text background
    counts_size = cv2.getTextSize(counts, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(frame, (5, 40), (15 + counts_size[0], 65), (0, 0, 0), -1)  # Black background
    cv2.putText(frame, counts, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)


def process_frame(
    frame,
    persons: List[BBox],
    hands: List[BBox],
    objects: List[BBox],
    poses: List[Pose],
    tracker: SuspicionTracker,
    dist_thresh_px: float,
    out_dir: Optional[str],
    frame_idx: int,
    display: bool,
    pocket_padding: int = 25,
) -> FrameResult:
    pocket_rois_all: List[BBox] = []
    hand_keypoints: List[Point] = []

    for pose in poses:
        pocket_rois_all.extend(find_pocket_rois(pose, padding=pocket_padding))
        # collect hand keypoints (wrists) if present
        lw, rw = pose.get(9), pose.get(10)
        if lw:
            hand_keypoints.append(lw)
        if rw:
            hand_keypoints.append(rw)

    interactions = check_interactions(hand_keypoints, hands, objects, pocket_rois_all, dist_thresh_px)
    tracker.update_interactions(interactions)

    hands_in_frame = len(hands)
    tracker.update_hands_presence(hands_in_frame)

    low_elbow = any(any_elbow_low_angle(p, tracker.elbow_angle_threshold_deg) for p in poses)
    tracker.update_elbow_angle(low_elbow)

    # More reasonable immediate suspicious condition for visual display
    # Show as suspicious if has strong activity in any area plus some activity in another
    # This matches the logging logic better than requiring ALL 3 simultaneously
    suspicious_now = (tracker.sus1 and (tracker.sus2 or tracker.sus3)) or \
                    (tracker.sus2 and (tracker.sus1 or tracker.sus3)) or \
                    (tracker.sus3 and (tracker.sus1 or tracker.sus2))

    draw_overlays(
        frame=frame,
        persons=persons,
        hands=hands,
        objects=objects,
        rois=pocket_rois_all,
        poses=poses,
        interactions_in_frame=interactions,
        hands_in_frame=hands_in_frame,
        sus1=tracker.sus1,
        sus2=tracker.sus2,
        sus3=tracker.sus3,
        suspicious_now=suspicious_now,
    )

    tracker.add_frame(frame)

    if suspicious_now and out_dir:
        tracker.dump_suspicious(out_dir, frame_idx)

    if display:
        cv2.imshow("stream", frame)
        cv2.waitKey(1)

    return FrameResult(
        interactions_in_frame=interactions,
        hands_in_frame=hands_in_frame,
        sus1=tracker.sus1,
        sus2=tracker.sus2,
        sus3=tracker.sus3,
        suspicious_now=suspicious_now,
    )


def process_frame_person_specific(
    frame,
    persons_tracked: List[BBox],
    hands: List[BBox],
    objects: List[BBox],
    poses: List[Pose],
    manager: MultiPersonSuspicion,
    dist_thresh_px: float,
    out_dir: Optional[str],
    frame_idx: int,
    display: bool,
    pocket_padding: int = 25,
) -> List[PersonResult]:
    results: List[PersonResult] = []
    active_ids = [p.track_id for p in persons_tracked if p.track_id is not None]

    for person in persons_tracked:
        if person.track_id is None:
            continue
        tracker = manager.get(person.track_id)

        pose = assign_pose_to_person(person, poses)
        pocket_rois: List[BBox] = find_pocket_rois(pose, padding=pocket_padding) if pose else []
        hand_keypoints: List[Point] = []
        if pose:
            lw, rw = pose.get(9), pose.get(10)
            if lw: hand_keypoints.append(lw)
            if rw: hand_keypoints.append(rw)

        hands_owned = assign_hands_to_person(person, hands)
        # Optionally restrict objects to nearest few if you want per-person objects
        interactions = check_interactions(hand_keypoints, hands_owned, objects, pocket_rois, dist_thresh_px)
        tracker.update_interactions(interactions)

        hands_count = len(hands_owned)
        tracker.update_hands_presence(hands_count)

        low_elbow = any_elbow_low_angle(pose, tracker.elbow_angle_threshold_deg) if pose else False
        tracker.update_elbow_angle(low_elbow)

        # More reasonable immediate suspicious condition for visual display
        # Show as suspicious if person has strong activity in any area
        # This matches the logging logic better than requiring ALL 3 simultaneously
        suspicious_now = (tracker.sus1 and (tracker.sus2 or tracker.sus3)) or \
                        (tracker.sus2 and (tracker.sus1 or tracker.sus3)) or \
                        (tracker.sus3 and (tracker.sus1 or tracker.sus2))

        # Draw overlays per person - bright red/thick for suspicious, blue for normal
        if suspicious_now:
            person_color = (0, 0, 255)  # Bright red for suspicious
            person_thickness = 6  # Much thicker for suspicious persons
            person_label = f"🚨 SUSPICIOUS #{person.track_id} 🚨"
        else:
            person_color = (255, 0, 0)  # Blue for normal
            person_thickness = 2  # Normal thickness
            person_label = f"person#{person.track_id}"
        
        _draw_box(frame, person, person_color, person_label, thickness=person_thickness)
        for h in hands_owned:
            _draw_box(frame, h, (0, 255, 0), h.label or "hand")
        for r in pocket_rois:
            _draw_box(frame, r, (255, 0, 255), r.label or "pocket")
        if pose:
            _draw_pose(frame, pose, pocket_rois)

        # Enhanced text overlay for suspicious persons
        text_color = (0, 0, 255) if suspicious_now else (255, 255, 255)
        text_bg_color = (255, 255, 255) if suspicious_now else (0, 0, 0)
        
        # Main status text
        main_text = f"ID:{person.track_id} s1:{int(tracker.sus1)} s2:{int(tracker.sus2)} s3:{int(tracker.sus3)} S:{int(suspicious_now)}"
        
        # Add prominent warning banner for suspicious persons
        if suspicious_now:
            warning_text = f"🚨 SUSPICIOUS PERSON {person.track_id} 🚨"
            # Use larger, more visible font
            font_scale = 0.8
            font_thickness = 3
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            
            # Draw bright red background rectangle for warning with padding
            banner_y1 = max(0, person.y1 - 45)
            banner_y2 = max(0, person.y1 - 5)
            cv2.rectangle(frame, 
                         (person.x1 - 5, banner_y1), 
                         (person.x1 + text_size[0] + 15, banner_y2), 
                         (0, 0, 255), -1)  # Bright red background
            
            # Draw white border around banner for extra visibility
            cv2.rectangle(frame, 
                         (person.x1 - 5, banner_y1), 
                         (person.x1 + text_size[0] + 15, banner_y2), 
                         (255, 255, 255), 2)  # White border
                         
            # Draw bright white text
            cv2.putText(frame, warning_text,
                       (person.x1, max(0, person.y1 - 15)),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        # Draw main status text with background for better visibility
        status_font_scale = 0.5
        status_font_thickness = 2 if suspicious_now else 1
        status_text_size = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, status_font_scale, status_font_thickness)[0]
        status_y = max(0, person.y1 - 50) if suspicious_now else max(0, person.y1 - 10)
        
        # Draw background rectangle for status text
        cv2.rectangle(frame,
                     (person.x1 - 2, status_y - status_text_size[1] - 2),
                     (person.x1 + status_text_size[0] + 2, status_y + 2),
                     (0, 0, 0), -1)  # Black background
        
        cv2.putText(
            frame, main_text,
            (person.x1, status_y),
            cv2.FONT_HERSHEY_SIMPLEX, status_font_scale,
            text_color, status_font_thickness
        )

        if suspicious_now and out_dir:
            tracker.dump_suspicious(out_dir, frame_idx)

        results.append(PersonResult(
            track_id=person.track_id,
            interactions=interactions,
            hands_in_frame=hands_count,
            sus1=tracker.sus1,
            sus2=tracker.sus2,
            sus3=tracker.sus3,
            suspicious_now=suspicious_now,
        ))

        tracker.add_frame(frame)

    manager.prune([tid for tid in active_ids if tid is not None])

    if display:
        cv2.imshow("stream", frame)
        cv2.waitKey(1)

    return results
