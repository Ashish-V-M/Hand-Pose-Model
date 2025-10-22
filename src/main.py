import argparse
import os
import time
import cv2
import glob
import shutil
import logging
from datetime import datetime
from pathlib import Path
from src.models.adapters import HOIDetector, PoseEstimator
from src.config import HOI_MODEL_PATH, POSE_MODEL_PATH, DEVICE, POSE_CONF_THRESHOLD, POCKET_PADDING_PX
from src.pipeline.processor import SuspicionTracker, process_frame, MultiPersonSuspicion, process_frame_person_specific
from src.pipeline.tracking import SimpleIOUTracker
from src.analyzer import VideoAnalyzer


def parse_args():
    ap = argparse.ArgumentParser(description="Hand/Object Interaction Suspicion Pipeline")
    ap.add_argument("--source", type=str, required=True, help="Video path/RTSP URL or folder containing videos")
    ap.add_argument("--output", type=str, default="output", help="Directory to save suspicious frames")
    ap.add_argument("--output_videos", type=str, default="output_videos", help="Directory to save processed videos")
    ap.add_argument("--display", type=int, default=0, help="Show window (1) or not (0)")
    ap.add_argument("--save_video", type=int, default=1, help="Save processed video (1) or not (0)")
    ap.add_argument("--batch_process", type=int, default=0, help="Process all videos in folder (1) or single video (0)")
    ap.add_argument("--interaction_threshold", type=int, default=3)
    ap.add_argument("--no_interaction_reset_frames", type=int, default=3)
    ap.add_argument("--hands_presence_needed", type=int, default=5)
    ap.add_argument("--hands_absent_trigger_frames", type=int, default=3)
    ap.add_argument("--elbow_angle_threshold_deg", type=float, default=160.0)
    ap.add_argument("--elbow_low_min_frames", type=int, default=3)
    ap.add_argument("--elbow_reset_frames", type=int, default=2)
    ap.add_argument("--dist_thresh_px", type=float, default=50.0)
    ap.add_argument("--print_every", type=int, default=30, help="Print sus states every N frames (0 to disable)")
    ap.add_argument("--person_specific", type=int, default=1, help="Use person-specific tracking (1) or frame-level (0)")
    ap.add_argument("--pose_conf_threshold", type=float, default=POSE_CONF_THRESHOLD, help="Minimum confidence for pose keypoints")
    ap.add_argument("--pocket_padding", type=int, default=POCKET_PADDING_PX, help="Padding in pixels to expand pocket regions")
    ap.add_argument("--sus_threshold", type=float, default=10.0, help="[DEPRECATED] Percentage threshold - now all SUS conditions use FPS-based thresholds")
    ap.add_argument("--sus1_fps_multiplier", type=float, default=1.0, help="FPS multiplier for SUS1 threshold (default: 1.0 = 1 second)")
    ap.add_argument("--sus1_reset_frames", type=int, default=3, help="[IGNORED] SUS1 no longer resets - accumulates for theft detection")
    ap.add_argument("--sus2_fps_multiplier", type=float, default=1.0, help="FPS multiplier for SUS2 threshold (default: 1.0 = 1 second)")
    ap.add_argument("--sus2_reset_frames", type=int, default=3, help="Frames of inactivity before SUS2 window resets (default: 3)")
    ap.add_argument("--sus3_fps_multiplier", type=float, default=1.0, help="[IGNORED] SUS3 uses fixed 3-frame threshold")
    ap.add_argument("--sus3_reset_frames", type=int, default=3, help="Frames of inactivity before SUS3 window resets (default: 3)")
    ap.add_argument("--log_file", type=str, default="suspicious_events.log", help="Log file path for recording suspicious events")
    ap.add_argument("--device", type=str, default=None, help="Override device (e.g., 'cuda:0', 'cpu'). If not specified, auto-detects best device")
    ap.add_argument("--analyze", type=int, default=0, help="Run analysis mode (1) or detection mode (0)")
    ap.add_argument("--analysis_output", type=str, default="analysis_output", help="Output directory for analysis results")
    return ap.parse_args()


def get_video_files(source_path):
    """Get all video files from a directory"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
    video_files = []
    
    if os.path.isfile(source_path):
        return [source_path]
    elif os.path.isdir(source_path):
        # Get all files in directory and filter by extension
        all_files = glob.glob(os.path.join(source_path, "*"))
        print(f"Debug: Found {len(all_files)} total items in directory")
        
        for file_path in all_files:
            if os.path.isfile(file_path):
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext in video_extensions:
                    video_files.append(file_path)
        
        print(f"Debug: Found {len(video_files)} video files before deduplication")
        
        # Remove duplicates and sort
        video_files = list(set(video_files))
        print(f"Debug: Found {len(video_files)} unique video files after deduplication")
        
        return sorted(video_files)
    else:
        return [source_path]  # Assume it's a URL or stream


def setup_suspicious_logging(log_file_path):
    """Setup logging for suspicious events"""
    # Create a logger for suspicious events
    sus_logger = logging.getLogger('suspicious_events')
    sus_logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in sus_logger.handlers[:]:
        sus_logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    sus_logger.addHandler(file_handler)
    
    # Add header if this is a new log file
    if not os.path.exists(log_file_path) or os.path.getsize(log_file_path) == 0:
        sus_logger.info("=== SUSPICIOUS EVENTS LOG ===")
        sus_logger.info("Format: TIMESTAMP | VIDEO: filename | PERSON: ID | FRAME: number | VIDEO_TIME: MM:SS.sss | SUS1: bool | SUS2: bool | SUS3: bool | REASON: detection_type")
        sus_logger.info("=" * 120)
    
    return sus_logger


def log_suspicious_event(logger, video_name, person_id, frame_idx, fps, sus1, sus2, sus3, reason="immediate"):
    """Log a suspicious event with details"""
    if logger is None:
        return
    
    # Calculate timestamp in video (more detailed format)
    timestamp_seconds = frame_idx / fps if fps > 0 else 0
    hours = int(timestamp_seconds // 3600)
    minutes = int((timestamp_seconds % 3600) // 60)
    seconds = timestamp_seconds % 60
    
    # Format timestamp as HH:MM:SS.MS for better readability
    if hours > 0:
        video_time = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    else:
        video_time = f"{minutes:02d}:{seconds:06.3f}"
    
    log_entry = (
        f"VIDEO: {video_name} | "
        f"PERSON: {person_id} | "
        f"FRAME: {frame_idx} | "
        f"VIDEO_TIME: {video_time} | "
        f"SUS1: {sus1} | SUS2: {sus2} | SUS3: {sus3} | "
        f"REASON: {reason}"
    )
    
    # Also print to console for immediate feedback
    print(f"📝 LOGGED: Person {person_id} suspicious at {video_time} in {video_name}")
    
    logger.info(log_entry)


def process_single_video(
    video_path, 
    hoi_detector, 
    pose_estimator, 
    person_tracker, 
    mp_manager,
    tracker,
    args,
    sus_logger=None
):
    """Process a single video file"""
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(video_path)}")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    video_name = os.path.basename(video_path)
    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Setup video writer if saving video
    video_writer = None
    if args.save_video:
        os.makedirs(args.output_videos, exist_ok=True)
        video_name = Path(video_path).stem
        output_video_path = os.path.join(args.output_videos, f"{video_name}_processed.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (640, 640))  # Resized dimensions
        print(f"Saving processed video to: {output_video_path}")
    
    # Create output directory for suspicious frames
    video_output_dir = os.path.join(args.output, Path(video_path).stem)
    os.makedirs(video_output_dir, exist_ok=True)
    
    frame_idx = 0
    start_time = time.time()
    video_has_suspicious_behavior = False  # Track if any suspicious behavior occurred in the entire video
    
    # Track individual sus conditions throughout the video
    sus1_frame_count = 0
    sus2_frame_count = 0  
    sus3_frame_count = 0
    total_frames_processed = 0
    
    # Windowed tracking for all SUS conditions (similar to original SUS1 logic)
    # SUS1 windowed tracking
    sus1_current_streak = 0  # Current consecutive frames with sus1=True
    sus1_no_activity_streak = 0  # Consecutive frames with sus1=False
    sus1_threshold_met = False  # Whether FPS threshold was met in current window
    sus1_reset_frames = args.sus1_reset_frames  # Frames of inactivity before reset
    
    # SUS2 windowed tracking
    sus2_current_streak = 0  # Current consecutive frames with sus2=True
    sus2_no_activity_streak = 0  # Consecutive frames with sus2=False
    sus2_threshold_met = False  # Whether FPS threshold was met in current window
    sus2_reset_frames = args.sus2_reset_frames  # Frames of inactivity before reset
    
    # SUS3 windowed tracking
    sus3_current_streak = 0  # Current consecutive frames with sus3=True
    sus3_no_activity_streak = 0  # Consecutive frames with sus3=False
    sus3_threshold_met = False  # Whether FPS threshold was met in current window
    sus3_reset_frames = args.sus3_reset_frames  # Frames of inactivity before reset
    
    # Track suspicious persons (for person-specific mode)
    suspicious_persons = set()  # Track IDs of persons who showed suspicious behavior
    person_sus_details = {}  # Track detailed sus info per person
    person_first_sus_frame = {}  # Track the first frame where each person became suspicious
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Resize frame to 640x640
            frame = cv2.resize(frame, (640, 640))
            
            persons, hands, objects = hoi_detector.predict(frame)
            poses = pose_estimator.predict(frame)

            if args.person_specific:
                # Person-specific tracking
                persons_tracked = person_tracker.update(persons)
                
                per_person = process_frame_person_specific(
                    frame=frame,
                    persons_tracked=persons_tracked,
                    hands=hands,
                    objects=objects,
                    poses=poses,
                    manager=mp_manager,
                    dist_thresh_px=args.dist_thresh_px,
                    out_dir=video_output_dir,
                    frame_idx=frame_idx,
                    display=bool(args.display),
                    pocket_padding=args.pocket_padding,
                )

                # Track individual sus conditions for person-specific mode
                if per_person:
                    frame_has_sus1 = any(r.sus1 for r in per_person)
                    frame_has_sus2 = any(r.sus2 for r in per_person)
                    frame_has_sus3 = any(r.sus3 for r in per_person)
                    
                    # Windowed logic for all SUS conditions
                    # SUS1 accumulation logic (NO RESET for theft detection)
                    if frame_has_sus1:
                        sus1_current_streak += 1  # Track current streak for reporting
                        sus1_no_activity_streak = 0
                        sus1_frame_count += 1  # Keep total count for reporting
                        
                        # Check if we've met the FPS threshold (total accumulated frames)
                        sus1_fps_threshold = max(1, int(fps * args.sus1_fps_multiplier))
                        if sus1_frame_count >= sus1_fps_threshold and not sus1_threshold_met:
                            sus1_threshold_met = True
                            print(f"[Frame {frame_idx}] SUS1 threshold met: {sus1_frame_count} total frames (≥{sus1_fps_threshold}) - NO RESET for theft detection")
                    else:
                        sus1_no_activity_streak += 1
                        sus1_current_streak = 0  # Reset streak counter for reporting
                        
                        # NO RESET LOGIC - SUS1 accumulates throughout entire video for theft detection
                        # Once threshold is met, it stays met for the entire video
                    
                    # SUS2 windowed logic
                    if frame_has_sus2:
                        sus2_current_streak += 1
                        sus2_no_activity_streak = 0
                        sus2_frame_count += 1  # Keep total count for reporting
                        
                        # Check if we've met the FPS threshold in current window
                        sus2_fps_threshold = max(1, int(fps * args.sus2_fps_multiplier))
                        if sus2_current_streak >= sus2_fps_threshold and not sus2_threshold_met:
                            sus2_threshold_met = True
                            print(f"[Frame {frame_idx}] SUS2 threshold met: {sus2_current_streak} consecutive frames (≥{sus2_fps_threshold})")
                    else:
                        sus2_no_activity_streak += 1
                        sus2_current_streak = 0  # Reset streak counter
                        
                        # Reset window if no activity for specified frames
                        if sus2_no_activity_streak >= sus2_reset_frames:
                            if sus2_threshold_met:
                                print(f"[Frame {frame_idx}] SUS2 window reset after {sus2_no_activity_streak} inactive frames")
                            sus2_threshold_met = False  # Reset threshold flag for new window
                    
                    # SUS3 windowed logic (fixed 3-frame threshold)
                    if frame_has_sus3:
                        sus3_current_streak += 1
                        sus3_no_activity_streak = 0
                        sus3_frame_count += 1  # Keep total count for reporting
                        
                        # Check if we've met the fixed 3-frame threshold
                        sus3_fixed_threshold = 3  # Always 3 frames for SUS3
                        if sus3_current_streak >= sus3_fixed_threshold and not sus3_threshold_met:
                            sus3_threshold_met = True
                            print(f"[Frame {frame_idx}] SUS3 threshold met: {sus3_current_streak} consecutive frames (≥{sus3_fixed_threshold})")
                    else:
                        sus3_no_activity_streak += 1
                        sus3_current_streak = 0  # Reset streak counter
                        
                        # Reset window if no activity for specified frames
                        if sus3_no_activity_streak >= sus3_reset_frames:
                            if sus3_threshold_met:
                                print(f"[Frame {frame_idx}] SUS3 window reset after {sus3_no_activity_streak} inactive frames")
                            sus3_threshold_met = False  # Reset threshold flag for new window
                    
                    # Track per-person suspicious behavior details
                    for r in per_person:
                        person_id = r.track_id
                        if person_id not in person_sus_details:
                            person_sus_details[person_id] = {
                                'sus1_frames': 0, 'sus2_frames': 0, 'sus3_frames': 0,
                                'total_frames': 0, 'immediate_sus_frames': 0
                            }
                        
                        person_sus_details[person_id]['total_frames'] += 1
                        
                        # Track first suspicious frame for this person (any condition)
                        if (r.sus1 or r.sus2 or r.sus3) and person_id not in person_first_sus_frame:
                            person_first_sus_frame[person_id] = frame_idx
                            
                        if r.sus1:
                            person_sus_details[person_id]['sus1_frames'] += 1
                        if r.sus2:
                            person_sus_details[person_id]['sus2_frames'] += 1
                        if r.sus3:
                            person_sus_details[person_id]['sus3_frames'] += 1
                        if r.suspicious_now:
                            person_sus_details[person_id]['immediate_sus_frames'] += 1
                            suspicious_persons.add(person_id)
                            print(f"🚨 Person {person_id} flagged as SUSPICIOUS in frame {frame_idx}!")
                            
                            # Log the suspicious event
                            log_suspicious_event(
                                sus_logger, video_name, person_id, frame_idx, fps,
                                r.sus1, r.sus2, r.sus3, "immediate"
                            )
                    
                    # Keep old logic for immediate suspicious frame detection
                    frame_is_suspicious = any(r.suspicious_now for r in per_person)
                    if frame_is_suspicious:
                        video_has_suspicious_behavior = True

                if args.print_every > 0 and frame_idx % args.print_every == 0:
                    if per_person:
                        print(f"Frame {frame_idx}:")
                        for r in per_person:
                            print(f"  Person {r.track_id}: sus1={r.sus1} sus2={r.sus2} sus3={r.sus3} SUS={r.suspicious_now}")
                    else:
                        print(f"Frame {frame_idx}: No persons detected")
            else:
                # Original frame-level tracking
                _ = process_frame(
                    frame=frame,
                    persons=persons,
                    hands=hands,
                    objects=objects,
                    poses=poses,
                    tracker=tracker,
                    dist_thresh_px=args.dist_thresh_px,
                    out_dir=video_output_dir,
                    frame_idx=frame_idx,
                    display=bool(args.display),
                    pocket_padding=args.pocket_padding,
                )

                # Track individual sus conditions for frame-level mode
                # Mixed logic: SUS1 accumulation, SUS2/SUS3 windowed
                # SUS1 accumulation logic (NO RESET for theft detection)
                if tracker.sus1:
                    sus1_current_streak += 1  # Track current streak for reporting
                    sus1_no_activity_streak = 0
                    sus1_frame_count += 1  # Keep total count for reporting
                    
                    # Check if we've met the FPS threshold (total accumulated frames)
                    sus1_fps_threshold = max(1, int(fps * args.sus1_fps_multiplier))
                    if sus1_frame_count >= sus1_fps_threshold and not sus1_threshold_met:
                        sus1_threshold_met = True
                        print(f"[Frame {frame_idx}] SUS1 threshold met: {sus1_frame_count} total frames (≥{sus1_fps_threshold}) - NO RESET for theft detection")
                else:
                    sus1_no_activity_streak += 1
                    sus1_current_streak = 0  # Reset streak counter for reporting
                    
                    # NO RESET LOGIC - SUS1 accumulates throughout entire video for theft detection
                    # Once threshold is met, it stays met for the entire video
                
                # SUS2 windowed logic
                if tracker.sus2:
                    sus2_current_streak += 1
                    sus2_no_activity_streak = 0
                    sus2_frame_count += 1  # Keep total count for reporting
                    
                    # Check if we've met the FPS threshold in current window
                    sus2_fps_threshold = max(1, int(fps * args.sus2_fps_multiplier))
                    if sus2_current_streak >= sus2_fps_threshold and not sus2_threshold_met:
                        sus2_threshold_met = True
                        print(f"[Frame {frame_idx}] SUS2 threshold met: {sus2_current_streak} consecutive frames (≥{sus2_fps_threshold})")
                else:
                    sus2_no_activity_streak += 1
                    sus2_current_streak = 0  # Reset streak counter
                    
                    # Reset window if no activity for specified frames
                    if sus2_no_activity_streak >= sus2_reset_frames:
                        if sus2_threshold_met:
                            print(f"[Frame {frame_idx}] SUS2 window reset after {sus2_no_activity_streak} inactive frames")
                        sus2_threshold_met = False  # Reset threshold flag for new window
                
                # SUS3 windowed logic (fixed 3-frame threshold)
                if tracker.sus3:
                    sus3_current_streak += 1
                    sus3_no_activity_streak = 0
                    sus3_frame_count += 1  # Keep total count for reporting
                    
                    # Check if we've met the fixed 3-frame threshold
                    sus3_fixed_threshold = 3  # Always 3 frames for SUS3
                    if sus3_current_streak >= sus3_fixed_threshold and not sus3_threshold_met:
                        sus3_threshold_met = True
                        print(f"[Frame {frame_idx}] SUS3 threshold met: {sus3_current_streak} consecutive frames (≥{sus3_fixed_threshold})")
                else:
                    sus3_no_activity_streak += 1
                    sus3_current_streak = 0  # Reset streak counter
                    
                    # Reset window if no activity for specified frames
                    if sus3_no_activity_streak >= sus3_reset_frames:
                        if sus3_threshold_met:
                            print(f"[Frame {frame_idx}] SUS3 window reset after {sus3_no_activity_streak} inactive frames")
                        sus3_threshold_met = False  # Reset threshold flag for new window
                
                # Keep old logic for immediate suspicious frame detection
                frame_is_suspicious = tracker.sus1 and tracker.sus2 and tracker.sus3
                if frame_is_suspicious:
                    video_has_suspicious_behavior = True

                if args.print_every > 0 and frame_idx % args.print_every == 0:
                    print(f"Frame {frame_idx}: sus1={tracker.sus1} sus2={tracker.sus2} sus3={tracker.sus3} SUS={frame_is_suspicious}")

            # Write frame to output video
            if video_writer is not None:
                video_writer.write(frame)
            
            frame_idx += 1
            total_frames_processed += 1
            
            # Progress update
            if frame_idx % (fps * 5) == 0:  # Every 5 seconds
                elapsed = time.time() - start_time
                progress = (frame_idx / total_frames) * 100
                fps_processed = frame_idx / elapsed
                print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames}) - {fps_processed:.1f} fps")
                
    except KeyboardInterrupt:
        print(f"\nInterrupted by user at frame {frame_idx}")
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
    
    elapsed_time = time.time() - start_time
    avg_fps = frame_idx / elapsed_time if elapsed_time > 0 else 0
    print(f"Completed: {frame_idx} frames in {elapsed_time:.1f}s (avg {avg_fps:.1f} fps)")
    
    # [DEPRECATED] Old percentage thresholds - kept for potential legacy compatibility
    sus_threshold_percentage = args.sus_threshold  # No longer used for thresholds
    min_frames_for_sus = max(1, int(total_frames_processed * sus_threshold_percentage / 100))  # No longer used
    
    sus1_percentage = (sus1_frame_count / total_frames_processed * 100) if total_frames_processed > 0 else 0
    sus2_percentage = (sus2_frame_count / total_frames_processed * 100) if total_frames_processed > 0 else 0
    sus3_percentage = (sus3_frame_count / total_frames_processed * 100) if total_frames_processed > 0 else 0
    
    # Check which conditions meet their respective thresholds
    # SUS1: Accumulation (no reset), SUS2: Windowed, SUS3: Fixed 3-frame threshold
    sus1_fps_threshold = max(1, int(fps * args.sus1_fps_multiplier))
    sus2_fps_threshold = max(1, int(fps * args.sus2_fps_multiplier))
    sus3_fixed_threshold = 3  # Fixed threshold for SUS3
    
    sus1_meets_threshold = sus1_threshold_met  # Use accumulation logic result (no reset)
    sus2_meets_threshold = sus2_threshold_met  # Use windowed logic result
    sus3_meets_threshold = sus3_threshold_met  # Use windowed logic result
    
    # Check which conditions have at least 1 frame
    sus1_has_activity = sus1_frame_count >= 1
    sus2_has_activity = sus2_frame_count >= 1
    sus3_has_activity = sus3_frame_count >= 1
    
    # New combined logic: One condition ≥threshold AND at least one other condition ≥1 frame
    sus_by_combined_logic = False
    combination_details = []
    
    if sus1_meets_threshold and (sus2_has_activity or sus3_has_activity):
        sus_by_combined_logic = True
        other_conditions = []
        if sus2_has_activity:
            other_conditions.append(f"SUS2({sus2_frame_count})")
        if sus3_has_activity:
            other_conditions.append(f"SUS3({sus3_frame_count})")
        combination_details.append(f"SUS1≥ACCUMULATED + {'+'.join(other_conditions)}≥1")
        
    if sus2_meets_threshold and (sus1_has_activity or sus3_has_activity):
        sus_by_combined_logic = True
        other_conditions = []
        if sus1_has_activity:
            other_conditions.append(f"SUS1({sus1_frame_count})")
        if sus3_has_activity:
            other_conditions.append(f"SUS3({sus3_frame_count})")
        combination_details.append(f"SUS2≥WINDOWED + {'+'.join(other_conditions)}≥1")
        
    if sus3_meets_threshold and (sus1_has_activity or sus2_has_activity):
        sus_by_combined_logic = True
        other_conditions = []
        if sus1_has_activity:
            other_conditions.append(f"SUS1({sus1_frame_count})")
        if sus2_has_activity:
            other_conditions.append(f"SUS2({sus2_frame_count})")
        combination_details.append(f"SUS3≥WINDOWED + {'+'.join(other_conditions)}≥1")
    
    print(f"Suspicion Analysis:")
    print(f"  SUS1 (interactions): {sus1_frame_count}/{total_frames_processed} frames ({sus1_percentage:.1f}%) {'✓ACCUMULATED' if sus1_meets_threshold else '✓≥1' if sus1_has_activity else '✗'}")
    print(f"  SUS2 (hands absent): {sus2_frame_count}/{total_frames_processed} frames ({sus2_percentage:.1f}%) {'✓WINDOWED' if sus2_meets_threshold else '✓≥1' if sus2_has_activity else '✗'}")
    print(f"  SUS3 (low elbow): {sus3_frame_count}/{total_frames_processed} frames ({sus3_percentage:.1f}%) {'✓WINDOWED' if sus3_meets_threshold else '✓≥1' if sus3_has_activity else '✗'}")
    print(f"  Thresholds for suspicion:")
    print(f"    SUS1: {sus1_fps_threshold} total frames ({args.sus1_fps_multiplier} seconds @ {fps} FPS) [NO RESET - accumulates for theft detection]")
    print(f"    SUS2: {sus2_fps_threshold} consecutive frames ({args.sus2_fps_multiplier} seconds @ {fps} FPS) [resets after {sus2_reset_frames} inactive frames]")
    print(f"    SUS3: {sus3_fixed_threshold} consecutive frames (FIXED threshold) [resets after {sus3_reset_frames} inactive frames]")
    print(f"  Combined logic: Any major condition + 1 frame from another condition")
    
    # Update suspicious behavior flag with new combined logic
    video_has_suspicious_behavior = video_has_suspicious_behavior or sus_by_combined_logic
    
    if sus_by_combined_logic:
        print(f"  🚨 Video flagged as SUSPICIOUS by combined logic: {' | '.join(combination_details)}")
    elif video_has_suspicious_behavior:
        print(f"  🚨 Video flagged as SUSPICIOUS by immediate detection!")
    else:
        print(f"  ✓ Video is CLEAN - insufficient suspicious behavior")
    
    # Detailed per-person analysis (for person-specific mode)
    if args.person_specific and person_sus_details:
        print(f"\nPer-Person Analysis:")
        for person_id, details in person_sus_details.items():
            total_frames = details['total_frames']
            if total_frames > 0:
                sus1_pct = (details['sus1_frames'] / total_frames) * 100
                sus2_pct = (details['sus2_frames'] / total_frames) * 100
                sus3_pct = (details['sus3_frames'] / total_frames) * 100
                immediate_pct = (details['immediate_sus_frames'] / total_frames) * 100
                
                # Check if this person individually meets the combined threshold
                min_frames_person = max(1, int(total_frames * args.sus_threshold / 100))
                # SUS1: FPS-based threshold for this person
                person_sus1_meets = details['sus1_frames'] >= sus1_fps_threshold
                person_sus2_meets = details['sus2_frames'] >= min_frames_person  
                person_sus3_meets = details['sus3_frames'] >= min_frames_person
                person_has_sus1 = details['sus1_frames'] >= 1
                person_has_sus2 = details['sus2_frames'] >= 1
                person_has_sus3 = details['sus3_frames'] >= 1
                
                person_meets_combined = False
                person_reasons = []
                if person_sus1_meets and (person_has_sus2 or person_has_sus3):
                    person_meets_combined = True
                    person_reasons.append("SUS1≥FPS")
                if person_sus2_meets and (person_has_sus1 or person_has_sus3):
                    person_meets_combined = True
                    person_reasons.append("SUS2≥10%")
                if person_sus3_meets and (person_has_sus1 or person_has_sus2):
                    person_meets_combined = True
                    person_reasons.append("SUS3≥10%")
                
                status_icon = "🚨" if person_meets_combined or person_id in suspicious_persons else "✓"
                status_text = "SUSPICIOUS" if person_meets_combined or person_id in suspicious_persons else "CLEAN"
                
                print(f"  {status_icon} Person {person_id}: {status_text} ({total_frames} frames)")
                print(f"    SUS1: {details['sus1_frames']} frames ({sus1_pct:.1f}%)")
                print(f"    SUS2: {details['sus2_frames']} frames ({sus2_pct:.1f}%)")
                print(f"    SUS3: {details['sus3_frames']} frames ({sus3_pct:.1f}%)")
                if immediate_pct > 0:
                    print(f"    Immediate detections: {details['immediate_sus_frames']} frames ({immediate_pct:.1f}%)")
                if person_reasons:
                    print(f"    Flagged by: {', '.join(person_reasons)}")
                
                # Log percentage-based detection
                if person_meets_combined and person_id not in suspicious_persons:
                    # This person was flagged by percentage analysis, not immediate detection
                    # Use the first frame where this person showed suspicious behavior
                    first_sus_frame = person_first_sus_frame.get(person_id, total_frames_processed)
                    log_suspicious_event(
                        sus_logger, video_name, person_id, first_sus_frame, fps,
                        person_has_sus1, person_has_sus2, person_has_sus3, 
                        f"percentage_analysis_{'+'.join(person_reasons)}"
                    )
        
        if suspicious_persons:
            print(f"\n🚨 SUSPICIOUS PERSONS DETECTED: {', '.join(map(str, sorted(suspicious_persons)))}")
        else:
            print(f"\n✓ No suspicious persons detected")
    
    # Move suspicious videos to SUS folder
    if video_has_suspicious_behavior and args.save_video:
        # Create SUS folder
        sus_folder = os.path.join(args.output_videos, "SUS")
        os.makedirs(sus_folder, exist_ok=True)
        
        # Move the video to SUS folder (video_writer is already released in finally block)
        original_path = output_video_path
        sus_path = os.path.join(sus_folder, os.path.basename(output_video_path))
        
        try:
            if os.path.exists(original_path):
                shutil.move(original_path, sus_path)
                print(f"🚨 SUSPICIOUS VIDEO detected! Moved to: {sus_path}")
            else:
                print(f"Warning: Could not find video file to move: {original_path}")
        except Exception as e:
            print(f"Warning: Failed to move suspicious video: {e}")
    
    return True


def main():
    args = parse_args()

    # Determine device to use
    if args.device:
        device = args.device
        print(f"Using specified device: {device}")
    else:
        device = DEVICE
    
    print(f"Initializing models on device: {device}")
    
    # Initialize models from config
    hoi_detector = HOIDetector(model_path=HOI_MODEL_PATH, device=device)
    pose_estimator = PoseEstimator(model_path=POSE_MODEL_PATH, device=device, conf_threshold=args.pose_conf_threshold)
    
    # Check if we should run in analysis mode
    if args.analyze:
        print("Running in ANALYSIS MODE")
        print("=" * 50)
        
        # Create analyzer
        analyzer = VideoAnalyzer(
            hoi_detector=hoi_detector,
            pose_estimator=pose_estimator,
            dist_thresh_px=args.dist_thresh_px,
            pocket_padding=args.pocket_padding
        )
        
        # Run analysis
        try:
            if os.path.isfile(args.source):
                # Single video analysis
                print(f"Analyzing single video: {args.source}")
                result = analyzer.analyze_video(args.source, args.analysis_output)
                
                print(f"\nAnalysis Results:")
                print(f"  Video: {result.video_name}")
                print(f"  Total frames: {result.total_frames}")
                print(f"  Hand presence: {result.hand_presence_percentage:.1f}%")
                print(f"  Total interactions: {result.interaction_count}")
                print(f"  Frames with hands: {result.frames_with_hands}")
                print(f"  Frames with interactions: {result.frames_with_interactions}")
                
            elif os.path.isdir(args.source):
                # Folder analysis
                print(f"Analyzing folder: {args.source}")
                results = analyzer.analyze_folder(args.source, args.analysis_output)
                
                print(f"\nAnalysis Complete!")
                print(f"  Videos analyzed: {len(results)}")
                print(f"  Results saved to: {args.analysis_output}")
                
                # Print summary
                for result in results:
                    print(f"  {result.video_name}: {result.hand_presence_percentage:.1f}% hands, {result.interaction_count} interactions")
            else:
                print(f"Error: {args.source} is not a valid file or directory")
                return
                
        except KeyboardInterrupt:
            print(f"\n⚠️  Analysis interrupted by user")
            return
        except Exception as e:
            print(f"❌ Analysis failed with error: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return
        
        print(f"\nAnalysis complete! Check {args.analysis_output} for graphs and detailed results.")
        return

    # Setup suspicious event logging
    sus_logger = setup_suspicious_logging(args.log_file)
    print(f"Suspicious events will be logged to: {args.log_file}")

    # Get video files to process
    if args.batch_process or os.path.isdir(args.source):
        video_files = get_video_files(args.source)
        if not video_files:
            print(f"No video files found in: {args.source}")
            return
        print(f"Found {len(video_files)} video files to process")
        args.batch_process = 1  # Force batch processing if directory provided
    else:
        video_files = [args.source]
        print(f"Processing single video: {args.source}")
    
    # Create output directories
    os.makedirs(args.output, exist_ok=True)
    if args.save_video:
        os.makedirs(args.output_videos, exist_ok=True)

    # Process each video
    total_videos = len(video_files)
    successful_videos = 0
    
    for video_idx, video_path in enumerate(video_files, 1):
        print(f"\n{'='*80}")
        print(f"VIDEO {video_idx}/{total_videos}: {os.path.basename(video_path)}")
        print(f"{'='*80}")
        
        # Initialize fresh tracking systems for each video
        person_tracker = SimpleIOUTracker()
        
        if args.person_specific:
            mp_manager = MultiPersonSuspicion(
                interaction_threshold=args.interaction_threshold,
                no_interaction_reset_frames=args.no_interaction_reset_frames,
                hands_presence_needed=args.hands_presence_needed,
                hands_absent_trigger_frames=args.hands_absent_trigger_frames,
                elbow_angle_threshold_deg=args.elbow_angle_threshold_deg,
                elbow_low_min_frames=args.elbow_low_min_frames,
                elbow_reset_frames=args.elbow_reset_frames,
            )
            tracker = None
        else:
            tracker = SuspicionTracker(
                interaction_threshold=args.interaction_threshold,
                no_interaction_reset_frames=args.no_interaction_reset_frames,
                hands_presence_needed=args.hands_presence_needed,
                hands_absent_trigger_frames=args.hands_absent_trigger_frames,
                elbow_angle_threshold_deg=args.elbow_angle_threshold_deg,
                elbow_low_min_frames=args.elbow_low_min_frames,
                elbow_reset_frames=args.elbow_reset_frames,
            )
            mp_manager = None

        # Process the video
        success = process_single_video(
            video_path=video_path,
            hoi_detector=hoi_detector,
            pose_estimator=pose_estimator,
            person_tracker=person_tracker,
            mp_manager=mp_manager,
            tracker=tracker,
            args=args,
            sus_logger=sus_logger
        )
        
        if success:
            successful_videos += 1
            print(f"✓ Successfully processed: {os.path.basename(video_path)}")
        else:
            print(f"✗ Failed to process: {os.path.basename(video_path)}")
    
    # Count suspicious videos in SUS folder
    sus_folder = os.path.join(args.output_videos, "SUS")
    sus_count = 0
    if os.path.exists(sus_folder):
        sus_count = len([f for f in os.listdir(sus_folder) if f.endswith('.mp4')])

    # Final summary
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total videos: {total_videos}")
    print(f"Successfully processed: {successful_videos}")
    print(f"Failed: {total_videos - successful_videos}")
    if args.save_video:
        print(f"Output videos saved to: {args.output_videos}")
        if sus_count > 0:
            print(f"🚨 Suspicious videos found: {sus_count} (moved to {sus_folder})")
        else:
            print(f"✓ No suspicious behavior detected in any videos")
    print(f"Suspicious frames saved to: {args.output}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
