import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from src.models.adapters import HOIDetector, PoseEstimator, Pose
from src.pipeline.processor import find_pocket_rois, assign_pose_to_person, assign_hands_to_person
from src.pipeline.tracking import SimpleIOUTracker
from src.utils.geometry import BBox, Point, angle_between, l2_distance


@dataclass
class AnalysisResult:
    """Results from video analysis"""
    video_name: str
    total_frames: int
    hand_presence_percentage: float
    interaction_count: int
    avg_distance: float  # Average distance between pockets and hands for this video
    avg_left_elbow_angle: float  # Average left elbow angle for this video
    avg_right_elbow_angle: float  # Average right elbow angle for this video
    frames_with_hands: int
    frames_with_interactions: int
    interaction_frames: List[int]  # Frame numbers where interactions occurred


class VideoAnalyzer:
    """Analyzes videos to generate graphs and statistics"""
    
    def __init__(self, hoi_detector: HOIDetector, pose_estimator: PoseEstimator, 
                 dist_thresh_px: float = 50.0, pocket_padding: int = 25):
        self.hoi_detector = hoi_detector
        self.pose_estimator = pose_estimator
        self.dist_thresh_px = dist_thresh_px
        self.pocket_padding = pocket_padding
        
    def analyze_video(self, video_path: str, output_dir: str = "analysis_output") -> AnalysisResult:
        """Analyze a single video and generate graphs and statistics"""
        print(f"Analyzing video: {os.path.basename(video_path)}")
        
        # Force garbage collection before starting to free memory
        import gc
        gc.collect()
        
        # Clear CUDA cache if using GPU
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if video file exists and is readable
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get and validate video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or total_frames <= 0:
            cap.release()
            raise ValueError(f"Invalid video properties: FPS={fps}, Frames={total_frames}")
        
        # Limit processing to reasonable number of frames to avoid timeouts
        max_frames = min(total_frames, 1000)  # Process max 1000 frames per video
        if total_frames > max_frames:
            print(f"   ⚠️  Video has {total_frames} frames, processing first {max_frames} frames")
        
        video_name = Path(video_path).stem
        
        # Initialize tracking
        person_tracker = SimpleIOUTracker()
        
        # Data collection lists for calculating averages
        distance_data = []
        left_elbow_angles = []
        right_elbow_angles = []
        frames_with_hands = 0
        frames_with_interactions = 0
        interaction_count = 0
        interaction_frames = []  # Track which frames have interactions
        
        frame_idx = 0
        
        try:
            frame_count = 0
            while frame_count < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                
                # Resize frame to 640x640 for consistency
                frame = cv2.resize(frame, (640, 640))
                
                # Run detection
                persons, hands, objects = self.hoi_detector.predict(frame)
                poses = self.pose_estimator.predict(frame)
                
                # Track persons
                persons_tracked = person_tracker.update(persons)
                
                # Initialize frame data
                frame_distance = 0.0
                frame_left_elbow_angle = 0.0
                frame_right_elbow_angle = 0.0
                frame_has_hands = len(hands) > 0
                frame_interactions = 0
                
                if frame_has_hands:
                    frames_with_hands += 1
                
                # Process each person
                for person in persons_tracked:
                    if person.track_id is None:
                        continue
                    
                    # Assign pose to person
                    pose = assign_pose_to_person(person, poses)
                    if pose is None:
                        continue
                    
                    # Get pocket regions
                    pocket_rois = find_pocket_rois(pose, padding=self.pocket_padding)
                    
                    # Get hand keypoints (wrists)
                    left_wrist = pose.get(9)  # left wrist
                    right_wrist = pose.get(10)  # right wrist
                    
                    # Calculate distance between pockets and hand keypoints
                    if pocket_rois and (left_wrist or right_wrist):
                        min_distance = float('inf')
                        for roi in pocket_rois:
                            if left_wrist:
                                dist = l2_distance(roi.center(), left_wrist)
                                min_distance = min(min_distance, dist)
                            if right_wrist:
                                dist = l2_distance(roi.center(), right_wrist)
                                min_distance = min(min_distance, dist)
                        
                        if min_distance != float('inf'):
                            frame_distance = min_distance
                    
                    # Calculate elbow angles
                    left_shoulder = pose.get(5)
                    left_elbow = pose.get(7)
                    right_shoulder = pose.get(6)
                    right_elbow = pose.get(8)
                    
                    if left_shoulder and left_elbow and left_wrist:
                        frame_left_elbow_angle = angle_between(left_shoulder, left_elbow, left_wrist)
                    
                    if right_shoulder and right_elbow and right_wrist:
                        frame_right_elbow_angle = angle_between(right_shoulder, right_elbow, right_wrist)
                    
                    # Count interactions (simplified version)
                    hands_owned = assign_hands_to_person(person, hands)
                    if hands_owned and objects:
                        for hand in hands_owned:
                            for obj in objects:
                                hand_center = hand.center()
                                obj_center = obj.center()
                                dist = l2_distance(hand_center, obj_center)
                                if dist <= self.dist_thresh_px:
                                    frame_interactions += 1
                                    break
                
                # Store frame data
                distance_data.append(frame_distance)
                left_elbow_angles.append(frame_left_elbow_angle)
                right_elbow_angles.append(frame_right_elbow_angle)
                
                if frame_interactions > 0:
                    frames_with_interactions += 1
                    interaction_count += frame_interactions
                    interaction_frames.append(frame_idx)  # Track this frame as having interactions
                
                frame_idx += 1
                
                # Progress update
                if frame_idx % (fps * 5) == 0:  # Every 5 seconds
                    progress = (frame_idx / max_frames) * 100
                    print(f"Analysis progress: {progress:.1f}% ({frame_idx}/{max_frames})")
                
                frame_count += 1
        
        finally:
            cap.release()
        
        # Calculate statistics based on actually processed frames
        processed_frames = frame_count
        hand_presence_percentage = (frames_with_hands / processed_frames) * 100 if processed_frames > 0 else 0
        
        # Calculate averages
        valid_distances = [d for d in distance_data if d > 0]
        avg_distance = np.mean(valid_distances) if valid_distances else 0.0
        
        valid_left_angles = [a for a in left_elbow_angles if a > 0]
        avg_left_elbow_angle = np.mean(valid_left_angles) if valid_left_angles else 0.0
        
        valid_right_angles = [a for a in right_elbow_angles if a > 0]
        avg_right_elbow_angle = np.mean(valid_right_angles) if valid_right_angles else 0.0
        
        # Create analysis result
        result = AnalysisResult(
            video_name=video_name,
            total_frames=processed_frames,  # Use processed frames instead of total
            hand_presence_percentage=hand_presence_percentage,
            interaction_count=interaction_count,
            avg_distance=avg_distance,
            avg_left_elbow_angle=avg_left_elbow_angle,
            avg_right_elbow_angle=avg_right_elbow_angle,
            frames_with_hands=frames_with_hands,
            frames_with_interactions=frames_with_interactions,
            interaction_frames=interaction_frames
        )
        
        # DON'T generate detailed log for single video to avoid overwriting folder analysis
        # Instead, generate a single video summary
        self._generate_single_video_summary(result, output_dir)
        
        return result
    
    def _generate_consolidated_graphs(self, results: List[AnalysisResult], output_dir: str):
        """Generate consolidated graphs across all videos"""
        if not results:
            print("⚠️  No results to generate graphs from")
            return
        
        # Generate graphs even for single videos
        if len(results) < 2:
            print(f"ℹ️  Generating single video graph for {len(results)} video(s) analyzed")
        else:
            print(f"ℹ️  Generating consolidated graphs for {len(results)} videos")
        
        # Prepare data for plotting
        video_names = [result.video_name for result in results]
        video_indices = list(range(1, len(video_names) + 1))
        
        # Extract average distances and angles
        avg_distances = [result.avg_distance for result in results]
        avg_left_angles = [result.avg_left_elbow_angle for result in results]
        avg_right_angles = [result.avg_right_elbow_angle for result in results]
        
        # Calculate overall averages
        overall_avg_distance = np.mean(avg_distances)
        overall_avg_left_angle = np.mean(avg_left_angles)
        overall_avg_right_angle = np.mean(avg_right_angles)
        
        # Create figure with two subplots
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        except Exception as e:
            print(f"❌ Error creating subplots: {e}")
            # Fallback to single plot
            fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))
            ax2 = None
        
        # Graph 1: Average distance between pockets and hands per video
        bars1 = ax1.bar(video_indices, avg_distances, color='skyblue', alpha=0.7, edgecolor='navy', linewidth=0.5)
        ax1.set_xlabel('Video Number')
        ax1.set_ylabel('Average Distance (pixels)')
        ax1.set_title('Average Distance Between Pockets and Hands Across All Videos')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add overall average line
        ax1.axhline(y=overall_avg_distance, color='red', linestyle='--', linewidth=2, 
                   label=f'Overall Average: {overall_avg_distance:.1f}px')
        ax1.legend()
        
        # Add value labels on bars
        for i, (bar, dist) in enumerate(zip(bars1, avg_distances)):
            if dist > 0:  # Only label non-zero values
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{dist:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Graph 2: Average elbow angles per video
        width = 0.35
        x_pos = np.arange(len(video_indices))
        
        bars2_left = ax2.bar(x_pos - width/2, avg_left_angles, width, 
                            color='lightgreen', alpha=0.7, edgecolor='darkgreen', 
                            linewidth=0.5, label='Left Elbow')
        bars2_right = ax2.bar(x_pos + width/2, avg_right_angles, width, 
                             color='lightcoral', alpha=0.7, edgecolor='darkred', 
                             linewidth=0.5, label='Right Elbow')
        
        ax2.set_xlabel('Video Number')
        ax2.set_ylabel('Average Angle (degrees)')
        ax2.set_title('Average Elbow Angles Across All Videos')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(video_indices)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend()
        
        # Add overall average lines
        ax2.axhline(y=overall_avg_left_angle, color='green', linestyle='--', linewidth=2, 
                   label=f'Left Overall Avg: {overall_avg_left_angle:.1f}°')
        ax2.axhline(y=overall_avg_right_angle, color='red', linestyle='--', linewidth=2, 
                   label=f'Right Overall Avg: {overall_avg_right_angle:.1f}°')
        ax2.legend()
        
        # Add value labels on bars
        for i, (bar_left, bar_right, left_angle, right_angle) in enumerate(zip(bars2_left, bars2_right, avg_left_angles, avg_right_angles)):
            if left_angle > 0:  # Only label non-zero values
                ax2.text(bar_left.get_x() + bar_left.get_width()/2, bar_left.get_height() + 1, 
                        f'{left_angle:.1f}', ha='center', va='bottom', fontsize=7)
            if right_angle > 0:  # Only label non-zero values
                ax2.text(bar_right.get_x() + bar_right.get_width()/2, bar_right.get_height() + 1, 
                        f'{right_angle:.1f}', ha='center', va='bottom', fontsize=7)
        
        plt.tight_layout()
        
        # Save the consolidated graphs
        graph_path = os.path.join(output_dir, "consolidated_analysis.png")
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Consolidated graphs saved to: {graph_path}")
        print(f"Overall average distance: {overall_avg_distance:.1f}px")
        print(f"Overall average left elbow angle: {overall_avg_left_angle:.1f}°")
        print(f"Overall average right elbow angle: {overall_avg_right_angle:.1f}°")
    
    def analyze_folder(self, folder_path: str, output_dir: str = "analysis_output") -> List[AnalysisResult]:
        """Analyze all videos in a folder"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
        video_files = []
        
        for file_path in os.listdir(folder_path):
            if any(file_path.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(folder_path, file_path))
        
        if not video_files:
            print(f"No video files found in: {folder_path}")
            return []
        
        print(f"Found {len(video_files)} video files to analyze")
        
        results = []
        successful_videos = 0
        failed_videos = 0
        error_log = []
        
        try:
            for i, video_path in enumerate(video_files, 1):
                print(f"\n{'='*60}")
                print(f"ANALYZING VIDEO {i}/{len(video_files)}: {os.path.basename(video_path)}")
                print(f"{'='*60}")
                
                try:
                    # Add timeout and better error handling
                    result = self.analyze_video(video_path, output_dir)
                    results.append(result)
                    successful_videos += 1
                    
                    # Print summary for this video
                    print(f"\nAnalysis Summary for {result.video_name}:")
                    print(f"  Total frames: {result.total_frames}")
                    print(f"  Hand presence: {result.hand_presence_percentage:.1f}% ({result.frames_with_hands} frames)")
                    print(f"  Interactions: {result.interaction_count} total interactions")
                    print(f"  Frames with interactions: {result.frames_with_interactions}")
                    print(f"  ✅ Video {i} completed successfully")
                    
                    # Clean up memory after each video
                    import gc
                    gc.collect()
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except:
                        pass
                    
                    # Monitor memory usage every 10 videos
                    if successful_videos % 10 == 0:
                        try:
                            import psutil
                            process = psutil.Process()
                            mem_info = process.memory_info()
                            mem_mb = mem_info.rss / 1024 / 1024
                            
                            # Get available system memory
                            vm = psutil.virtual_memory()
                            available_mb = vm.available / 1024 / 1024
                            used_percent = vm.percent
                            
                            print(f"   📊 Memory usage: {mem_mb:.1f} MB (System: {used_percent:.1f}% used, {available_mb:.1f} MB available)")
                            
                            # Warning if memory usage is getting high
                            if used_percent > 85:
                                print(f"   ⚠️  WARNING: System memory usage is high ({used_percent:.1f}%)")
                                print(f"   💾 Forcing aggressive garbage collection...")
                                import gc
                                gc.collect()
                                gc.collect()
                                try:
                                    import torch
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                        print(f"   🧹 CUDA cache cleared")
                                except:
                                    pass
                        except:
                            pass
                    
                    # Save intermediate results every 10 videos
                    if successful_videos % 10 == 0:
                        print(f"\n💾 Saving intermediate results... ({successful_videos} videos processed)")
                        self._save_intermediate_results(results, error_log, output_dir, successful_videos, failed_videos)
                        # Also save detailed log incrementally
                        try:
                            self._generate_detailed_log(results, output_dir)
                            print(f"   📝 Detailed log updated with {len(results)} videos")
                        except Exception as e:
                            print(f"   ⚠️  Could not update detailed log: {e}")
                    
                except KeyboardInterrupt:
                    print(f"\n⚠️  Analysis interrupted by user at video {i}")
                    break
                except MemoryError as e:
                    print(f"\n❌ MEMORY ERROR at video {i}: {e}")
                    print(f"   System ran out of memory!")
                    print(f"   Processed {successful_videos} videos before memory exhaustion")
                    print(f"   Saving results and exiting gracefully...")
                    failed_videos += 1
                    break  # Exit loop due to memory error
                    
                except Exception as e:
                    failed_videos += 1
                    error_info = {
                        'video': os.path.basename(video_path),
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'traceback': traceback.format_exc()
                    }
                    error_log.append(error_info)
                    
                    print(f"❌ Error analyzing {video_path}: {e}")
                    print(f"   Error type: {type(e).__name__}")
                    import traceback
                    print(f"   Traceback: {traceback.format_exc()}")
                    print(f"   Continuing with next video...")
                    
                    # Check if this might be a critical error
                    if "CUDA" in str(e) or "memory" in str(e).lower():
                        print(f"   ⚠️  Detected potential memory/CUDA issue")
                        print(f"   🧹 Attempting aggressive cleanup...")
                        import gc
                        gc.collect()
                        gc.collect()
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except:
                            pass
                    
                    # Continue processing even if this video failed
                    continue
        
        except Exception as outer_e:
            # Catch any exceptions that might break the entire loop
            print(f"\n❌ CRITICAL ERROR in analysis loop: {outer_e}")
            print(f"   Error type: {type(outer_e).__name__}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            print(f"   Processed {successful_videos} videos before critical error")
            print(f"   Saving results and exiting...")
        
        finally:
            # ALWAYS save the final detailed log, even if interrupted or crashed
            if results:
                print(f"\n💾 Saving final results with {len(results)} videos...")
                try:
                    self._generate_detailed_log(results, output_dir)
                    print(f"   ✅ Final detailed log saved with {len(results)} videos")
                except Exception as e:
                    print(f"   ❌ Error saving final detailed log: {e}")
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Total videos found: {len(video_files)}")
        print(f"Successfully analyzed: {successful_videos}")
        print(f"Failed to analyze: {failed_videos}")
        print(f"Results collected: {len(results)}")
        
        # Generate consolidated graphs across all videos (even if only 1 video)
        try:
            self._generate_consolidated_graphs(results, output_dir)
        except Exception as e:
            print(f"❌ Error generating consolidated graphs: {e}")
            print("   Attempting to generate basic graph...")
            self._generate_basic_graph(results, output_dir)
        
        # Generate summary report
        try:
            self._generate_summary_report(results, output_dir)
        except Exception as e:
            print(f"❌ Error generating summary report: {e}")
        
        # Generate detailed log file with percentages
        try:
            self._generate_detailed_log(results, output_dir)
        except Exception as e:
            print(f"❌ Error generating detailed log: {e}")
        
        # Save error log if there were failures
        if error_log:
            try:
                self._save_error_log(error_log, output_dir)
            except Exception as e:
                print(f"❌ Error saving error log: {e}")
        
        return results
    
    def _generate_summary_report(self, results: List[AnalysisResult], output_dir: str):
        """Generate a summary report for all analyzed videos"""
        if not results:
            return
        
        report_path = os.path.join(output_dir, "analysis_summary.txt")
        
        with open(report_path, 'w') as f:
            f.write("VIDEO ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            for result in results:
                f.write(f"Video: {result.video_name}\n")
                f.write(f"  Total frames: {result.total_frames}\n")
                f.write(f"  Hand presence percentage: {result.hand_presence_percentage:.2f}%\n")
                f.write(f"  Total interactions: {result.interaction_count}\n")
                f.write(f"  Frames with hands: {result.frames_with_hands}\n")
                f.write(f"  Frames with interactions: {result.frames_with_interactions}\n")
                
                # Distance statistics
                if result.avg_distance > 0:
                    f.write(f"  Average pocket-hand distance: {result.avg_distance:.2f}px\n")
                
                # Angle statistics
                if result.avg_left_elbow_angle > 0:
                    f.write(f"  Average left elbow angle: {result.avg_left_elbow_angle:.2f}°\n")
                if result.avg_right_elbow_angle > 0:
                    f.write(f"  Average right elbow angle: {result.avg_right_elbow_angle:.2f}°\n")
                
                f.write("\n")
        
        print(f"Summary report saved to: {report_path}")
    
    def _generate_single_video_summary(self, result: AnalysisResult, output_dir: str):
        """Generate a summary for a single video (doesn't overwrite folder analysis log)"""
        summary_path = os.path.join(output_dir, f"{result.video_name}_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("SINGLE VIDEO ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Video: {result.video_name}\n")
            f.write(f"Analysis Date: {self._get_timestamp()}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("VIDEO STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total frames: {result.total_frames:,}\n")
            f.write(f"Hand presence percentage: {result.hand_presence_percentage:.2f}%\n")
            f.write(f"Frames with hands: {result.frames_with_hands:,} / {result.total_frames:,}\n")
            f.write(f"Total interactions: {result.interaction_count:,}\n")
            f.write(f"Frames with interactions: {result.frames_with_interactions:,}\n")
            f.write(f"Interaction rate: {(result.frames_with_interactions/result.total_frames*100):.2f}%\n\n")
            
            # Distance statistics
            if result.avg_distance > 0:
                f.write(f"Average pocket-hand distance: {result.avg_distance:.2f}px\n")
            
            # Angle statistics
            if result.avg_left_elbow_angle > 0:
                f.write(f"Left elbow angle - Avg: {result.avg_left_elbow_angle:.2f}°\n")
            if result.avg_right_elbow_angle > 0:
                f.write(f"Right elbow angle - Avg: {result.avg_right_elbow_angle:.2f}°\n")
            
            # Interaction frame details
            if result.interaction_frames:
                f.write(f"\nInteraction frames: {len(result.interaction_frames)}\n")
                f.write(f"First interaction at frame: {min(result.interaction_frames)}\n")
                f.write(f"Last interaction at frame: {max(result.interaction_frames)}\n")
                
                # Interaction clusters
                clusters = self._find_interaction_clusters(result.interaction_frames)
                f.write(f"Interaction clusters: {len(clusters)}\n")
                for j, cluster in enumerate(clusters[:10], 1):  # Show first 10 clusters
                    f.write(f"  Cluster {j}: frames {cluster[0]}-{cluster[1]} ({cluster[1]-cluster[0]+1} frames)\n")
                if len(clusters) > 10:
                    f.write(f"  ... and {len(clusters)-10} more clusters\n")
        
        print(f"Single video summary saved to: {summary_path}")
    
    def _generate_detailed_log(self, results: List[AnalysisResult], output_dir: str):
        """Generate a detailed log file with percentages and statistics"""
        if not results:
            return
        
        log_path = os.path.join(output_dir, "analysis_detailed_log.txt")
        
        with open(log_path, 'w') as f:
            f.write("DETAILED VIDEO ANALYSIS LOG\n")
            f.write("=" * 60 + "\n")
            f.write(f"Analysis Date: {self._get_timestamp()}\n")
            f.write(f"Total Videos Analyzed: {len(results)}\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall statistics
            total_frames = sum(r.total_frames for r in results)
            total_interactions = sum(r.interaction_count for r in results)
            total_hand_frames = sum(r.frames_with_hands for r in results)
            total_interaction_frames = sum(r.frames_with_interactions for r in results)
            
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total frames processed: {total_frames:,}\n")
            f.write(f"Total interactions detected: {total_interactions:,}\n")
            f.write(f"Total frames with hands: {total_hand_frames:,}\n")
            f.write(f"Total frames with interactions: {total_interaction_frames:,}\n")
            f.write(f"Overall hand presence: {(total_hand_frames/total_frames*100):.2f}%\n")
            f.write(f"Overall interaction rate: {(total_interaction_frames/total_frames*100):.2f}%\n\n")
            
            # Individual video details
            f.write("INDIVIDUAL VIDEO ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"VIDEO {i}: {result.video_name}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total frames: {result.total_frames:,}\n")
                f.write(f"Hand presence percentage: {result.hand_presence_percentage:.2f}%\n")
                f.write(f"Frames with hands: {result.frames_with_hands:,} / {result.total_frames:,}\n")
                f.write(f"Total interactions: {result.interaction_count:,}\n")
                f.write(f"Frames with interactions: {result.frames_with_interactions:,}\n")
                f.write(f"Interaction rate: {(result.frames_with_interactions/result.total_frames*100):.2f}%\n")
                
                # Distance statistics
                if result.avg_distance > 0:
                    f.write(f"Average pocket-hand distance: {result.avg_distance:.2f}px\n")
                
                # Angle statistics
                if result.avg_left_elbow_angle > 0:
                    f.write(f"Left elbow angle - Avg: {result.avg_left_elbow_angle:.2f}°\n")
                if result.avg_right_elbow_angle > 0:
                    f.write(f"Right elbow angle - Avg: {result.avg_right_elbow_angle:.2f}°\n")
                
                # Interaction frame details
                if result.interaction_frames:
                    f.write(f"Interaction frames: {len(result.interaction_frames)}\n")
                    f.write(f"First interaction at frame: {min(result.interaction_frames)}\n")
                    f.write(f"Last interaction at frame: {max(result.interaction_frames)}\n")
                    
                    # Interaction clusters (consecutive frames)
                    clusters = self._find_interaction_clusters(result.interaction_frames)
                    f.write(f"Interaction clusters: {len(clusters)}\n")
                    for j, cluster in enumerate(clusters[:5], 1):  # Show first 5 clusters
                        f.write(f"  Cluster {j}: frames {cluster[0]}-{cluster[1]} ({cluster[1]-cluster[0]+1} frames)\n")
                    if len(clusters) > 5:
                        f.write(f"  ... and {len(clusters)-5} more clusters\n")
                
                f.write("\n")
            
            # Summary by video type (if applicable)
            f.write("SUMMARY BY VIDEO TYPE\n")
            f.write("=" * 60 + "\n")
            
            # Group by video name patterns if they exist
            video_groups = {}
            for result in results:
                # Extract base name (remove numbers/suffixes)
                base_name = result.video_name.split('_')[0] if '_' in result.video_name else result.video_name
                if base_name not in video_groups:
                    video_groups[base_name] = []
                video_groups[base_name].append(result)
            
            for group_name, group_results in video_groups.items():
                if len(group_results) > 1:  # Only show groups with multiple videos
                    avg_hand_presence = np.mean([r.hand_presence_percentage for r in group_results])
                    avg_interactions = np.mean([r.interaction_count for r in group_results])
                    avg_distance = np.mean([r.avg_distance for r in group_results if r.avg_distance > 0])
                    avg_left_angle = np.mean([r.avg_left_elbow_angle for r in group_results if r.avg_left_elbow_angle > 0])
                    avg_right_angle = np.mean([r.avg_right_elbow_angle for r in group_results if r.avg_right_elbow_angle > 0])
                    
                    f.write(f"{group_name} group ({len(group_results)} videos):\n")
                    f.write(f"  Average hand presence: {avg_hand_presence:.2f}%\n")
                    f.write(f"  Average interactions: {avg_interactions:.1f}\n")
                    if avg_distance > 0:
                        f.write(f"  Average distance: {avg_distance:.2f}px\n")
                    if avg_left_angle > 0:
                        f.write(f"  Average left elbow angle: {avg_left_angle:.2f}°\n")
                    if avg_right_angle > 0:
                        f.write(f"  Average right elbow angle: {avg_right_angle:.2f}°\n")
                    f.write("\n")
        
        print(f"Detailed log saved to: {log_path}")
    
    def _save_error_log(self, error_log: List[Dict], output_dir: str):
        """Save error log for failed video analyses"""
        error_log_path = os.path.join(output_dir, "analysis_errors.txt")
        
        with open(error_log_path, 'w') as f:
            f.write("VIDEO ANALYSIS ERROR LOG\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total failed videos: {len(error_log)}\n")
            f.write(f"Error log generated: {self._get_timestamp()}\n")
            f.write("=" * 50 + "\n\n")
            
            for i, error in enumerate(error_log, 1):
                f.write(f"ERROR {i}: {error['video']}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Error Type: {error['error_type']}\n")
                f.write(f"Error Message: {error['error']}\n")
                f.write(f"Traceback:\n{error['traceback']}\n\n")
        
        print(f"Error log saved to: {error_log_path}")
    
    def _save_intermediate_results(self, results: List[AnalysisResult], error_log: List[Dict], 
                                 output_dir: str, successful_videos: int, failed_videos: int):
        """Save intermediate results during processing"""
        try:
            # Save current progress
            progress_path = os.path.join(output_dir, "progress.txt")
            with open(progress_path, 'w') as f:
                f.write(f"ANALYSIS PROGRESS\n")
                f.write(f"================\n")
                f.write(f"Successful videos: {successful_videos}\n")
                f.write(f"Failed videos: {failed_videos}\n")
                f.write(f"Total processed: {successful_videos + failed_videos}\n")
                f.write(f"Results collected: {len(results)}\n")
                f.write(f"Last updated: {self._get_timestamp()}\n")
            
            # Generate intermediate graph if we have results
            if results:
                try:
                    self._generate_consolidated_graphs(results, output_dir)
                    print(f"   📊 Intermediate graph generated")
                except Exception as e:
                    print(f"   ⚠️  Could not generate intermediate graph: {e}")
            
            # Save error log if there are errors
            if error_log:
                try:
                    self._save_error_log(error_log, output_dir)
                    print(f"   📝 Error log updated")
                except Exception as e:
                    print(f"   ⚠️  Could not save error log: {e}")
                    
        except Exception as e:
            print(f"   ❌ Error saving intermediate results: {e}")
    
    def _generate_basic_graph(self, results: List[AnalysisResult], output_dir: str):
        """Generate a basic graph as fallback when consolidated graph fails"""
        if not results:
            print("⚠️  No results to generate basic graph from")
            return
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            print(f"ℹ️  Generating basic graph for {len(results)} video(s)")
            
            # Prepare data
            video_names = [result.video_name for result in results]
            video_indices = list(range(1, len(video_names) + 1))
            avg_distances = [result.avg_distance for result in results]
            avg_left_angles = [result.avg_left_elbow_angle for result in results]
            avg_right_angles = [result.avg_right_elbow_angle for result in results]
            
            # Create a simple single plot
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Plot average distances
            bars = ax.bar(video_indices, avg_distances, color='skyblue', alpha=0.7, edgecolor='navy')
            ax.set_xlabel('Video Number')
            ax.set_ylabel('Average Distance (pixels)')
            ax.set_title(f'Average Distance Between Pockets and Hands - {len(results)} Video(s)')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, (bar, dist) in enumerate(zip(bars, avg_distances)):
                if dist > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                           f'{dist:.1f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            
            # Save the basic graph
            graph_path = os.path.join(output_dir, "basic_analysis.png")
            plt.savefig(graph_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Basic graph saved to: {graph_path}")
            
            # Also create a simple text summary
            summary_path = os.path.join(output_dir, "basic_summary.txt")
            with open(summary_path, 'w') as f:
                f.write("BASIC ANALYSIS SUMMARY\n")
                f.write("=" * 30 + "\n")
                f.write(f"Total videos analyzed: {len(results)}\n")
                f.write(f"Average distance: {np.mean(avg_distances):.2f}px\n")
                f.write(f"Average left elbow angle: {np.mean(avg_left_angles):.2f}°\n")
                f.write(f"Average right elbow angle: {np.mean(avg_right_angles):.2f}°\n")
                f.write("\nIndividual video results:\n")
                for i, result in enumerate(results, 1):
                    f.write(f"Video {i}: {result.video_name}\n")
                    f.write(f"  Distance: {result.avg_distance:.2f}px\n")
                    f.write(f"  Left elbow: {result.avg_left_elbow_angle:.2f}°\n")
                    f.write(f"  Right elbow: {result.avg_right_elbow_angle:.2f}°\n")
            
            print(f"Basic summary saved to: {summary_path}")
            
        except Exception as e:
            print(f"❌ Error generating basic graph: {e}")
            # Create a minimal text file as last resort
            try:
                minimal_path = os.path.join(output_dir, "minimal_results.txt")
                with open(minimal_path, 'w') as f:
                    f.write("MINIMAL ANALYSIS RESULTS\n")
                    f.write("=" * 30 + "\n")
                    f.write(f"Total videos analyzed: {len(results)}\n")
                    for i, result in enumerate(results, 1):
                        f.write(f"Video {i}: {result.video_name}\n")
                        f.write(f"  Frames: {result.total_frames}\n")
                        f.write(f"  Hand presence: {result.hand_presence_percentage:.2f}%\n")
                        f.write(f"  Interactions: {result.interaction_count}\n")
                print(f"Minimal results saved to: {minimal_path}")
            except Exception as e2:
                print(f"❌ Error creating minimal results: {e2}")
    
    def _find_interaction_clusters(self, interaction_frames: List[int]) -> List[Tuple[int, int]]:
        """Find clusters of consecutive interaction frames"""
        if not interaction_frames:
            return []
        
        sorted_frames = sorted(interaction_frames)
        clusters = []
        start = sorted_frames[0]
        end = sorted_frames[0]
        
        for i in range(1, len(sorted_frames)):
            if sorted_frames[i] == end + 1:
                end = sorted_frames[i]
            else:
                clusters.append((start, end))
                start = sorted_frames[i]
                end = sorted_frames[i]
        
        clusters.append((start, end))
        return clusters
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for logging"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """Main function to run video analysis"""
    import argparse
    from src.config import HOI_MODEL_PATH, POSE_MODEL_PATH, DEVICE
    
    parser = argparse.ArgumentParser(description="Analyze videos for hand pose and interaction patterns")
    parser.add_argument("--source", type=str, required=True, help="Video file or folder containing videos")
    parser.add_argument("--output", type=str, default="analysis_output", help="Output directory for analysis results")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda:0, cpu, etc.)")
    parser.add_argument("--dist_thresh", type=float, default=50.0, help="Distance threshold for interactions")
    parser.add_argument("--pocket_padding", type=int, default=25, help="Padding for pocket regions")
    
    args = parser.parse_args()
    
    # Determine device
    device = args.device if args.device else DEVICE
    
    # Initialize models
    print(f"Initializing models on device: {device}")
    hoi_detector = HOIDetector(model_path=HOI_MODEL_PATH, device=device)
    pose_estimator = PoseEstimator(model_path=POSE_MODEL_PATH, device=device)
    
    # Create analyzer
    analyzer = VideoAnalyzer(
        hoi_detector=hoi_detector,
        pose_estimator=pose_estimator,
        dist_thresh_px=args.dist_thresh,
        pocket_padding=args.pocket_padding
    )
    
    # Run analysis
    if os.path.isfile(args.source):
        # Single video
        result = analyzer.analyze_video(args.source, args.output)
        print(f"\nAnalysis complete! Results saved to: {args.output}")
    elif os.path.isdir(args.source):
        # Folder of videos
        results = analyzer.analyze_folder(args.source, args.output)
        print(f"\nAnalysis complete! {len(results)} videos analyzed. Results saved to: {args.output}")
    else:
        print(f"Error: {args.source} is not a valid file or directory")


if __name__ == "__main__":
    main()
