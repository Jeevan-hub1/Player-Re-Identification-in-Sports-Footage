import json
import numpy as np
from collections import defaultdict

class Evaluator:
    """Evaluate tracking performance"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_metrics(self, results):
        """Calculate tracking metrics"""
        # Extract player trajectories
        trajectories = defaultdict(list)
        for frame in results:
            for detection in frame['detections']:
                player_id = detection['player_id']
                frame_num = frame['frame_number']
                trajectories[player_id].append(frame_num)
        
        # Calculate metrics
        total_players = len(trajectories)
        total_frames = len(results)
        
        # Track continuity
        fragmented_tracks = 0
        total_gaps = 0
        
        for player_id, frames in trajectories.items():
            if len(frames) < 2:
                continue
                
            # Check for gaps in trajectory
            gaps = []
            for i in range(1, len(frames)):
                gap = frames[i] - frames[i-1]
                if gap > 1:
                    gaps.append(gap - 1)
            
            if gaps:
                fragmented_tracks += 1
                total_gaps += len(gaps)
        
        # Calculate detection rate
        detections_per_frame = [len(frame['detections']) for frame in results]
        avg_detections = np.mean(detections_per_frame)
        
        self.metrics = {
            'total_players': total_players,
            'total_frames': total_frames,
            'fragmented_tracks': fragmented_tracks,
            'fragmentation_rate': fragmented_tracks / total_players if total_players > 0 else 0,
            'total_gaps': total_gaps,
            'avg_detections_per_frame': avg_detections,
            'detection_consistency': np.std(detections_per_frame)
        }
        
        return self.metrics
    
    def print_metrics(self):
        """Print evaluation metrics"""
        if not self.metrics:
            print("No metrics calculated yet.")
            return
        
        print("\n=== EVALUATION METRICS ===")
        print(f"Total Players Identified: {self.metrics['total_players']}")
        print(f"Total Frames Processed: {self.metrics['total_frames']}")
        print(f"Fragmented Tracks: {self.metrics['fragmented_tracks']}")
        print(f"Fragmentation Rate: {self.metrics['fragmentation_rate']:.2%}")
        print(f"Total Track Gaps: {self.metrics['total_gaps']}")
        print(f"Avg Detections per Frame: {self.metrics['avg_detections_per_frame']:.1f}")
        print(f"Detection Consistency (std): {self.metrics['detection_consistency']:.2f}")
    
    def save_metrics(self, output_file):
        """Save metrics to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)