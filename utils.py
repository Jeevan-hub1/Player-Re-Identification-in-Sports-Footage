import cv2
import numpy as np
import json
from typing import List, Dict, Tuple

def visualize_tracks(results, video_path, output_path):
    """Create visualization of player tracks"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Track centers for trajectory visualization
    trajectories = {}
    
    for frame_idx, frame_result in enumerate(results):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update trajectories
        for detection in frame_result['detections']:
            player_id = detection['player_id']
            x, y, w, h = detection['bbox']
            center = (x + w//2, y + h//2)
            
            if player_id not in trajectories:
                trajectories[player_id] = []
            trajectories[player_id].append(center)
        
        # Draw trajectories
        for player_id, centers in trajectories.items():
            if len(centers) > 1:
                # Draw trajectory line
                for i in range(1, len(centers)):
                    if i < len(centers) - 10:  # Fade older points
                        continue
                    cv2.line(frame, centers[i-1], centers[i], (0, 255, 0), 2)
        
        # Draw current detections
        for detection in frame_result['detections']:
            player_id = detection['player_id']
            x, y, w, h = detection['bbox']
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Draw player ID
            cv2.putText(frame, f"ID: {player_id}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
    print(f"Trajectory visualization saved to: {output_path}")

def analyze_player_movements(results):
    """Analyze player movement patterns"""
    movements = {}
    
    for frame_result in results:
        for detection in frame_result['detections']:
            player_id = detection['player_id']
            x, y, w, h = detection['bbox']
            center = (x + w//2, y + h//2)
            
            if player_id not in movements:
                movements[player_id] = []
            movements[player_id].append({
                'frame': frame_result['frame_number'],
                'center': center,
                'bbox_area': w * h
            })
    
    # Calculate movement statistics
    stats = {}
    for player_id, positions in movements.items():
        if len(positions) < 2:
            continue
        
        # Calculate total distance
        total_distance = 0
        speeds = []
        
        for i in range(1, len(positions)):
            prev_pos = positions[i-1]['center']
            curr_pos = positions[i]['center']
            
            distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + 
                             (curr_pos[1] - prev_pos[1])**2)
            total_distance += distance
            speeds.append(distance)
        
        stats[player_id] = {
            'total_distance': total_distance,
            'avg_speed': np.mean(speeds) if speeds else 0,
            'max_speed': np.max(speeds) if speeds else 0,
            'frames_visible': len(positions),
            'avg_size': np.mean([p['bbox_area'] for p in positions])
        }
    
    return stats

def create_summary_report(results, metrics, output_file):
    """Create a comprehensive summary report"""
    report = {
        'summary': {
            'total_frames': len(results),
            'total_detections': sum(len(frame['detections']) for frame in results),
            'unique_players': len(set(det['player_id'] for frame in results for det in frame['detections']))
        },
        'metrics': metrics,
        'player_analysis': analyze_player_movements(results)
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Summary report saved to: {output_file}")
    return report