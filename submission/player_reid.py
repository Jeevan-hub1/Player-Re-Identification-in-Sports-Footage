import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.spatial.distance import cosine, euclidean
from sklearn.cluster import KMeans
import json
import os
from collections import defaultdict, deque
import argparse
from typing import Dict, List, Tuple, Optional
import time

class PlayerFeatureExtractor:
    """Extract various features from player detections"""
    
    def __init__(self):
        self.color_bins = 32
        
    def extract_color_histogram(self, roi):
        """Extract color histogram from player ROI"""
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram for each channel
        h_hist = cv2.calcHist([hsv], [0], None, [self.color_bins], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [self.color_bins], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [self.color_bins], [0, 256])
        
        # Normalize histograms
        h_hist = h_hist.flatten() / np.sum(h_hist)
        s_hist = s_hist.flatten() / np.sum(s_hist)
        v_hist = v_hist.flatten() / np.sum(v_hist)
        
        return np.concatenate([h_hist, s_hist, v_hist])
    
    def extract_shape_features(self, roi):
        """Extract shape-based features from player ROI"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate moments
        moments = cv2.moments(gray)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
        else:
            cx, cy = roi.shape[1] // 2, roi.shape[0] // 2
        
        # Aspect ratio
        h, w = roi.shape[:2]
        aspect_ratio = w / h if h > 0 else 1.0
        
        # Contour features
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            compactness = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        else:
            area = 0
            compactness = 0
        
        return np.array([cx/w, cy/h, aspect_ratio, area/(w*h), compactness])
    
    def extract_features(self, roi, bbox):
        """Extract combined features from player detection"""
        # Color features
        color_features = self.extract_color_histogram(roi)
        
        # Shape features
        shape_features = self.extract_shape_features(roi)
        
        # Spatial features (normalized bbox)
        x, y, w, h = bbox
        spatial_features = np.array([x, y, w, h, w*h])  # x, y, width, height, area
        
        return {
            'color': color_features,
            'shape': shape_features,
            'spatial': spatial_features,
            'combined': np.concatenate([color_features, shape_features, spatial_features])
        }

class PlayerTracker:
    """Track players across frames with re-identification"""
    
    def __init__(self, max_disappeared=15, similarity_threshold=0.5):
        self.next_id = 0
        self.players = {}  # Active players
        self.disappeared = {}  # Players that have disappeared
        self.max_disappeared = max_disappeared
        self.similarity_threshold = similarity_threshold
        self.feature_extractor = PlayerFeatureExtractor()
        self.history = defaultdict(lambda: deque(maxlen=10))  # Track history for smoothing
        
    def register_player(self, bbox, roi, frame_num):
        """Register a new player"""
        player_id = self.next_id
        features = self.feature_extractor.extract_features(roi, bbox)
        
        self.players[player_id] = {
            'bbox': bbox,
            'features': features,
            'frame': frame_num,
            'trajectory': deque(maxlen=20),
            'feature_history': deque(maxlen=5)
        }
        
        self.players[player_id]['trajectory'].append(bbox)
        self.players[player_id]['feature_history'].append(features)
        
        self.next_id += 1
        return player_id
    
    def calculate_similarity(self, features1, features2):
        """Calculate similarity between two feature sets"""
        # Color similarity
        color_sim = 1 - cosine(features1['color'], features2['color'])
        
        # Shape similarity
        shape_sim = 1 - cosine(features1['shape'], features2['shape'])
        
        # Spatial similarity (based on position)
        spatial_dist = euclidean(features1['spatial'][:2], features2['spatial'][:2])
        spatial_sim = 1 / (1 + spatial_dist / 100)  # Normalize spatial distance
        
        # Combined similarity with weights
        combined_sim = 0.5 * color_sim + 0.3 * shape_sim + 0.2 * spatial_sim
        
        return combined_sim
    
    def predict_next_position(self, player_id):
        """Predict next position based on trajectory"""
        if player_id not in self.players:
            return None
            
        trajectory = self.players[player_id]['trajectory']
        if len(trajectory) < 2:
            return trajectory[-1] if trajectory else None
        
        # Simple linear prediction
        last_pos = np.array(trajectory[-1][:2])
        prev_pos = np.array(trajectory[-2][:2])
        velocity = last_pos - prev_pos
        
        predicted_pos = last_pos + velocity
        return predicted_pos
    
    def update(self, detections, frame, frame_num):
        """Update tracker with new detections"""
        if len(detections) == 0:
            # Mark all players as disappeared
            for player_id in list(self.players.keys()):
                self.disappeared[player_id] = self.disappeared.get(player_id, 0) + 1
                
                if self.disappeared[player_id] > self.max_disappeared:
                    del self.players[player_id]
                    if player_id in self.disappeared:
                        del self.disappeared[player_id]
            
            return []
        
        # Extract features for all detections
        detection_features = []
        for det in detections:
            x, y, w, h = det['bbox']
            roi = frame[y:y+h, x:x+w]
            if roi.size > 0:
                features = self.feature_extractor.extract_features(roi, det['bbox'])
                detection_features.append(features)
            else:
                detection_features.append(None)
        
        # If no existing players, register all detections
        if len(self.players) == 0:
            results = []
            for i, det in enumerate(detections):
                if detection_features[i] is not None:
                    x, y, w, h = det['bbox']
                    roi = frame[y:y+h, x:x+w]
                    player_id = self.register_player(det['bbox'], roi, frame_num)
                    results.append({
                        'player_id': player_id,
                        'bbox': det['bbox'],
                        'confidence': det['confidence']
                    })
            return results
        
        # Match detections to existing players
        similarity_matrix = []
        player_ids = list(self.players.keys())
        
        for player_id in player_ids:
            player_similarities = []
            for det_features in detection_features:
                if det_features is not None:
                    # Use average of recent features for better matching
                    recent_features = self.players[player_id]['feature_history']
                    if recent_features:
                        avg_features = {
                            'color': np.mean([f['color'] for f in recent_features], axis=0),
                            'shape': np.mean([f['shape'] for f in recent_features], axis=0),
                            'spatial': np.mean([f['spatial'] for f in recent_features], axis=0)
                        }
                        similarity = self.calculate_similarity(avg_features, det_features)
                    else:
                        similarity = self.calculate_similarity(self.players[player_id]['features'], det_features)
                    
                    player_similarities.append(similarity)
                else:
                    player_similarities.append(0)
            
            similarity_matrix.append(player_similarities)
        
        # Hungarian algorithm for optimal assignment (simplified version)
        assignments = self.simple_assignment(similarity_matrix, self.similarity_threshold)
        
        # Update matched players
        used_detection_indices = set()
        results = []
        
        for player_idx, det_idx in assignments:
            if det_idx is not None:
                player_id = player_ids[player_idx]
                det = detections[det_idx]
                
                # Update player information
                self.players[player_id]['bbox'] = det['bbox']
                self.players[player_id]['features'] = detection_features[det_idx]
                self.players[player_id]['frame'] = frame_num
                self.players[player_id]['trajectory'].append(det['bbox'])
                self.players[player_id]['feature_history'].append(detection_features[det_idx])
                
                # Remove from disappeared list
                if player_id in self.disappeared:
                    del self.disappeared[player_id]
                
                used_detection_indices.add(det_idx)
                results.append({
                    'player_id': player_id,
                    'bbox': det['bbox'],
                    'confidence': det['confidence']
                })
        
        # Handle unmatched players (mark as disappeared)
        for player_idx, det_idx in enumerate(assignments):
            if det_idx is None:
                player_id = player_ids[player_idx]
                self.disappeared[player_id] = self.disappeared.get(player_id, 0) + 1
                
                if self.disappeared[player_id] > self.max_disappeared:
                    del self.players[player_id]
                    if player_id in self.disappeared:
                        del self.disappeared[player_id]
        
        # Register new players for unmatched detections
        for i, det in enumerate(detections):
            if i not in used_detection_indices and detection_features[i] is not None:
                x, y, w, h = det['bbox']
                roi = frame[y:y+h, x:x+w]
                player_id = self.register_player(det['bbox'], roi, frame_num)
                results.append({
                    'player_id': player_id,
                    'bbox': det['bbox'],
                    'confidence': det['confidence']
                })
        
        return results
    
    def simple_assignment(self, similarity_matrix, threshold):
        """Simple greedy assignment algorithm"""
        if not similarity_matrix:
            return []
        
        assignments = []
        used_detections = set()
        
        for player_idx, similarities in enumerate(similarity_matrix):
            best_det_idx = None
            best_similarity = threshold
            
            for det_idx, similarity in enumerate(similarities):
                if det_idx not in used_detections and similarity > best_similarity:
                    best_similarity = similarity
                    best_det_idx = det_idx
            
            if best_det_idx is not None:
                used_detections.add(best_det_idx)
                assignments.append((player_idx, best_det_idx))
            else:
                assignments.append((player_idx, None))
        
        return assignments

class PlayerReID:
    """Main class for player re-identification"""
    
    def __init__(self, model_path, confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.tracker = PlayerTracker()
        
    def detect_players(self, frame):
        """Detect players in frame using YOLO"""
        results = self.model(frame)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class (assuming class 0 is person/player)
                    class_id = int(box.cls.cpu().numpy()[0])
                    confidence = float(box.conf.cpu().numpy()[0])
                    
                    if confidence > self.confidence_threshold:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                        x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                        
                        detections.append({
                            'bbox': (x, y, w, h),
                            'confidence': confidence,
                            'class': class_id
                        })
        
        return detections
    
    def process_video(self, video_path, output_path=None):
        """Process video and track players"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
        
        # Initialize video writer if output path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results = []
        frame_num = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect players
            detections = self.detect_players(frame)
            
            # Update tracker
            tracked_players = self.tracker.update(detections, frame, frame_num)
            
            # Store results
            frame_result = {
                'frame_number': frame_num,
                'detections': tracked_players
            }
            results.append(frame_result)
            
            # Draw results on frame if output is requested
            if output_path:
                annotated_frame = self.draw_results(frame, tracked_players)
                out.write(annotated_frame)
            
            # Progress update
            if frame_num % 30 == 0:
                progress = (frame_num / total_frames) * 100
                print(f"Progress: {progress:.1f}% - Frame {frame_num}/{total_frames}")
            
            frame_num += 1
        
        cap.release()
        if output_path:
            out.release()
        
        return results
    
    def draw_results(self, frame, tracked_players):
        """Draw tracking results on frame"""
        annotated_frame = frame.copy()
        
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 255), (255, 20, 147), (0, 255, 127), (255, 69, 0)
        ]
        
        for player in tracked_players:
            player_id = player['player_id']
            x, y, w, h = player['bbox']
            confidence = player['confidence']
            
            # Choose color based on player ID
            color = colors[player_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw player ID and confidence
            label = f"ID: {player_id} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            cv2.putText(annotated_frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated_frame
    
    def save_results(self, results, output_file):
        """Save tracking results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Player Re-identification in Single Feed")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--model", required=True, help="Path to YOLO model")
    parser.add_argument("--output_video", help="Path to output video (optional)")
    parser.add_argument("--output_json", help="Path to output JSON results")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Initialize player re-identification system
    reid_system = PlayerReID(args.model, args.confidence)
    
    # Process video
    start_time = time.time()
    results = reid_system.process_video(args.video, args.output_video)
    end_time = time.time()
    
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    
    # Save results if requested
    if args.output_json:
        reid_system.save_results(results, args.output_json)
    
    # Print summary
    total_frames = len(results)
    total_detections = sum(len(frame['detections']) for frame in results)
    unique_players = len(set(det['player_id'] for frame in results for det in frame['detections']))
    
    print(f"\nSummary:")
    print(f"Total frames processed: {total_frames}")
    print(f"Total detections: {total_detections}")
    print(f"Unique players identified: {unique_players}")

if __name__ == "__main__":
    main()