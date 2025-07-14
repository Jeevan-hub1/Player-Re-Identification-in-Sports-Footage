# test_system.py
import os
import json
import cv2
from player_reid import PlayerReID

def test_basic_functionality():
    """Test basic system functionality"""
    print("Testing Player Re-identification System...")
    
    # Configuration
    model_path = "models/best.pt"
    video_path = "data/15sec_input_720p.mp4"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return False
    
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found at {video_path}")
        return False
    
    try:
        # Initialize system
        reid_system = PlayerReID(model_path, confidence_threshold=0.5)
        print("✓ System initialized successfully")
        
        # Test video reading
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("ERROR: Cannot open video file")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read frame from video")
            return False
        
        cap.release()
        print("✓ Video file readable")
        
        # Test detection on single frame
        detections = reid_system.detect_players(frame)
        print(f"✓ Detected {len(detections)} players in test frame")
        
        # Test feature extraction
        if detections:
            x, y, w, h = detections[0]['bbox']
            roi = frame[y:y+h, x:x+w]
            features = reid_system.tracker.feature_extractor.extract_features(roi, detections[0]['bbox'])
            print(f"✓ Feature extraction successful - {len(features['combined'])} features")
        
        print("✓ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"ERROR: Test failed with exception: {e}")
        return False

def test_full_processing():
    """Test full video processing"""
    print("\nTesting full video processing...")
    
    model_path = "models/best.pt"
    video_path = "data/15sec_input_720p.mp4"
    output_json = "test_results.json"
    
    try:
        reid_system = PlayerReID(model_path, confidence_threshold=0.5)
        
        # Process first 30 frames only for testing
        cap = cv2.VideoCapture(video_path)
        results = []
        frame_num = 0
        
        while frame_num < 30:  # Test with first 30 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = reid_system.detect_players(frame)
            tracked_players = reid_system.tracker.update(detections, frame, frame_num)
            
            results.append({
                'frame_number': frame_num,
                'detections': tracked_players
            })
            
            frame_num += 1
        
        cap.release()
        
        # Save test results
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Analyze results
        total_detections = sum(len(frame['detections']) for frame in results)
        unique_players = len(set(det['player_id'] for frame in results for det in frame['detections']))
        
        print(f"✓ Processed {len(results)} frames")
        print(f"✓ Total detections: {total_detections}")
        print(f"✓ Unique players: {unique_players}")
        print(f"✓ Results saved to {output_json}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Full processing test failed: {e}")
        return False

if __name__ == "__main__":
    # Run tests
    success = test_basic_functionality()
    if success:
        test_full_processing()
    else:
        print("Basic tests failed. Please check your setup.")