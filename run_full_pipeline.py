import argparse
import time
import os
from player_reid import PlayerReID
from evaluation import Evaluator

def run_full_pipeline(video_path, model_path, output_dir="results"):
    """Run the complete pipeline with evaluation"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize system
    print("Initializing Player Re-identification System...")
    reid_system = PlayerReID(model_path, confidence_threshold=0.5)
    
    # Process video
    print(f"Processing video: {video_path}")
    start_time = time.time()
    
    results = reid_system.process_video(
        video_path, 
        output_path=os.path.join(output_dir, "tracked_video.mp4")
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Save results
    results_file = os.path.join(output_dir, "tracking_results.json")
    reid_system.save_results(results, results_file)
    
    # Evaluate performance
    print("Evaluating performance...")
    evaluator = Evaluator()
    metrics = evaluator.calculate_metrics(results)
    evaluator.print_metrics()
    
    # Save evaluation metrics
    metrics_file = os.path.join(output_dir, "evaluation_metrics.json")
    evaluator.save_metrics(metrics_file)
    
    # Print summary
    print(f"\n=== PROCESSING SUMMARY ===")
    print(f"Processing Time: {processing_time:.2f} seconds")
    print(f"Frames Processed: {len(results)}")
    print(f"Processing Speed: {len(results)/processing_time:.1f} FPS")
    print(f"Results saved to: {output_dir}")
    
    return results, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full player re-identification pipeline")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--model", required=True, help="Path to YOLO model")
    parser.add_argument("--output", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    run_full_pipeline(args.video, args.model, args.output)