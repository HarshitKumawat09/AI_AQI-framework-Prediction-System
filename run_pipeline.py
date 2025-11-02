"""
Master Execution Script - Run Complete Research Pipeline
Execute all components in proper sequence
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


class ResearchPipeline:
    """
    Master pipeline orchestrator for the complete research project
    """
    
    def __init__(self):
        self.start_time = None
        self.steps_completed = []
        self.steps_failed = []
    
    def log(self, message, level="INFO"):
        """Log messages with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
    
    def run_script(self, script_path, description):
        """
        Run a Python script and track its execution
        
        Parameters:
        -----------
        script_path : str
            Path to the script
        description : str
            Description of the step
        """
        self.log(f"Starting: {description}", "INFO")
        self.log(f"Executing: {script_path}", "INFO")
        
        step_start = time.time()
        
        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Log output
            if result.stdout:
                print(result.stdout)
            
            duration = time.time() - step_start
            self.log(f"✓ Completed: {description} (Duration: {duration:.2f}s)", "SUCCESS")
            self.steps_completed.append({
                'step': description,
                'duration': duration,
                'status': 'SUCCESS'
            })
            
            return True
            
        except subprocess.CalledProcessError as e:
            duration = time.time() - step_start
            self.log(f"✗ Failed: {description}", "ERROR")
            self.log(f"Error: {e.stderr}", "ERROR")
            self.steps_failed.append({
                'step': description,
                'duration': duration,
                'status': 'FAILED',
                'error': str(e.stderr)
            })
            
            return False
    
    def print_banner(self, text):
        """Print a formatted banner"""
        print("\n" + "="*80)
        print(text.center(80))
        print("="*80 + "\n")
    
    def run_full_pipeline(self, skip_preprocessing=False, skip_dl=False, quick_mode=False):
        """
        Execute the complete research pipeline
        
        Parameters:
        -----------
        skip_preprocessing : bool
            Skip data preprocessing if already done
        skip_dl : bool
            Skip deep learning models (time-consuming)
        quick_mode : bool
            Use faster settings (less thorough but quicker)
        """
        self.start_time = time.time()
        
        self.print_banner("STARTING COMPLETE RESEARCH PIPELINE")
        self.log(f"Start Time: {datetime.now()}")
        self.log(f"Quick Mode: {quick_mode}")
        self.log(f"Skip Preprocessing: {skip_preprocessing}")
        self.log(f"Skip Deep Learning: {skip_dl}")
        
        # Step 1: Data Preprocessing
        if not skip_preprocessing:
            self.print_banner("PHASE 1: DATA PREPROCESSING & FEATURE ENGINEERING")
            success = self.run_script(
                'scripts/data_preprocessing.py',
                'Data Preprocessing & Feature Engineering'
            )
            if not success and not quick_mode:
                self.log("Preprocessing failed. Aborting pipeline.", "ERROR")
                return
        else:
            self.log("Skipping preprocessing (already completed)", "INFO")
        
        # Step 2: Classical ML Benchmarking
        self.print_banner("PHASE 2: CLASSICAL ML MODEL BENCHMARKING")
        success = self.run_script(
            'scripts/model_benchmarking.py',
            'Classical ML Model Benchmarking'
        )
        
        # Step 3: Deep Learning Models
        if not skip_dl:
            self.print_banner("PHASE 3: DEEP LEARNING MODELS")
            success = self.run_script(
                'scripts/deep_learning_models.py',
                'Deep Learning Models (LSTM, GRU, CNN)'
            )
        else:
            self.log("Skipping deep learning models", "INFO")
        
        # Step 4: Hybrid Model
        self.print_banner("PHASE 4: HYBRID MODEL DEVELOPMENT")
        success = self.run_script(
            'scripts/hybrid_model.py',
            'Hybrid Model (ML + Time-Series)'
        )
        
        # Step 5: Comprehensive Explainability
        self.print_banner("PHASE 5: COMPREHENSIVE EXPLAINABILITY ANALYSIS")
        success = self.run_script(
            'scripts/comprehensive_explainability.py',
            'Multi-Level Explainability Analysis'
        )
        
        # Print Summary
        self.print_summary()
    
    def print_summary(self):
        """Print execution summary"""
        total_duration = time.time() - self.start_time
        
        self.print_banner("PIPELINE EXECUTION SUMMARY")
        
        print(f"Total Duration: {total_duration/60:.2f} minutes\n")
        
        print("Completed Steps:")
        print("-" * 80)
        for step in self.steps_completed:
            print(f"✓ {step['step']:<60} {step['duration']:.2f}s")
        
        if self.steps_failed:
            print("\nFailed Steps:")
            print("-" * 80)
            for step in self.steps_failed:
                print(f"✗ {step['step']:<60} {step['duration']:.2f}s")
        
        print("\n" + "="*80)
        print(f"SUCCESS: {len(self.steps_completed)} steps")
        print(f"FAILED:  {len(self.steps_failed)} steps")
        print("="*80)
        
        # Save summary
        self.save_summary()
    
    def save_summary(self):
        """Save execution summary to file"""
        summary = {
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_duration_minutes': (time.time() - self.start_time) / 60,
            'completed_steps': self.steps_completed,
            'failed_steps': self.steps_failed,
            'success_rate': len(self.steps_completed) / (len(self.steps_completed) + len(self.steps_failed)) if (self.steps_completed or self.steps_failed) else 0
        }
        
        import json
        with open('pipeline_execution_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
        
        self.log("Execution summary saved to: pipeline_execution_summary.json", "INFO")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run AQI Research Pipeline')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip data preprocessing step')
    parser.add_argument('--skip-dl', action='store_true',
                       help='Skip deep learning models (faster execution)')
    parser.add_argument('--quick-mode', action='store_true',
                       help='Quick mode with faster settings')
    parser.add_argument('--step', type=str, choices=['preprocess', 'benchmark', 'dl', 'hybrid', 'explain'],
                       help='Run only a specific step')
    
    args = parser.parse_args()
    
    pipeline = ResearchPipeline()
    
    if args.step:
        # Run specific step
        pipeline.print_banner(f"RUNNING SPECIFIC STEP: {args.step.upper()}")
        
        step_map = {
            'preprocess': ('scripts/data_preprocessing.py', 'Data Preprocessing'),
            'benchmark': ('scripts/model_benchmarking.py', 'Model Benchmarking'),
            'dl': ('scripts/deep_learning_models.py', 'Deep Learning Models'),
            'hybrid': ('scripts/hybrid_model.py', 'Hybrid Model'),
            'explain': ('scripts/comprehensive_explainability.py', 'Explainability Analysis')
        }
        
        script_path, description = step_map[args.step]
        pipeline.run_script(script_path, description)
        
    else:
        # Run full pipeline
        pipeline.run_full_pipeline(
            skip_preprocessing=args.skip_preprocessing,
            skip_dl=args.skip_dl,
            quick_mode=args.quick_mode
        )


if __name__ == "__main__":
    main()
