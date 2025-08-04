"""
Experiments for single binary treatment
"""

import torch
import numpy as np
from gcmp_tuning_framework import GCMPTuningFramework

def run_single_binary_experiments():
    """Run all experiments for single binary treatment"""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize framework with custom directory
    framework = GCMPTuningFramework(base_dir='./tuning_results_single_binary')
    
    treatment_type = 'single_binary'
    
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENTS FOR {treatment_type.upper()}")
    print(f"{'='*80}")
    
    # 1. Hyperparameter tuning (reduced to 50 trials)
    print("\n1. Hyperparameter Tuning")
    best_params = framework.tune_treatment_type(treatment_type, n_trials=50)
    
    # 2. Multi-seed evaluation (reduced seeds for speed)
    print("\n2. Multi-seed Evaluation")
    seeds = [42, 123, 456, 789, 1011]  # 5 seeds instead of 10
    multi_seed_results = framework.run_multi_seed_evaluation(
        treatment_type, best_params, seeds=seeds
    )
    
    # 3. Characteristic experiments
    print("\n3. Characteristic Experiments")
    framework.run_characteristic_experiments(treatment_type, best_params)
    
    # Generate final report
    all_results = {
        treatment_type: {
            'best_params': best_params,
            'multi_seed_results': multi_seed_results
        }
    }
    framework._generate_final_report(all_results)
    
    print(f"\n{'='*80}")
    print(f"COMPLETED EXPERIMENTS FOR {treatment_type.upper()}")
    print(f"Results saved to: {framework.session_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    run_single_binary_experiments()