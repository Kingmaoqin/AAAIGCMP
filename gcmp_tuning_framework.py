"""
Comprehensive hyperparameter tuning and evaluation for GCMP
Includes tuning for all treatment types and detailed analysis
"""

import optuna
import json
import time
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
from multiprocessing import Pool
from functools import partial

# Import project modules
from utils import get_default_config, save_config, CausalEvaluator
from main import GCMPTrainer
from data import DataModule, TreatmentType
from mixed_treatment import TreatmentSpec

class GCMPTuningFramework:
    """Comprehensive tuning framework for GCMP across all treatment types"""
    
    def __init__(self, base_dir: str = './tuning_results'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this tuning session
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_dir / self.timestamp
        self.session_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Treatment type configurations
        self.treatment_configs = {
            'single_continuous': {
                'treatment_type': 'continuous',
                'treatment_dim': 1,
                'description': 'Single continuous treatment'
            },
            'single_binary': {
                'treatment_type': 'binary',
                'treatment_dim': 1,
                'description': 'Single binary treatment'
            },
            'multi_continuous': {
                'treatment_type': 'continuous',
                'treatment_dim': 3,
                'description': 'Multiple continuous treatments'
            },
            'multi_binary': {
                'treatment_type': 'multilabel',
                'treatment_dim': 3,
                'description': 'Multiple binary treatments'
            },
            'mixed': {
                'treatment_type': 'mixed',
                'treatment_spec': {
                    'continuous_dims': [0, 1],
                    'discrete_dims': [2, 3],
                    'discrete_levels': {2: 2, 3: 3}
                },
                'description': 'Mixed continuous and discrete treatments'
            }
        }


    @staticmethod
    def _convert_to_json_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: GCMPTuningFramework._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [GCMPTuningFramework._convert_to_json_serializable(v) for v in obj]
        else:
            return obj

    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.session_dir / 'tuning.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def define_search_space(self, trial: optuna.Trial, treatment_type: str) -> Dict:
        """Define hyperparameter search space based on treatment type"""
        params = {}
        
        # Model architecture parameters
        params["model.dropout"] = trial.suggest_float("model.dropout", 0.05, 0.3)
        
        # Adjust latent dimension based on treatment complexity
        if 'multi' in treatment_type or treatment_type == 'mixed':
            params["model.latent_dim"] = trial.suggest_categorical("model.latent_dim", [16, 24, 32, 48])
        else:
            params["model.latent_dim"] = trial.suggest_categorical("model.latent_dim", [8, 16, 24])
        
        params["model.ssl_output_dim"] = trial.suggest_categorical("model.ssl_output_dim", [32, 64, 96, 128])
        
        # Network architecture choices
        ssl_hidden_choice = trial.suggest_categorical("ssl_hidden_choice", ["small", "medium", "large"])
        ssl_architectures = {
            "small": [128, 64],
            "medium": [256, 128],
            "large": [512, 256, 128]
        }
        params["model.ssl_hidden_dims"] = ssl_architectures[ssl_hidden_choice]
        
        vi_hidden_choice = trial.suggest_categorical("vi_hidden_choice", ["small", "medium", "large"])
        vi_architectures = {
            "small": [64, 32],
            "medium": [128, 64],
            "large": [256, 128, 64]
        }
        params["model.vi_hidden_dims"] = vi_architectures[vi_hidden_choice]
        
        # SSL parameters
        params["ssl.lr"] = trial.suggest_float("ssl.lr", 1e-4, 1e-2, log=True)
        params["ssl.temperature"] = trial.suggest_float("ssl.temperature", 0.1, 1.0)
        params["ssl.lambda_preserve"] = trial.suggest_float("ssl.lambda_preserve", 0.01, 0.5, log=True)
        params["ssl.lambda_diverse"] = trial.suggest_float("ssl.lambda_diverse", 0.001, 0.1, log=True)
        params["training.n_epochs_ssl"] = trial.suggest_categorical("training.n_epochs_ssl", [30, 50, 70])
        
        # Diffusion parameters
        params["diffusion_training.lr"] = trial.suggest_float("diffusion_training.lr", 1e-5, 1e-3, log=True)
        params["diffusion_training.lambda_orthogonal"] = trial.suggest_float(
            "diffusion_training.lambda_orthogonal", 0.01, 1.0, log=True
        )
        params["diffusion_training.lambda_outcome"] = trial.suggest_float(
            "diffusion_training.lambda_outcome", 0.1, 2.0
        )
        params["training.n_epochs_diffusion"] = trial.suggest_categorical(
            "training.n_epochs_diffusion", [40, 60, 80]
        )
        
        # Diffusion sampling parameters
        params["diffusion_sampling.n_steps"] = trial.suggest_categorical(
            "diffusion_sampling.n_steps", [20, 50, 100]
        )
        
        # VI parameters
        params["vi_training.lr"] = trial.suggest_float("vi_training.lr", 1e-4, 1e-2, log=True)
        params["vi_training.lambda_entropy"] = trial.suggest_float(
            "vi_training.lambda_entropy", 1e-3, 0.1, log=True
        )
        params["vi_training.weight_decay"] = trial.suggest_float(
            "vi_training.weight_decay", 1e-6, 1e-4, log=True
        )
        params["training.n_epochs_vi"] = trial.suggest_categorical("training.n_epochs_vi", [30, 40, 50])
        
        # Cross-fitting parameters
        params["training.n_folds"] = trial.suggest_categorical("training.n_folds", [3, 5])
        
        # Treatment-specific adjustments
        if treatment_type == 'mixed':
            # Mixed treatments might benefit from larger models
            params["model.manifold_dim"] = trial.suggest_categorical("model.manifold_dim", [3, 5, 7])
        
        return params
    
    def create_objective(self, treatment_type: str):
        """Create an objective function for a specific treatment type"""
        
        def objective(trial: optuna.Trial) -> float:
            """Objective function for Optuna optimization"""
            # Get base configuration
            base_config = get_default_config()
            
            # Configure for specific treatment type
            treatment_config = self.treatment_configs[treatment_type]
            base_config['data']['dataset_kwargs']['treatment_type'] = treatment_config.get(
                'treatment_type', 'continuous'
            )
            
            if 'treatment_dim' in treatment_config:
                base_config['data']['dataset_kwargs']['treatment_dim'] = treatment_config['treatment_dim']
            
            if 'treatment_spec' in treatment_config:
                spec_config = treatment_config['treatment_spec']
                # 同样，创建一个全新的、干净的 kwargs 字典
                base_config['data']['dataset_kwargs'] = {
                    'treatment_spec': TreatmentSpec(**spec_config)
                }
                base_config['data']['dataset_name'] = 'mixed_synthetic'
            # Reduced data size for faster tuning
            base_config['data']['n_samples'] = 5000
            base_config['training']['batch_size'] = 256
            
            # Get trial parameters
            trial_params = self.define_search_space(trial, treatment_type)
            
            # Log trial info
            self.logger.info(f"\nTrial {trial.number} for {treatment_type}")
            self.logger.info(f"Parameters: {json.dumps(trial_params, indent=2)}")
            
            # Apply parameters to config
            for key, value in trial_params.items():
                self._set_nested_param(base_config, key, value)
            
            # Create unique directories for this trial
            trial_dir = self.session_dir / treatment_type / f"trial_{trial.number}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Initialize trainer
                trainer = GCMPTrainer(
                    config=base_config,
                    log_dir=str(trial_dir / 'logs'),
                    checkpoint_dir=str(trial_dir / 'checkpoints'),
                    use_wandb=False
                )
                
                # Train model
                n_epochs = {
                    'ssl': base_config['training']['n_epochs_ssl'],
                    'diffusion': base_config['training']['n_epochs_diffusion'],
                    'vi': base_config['training']['n_epochs_vi']
                }
                
                trainer.train(n_epochs, n_folds=base_config['training']['n_folds'])
                
                # Evaluate
                results = trainer.evaluate()
                
                # Primary metric: PEHE
                pehe = results.get('pehe', float('inf'))
                
                # Secondary metrics for multi-objective optimization
                ate_error = results.get('ate_error', float('inf'))
                coverage = results.get('coverage_95', 0.0)
                
                # Composite score (you can adjust weights)
                score = pehe + 0.1 * ate_error + 0.1 * abs(coverage - 0.95)
                
                # Log results
                self.logger.info(f"Trial {trial.number} Results:")
                self.logger.info(f"  PEHE: {pehe:.4f}")
                self.logger.info(f"  ATE Error: {ate_error:.4f}")
                self.logger.info(f"  Coverage: {coverage:.4f}")
                self.logger.info(f"  Score: {score:.4f}")
                
                # Save trial results
                trial_results = {
                    'trial_number': trial.number,
                    'parameters': trial_params,
                    'results': results,
                    'score': score
                }
                
                with open(trial_dir / 'results.json', 'w') as f:
                    json.dump(trial_results, f, indent=2)
                
                return score
                
            except Exception as e:
                self.logger.error(f"Trial {trial.number} failed: {str(e)}")
                return float('inf')
        
        return objective
    
    def tune_treatment_type(self, treatment_type: str, n_trials: int = 100) -> Dict:
        """Run hyperparameter tuning for a specific treatment type"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Tuning {treatment_type}")
        self.logger.info(f"{'='*60}")
        
        # Create study
        study_name = f"gcmp_{treatment_type}_{self.timestamp}"
        storage_path = self.session_dir / f"{treatment_type}_study.db"
        
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=f"sqlite:///{storage_path}",
            load_if_exists=True
        )
        
        # Run optimization
        objective = self.create_objective(treatment_type)
        study.optimize(objective, n_trials=n_trials)
        
        # Get best trial
        best_trial = study.best_trial
        
        # Save best parameters
        best_params = {
            'treatment_type': treatment_type,
            'best_trial_number': best_trial.number,
            'best_score': best_trial.value,
            'best_parameters': best_trial.params,
            'n_trials': len(study.trials)
        }
        
        save_path = self.session_dir / f"{treatment_type}_best_params.json"
        with open(save_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        self.logger.info(f"\nBest trial for {treatment_type}:")
        self.logger.info(f"  Trial number: {best_trial.number}")
        self.logger.info(f"  Score: {best_trial.value:.4f}")
        self.logger.info(f"  Parameters saved to: {save_path}")
        
        return best_params
    
    def run_multi_seed_evaluation(self, treatment_type: str, best_params: Dict, 
                                  seeds: List[int] = None) -> Dict:
        """Run evaluation with multiple seeds using best parameters"""
        if seeds is None:
            seeds = [42, 123, 456, 789, 1011, 1314, 1617, 1920, 2223, 2526]
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Multi-seed evaluation for {treatment_type}")
        self.logger.info(f"{'='*60}")
        
        results_all_seeds = []
        
        for seed in seeds:
            self.logger.info(f"\nRunning with seed {seed}")
            
            # Get base configuration
            base_config = get_default_config()
            
            # Apply treatment configuration
            treatment_config = self.treatment_configs[treatment_type]
            base_config['data']['dataset_kwargs']['treatment_type'] = treatment_config.get(
                'treatment_type', 'continuous'
            )
            
            if 'treatment_dim' in treatment_config:
                base_config['data']['dataset_kwargs']['treatment_dim'] = treatment_config['treatment_dim']
            
            if 'treatment_spec' in treatment_config:
                spec_config = treatment_config['treatment_spec']
                # 最后一次，创建全新的、干净的 kwargs 字典
                base_config['data']['dataset_kwargs'] = {
                    'treatment_spec': TreatmentSpec(**spec_config)
                }
                base_config['data']['dataset_name'] = 'mixed_synthetic'
            
            # Use full dataset for final evaluation
            base_config['data']['n_samples'] = 10000
            base_config['seed'] = seed
            base_config['data']['dataset_kwargs']['seed'] = seed
            
            # Apply best parameters
            for key, value in best_params['best_parameters'].items():
                self._set_nested_param(base_config, key, value)
            
            # Create directories
            eval_dir = self.session_dir / treatment_type / f"eval_seed_{seed}"
            eval_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Initialize trainer
                trainer = GCMPTrainer(
                    config=base_config,
                    log_dir=str(eval_dir / 'logs'),
                    checkpoint_dir=str(eval_dir / 'checkpoints'),
                    use_wandb=False
                )
                
                # Train model
                n_epochs = {
                    'ssl': base_config['training']['n_epochs_ssl'],
                    'diffusion': base_config['training']['n_epochs_diffusion'],
                    'vi': base_config['training']['n_epochs_vi']
                }
                
                trainer.train(n_epochs, n_folds=base_config['training']['n_folds'])
                
                # Evaluate
                results = trainer.evaluate()
                results['seed'] = seed
                results_all_seeds.append(results)
                
                # Save individual seed results
                with open(eval_dir / 'results.json', 'w') as f:
                    json.dump(self._convert_to_json_serializable(results), f, indent=2)
                
            except Exception as e:
                self.logger.error(f"Seed {seed} failed: {str(e)}")
                continue
        
        # Aggregate results
        aggregated_results = self._aggregate_seed_results(results_all_seeds)
        
        # Save aggregated results
        save_path = self.session_dir / f"{treatment_type}_multi_seed_results.json"
        with open(save_path, 'w') as f:
            json.dump(self._convert_to_json_serializable(aggregated_results), f, indent=2)
        
        self.logger.info(f"\nAggregated results for {treatment_type}:")
        for metric, stats in aggregated_results['metrics'].items():
            if isinstance(stats, dict) and 'mean' in stats:
                self.logger.info(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        return aggregated_results
    
    def run_characteristic_experiments(self, treatment_type: str, best_params: Dict):
        """Run characteristic experiments specific to GCMP"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Characteristic experiments for {treatment_type}")
        self.logger.info(f"{'='*60}")
        
        experiments_dir = self.session_dir / treatment_type / "characteristic_experiments"
        experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Orthogonality violation sensitivity
        self._run_orthogonality_sensitivity(treatment_type, best_params, experiments_dir)
        
        # 2. Sample size efficiency
        self._run_sample_efficiency(treatment_type, best_params, experiments_dir)
        
        # 3. Confounder dimension analysis
        self._run_confounder_dimension_analysis(treatment_type, best_params, experiments_dir)
        
        # 4. Ablation study
        self._run_ablation_study(treatment_type, best_params, experiments_dir)
        
        # 5. Convergence analysis
        self._run_convergence_analysis(treatment_type, best_params, experiments_dir)
    
    def _run_orthogonality_sensitivity(self, treatment_type: str, best_params: Dict, 
                                    save_dir: Path, delta_values: list = None):
        if delta_values is None:
            delta_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        results = {}
        
        for delta in delta_values:
            self.logger.info(f"  Testing δ = {delta}")
            
            # Configure with orthogonality violation
            config = self._create_config_from_best_params(treatment_type, best_params)
            config['data']['dataset_kwargs']['orthogonality_violation'] = delta
            config['data']['n_samples'] = 5000  # Smaller for efficiency
            
            # Run experiment
            exp_results = self._run_single_experiment(
                config, 
                save_dir / f"ortho_delta_{delta}"
            )
            results[delta] = exp_results
        
        # Plot results
        self._plot_orthogonality_results(results, save_dir)
        
        # Save results
        with open(save_dir / "orthogonality_sensitivity.json", 'w') as f:
            json.dump(self._convert_to_json_serializable(results), f, indent=2)
    
    def _run_sample_efficiency(self, treatment_type: str, best_params: Dict, 
                            save_dir: Path, sample_sizes: list = None):
        if sample_sizes is None:
            sample_sizes = [500, 1000, 2000, 5000, 10000, 20000]
        results = {}
        
        for n in sample_sizes:
            self.logger.info(f"  Testing n = {n}")
            
            config = self._create_config_from_best_params(treatment_type, best_params)
            config['data']['n_samples'] = n
            
            # Adjust batch size for large datasets
            if n > 10000:
                config['training']['batch_size'] = 512
            
            # Run experiment
            exp_results = self._run_single_experiment(
                config,
                save_dir / f"sample_size_{n}"
            )
            results[n] = exp_results
        
        # Plot results
        self._plot_sample_efficiency(results, save_dir)
        
        # Save results
        with open(save_dir / "sample_efficiency.json", 'w') as f:
            json.dump(self._convert_to_json_serializable(results), f, indent=2)
    
    def _run_confounder_dimension_analysis(self, treatment_type: str, best_params: Dict,
                                        save_dir: Path, confounder_dims: list = None):

        """Test confounder dimensionality impact"""
        self.logger.info("\nRunning confounder dimension analysis...")

        if confounder_dims is None:
            confounder_dims = [1, 3, 5, 7, 10]
        results = {}
        
        for dim in confounder_dims:
            self.logger.info(f"  Testing dimension = {dim}")
            
            config = self._create_config_from_best_params(treatment_type, best_params)
            config['data']['dataset_kwargs']['confounder_dim'] = dim
            config['model']['manifold_dim'] = min(dim, 5)
            config['data']['n_samples'] = 5000
            
            # Run experiment
            exp_results = self._run_single_experiment(
                config,
                save_dir / f"confounder_dim_{dim}"
            )
            results[dim] = exp_results
        
        # Plot results
        self._plot_confounder_dimension_results(results, save_dir)
        
        # Save results
        with open(save_dir / "confounder_dimension.json", 'w') as f:
            json.dump(self._convert_to_json_serializable(results), f, indent=2)
    
    def _run_ablation_study(self, treatment_type: str, best_params: Dict,
                            save_dir: Path, ablations_to_run: Optional[List[str]] = None):
        """
        Run ablation study.

        Args:
            ...
            ablations_to_run: An optional list of ablation keys to run. 
                              If None, all ablations are run.
        """
        self.logger.info("\nRunning ablation study...")
        
        # 1. Define all possible ablation configurations
        all_ablations = {
            'full_model': {},
            'no_ssl': {'training.n_epochs_ssl': 0},
            'no_orthogonality': {'diffusion_training.lambda_orthogonal': 0},
            'no_entropy': {'vi_training.lambda_entropy': 0},
            'no_cross_fitting': {'training.n_folds': 1},
            'simple_architecture': {
                'model.ssl_hidden_dims': [64],
                'model.vi_hidden_dims': [32]
            }
        }
        
        # 2. Determine which ablations to run based on the new parameter
        if ablations_to_run is None:
            ablations_to_run_dict = all_ablations
        else:
            ablations_to_run_dict = {
                k: v for k, v in all_ablations.items() if k in ablations_to_run
            }
            self.logger.info(f"  (Running in QUICK mode with settings: {ablations_to_run})")

        results = {}
        
        for ablation_name, modifications in ablations_to_run_dict.items():
            self.logger.info(f"  Testing {ablation_name}")
            
            config = self._create_config_from_best_params(treatment_type, best_params)
            # Use a smaller sample size for these experiments to keep them fast
            config['data']['n_samples'] = 5000 
            
            # Apply modifications for the current ablation setting
            for key, value in modifications.items():
                self._set_nested_param(config, key, value)
            
            # Run the actual experiment for this setting
            exp_results = self._run_single_experiment(
                config,
                save_dir / f"ablation_{ablation_name}"
            )
            results[ablation_name] = exp_results
        
        # Plot and save the results
        self._plot_ablation_results(results, save_dir)
        
        with open(save_dir / "ablation_study.json", 'w') as f:
            # 3. Use the JSON serialization helper for robust saving
            serializable_results = self._convert_to_json_serializable(results)
            json.dump(serializable_results, f, indent=2)
    
    def _run_convergence_analysis(self, treatment_type: str, best_params: Dict,
                                  save_dir: Path):
        """Analyze convergence behavior"""
        self.logger.info("\nRunning convergence analysis...")
        
        config = self._create_config_from_best_params(treatment_type, best_params)
        config['data']['n_samples'] = 5000
        
        # Enable detailed logging for convergence analysis
        config['logging'] = {
            'log_interval': 10,
            'save_checkpoints': True,
            'checkpoint_interval': 10
        }
        
        # Run experiment with convergence tracking
        conv_dir = save_dir / "convergence"
        conv_dir.mkdir(exist_ok=True)
        
        trainer = GCMPTrainer(
            config=config,
            log_dir=str(conv_dir / 'logs'),
            checkpoint_dir=str(conv_dir / 'checkpoints'),
            use_wandb=False
        )
        
        # Track losses during training
        losses = {
            'ssl': [],
            'diffusion': [],
            'vi': []
        }
        
        # This would require modifying the trainer to return loss histories
        # For now, we'll simulate this
        n_epochs = {
            'ssl': config['training']['n_epochs_ssl'],
            'diffusion': config['training']['n_epochs_diffusion'],
            'vi': config['training']['n_epochs_vi']
        }
        
        trainer.train(n_epochs, n_folds=config['training']['n_folds'])
        results = trainer.evaluate()
        
        # Plot convergence curves
        self._plot_convergence_curves(losses, save_dir)
        
        # Save results
        with open(save_dir / "convergence_analysis.json", 'w') as f:
            json.dump(self._convert_to_json_serializable({
                'losses': losses,
                'final_results': results
            }), f, indent=2)
    
    def run_all_experiments(self):
        """Run complete tuning and evaluation pipeline"""
        self.logger.info("="*80)
        self.logger.info("GCMP COMPREHENSIVE TUNING AND EVALUATION")
        self.logger.info(f"Session: {self.timestamp}")
        self.logger.info("="*80)
        
        all_results = {}
        
        for treatment_type in self.treatment_configs.keys():
            self.logger.info(f"\n{'#'*80}")
            self.logger.info(f"Processing {treatment_type}")
            self.logger.info(f"{'#'*80}")
            
            # 1. Hyperparameter tuning
            best_params = self.tune_treatment_type(treatment_type, n_trials=100)
            
            # 2. Multi-seed evaluation
            multi_seed_results = self.run_multi_seed_evaluation(treatment_type, best_params)
            
            # 3. Characteristic experiments
            self.run_characteristic_experiments(treatment_type, best_params)
            
            # Store results
            all_results[treatment_type] = {
                'best_params': best_params,
                'multi_seed_results': multi_seed_results
            }
        
        # Generate comprehensive report
        self._generate_final_report(all_results)
        
        self.logger.info("\n" + "="*80)
        self.logger.info("ALL EXPERIMENTS COMPLETED!")
        self.logger.info(f"Results saved to: {self.session_dir}")
        self.logger.info("="*80)
    
    # Helper methods
    
    def _set_nested_param(self, config: Dict, param_path: str, value):
        """Set nested parameter in config"""
        keys = param_path.split('.')
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    def _create_config_from_best_params(self, treatment_type: str, best_params: Dict) -> Dict:
        """Create configuration from best parameters"""
        base_config = get_default_config()
        
        # Apply treatment configuration
        treatment_config = self.treatment_configs[treatment_type]
        base_config['data']['dataset_kwargs']['treatment_type'] = treatment_config.get(
            'treatment_type', 'continuous'
        )
        
        if 'treatment_dim' in treatment_config:
            base_config['data']['dataset_kwargs']['treatment_dim'] = treatment_config['treatment_dim']
        
        if 'treatment_spec' in treatment_config:
            # Handle mixed treatment configuration
            spec_config = treatment_config['treatment_spec']
            base_config['data']['dataset_kwargs'] = {
                'treatment_spec': TreatmentSpec(**spec_config)
            }
            base_config['data']['dataset_name'] = 'mixed_synthetic'
        
        # Apply best parameters
        for key, value in best_params['best_parameters'].items():
            self._set_nested_param(base_config, key, value)
        
        return base_config
    
    def _run_single_experiment(self, config: Dict, save_dir: Path) -> Dict:
        """Run a single experiment"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            trainer = GCMPTrainer(
                config=config,
                log_dir=str(save_dir / 'logs'),
                checkpoint_dir=str(save_dir / 'checkpoints'),
                use_wandb=False
            )
            
            n_epochs = {
                'ssl': config['training']['n_epochs_ssl'],
                'diffusion': config['training']['n_epochs_diffusion'],
                'vi': config['training']['n_epochs_vi']
            }
            
            trainer.train(n_epochs, n_folds=config['training']['n_folds'])
            results = trainer.evaluate()
            
            # Save results
            with open(save_dir / 'results.json', 'w') as f:
                json.dump(self._convert_to_json_serializable(results), f, indent=2)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {str(e)}")
            return {'error': str(e)}
    
    def _aggregate_seed_results(self, results: List[Dict]) -> Dict:
        """Aggregate results from multiple seeds"""
        if not results:
            return {}
        
        # Get all metrics
        metrics = {}
        for key in results[0].keys():
            if key == 'seed' or key == 'config':
                continue
            
            values = []
            for r in results:
                if key in r and isinstance(r[key], (int, float)):
                    values.append(r[key])
            
            if values:
                metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
        
        return {
            'n_seeds': len(results),
            'metrics': metrics,
            'raw_results': results
        }
    
    # Plotting methods
    
    def _plot_orthogonality_results(self, results: Dict, save_dir: Path):
        """Plot orthogonality sensitivity results"""
        plt.figure(figsize=(10, 6))
        
        deltas = sorted(results.keys())
        pehe_values = [results[d].get('pehe', np.nan) for d in deltas]
        ate_errors = [results[d].get('ate_error', np.nan) for d in deltas]
        coverage_values = [results[d].get('coverage_95', np.nan) for d in deltas]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # PEHE vs delta
        axes[0].plot(deltas, pehe_values, 'o-', markersize=8, linewidth=2)
        axes[0].set_xlabel('Orthogonality Violation (δ)')
        axes[0].set_ylabel('PEHE')
        axes[0].set_title('Prediction Error vs Orthogonality Violation')
        axes[0].grid(True, alpha=0.3)

        # ATE error vs delta
        axes[1].plot(deltas, ate_errors, 's-', markersize=8, linewidth=2, color='orange')
        axes[1].set_xlabel('Orthogonality Violation (δ)')
        axes[1].set_ylabel('ATE Error')
        axes[1].set_title('ATE Error vs Orthogonality Violation')
        axes[1].grid(True, alpha=0.3)

        # Coverage vs delta
        axes[2].plot(deltas, coverage_values, '^-', markersize=8, linewidth=2, color='green')
        axes[2].axhline(y=0.95, color='red', linestyle='--', label='Target Coverage')
        axes[2].set_xlabel('Orthogonality Violation (δ)')
        axes[2].set_ylabel('95% CI Coverage')
        axes[2].set_title('Coverage vs Orthogonality Violation')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(save_dir / 'orthogonality_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.close()
   
    def _plot_sample_efficiency(self, results: Dict, save_dir: Path):
        """Plot sample efficiency results"""
        sample_sizes = sorted(results.keys())
        pehe_values = [results[n].get('pehe', np.nan) for n in sample_sizes]
        runtime_values = [results[n].get('runtime', np.nan) for n in sample_sizes]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # PEHE vs sample size
        axes[0].loglog(sample_sizes, pehe_values, 'o-', markersize=8, linewidth=2)

        # Fit power law
        valid_indices = ~np.isnan(pehe_values)
        if np.sum(valid_indices) > 2:
            log_n = np.log(np.array(sample_sizes)[valid_indices])
            log_pehe = np.log(np.array(pehe_values)[valid_indices])
            slope, intercept = np.polyfit(log_n, log_pehe, 1)
            fitted_pehe = np.exp(intercept) * np.array(sample_sizes) ** slope
            axes[0].loglog(sample_sizes, fitted_pehe, 'r--', linewidth=2, 
                            label=f'Fit: n^{slope:.2f}')
            axes[0].legend()

        axes[0].set_xlabel('Sample Size (n)')
        axes[0].set_ylabel('PEHE')
        axes[0].set_title('Sample Efficiency')
        axes[0].grid(True, alpha=0.3)

        # Runtime vs sample size
        axes[1].loglog(sample_sizes, runtime_values, 's-', markersize=8, linewidth=2, color='orange')
        axes[1].set_xlabel('Sample Size (n)')
        axes[1].set_ylabel('Runtime (seconds)')
        axes[1].set_title('Computational Scaling')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / 'sample_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()
   
    def _plot_confounder_dimension_results(self, results: Dict, save_dir: Path):
        """Plot confounder dimension analysis results"""
        dims = sorted(results.keys())
        pehe_values = [results[d].get('pehe', np.nan) for d in dims]
        ortho_scores = [results[d].get('orthogonality_score', np.nan) for d in dims]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # PEHE vs dimension
        axes[0].plot(dims, pehe_values, 'o-', markersize=8, linewidth=2)
        axes[0].set_xlabel('Confounder Dimension')
        axes[0].set_ylabel('PEHE')
        axes[0].set_title('Error vs Confounder Dimensionality')
        axes[0].grid(True, alpha=0.3)
        
        # Orthogonality score vs dimension
        axes[1].plot(dims, ortho_scores, 's-', markersize=8, linewidth=2, color='green')
        axes[1].set_xlabel('Confounder Dimension')
        axes[1].set_ylabel('Orthogonality Score')
        axes[1].set_title('Orthogonality Quality vs Dimensionality')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'confounder_dimension_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
   
    def _plot_ablation_results(self, results: Dict, save_dir: Path):
        """Plot ablation study results"""
        ablations = list(results.keys())

        # Get metrics
        metrics = ['pehe', 'ate_error', 'coverage_95', 'runtime']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            values = [results[abl].get(metric, np.nan) for abl in ablations]
            
            # Normalize by full model performance
            if 'full_model' in results and metric in results['full_model']:
                full_value = results['full_model'][metric]
                if metric in ['pehe', 'ate_error', 'runtime']:  # Lower is better
                    relative_values = [(v / full_value - 1) * 100 for v in values]
                else:  # Higher is better
                    relative_values = [(v - full_value) / full_value * 100 for v in values]
            else:
                relative_values = values
            
            ax = axes[i]
            bars = ax.bar(range(len(ablations)), relative_values)
            
            # Color code bars
            for j, (bar, val) in enumerate(zip(bars, relative_values)):
                if ablations[j] == 'full_model':
                    bar.set_color('blue')
                elif val > 0 and metric in ['pehe', 'ate_error', 'runtime']:
                    bar.set_color('red')
                elif val < 0 and metric not in ['pehe', 'ate_error', 'runtime']:
                    bar.set_color('red')
                else:
                    bar.set_color('green')
            
            ax.set_xticks(range(len(ablations)))
            ax.set_xticklabels([abl.replace('_', ' ').title() for abl in ablations], 
                                rotation=45, ha='right')
            ax.set_ylabel(f'Relative Change in {metric} (%)')
            ax.set_title(f'{metric.upper()} - Ablation Study')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        plt.tight_layout()
        plt.savefig(save_dir / 'ablation_study.png', dpi=300, bbox_inches='tight')
        plt.close()
   
    def _plot_convergence_curves(self, losses: Dict, save_dir: Path):
        """Plot convergence curves"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        stages = ['ssl', 'diffusion', 'vi']
        colors = ['blue', 'orange', 'green']

        for i, (stage, color) in enumerate(zip(stages, colors)):
            if stage in losses and losses[stage]:
                ax = axes[i]
                epochs = range(1, len(losses[stage]) + 1)
                ax.plot(epochs, losses[stage], linewidth=2, color=color)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(f'{stage.upper()} Training Loss')
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(save_dir / 'convergence_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
   
    def _generate_final_report(self, all_results: Dict):
        """Generate comprehensive final report"""
        report_path = self.session_dir / 'final_report.md'

        with open(report_path, 'w') as f:
            f.write("# GCMP Hyperparameter Tuning and Evaluation Report\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Session**: {self.timestamp}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This report presents comprehensive hyperparameter tuning results ")
            f.write("for the GCMP framework across different treatment types.\n\n")
            
            # Best results summary table
            f.write("## Best Results Summary\n\n")
            f.write("| Treatment Type | PEHE (mean ± std) | ATE Error (mean ± std) | Coverage (mean ± std) |\n")
            f.write("|----------------|-------------------|------------------------|----------------------|\n")
            
            for treatment_type, results in all_results.items():
                multi_seed = results['multi_seed_results']['metrics']
                pehe = multi_seed.get('pehe', {})
                ate = multi_seed.get('ate_error', {})
                cov = multi_seed.get('coverage_95', {})
                
                f.write(f"| {treatment_type} | ")
                f.write(f"{pehe.get('mean', 'N/A'):.4f} ± {pehe.get('std', 0):.4f} | ")
                f.write(f"{ate.get('mean', 'N/A'):.4f} ± {ate.get('std', 0):.4f} | ")
                f.write(f"{cov.get('mean', 'N/A'):.4f} ± {cov.get('std', 0):.4f} |\n")
            
            f.write("\n## Detailed Results by Treatment Type\n\n")
            
            for treatment_type, results in all_results.items():
                f.write(f"### {treatment_type.replace('_', ' ').title()}\n\n")
                
                # Best parameters
                f.write("#### Best Hyperparameters\n\n")
                f.write("```json\n")
                f.write(json.dumps(results['best_params']['best_parameters'], indent=2))
                f.write("\n```\n\n")
                
                # Multi-seed results
                f.write("#### Multi-Seed Evaluation Results\n\n")
                multi_seed = results['multi_seed_results']['metrics']
                for metric, stats in multi_seed.items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        f.write(f"- **{metric}**: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                f.write("\n")
            
            f.write("## Key Findings\n\n")
            f.write("1. **Treatment Type Performance**: Analysis of performance across treatment types\n")
            f.write("2. **Hyperparameter Sensitivity**: Key hyperparameters that impact performance\n")
            f.write("3. **Robustness**: Model robustness to various assumption violations\n")
            f.write("4. **Sample Efficiency**: Performance scaling with dataset size\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("Based on the tuning results, we recommend:\n\n")
            f.write("1. Using treatment-specific hyperparameters for optimal performance\n")
            f.write("2. Paying special attention to orthogonality regularization strength\n")
            f.write("3. Adjusting network architectures based on treatment complexity\n")
            f.write("4. Using cross-fitting with at least 3 folds for stable results\n")

        # Generate LaTeX tables for paper
        self._generate_latex_tables(all_results)

        self.logger.info(f"\nFinal report saved to: {report_path}")
   
    def _generate_latex_tables(self, all_results: Dict):
        """Generate LaTeX tables for paper"""
        latex_dir = self.session_dir / 'latex_tables'
        latex_dir.mkdir(exist_ok=True)

        # Main results table
        with open(latex_dir / 'main_results.tex', 'w') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{GCMP Performance Across Treatment Types}\n")
            f.write("\\label{tab:gcmp_results}\n")
            f.write("\\begin{tabular}{lccc}\n")
            f.write("\\toprule\n")
            f.write("Treatment Type & PEHE & ATE Error & Coverage \\\\\n")
            f.write("\\midrule\n")
            
            for treatment_type, results in all_results.items():
                multi_seed = results['multi_seed_results']['metrics']
                pehe = multi_seed.get('pehe', {})
                ate = multi_seed.get('ate_error', {})
                cov = multi_seed.get('coverage_95', {})
                
                treatment_name = treatment_type.replace('_', ' ').title()
                f.write(f"{treatment_name} & ")
                f.write(f"${pehe.get('mean', 0):.3f} \\pm {pehe.get('std', 0):.3f}$ & ")
                f.write(f"${ate.get('mean', 0):.3f} \\pm {ate.get('std', 0):.3f}$ & ")
                f.write(f"${cov.get('mean', 0):.3f} \\pm {cov.get('std', 0):.3f}$ \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        self.logger.info("LaTeX tables generated in: " + str(latex_dir))


def main():
    """Main function to run all experiments"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize framework
    framework = GCMPTuningFramework()

    # Run all experiments
    framework.run_all_experiments()


if __name__ == "__main__":
    main()        