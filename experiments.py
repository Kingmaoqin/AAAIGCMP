"""
Comprehensive experiments for GCMP framework
Tests various data scenarios including multiple treatments
"""

import numpy as np
import torch
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# Import GCMP modules
from data import DataModule, TreatmentType
from models import GCMP
from main import GCMPTrainer
from baselines import create_baseline
from utils import get_default_config, save_config, compare_methods, CausalEvaluator

# Mixed treatment support (optional - will work without it)
try:
    from mixed_treatment import TreatmentSpec, MixedTreatmentDataset, MixedTreatmentVIModel
    MIXED_TREATMENT_AVAILABLE = True
except ImportError:
    MIXED_TREATMENT_AVAILABLE = False
    print("Warning: mixed_treatment module not available. Mixed treatment experiments will be skipped.")


class ExperimentSuite:
    """Run comprehensive experiments on GCMP"""
    
    def __init__(self, base_dir: str = './experiment_results'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def run_all_experiments(self):
        """Run all experiment types"""
        print("=" * 80)
        print("GCMP COMPREHENSIVE EXPERIMENT SUITE")
        print("=" * 80)
        
        # 1. Single continuous treatment
        print("\n1. SINGLE CONTINUOUS TREATMENT EXPERIMENT")
        self.run_continuous_treatment_experiment()
        
        # 2. Single binary treatment  
        print("\n2. SINGLE BINARY TREATMENT EXPERIMENT")
        self.run_binary_treatment_experiment()
        
        # 3. Multi-label treatment
        print("\n3. MULTI-LABEL TREATMENT EXPERIMENT")
        self.run_multilabel_treatment_experiment()
        
        # 4. Multiple continuous treatments
        print("\n4. MULTIPLE CONTINUOUS TREATMENTS EXPERIMENT")
        self.run_multiple_continuous_experiment()
        
        # 5. Mixed treatment types
        print("\n5. MIXED TREATMENT TYPES EXPERIMENT")
        self.run_mixed_treatment_experiment()
        
        # 6. Orthogonality violation sensitivity
        print("\n6. ORTHOGONALITY VIOLATION SENSITIVITY ANALYSIS")
        self.run_orthogonality_sensitivity()
        
        # 7. Sample size scaling
        print("\n7. SAMPLE SIZE SCALING ANALYSIS")
        self.run_sample_size_scaling()
        
        # 8. Confounder dimensionality
        print("\n8. CONFOUNDER DIMENSIONALITY ANALYSIS")
        self.run_confounder_dim_analysis()
        
        print("\n" + "=" * 80)
        print("ALL EXPERIMENTS COMPLETED!")
        print(f"Results saved to: {self.base_dir}")
    
    def run_continuous_treatment_experiment(self):
        """Experiment 1: Single continuous treatment (drug dosage)"""
        exp_dir = self.base_dir / "continuous_treatment"
        exp_dir.mkdir(exist_ok=True)
        
        # Configure
        config = get_default_config()
        config['data']['n_samples'] = 5000  # Set sample size at data level
        config['data']['dataset_kwargs'].update({
            'treatment_type': 'continuous',
            'orthogonality_violation': 0.2
        })
        config['training']['n_epochs_ssl'] = 20
        config['training']['n_epochs_diffusion'] = 30
        config['training']['n_epochs_vi'] = 20
        
        # Run GCMP
        results = self._run_gcmp(config, exp_dir)
        
        # Run baselines
        baseline_results = self._run_baselines(config, ['naive', 'ipw', 'contivae'])
        
        # Compare
        all_results = {'gcmp': results, **baseline_results}
        comparison = compare_methods(all_results, ['pehe', 'ate_error', 'runtime'])
        
        # Visualize dose-response curve
        self._plot_dose_response(config, exp_dir)
        
        # Save results
        self._save_results(all_results, exp_dir / "results.json")
    
    def run_binary_treatment_experiment(self):
        """Experiment 2: Single binary treatment"""
        exp_dir = self.base_dir / "binary_treatment"
        exp_dir.mkdir(exist_ok=True)
        
        # Configure
        config = get_default_config()
        config['data']['n_samples'] = 5000  # Set sample size at data level
        config['data']['dataset_kwargs'].update({
            'treatment_type': 'binary',
            'orthogonality_violation': 0.1
        })
        config['training']['n_epochs_ssl'] = 20
        config['training']['n_epochs_diffusion'] = 30
        config['training']['n_epochs_vi'] = 20
        
        # Run GCMP
        results = self._run_gcmp(config, exp_dir)
        
        # Run baselines
        baseline_results = self._run_baselines(config, ['naive', 'ipw', 'cevae', 'deconfounder'])
        
        # Compare
        all_results = {'gcmp': results, **baseline_results}
        comparison = compare_methods(all_results, ['pehe', 'ate_error', 'coverage_95', 'runtime'])
        
        # Save results
        self._save_results(all_results, exp_dir / "results.json")
    
    def run_multilabel_treatment_experiment(self):
        """Experiment 3: Multi-label treatment (combination therapy)"""
        exp_dir = self.base_dir / "multilabel_treatment"
        exp_dir.mkdir(exist_ok=True)
        
        # Configure - 3 binary treatments
        config = get_default_config()
        config['data']['n_samples'] = 8000  # More samples for higher complexity
        config['data']['dataset_kwargs'].update({
            'treatment_type': 'multilabel',
            'treatment_dim': 3,
            'orthogonality_violation': 0.15
        })
        config['training']['n_epochs_ssl'] = 25
        config['training']['n_epochs_diffusion'] = 40
        config['training']['n_epochs_vi'] = 25
        
        # Run GCMP
        results = self._run_gcmp(config, exp_dir)
        
        # Analyze treatment combinations
        self._analyze_treatment_combinations(config, exp_dir)
        
        # Save results
        self._save_results({'gcmp': results}, exp_dir / "results.json")
    
    def run_multiple_continuous_experiment(self):
        """Experiment 4: Multiple continuous treatments"""
        exp_dir = self.base_dir / "multiple_continuous"
        exp_dir.mkdir(exist_ok=True)
        
        # Configure - 2 continuous treatments
        config = get_default_config()
        config['data']['n_samples'] = 6000
        config['data']['dataset_kwargs'].update({
            'treatment_type': 'continuous',
            'treatment_dim': 2,
            'orthogonality_violation': 0.1
        })
        config['training']['n_epochs_ssl'] = 25
        config['training']['n_epochs_diffusion'] = 35
        config['training']['n_epochs_vi'] = 25
        
        # Run GCMP
        results = self._run_gcmp(config, exp_dir)
        
        # Visualize interaction effects
        self._plot_treatment_interaction_surface(config, exp_dir)
        
        # Save results
        self._save_results({'gcmp': results}, exp_dir / "results.json")
    
    def run_mixed_treatment_experiment(self):
        """Experiment 5: Mixed continuous and discrete treatments"""
        exp_dir = self.base_dir / "mixed_treatment"
        exp_dir.mkdir(exist_ok=True)
        
        if not MIXED_TREATMENT_AVAILABLE:
            print("Skipping mixed treatment experiment - module not available")
            return
        
        # This requires the mixed treatment extension
        print("Mixed treatment experiment - using specialized dataset")
        
        # Create mixed treatment specification
        treatment_spec = TreatmentSpec(
            continuous_dims=[0, 1],      # 2 continuous (e.g., drug doses)
            discrete_dims=[2, 3],        # 2 discrete
            discrete_levels={2: 2, 3: 3}  # Binary + 3-level categorical
        )
        
        # Create dataset
        dataset = MixedTreatmentDataset(
            n_samples=5000,
            n_covariates=30,
            treatment_spec=treatment_spec,
            orthogonality_violation=0.15
        )
        
        # Analyze mixed effects
        self._analyze_mixed_effects(dataset, treatment_spec, exp_dir)
        
        # Save results
        results = {'mixed_treatment': 'See visualizations'}
        self._save_results(results, exp_dir / "results.json")
    
    def run_orthogonality_sensitivity(self):
        """Experiment 6: Sensitivity to orthogonality violation"""
        exp_dir = self.base_dir / "orthogonality_sensitivity"
        exp_dir.mkdir(exist_ok=True)
        
        delta_values = [0.0, 0.1, 0.2, 0.3, 0.5]
        results = {}
        
        for delta in delta_values:
            print(f"\nTesting δ = {delta}")
            
            config = get_default_config()
            config['data']['dataset_kwargs']['orthogonality_violation'] = delta
            config['data']['n_samples'] = 3000
            config['training']['n_epochs_ssl'] = 15
            config['training']['n_epochs_diffusion'] = 20
            config['training']['n_epochs_vi'] = 15
            
            # Run quick experiment
            exp_results = self._run_gcmp(config, exp_dir / f"delta_{delta}", quick=True)
            results[delta] = exp_results
        
        # Plot sensitivity analysis
        self._plot_sensitivity_analysis(results, exp_dir)
        
        # Save results
        self._save_results(results, exp_dir / "sensitivity_results.json")
    
    def run_sample_size_scaling(self):
        """Experiment 7: Sample size scaling analysis"""
        exp_dir = self.base_dir / "sample_size_scaling"
        exp_dir.mkdir(exist_ok=True)
        
        sample_sizes = [1000, 2000, 5000, 10000]
        results = {}
        
        for n in sample_sizes:
            print(f"\nTesting n = {n}")
            
            config = get_default_config()
            config['data']['n_samples'] = n
            config['data']['dataset_kwargs']['orthogonality_violation'] = 0.2
            config['training']['n_epochs_ssl'] = 15
            config['training']['n_epochs_diffusion'] = 20
            config['training']['n_epochs_vi'] = 15
            
            # Run quick experiment
            exp_results = self._run_gcmp(config, exp_dir / f"n_{n}", quick=True)
            results[n] = exp_results
        
        # Plot scaling analysis
        self._plot_scaling_analysis(results, exp_dir)
        
        # Save results
        self._save_results(results, exp_dir / "scaling_results.json")
    
    def run_confounder_dim_analysis(self):
        """Experiment 8: Confounder dimensionality analysis"""
        exp_dir = self.base_dir / "confounder_dim"
        exp_dir.mkdir(exist_ok=True)
        
        confounder_dims = [1, 3, 5, 10]
        results = {}
        
        for d in confounder_dims:
            print(f"\nTesting confounder_dim = {d}")
            
            config = get_default_config()
            config['data']['dataset_kwargs']['confounder_dim'] = d
            config['data']['n_samples'] = 3000
            config['model']['manifold_dim'] = d
            config['training']['n_epochs_ssl'] = 15
            config['training']['n_epochs_diffusion'] = 20
            config['training']['n_epochs_vi'] = 15
            
            # Run quick experiment
            exp_results = self._run_gcmp(config, exp_dir / f"dim_{d}", quick=True)
            results[d] = exp_results
        
        # Plot dimension analysis
        self._plot_dimension_analysis(results, exp_dir)
        
        # Save results
        self._save_results(results, exp_dir / "dimension_results.json")
    
    # Helper methods
    
    def _run_gcmp(self, config: Dict, save_dir: Path, quick: bool = False) -> Dict:
        """Run GCMP experiment"""
        save_dir.mkdir(exist_ok=True)
        save_config(config, save_dir / "config.json")
        
        # Reduce epochs for quick experiments
        if quick:
            config['training']['n_folds'] = 2
        
        # Train
        start_time = time.time()
        trainer = GCMPTrainer(
            config,
            log_dir=str(save_dir / 'logs'),
            checkpoint_dir=str(save_dir / 'checkpoints'),
            use_wandb=False
        )
        
        n_epochs = {
            'ssl': config['training']['n_epochs_ssl'],
            'diffusion': config['training']['n_epochs_diffusion'],
            'vi': config['training']['n_epochs_vi']
        }
        
        trainer.train(n_epochs, n_folds=config['training'].get('n_folds', 5))
        
        # Evaluate
        results = trainer.evaluate()
        results['runtime'] = time.time() - start_time
        
        return results
    
    def _run_baselines(self, config: Dict, methods: List[str]) -> Dict[str, Dict]:
        """Run baseline methods"""
        # Get data
        data_module = DataModule(**config['data'])
        
        # Prepare data
        train_dataset = data_module.train_dataset
        test_dataset = data_module.test_dataset
        
        X_train, T_train, Y_train = self._extract_data(train_dataset)
        X_test, T_test, Y_test = self._extract_data(test_dataset)
        
        # Get true effects if available
        true_effects = None
        if hasattr(data_module.dataset, 'get_true_effect'):
            true_effects = []
            for i in range(len(X_test)):
                effect = data_module.dataset.get_true_effect(
                    X_test[i:i+1], T_test[i:i+1]
                )
                true_effects.append(effect[0])
            true_effects = np.array(true_effects)
        
        results = {}
        
        for method in methods:
            print(f"Running {method}...")
            start_time = time.time()
            
            # Create and fit model
            if method in ['cevae', 'contivae']:
                model = create_baseline(
                    method,
                    input_dim=config['data']['n_covariates'],
                    treatment_type=config['data']['dataset_kwargs']['treatment_type']
                )
                model.fit(X_train, T_train, Y_train, n_epochs=30)
            else:
                model = create_baseline(
                    method,
                    treatment_type=config['data']['dataset_kwargs']['treatment_type']
                )
                model.fit(X_train, T_train, Y_train)
            
            # Predict
            if config['data']['dataset_kwargs']['treatment_type'] == 'continuous':
                t0 = np.zeros_like(T_test)
                t1 = np.ones_like(T_test)
            else:
                t0 = np.zeros_like(T_test)
                t1 = np.ones_like(T_test)
            
            pred_effects = model.predict_ite(X_test, t0, t1)
            
            # Evaluate
            method_results = {'runtime': time.time() - start_time}
            
            if true_effects is not None:
                method_results['pehe'] = np.sqrt(np.mean((pred_effects - true_effects)**2))
                method_results['ate_error'] = abs(pred_effects.mean() - true_effects.mean())
            
            results[method] = method_results
        
        return results
    
    def _extract_data(self, dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract numpy arrays from dataset"""
        X, T, Y = [], [], []
        for i in range(len(dataset)):
            sample = dataset[i]
            X.append(sample['covariates'].numpy())
            T.append(sample['treatment'].numpy())
            Y.append(sample['outcome'].numpy())
        
        X = np.array(X)
        T = np.array(T)
        Y = np.array(Y).flatten()
        
        return X, T, Y
    
    def _plot_dose_response(self, config: Dict, save_dir: Path):
        """Plot dose-response curves"""
        # Create a trained model
        trainer = GCMPTrainer(config, use_wandb=False)
        
        # Quick training for visualization
        trainer.train({'ssl': 5, 'diffusion': 10, 'vi': 5}, n_folds=2)
        
        # Generate dose-response curve
        test_loader = trainer.data_module.get_dataloader('test')
        batch = next(iter(test_loader))
        
        x = batch['covariates'][:20].to(trainer.device)
        doses = np.linspace(0, 2, 50)
        
        with torch.no_grad():
            phi_x = trainer.model.ssl_encoder.get_representation(x)
            
            outcomes = []
            for dose in doses:
                t = torch.full((20, 1), dose, device=trainer.device)
                y_pred, _ = trainer.model.vi_model.predict_outcome(phi_x, t)
                outcomes.append(y_pred.mean().item())
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(doses, outcomes, 'b-', linewidth=2)
        plt.xlabel('Treatment Dose')
        plt.ylabel('Expected Outcome')
        plt.title('Average Dose-Response Curve')
        plt.grid(True, alpha=0.3)
        plt.savefig(save_dir / 'dose_response.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_treatment_combinations(self, config: Dict, save_dir: Path):
        """Analyze multi-label treatment combinations"""
        # Create dataset
        data_module = DataModule(**config['data'])
        dataset = data_module.dataset
        
        # Get treatment combinations and their effects
        n_treatments = config['data']['dataset_kwargs']['treatment_dim']
        combinations = []
        effects = []
        
        # Sample some data points
        n_samples = 100
        X_sample = dataset.X[:n_samples]
        
        # Generate all 2^n combinations
        for i in range(2**n_treatments):
            combo = [(i >> j) & 1 for j in range(n_treatments)]
            t = np.array(combo, dtype=np.float32)
            
            # Compute average effect
            avg_effect = 0
            for j in range(n_samples):
                effect = dataset.get_true_effect(
                    X_sample[j:j+1],
                    t.reshape(1, -1),
                    np.zeros((1, n_treatments))
                )[0]
                avg_effect += effect
            
            combinations.append(combo)
            effects.append(avg_effect / n_samples)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        
        # Convert to dataframe for better visualization
        combo_strings = [''.join(map(str, c)) for c in combinations]
        
        # Sort by effect size
        sorted_idx = np.argsort(effects)[::-1]
        
        plt.barh(range(len(combinations)), 
                [effects[i] for i in sorted_idx],
                color=plt.cm.RdBu(np.array(effects)[sorted_idx] / max(abs(min(effects)), max(effects)) * 0.5 + 0.5))
        
        plt.yticks(range(len(combinations)), [combo_strings[i] for i in sorted_idx])
        plt.xlabel('Average Treatment Effect')
        plt.title(f'Effects of {n_treatments}-Treatment Combinations')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'treatment_combinations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_treatment_interaction_surface(self, config: Dict, save_dir: Path):
        """Plot interaction surface for 2 continuous treatments"""
        # Create dataset
        data_module = DataModule(**config['data'])
        dataset = data_module.dataset
        
        # Create grid
        t1_range = np.linspace(0, 1, 20)
        t2_range = np.linspace(0, 1, 20)
        T1, T2 = np.meshgrid(t1_range, t2_range)
        
        # Sample a few covariates
        X_sample = dataset.X[:10]
        
        # Compute effects
        effects = np.zeros_like(T1)
        
        for i in range(20):
            for j in range(20):
                t = np.array([T1[i, j], T2[i, j]]).reshape(1, -1)
                t0 = np.zeros((1, 2))
                
                # Average over samples
                avg_effect = 0
                for k in range(10):
                    effect = dataset.get_true_effect(X_sample[k:k+1], t, t0)[0]
                    avg_effect += effect
                
                effects[i, j] = avg_effect / 10
        
        # Plot surface
        fig = plt.figure(figsize=(12, 5))
        
        # 3D surface
        ax1 = fig.add_subplot(121, projection='3d')
        surf = ax1.plot_surface(T1, T2, effects, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('Treatment 1')
        ax1.set_ylabel('Treatment 2')
        ax1.set_zlabel('Average Effect')
        ax1.set_title('Treatment Interaction Surface')
        
        # Contour plot
        ax2 = fig.add_subplot(122)
        contour = ax2.contourf(T1, T2, effects, levels=20, cmap='viridis')
        fig.colorbar(contour, ax=ax2)
        ax2.set_xlabel('Treatment 1')
        ax2.set_ylabel('Treatment 2')
        ax2.set_title('Treatment Interaction Contours')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'treatment_interaction.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_mixed_effects(self, dataset, treatment_spec, save_dir: Path):
        """Analyze mixed treatment effects"""
        if not MIXED_TREATMENT_AVAILABLE:
            return
            
        # Sample data
        n_samples = 100
        X_sample = dataset.data['X'][:n_samples]
        
        # Analyze continuous treatment effects
        continuous_effects = []
        for i, idx in enumerate(treatment_spec.continuous_dims):
            effects = []
            for delta in np.linspace(0, 1, 10):
                avg_effect = 0
                for j in range(n_samples):
                    t_base = dataset.data['T'][j].copy()
                    t_new = t_base.copy()
                    t_new[idx] += delta
                    
                    # Compute effect
                    y_base = dataset._compute_structural_outcome(
                        X_sample[j:j+1], 
                        t_base.reshape(1, -1),
                        np.zeros((1, dataset.confounder_dim))
                    )[0]
                    y_new = dataset._compute_structural_outcome(
                        X_sample[j:j+1],
                        t_new.reshape(1, -1),
                        np.zeros((1, dataset.confounder_dim))
                    )[0]
                    
                    avg_effect += (y_new - y_base)
                
                effects.append(avg_effect / n_samples)
            continuous_effects.append(effects)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Continuous effects
        ax = axes[0]
        for i, effects in enumerate(continuous_effects):
            ax.plot(np.linspace(0, 1, 10), effects, label=f'Continuous {i+1}')
        ax.set_xlabel('Treatment Change')
        ax.set_ylabel('Average Effect')
        ax.set_title('Continuous Treatment Effects')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Discrete effects
        ax = axes[1]
        discrete_effects = {}
        for i, (idx, n_levels) in enumerate(treatment_spec.discrete_levels.items()):
            effects = []
            for level in range(n_levels):
                avg_effect = 0
                for j in range(n_samples):
                    t_base = dataset.data['T'][j].copy()
                    t_new = t_base.copy()
                    t_new[idx] = level
                    
                    # Compute effect relative to level 0
                    t_base[idx] = 0
                    
                    y_base = dataset._compute_structural_outcome(
                        X_sample[j:j+1],
                        t_base.reshape(1, -1),
                        np.zeros((1, dataset.confounder_dim))
                    )[0]
                    y_new = dataset._compute_structural_outcome(
                        X_sample[j:j+1],
                        t_new.reshape(1, -1),
                        np.zeros((1, dataset.confounder_dim))
                    )[0]
                    
                    avg_effect += (y_new - y_base)
                
                effects.append(avg_effect / n_samples)
            discrete_effects[f'Discrete {i+1}'] = effects
        
        # Plot discrete effects
        x_pos = 0
        for name, effects in discrete_effects.items():
            positions = x_pos + np.arange(len(effects))
            ax.bar(positions, effects, label=name, alpha=0.7, width=0.8)
            x_pos += len(effects) + 0.5
        
        ax.set_xlabel('Treatment Level')
        ax.set_ylabel('Average Effect')
        ax.set_title('Discrete Treatment Effects')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'mixed_treatment_effects.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sensitivity_analysis(self, results: Dict[float, Dict], save_dir: Path):
        """Plot sensitivity to orthogonality violation"""
        delta_values = sorted(results.keys())
        
        metrics = ['pehe', 'ate_error', 'coverage_95', 'orthogonality_score']
        metric_values = {m: [] for m in metrics}
        
        for delta in delta_values:
            for metric in metrics:
                value = results[delta].get(metric, np.nan)
                metric_values[metric].append(value)
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            ax.plot(delta_values, metric_values[metric], 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Orthogonality Violation (δ)')
            ax.set_ylabel(metric.upper().replace('_', ' '))
            ax.set_title(f'{metric.upper().replace("_", " ")} vs Orthogonality Violation')
            ax.grid(True, alpha=0.3)
            
            if metric == 'coverage_95':
                ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='Target')
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_scaling_analysis(self, results: Dict[int, Dict], save_dir: Path):
        """Plot sample size scaling"""
        n_values = sorted(results.keys())
        
        metrics = ['pehe', 'interval_width', 'runtime']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = [results[n].get(metric, np.nan) for n in n_values]
            
            ax.loglog(n_values, values, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Sample Size (n)')
            ax.set_ylabel(metric.upper().replace('_', ' '))
            ax.set_title(f'{metric.upper().replace("_", " ")} vs Sample Size')
            ax.grid(True, alpha=0.3)
            
            # Add theoretical scaling lines
            if metric == 'pehe':
                # Should scale as 1/sqrt(n)
                theory = values[0] * np.sqrt(n_values[0] / np.array(n_values))
                ax.loglog(n_values, theory, 'k--', alpha=0.5, label='O(1/√n)')
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'scaling_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_dimension_analysis(self, results: Dict[int, Dict], save_dir: Path):
        """Plot confounder dimension analysis"""
        dim_values = sorted(results.keys())
        
        metrics = ['pehe', 'runtime']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = [results[d].get(metric, np.nan) for d in dim_values]
            
            ax.plot(dim_values, values, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Confounder Dimension')
            ax.set_ylabel(metric.upper().replace('_', ' '))
            ax.set_title(f'{metric.upper().replace("_", " ")} vs Confounder Dimension')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'dimension_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self, results: Dict, path: Path):
        """Save results to JSON"""
        # Convert numpy values to Python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        results_converted = convert(results)
        
        with open(path, 'w') as f:
            json.dump(results_converted, f, indent=2)
        
        print(f"Results saved to: {path}")


def main():
    """Run all experiments"""
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create experiment suite
    suite = ExperimentSuite()
    
    # Run all experiments
    suite.run_all_experiments()


if __name__ == "__main__":
    main()