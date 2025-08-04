"""
Utility modules for GCMP framework
Includes diffusion utilities, evaluation metrics, and configuration
FIXED VERSION with proper orthogonality regularization in diffusion training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# ============== Diffusion Utilities ==============

class DiffusionSchedule:
    """Manages diffusion noise schedule"""
    
    def __init__(self, 
                 n_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 schedule_type: str = 'linear'):
        
        self.n_timesteps = n_timesteps
        
        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, n_timesteps)
        elif schedule_type == 'cosine':
            # Cosine schedule
            s = 0.008
            x = torch.linspace(0, n_timesteps, n_timesteps + 1)
            alphas_cumprod = torch.cos(((x / n_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # Pre-compute useful quantities
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # For sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # For posterior
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample from q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise


class DiffusionTrainer:
    """
    Trainer for conditional diffusion model
    FIXED: Now includes orthogonality regularization as per paper
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 config: Dict,
                 orthogonality_regularizer: Optional[nn.Module] = None):
        
        self.model = model
        self.device = device
        self.config = config
        self.orthogonality_regularizer = orthogonality_regularizer
        
        # Setup diffusion schedule
        self.schedule = DiffusionSchedule(
            n_timesteps=config.get('n_timesteps', 1000),
            beta_start=config.get('beta_start', 0.0001),
            beta_end=config.get('beta_end', 0.02),
            schedule_type=config.get('schedule_type', 'linear')
        )
        
        # Move schedule to device
        for attr in ['betas', 'alphas', 'alphas_cumprod', 'sqrt_alphas_cumprod', 
                     'sqrt_one_minus_alphas_cumprod', 'sqrt_recip_alphas']:
            setattr(self.schedule, attr, getattr(self.schedule, attr).to(device))
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 1e-4))
        
        # Loss weights
        self.lambda_outcome = config.get('lambda_outcome', 1.0)
        self.lambda_orthogonal = config.get('lambda_orthogonal', 0.1)
    
    def train_epoch(self, 
                   data_loader: torch.utils.data.DataLoader,
                   ssl_encoder: nn.Module,
                   outcome_model: nn.Module) -> float:
        """Train for one epoch with orthogonality regularization"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(data_loader, desc="Diffusion Training"):
            x = batch['covariates'].to(self.device)
            t = batch['treatment'].to(self.device)
            y = batch['outcome'].to(self.device)
            
            # Get SSL representations
            with torch.no_grad():
                phi_x = ssl_encoder.get_representation(x)
            
            # Sample timesteps
            batch_size = x.shape[0]
            timesteps = torch.randint(0, self.schedule.n_timesteps, (batch_size,), device=self.device)
            
            # Generate counterfactual treatments
            if t.shape[1] == 1:  # Continuous
                t_cf = torch.randn_like(t)
            else:  # Multi-label
                t_cf = 1 - t
            
            # Create conditioning
            condition = torch.cat([phi_x, t, y, t_cf], dim=-1)
            
            # Sample noise and perturbed data
            noise = torch.randn_like(phi_x)
            x_t = self.schedule.q_sample(phi_x, timesteps, noise)
            
            # Predict noise
            noise_pred = self.model(x_t, timesteps, condition)
            
            # Diffusion loss
            diff_loss = F.mse_loss(noise_pred, noise)
            
            # Outcome consistency loss
            outcome_loss = 0
            if self.lambda_outcome > 0:
                # Sample some perturbations using current model
                with torch.no_grad():
                    # Simple one-step denoising for efficiency during training
                    # Use a small batch to save memory
                    small_batch_size = min(16, batch_size)
                    t_zero = torch.zeros(small_batch_size, dtype=torch.long, device=self.device)
                    x_noisy = torch.randn(small_batch_size, phi_x.shape[1], device=self.device)
                    noise_pred_final = self.model(x_noisy, t_zero, condition[:small_batch_size])
                    delta_phi = x_noisy - noise_pred_final  # Approximation
                
                # Check outcome consistency
                phi_perturbed = phi_x[:small_batch_size] + delta_phi
                xt_cf = torch.cat([phi_perturbed, t_cf[:small_batch_size]], dim=-1)
                y_pred = outcome_model(xt_cf)
                
                outcome_loss = F.mse_loss(y_pred, y[:small_batch_size])
            
            # CRITICAL FIX: Orthogonality regularization on generated perturbations
            ortho_reg = 0
            if self.lambda_orthogonal > 0 and self.orthogonality_regularizer is not None:
                # Generate clean perturbations for regularization
                # Use smaller batch to save memory and computation
                reg_batch_size = min(8, batch_size)
                with torch.no_grad():
                    # Quick sampling with fewer steps
                    sampler = DiffusionSampler(self.model, self.device, {
                        'n_steps': 5,  # Reduced steps for training efficiency
                        'n_timesteps': self.schedule.n_timesteps,
                        'beta_start': 0.0001,
                        'beta_end': 0.02,
                        'schedule_type': 'linear'
                    })
                    # Use the appropriate condition size
                    delta_phi_clean = sampler.sample(reg_batch_size, condition[:reg_batch_size])
                
                # Compute orthogonality regularization
                ortho_reg = self.orthogonality_regularizer(
                    phi_x[:reg_batch_size], 
                    t[:reg_batch_size], 
                    delta_phi_clean
                )
            
            # Total loss (as per paper equation 8)
            loss = diff_loss
            if self.lambda_outcome > 0:
                loss = loss + self.lambda_outcome * outcome_loss
            if self.lambda_orthogonal > 0 and isinstance(ortho_reg, torch.Tensor):
                loss = loss + self.lambda_orthogonal * ortho_reg
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    def validate(self,
                val_loader: torch.utils.data.DataLoader,
                ssl_encoder: nn.Module,
                outcome_model: nn.Module) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['covariates'].to(self.device)
                t = batch['treatment'].to(self.device)
                y = batch['outcome'].to(self.device)
                
                phi_x = ssl_encoder.get_representation(x)
                
                batch_size = x.shape[0]
                timesteps = torch.randint(0, self.schedule.n_timesteps, (batch_size,), device=self.device)
                
                if t.shape[1] == 1:
                    t_cf = torch.randn_like(t)
                else:
                    t_cf = 1 - t
                
                condition = torch.cat([phi_x, t, y, t_cf], dim=-1)
                
                noise = torch.randn_like(phi_x)
                x_t = self.schedule.q_sample(phi_x, timesteps, noise)
                
                noise_pred = self.model(x_t, timesteps, condition)
                
                loss = F.mse_loss(noise_pred, noise)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)


class DiffusionSampler:
    """Sampler for conditional diffusion model"""
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 config: Dict):
        
        self.model = model
        self.device = device
        self.config = config
        
        # Setup schedule
        self.schedule = DiffusionSchedule(
            n_timesteps=config.get('n_timesteps', 1000),
            beta_start=config.get('beta_start', 0.0001),
            beta_end=config.get('beta_end', 0.02),
            schedule_type=config.get('schedule_type', 'linear')
        )
        
        # Move to device
        for attr in ['betas', 'alphas', 'alphas_cumprod', 'sqrt_recip_alphas', 'posterior_variance']:
            setattr(self.schedule, attr, getattr(self.schedule, attr).to(device))
        
        self.n_steps = config.get('n_steps', 50)  # DDIM steps
    
    @torch.no_grad()
    def sample(self, 
               batch_size: int,
               condition: torch.Tensor,
               return_trajectory: bool = False) -> torch.Tensor:
        """Sample perturbations using DDIM"""
        self.model.eval()
        
        # Start from noise
        dim = self.model.input_dim
        x = torch.randn(batch_size, dim, device=self.device)
        
        # DDIM sampling
        timesteps = torch.linspace(self.schedule.n_timesteps - 1, 0, self.n_steps, dtype=torch.long)
        
        trajectory = []
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.model(x, t_batch, condition)
            
            # DDIM update
            alpha_t = self.schedule.alphas_cumprod[t]
            if i < len(timesteps) - 1:
                alpha_prev = self.schedule.alphas_cumprod[timesteps[i+1]]
            else:
                alpha_prev = torch.tensor(1.0, device=self.device)
            
            # Ensure alpha values are tensors
            if not isinstance(alpha_t, torch.Tensor):
                alpha_t = torch.tensor(alpha_t, device=self.device)
            if not isinstance(alpha_prev, torch.Tensor):
                alpha_prev = torch.tensor(alpha_prev, device=self.device)
            
            # Predict x_0
            x_0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            
            # Sample x_{t-1}
            x = torch.sqrt(alpha_prev) * x_0_pred + torch.sqrt(1 - alpha_prev) * noise_pred
            
            if return_trajectory:
                trajectory.append(x.clone())
        
        if return_trajectory:
            return torch.stack(trajectory)
        else:
            return x


# ============== Evaluation Utilities ==============

class CausalEvaluator:
    """Evaluator for causal effect estimation"""
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 config: Dict):
        
        self.model = model
        self.device = device
        self.config = config
    
    def evaluate(self, test_loader: torch.utils.data.DataLoader, 
                 test_dataset: Optional[torch.utils.data.Dataset] = None) -> Dict[str, float]:
        """Comprehensive evaluation"""
        self.model.eval()
        
        predictions = []
        true_effects = []
        uncertainties = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                x = batch['covariates'].to(self.device)
                t = batch['treatment'].to(self.device)
                y = batch['outcome'].to(self.device)
                
                # Predict effects
                if t.shape[1] == 1:  # Continuous
                    t0 = torch.zeros_like(t)
                    t1 = torch.ones_like(t)
                else:  # Binary/Multi-label
                    t0 = torch.zeros_like(t)
                    t1 = torch.ones_like(t)
                
                # Get representations
                phi_x = self.model.ssl_encoder.get_representation(x)
                
                # Estimate effects with uncertainty
                effect_mean, effect_std = self.model.vi_model.estimate_treatment_effect(
                    phi_x, t0, t1, n_samples=100
                )
                
                # Ensure we're collecting 1D arrays
                batch_predictions = effect_mean.cpu().numpy().flatten()
                batch_uncertainties = effect_std.cpu().numpy().flatten()
                
                predictions.append(batch_predictions)
                uncertainties.append(batch_uncertainties)
                
                # Get true effects if available - MUST match the batch size
                if test_dataset is not None and hasattr(test_dataset.dataset, 'get_true_effect'):
                    
                    # Get the underlying dataset object, which has the get_true_effect method
                    actual_dataset = test_dataset.dataset if hasattr(test_dataset, 'dataset') else test_dataset
                    
                    # 修复：对当前批次进行标准化的真实效应计算
                    # 为当前批次的所有样本定义固定的反事实治疗 t=1 和 t=0
                    x_np = x.cpu().numpy()
                    t1_batch = np.ones((x_np.shape[0], t.shape[1]))
                    t0_batch = np.zeros((x_np.shape[0], t.shape[1]))

                    # 直接调用函数对整个批次进行计算
                    batch_true_effects = actual_dataset.get_true_effect(
                        x_np,
                        t1_batch,
                        t0_batch
                    )
                    true_effects.extend(batch_true_effects.flatten())
        
        # Concatenate results
        predictions = np.concatenate(predictions)
        uncertainties = np.concatenate(uncertainties)
        
        results = {}
        
        # If true effects available, compute metrics
        if len(true_effects) > 0:
            true_effects = np.array(true_effects)
            
            # Debug info
            if predictions.shape != true_effects.shape:
                print(f"WARNING: Shape mismatch - predictions: {predictions.shape}, true_effects: {true_effects.shape}")
                # Take the minimum length to avoid errors
                min_len = min(len(predictions), len(true_effects))
                predictions = predictions[:min_len]
                uncertainties = uncertainties[:min_len]
                true_effects = true_effects[:min_len]
            
            # PEHE (Precision in Estimation of Heterogeneous Effect)
            pehe = np.sqrt(np.mean((predictions - true_effects)**2))
            results['pehe'] = pehe
            
            # ATE error
            ate_pred = predictions.mean()
            ate_true = true_effects.mean()
            results['ate_error'] = abs(ate_pred - ate_true)
            
            # Coverage
            lower = predictions - 1.96 * uncertainties
            upper = predictions + 1.96 * uncertainties
            coverage = np.mean((true_effects >= lower) & (true_effects <= upper))
            results['coverage_95'] = coverage
            
            # Interval width
            results['interval_width'] = np.mean(upper - lower)
        
        # Outcome prediction metrics
        outcome_mse = self._evaluate_outcome_prediction(test_loader)
        results['outcome_mse'] = outcome_mse
        
        # Orthogonality diagnostic
        ortho_score = self._evaluate_orthogonality(test_loader, test_dataset)
        results['orthogonality_score'] = ortho_score
        
        return results
    
    def _evaluate_outcome_prediction(self, test_loader: torch.utils.data.DataLoader) -> float:
        """Evaluate outcome prediction accuracy"""
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                x = batch['covariates'].to(self.device)
                t = batch['treatment'].to(self.device)
                y = batch['outcome'].to(self.device)
                
                # Get representations
                phi_x = self.model.ssl_encoder.get_representation(x)
                
                # Predict outcome
                y_pred, _ = self.model.vi_model.predict_outcome(phi_x, t)
                
                predictions.append(y_pred.cpu().numpy())
                targets.append(y.cpu().numpy())
        
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        return mean_squared_error(targets, predictions)
    
    def _evaluate_orthogonality(self, test_loader: torch.utils.data.DataLoader,
                               test_dataset: Optional[torch.utils.data.Dataset] = None) -> float:
        """Evaluate effective orthogonality measure"""
        if test_dataset is None:
            return -1
            
        # Get the actual dataset (handle Subset wrapper)
        if hasattr(test_dataset, 'dataset'):
            actual_dataset = test_dataset.dataset
        else:
            actual_dataset = test_dataset
            
        if not hasattr(actual_dataset, 'get_orthogonality_measure'):
            return -1  # Not available
        
        ortho_scores = []
        
        for batch in test_loader:
            x = batch['covariates']
            
            for i in range(len(x)):
                x_i = x[i].cpu().numpy()
                score = actual_dataset.get_orthogonality_measure(x_i)
                ortho_scores.append(score)
        
        return np.mean(ortho_scores) if ortho_scores else -1
    
    def plot_results(self, results: Dict[str, float], save_path: Optional[str] = None):
        """Plot evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Metrics bar plot
        metrics = {k: v for k, v in results.items() if isinstance(v, (int, float)) and v >= 0}
        ax = axes[0, 0]
        ax.bar(metrics.keys(), metrics.values())
        ax.set_title('Evaluation Metrics')
        ax.set_xticklabels(metrics.keys(), rotation=45)
        
        # Placeholder for other plots
        for ax in axes.flat[1:]:
            ax.set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


# ============== Configuration Utilities ==============

def get_default_config() -> Dict:
    """Get default configuration for GCMP"""
    config = {
        "seed": 42,
        "data": {
            "dataset_name": "synthetic",
            "n_samples": 10000,
            "n_covariates": 50,
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "normalize": True,
            "dataset_kwargs": {
                "confounder_dim": 3,
                "treatment_type": "continuous",
                "orthogonality_violation": 0.0,
                "nonlinearity": "neural",
                "noise_level": 0.1
            }
        },
        "model": {
            "ssl_hidden_dims": [256, 128],
            "ssl_output_dim": 64,
            "diffusion_hidden_dims": [256, 512, 256],
            "vi_hidden_dims": [128, 64],
            "latent_dim": 10,
            "manifold_dim": 3,
            "dropout": 0.1
        },
        "ssl": {
            "lr": 0.001,
            "n_epochs": 100,
            "lambda_preserve": 0.1,
            "lambda_diverse": 0.01,
            "temperature": 0.5,
            "augment_noise": 0.1,
            "gamma": 1.0
        },
        "diffusion": {
            "time_embed_dim": 128,
            "num_heads": 8,
            "dropout": 0.1
        },
        "diffusion_training": {
            "lr": 0.0001,
            "n_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "schedule_type": "linear",
            "lambda_outcome": 1.0,
            "lambda_orthogonal": 0.1
        },
        "diffusion_sampling": {
            "n_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "schedule_type": "linear",
            "n_steps": 50
        },
        "vi": {
            "latent_dim": 10,
            "hidden_dims": [128, 64],
            "dropout": 0.1
        },
        "vi_training": {
            "lr": 0.001,
            "lambda_orthogonal": 0.1,
            "lambda_entropy": 0.01
        },
        "training": {
            "batch_size": 256,
            "n_epochs_ssl": 50,
            "n_epochs_diffusion": 100,
            "n_epochs_vi": 50,
            "n_folds": 5
        },
        "evaluation": {
            "n_bootstrap": 100,
            "confidence_level": 0.95
        }
    }
    
    return config


def save_config(config: Dict, path: str):
    """Save configuration to file"""
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(path: str) -> Dict:
    """Load configuration from file"""
    with open(path, 'r') as f:
        return json.load(f)


# ============== Experiment Utilities ==============

class ExperimentRunner:
    """Utility for running experiments with different configurations"""
    
    def __init__(self, base_config: Dict, results_dir: str = './results'):
        self.base_config = base_config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_orthogonality_experiment(self, delta_values: List[float]):
        """Run experiment varying orthogonality violation"""
        results = {}
        
        for delta in delta_values:
            print(f"\nRunning experiment with δ={delta}")
            
            # Update config
            config = self.base_config.copy()
            config['data']['dataset_kwargs']['orthogonality_violation'] = delta
            
            # Run experiment
            exp_results = self._run_single_experiment(config, f"delta_{delta}")
            results[delta] = exp_results
        
        # Plot results
        self._plot_orthogonality_results(results)
        
        return results
    
    def run_sample_size_experiment(self, sample_sizes: List[int]):
        """Run experiment varying sample size"""
        results = {}
        
        for n in sample_sizes:
            print(f"\nRunning experiment with n={n}")
            
            # Update config
            config = self.base_config.copy()
            config['data']['n_samples'] = n
            
            # Run experiment
            exp_results = self._run_single_experiment(config, f"n_{n}")
            results[n] = exp_results
        
        # Plot results
        self._plot_sample_size_results(results)
        
        return results
    
    def _run_single_experiment(self, config: Dict, exp_name: str) -> Dict:
        """Run a single experiment"""
        from main import GCMPTrainer
        
        # Create experiment directory
        exp_dir = self.results_dir / exp_name
        exp_dir.mkdir(exist_ok=True)
        
        # Save config
        save_config(config, exp_dir / 'config.json')
        
        # Run training
        trainer = GCMPTrainer(
            config,
            log_dir=str(exp_dir / 'logs'),
            checkpoint_dir=str(exp_dir / 'checkpoints'),
            use_wandb=False
        )
        
        n_epochs = {
            'ssl': config['training']['n_epochs_ssl'],
            'diffusion': config['training']['n_epochs_diffusion'],
            'vi': config['training']['n_epochs_vi']
        }
        
        trainer.train(n_epochs)
        
        # Evaluate
        results = trainer.evaluate()
        
        # Save results
        with open(exp_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _plot_orthogonality_results(self, results: Dict[float, Dict]):
        """Plot results from orthogonality experiment"""
        delta_values = sorted(results.keys())
        pehe_values = [results[d].get('pehe', np.nan) for d in delta_values]
        coverage_values = [results[d].get('coverage_95', np.nan) for d in delta_values]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # PEHE vs delta
        ax1.plot(delta_values, pehe_values, 'o-')
        ax1.set_xlabel('Orthogonality Violation (δ)')
        ax1.set_ylabel('PEHE')
        ax1.set_title('Estimation Error vs Orthogonality Violation')
        ax1.grid(True)
        
        # Coverage vs delta
        ax2.plot(delta_values, coverage_values, 'o-')
        ax2.axhline(y=0.95, color='r', linestyle='--', label='Target Coverage')
        ax2.set_xlabel('Orthogonality Violation (δ)')
        ax2.set_ylabel('95% CI Coverage')
        ax2.set_title('Coverage vs Orthogonality Violation')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'orthogonality_results.png')
        plt.close()
    
    def _plot_sample_size_results(self, results: Dict[int, Dict]):
        """Plot results from sample size experiment"""
        n_values = sorted(results.keys())
        pehe_values = [results[n].get('pehe', np.nan) for n in n_values]
        interval_widths = [results[n].get('interval_width', np.nan) for n in n_values]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # PEHE vs n
        ax1.loglog(n_values, pehe_values, 'o-')
        ax1.set_xlabel('Sample Size (n)')
        ax1.set_ylabel('PEHE')
        ax1.set_title('Estimation Error vs Sample Size')
        ax1.grid(True)
        
        # Interval width vs n
        ax2.loglog(n_values, interval_widths, 'o-')
        ax2.set_xlabel('Sample Size (n)')
        ax2.set_ylabel('95% CI Width')
        ax2.set_title('Uncertainty vs Sample Size')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'sample_size_results.png')
        plt.close()


# ============== Comparison Utilities ==============

def compare_methods(methods: Dict[str, Dict], metrics: List[str] = ['pehe', 'coverage_95', 'runtime']):
    """Compare multiple methods on key metrics"""
    
    # Create comparison table
    comparison = {}
    for metric in metrics:
        comparison[metric] = {}
        for method_name, results in methods.items():
            comparison[metric][method_name] = results.get(metric, np.nan)
    
    # Print table
    print("\nMethod Comparison:")
    print("-" * 60)
    print(f"{'Method':<20} " + " ".join([f"{m:<15}" for m in metrics]))
    print("-" * 60)
    
    for method_name in methods.keys():
        values = [f"{comparison[m][method_name]:<15.4f}" for m in metrics]
        print(f"{method_name:<20} " + " ".join(values))
    
    # Create bar plots
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        method_names = list(methods.keys())
        values = [comparison[metric][m] for m in method_names]
        
        bars = ax.bar(method_names, values)
        ax.set_title(metric.upper())
        ax.set_xticks(range(len(method_names)))
        ax.set_xticklabels(method_names, rotation=45)
        
        # Color best performer
        if metric == 'pehe' or metric == 'runtime':  # Lower is better
            best_idx = np.nanargmin(values)
        else:  # Higher is better
            best_idx = np.nanargmax(values)
        bars[best_idx].set_color('green')
    
    plt.tight_layout()
    plt.savefig('method_comparison.png')
    plt.show()
    
    return comparison