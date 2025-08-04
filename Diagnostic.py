"""
GCMP Diagnostic Analysis for Single Continuous Treatment
Implements three practical metrics: Orthogonality Diagnostic, Manifold Quality, and Proxy Validity
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import warnings
warnings.filterwarnings('ignore')
import time
import traceback
import sys
from tqdm import tqdm
# Import required modules from the GCMP framework
from data import DataModule, SyntheticCausalDataset
from models import GCMP
from main import GCMPTrainer
from utils import get_default_config

class GCMPDiagnostics:
    """Comprehensive diagnostic tools for GCMP model evaluation"""
    
    def __init__(self, model, data_module, device='cuda'):
        self.model = model
        self.data_module = data_module
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def _ensure_eval_mode(self):
        self.model.eval()
        if hasattr(self.model, 'ssl_encoder'):
            self.model.ssl_encoder.eval()
        if hasattr(self.model, 'diffusion_model'):
            self.model.diffusion_model.eval()
        if hasattr(self.model, 'vi_model'):
            self.model.vi_model.eval()
            if hasattr(self.model.vi_model, 'outcome_model'):
                self.model.vi_model.outcome_model.eval()
        if hasattr(self.model, 'proxy_projection'):
            self.model.proxy_projection.eval()  
        if hasattr(self.model, 'treatment_encoder'):
            self.model.treatment_encoder.eval()
        if hasattr(self.model, 'confounder_encoder'):
            self.model.confounder_encoder.eval()
        if hasattr(self.model, 'outcome_decoder'):
            self.model.outcome_decoder.eval()
        if hasattr(self.model, 'ssl_decoder'):
            self.model.ssl_decoder.eval()
        if hasattr(self.model, 'diffusion_decoder'):
            self.model.diffusion_decoder.eval()
        if hasattr(self.model, 'vi_decoder'):
            self.model.vi_decoder.eval()

    def compute_orthogonality_diagnostic(self, test_loader, n_samples=1000):
        """
        Compute and visualize the effective orthogonality measure ρ⊥(x) across the covariate space.
        FIXED: Ensures model is in eval mode to prevent BatchNorm errors with single samples.
        """
        print("Computing Orthogonality Diagnostic...")
        
        # Ensure all model components are in evaluation mode.
        # This is crucial to prevent BatchNorm layers from failing on single-sample inputs.
        self._ensure_eval_mode()
        
        orthogonality_scores = []
        covariates_list = []
        
        all_phi_x = []
        with torch.no_grad():
            for batch in test_loader:
                x = batch['covariates'].to(self.device)
                phi_x = self.model.ssl_encoder.get_representation(x)
                all_phi_x.append(phi_x)
                if sum(len(p) for p in all_phi_x) >= n_samples:
                    break
        
        all_phi_x = torch.cat(all_phi_x, dim=0)[:n_samples]

        # Process each sample to compute its orthogonality score
        for i in tqdm(range(len(all_phi_x)), desc="Orthogonality Calculation"):
            # Isolate one sample and enable gradient tracking for it
            phi_x_i = all_phi_x[i:i+1].detach().clone().requires_grad_(True)
            
            # Create a dummy treatment for gradient computation (e.g., t=0)
            t_dummy = torch.zeros(1, self.model.vi_model.treatment_dim, device=self.device)
            
            # Concatenate representation and treatment
            xt = torch.cat([phi_x_i, t_dummy], dim=-1)
            
            # Compute outcome prediction
            y_pred = self.model.vi_model.outcome_model(xt)
            
            # Compute the gradient of the outcome with respect to the representation phi_x_i
            grad = torch.autograd.grad(
                outputs=y_pred.sum(),
                inputs=phi_x_i,
                create_graph=False, # We don't need higher-order gradients
                retain_graph=False
            )[0]
            
            if grad is not None:
                # Estimate the local tangent space of the confounder manifold
                tangent_basis = self._estimate_local_tangent_space(all_phi_x, i)
                
                if tangent_basis is not None:
                    grad_norm = torch.norm(grad)
                    if grad_norm > 1e-8:
                        # Project the gradient onto the tangent space
                        grad_flat = grad.flatten()
                        proj_grad = tangent_basis @ (tangent_basis.T @ grad_flat)
                        proj_norm = torch.norm(proj_grad)
                        
                        # Orthogonality measure: 1 - ||proj(grad)||^2 / ||grad||^2
                        rho = 1 - (proj_norm**2 / grad_norm**2).item()
                        orthogonality_scores.append(rho)

        orthogonality_scores = np.array(orthogonality_scores)
        
        # Visualization (remains the same)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Distribution of ρ⊥
        ax = axes[0, 0]
        ax.hist(orthogonality_scores, bins=50, density=True, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(orthogonality_scores), color='red', linestyle='--', 
                label=f'Mean: {np.mean(orthogonality_scores):.3f}')
        ax.set_xlabel('Orthogonality Measure ρ⊥')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Orthogonality Measure')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Kernel density estimate
        ax = axes[0, 1]
        sns.kdeplot(orthogonality_scores, ax=ax, fill=True)
        ax.set_xlabel('Orthogonality Measure ρ⊥')
        ax.set_ylabel('Density')
        ax.set_title('Kernel Density Estimate of ρ⊥')
        ax.grid(True, alpha=0.3)
        
        # 3. Cumulative distribution
        ax = axes[1, 0]
        sorted_scores = np.sort(orthogonality_scores)
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        ax.plot(sorted_scores, cumulative, linewidth=2)
        ax.set_xlabel('Orthogonality Measure ρ⊥')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Cumulative Distribution of ρ⊥')
        ax.grid(True, alpha=0.3)
        
        # 4. Statistics summary
        ax = axes[1, 1]
        stats_text = f"""
        Orthogonality Statistics:
        
        Mean: {np.mean(orthogonality_scores):.4f}
        Median: {np.median(orthogonality_scores):.4f}
        Std Dev: {np.std(orthogonality_scores):.4f}
        Min: {np.min(orthogonality_scores):.4f}
        Max: {np.max(orthogonality_scores):.4f}
        
        Percentiles:
        25th: {np.percentile(orthogonality_scores, 25):.4f}
        75th: {np.percentile(orthogonality_scores, 75):.4f}
        95th: {np.percentile(orthogonality_scores, 95):.4f}
        
        Interpretation:
        Values close to 1 indicate good orthogonality
        Values close to 0 indicate poor orthogonality
        """
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, 
                fontsize=12, verticalalignment='center', fontfamily='monospace')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('orthogonality_diagnostic.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return orthogonality_scores
    
    def compute_manifold_quality(self, test_loader, n_samples=1000, k_neighbors=20):
        """
        Assess manifold quality using local intrinsic dimension estimation
        """
        print("\nComputing Manifold Quality...")
        
        self.model.eval()
        representations = []
        local_dimensions = []
        
        with torch.no_grad():
            sample_count = 0
            for batch in test_loader:
                if sample_count >= n_samples:
                    break
                    
                x = batch['covariates'].to(self.device)
                phi_x = self.model.ssl_encoder.get_representation(x)
                representations.append(phi_x.cpu().numpy())
                sample_count += phi_x.shape[0]
        
        representations = np.concatenate(representations)[:n_samples]
        
        # Estimate local intrinsic dimensions using MLE
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1)
        nbrs.fit(representations)
        
        for i in range(min(n_samples, len(representations))):
            distances, _ = nbrs.kneighbors(representations[i:i+1])
            distances = distances[0][1:]  # Exclude self
            
            # Maximum likelihood estimation of local dimension
            if np.all(distances > 0):
                log_ratios = np.log(distances[1:] / distances[:-1])
                local_dim = 1 / np.mean(log_ratios)
                local_dimensions.append(local_dim)
        
        local_dimensions = np.array(local_dimensions)
        
        # Global intrinsic dimension using PCA
        pca = PCA()
        pca.fit(representations)
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Find effective dimension (95% variance)
        effective_dim = np.argmax(cumulative_variance >= 0.95) + 1
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Distribution of local dimensions
        ax = axes[0, 0]
        ax.hist(local_dimensions, bins=50, density=True, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(local_dimensions), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(local_dimensions):.2f}')
        ax.set_xlabel('Local Intrinsic Dimension')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Local Intrinsic Dimensions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. PCA explained variance
        ax = axes[0, 1]
        ax.plot(range(1, min(21, len(explained_variance_ratio) + 1)), 
                explained_variance_ratio[:20], 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('PCA Explained Variance')
        ax.grid(True, alpha=0.3)
        
        # 3. Cumulative variance
        ax = axes[1, 0]
        ax.plot(range(1, min(21, len(cumulative_variance) + 1)), 
                cumulative_variance[:20], 'go-', linewidth=2, markersize=8)
        ax.axhline(y=0.95, color='red', linestyle='--', label='95% threshold')
        ax.axvline(x=effective_dim, color='red', linestyle=':', 
                   label=f'Effective dim: {effective_dim}')
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Explained Variance')
        ax.set_title('Cumulative PCA Variance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Statistics summary
        ax = axes[1, 1]
        stats_text = f"""
        Manifold Quality Statistics:
        
        Local Dimension (MLE):
        Mean: {np.mean(local_dimensions):.2f}
        Median: {np.median(local_dimensions):.2f}
        Std Dev: {np.std(local_dimensions):.2f}
        
        Global Dimension (PCA):
        95% Variance Dim: {effective_dim}
        99% Variance Dim: {np.argmax(cumulative_variance >= 0.99) + 1}
        
        Representation Space:
        Total Dimensions: {representations.shape[1]}
        Effective Rank: {np.linalg.matrix_rank(representations)}
        
        Quality Assessment:
        Local dimensions should be consistent
        and match the true manifold dimension
        """
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, 
                fontsize=12, verticalalignment='center', fontfamily='monospace')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('manifold_quality.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'local_dimensions': local_dimensions,
            'effective_dim_95': effective_dim,
            'explained_variance': explained_variance_ratio
        }
    
    def compute_proxy_validity(self, test_loader, n_samples=1000):
        """
        Test conditional independence: Z ⊥ Y | (X, T, U)
        Since U is unobserved, we test weaker conditions
        """
        print("\nComputing Proxy Validity...")
        
        self.model.eval()
        proxies = []
        outcomes = []
        treatments = []
        representations = []
        
        # Generate proxies using the trained model
        with torch.no_grad():
            sample_count = 0
            for batch in test_loader:
                if sample_count >= n_samples:
                    break
                    
                x = batch['covariates'].to(self.device)
                t = batch['treatment'].to(self.device)
                y = batch['outcome'].to(self.device)
                
                phi_x = self.model.ssl_encoder.get_representation(x)
                
                # Generate counterfactual treatment
                t_cf = torch.randn_like(t)
                
                # Use a simplified proxy generation for testing
                # In practice, this would use the full diffusion model
                z = self.model.proxy_projection(phi_x)
                
                proxies.append(z.cpu().numpy())
                outcomes.append(y.cpu().numpy())
                treatments.append(t.cpu().numpy())
                representations.append(phi_x.cpu().numpy())
                
                sample_count += x.shape[0]
        
        proxies = np.concatenate(proxies)[:n_samples]
        outcomes = np.concatenate(outcomes)[:n_samples]
        treatments = np.concatenate(treatments)[:n_samples]
        representations = np.concatenate(representations)[:n_samples]
        
        # Test 1: Correlation between proxy and outcome
        proxy_outcome_corr = np.corrcoef(proxies.flatten(), outcomes.flatten())[0, 1]
        
        # Test 2: Conditional independence test using residuals
        # Fit E[Y | X, T]
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(np.hstack([representations, treatments]), outcomes.flatten())
        y_residuals = outcomes.flatten() - rf.predict(np.hstack([representations, treatments]))
        
        # Check correlation between proxy and residuals
        proxy_residual_corr = np.corrcoef(proxies.flatten(), y_residuals)[0, 1]
        
        # Test 3: Predictive power of proxy for outcome
        from sklearn.model_selection import cross_val_score
        rf_with_proxy = RandomForestRegressor(n_estimators=100, random_state=42)
        features_with_proxy = np.hstack([representations, treatments, proxies])
        features_without_proxy = np.hstack([representations, treatments])
        
        scores_with = cross_val_score(rf_with_proxy, features_with_proxy, outcomes.flatten(), cv=5)
        scores_without = cross_val_score(RandomForestRegressor(n_estimators=100, random_state=42), 
                                         features_without_proxy, outcomes.flatten(), cv=5)
        
        improvement = np.mean(scores_with) - np.mean(scores_without)
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Proxy vs Outcome scatter
        ax = axes[0, 0]
        ax.scatter(proxies.flatten(), outcomes.flatten(), alpha=0.5, s=20)
        ax.set_xlabel('Proxy Z')
        ax.set_ylabel('Outcome Y')
        ax.set_title(f'Proxy vs Outcome (corr: {proxy_outcome_corr:.3f})')
        
        # Add regression line
        z = np.polyfit(proxies.flatten(), outcomes.flatten(), 1)
        p = np.poly1d(z)
        ax.plot(sorted(proxies.flatten()), p(sorted(proxies.flatten())), "r--", alpha=0.8)
        ax.grid(True, alpha=0.3)
        
        # 2. Proxy vs Residuals
        ax = axes[0, 1]
        ax.scatter(proxies.flatten(), y_residuals, alpha=0.5, s=20)
        ax.set_xlabel('Proxy Z')
        ax.set_ylabel('Outcome Residuals')
        ax.set_title(f'Proxy vs Residuals (corr: {proxy_residual_corr:.3f})')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        # 3. Proxy distribution
        ax = axes[1, 0]
        ax.hist(proxies.flatten(), bins=50, density=True, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Proxy Z')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Proxy Values')
        ax.grid(True, alpha=0.3)
        
        # 4. Validity summary
        ax = axes[1, 1]
        validity_text = f"""
        Proxy Validity Assessment:
        
        Basic Statistics:
        Proxy-Outcome Correlation: {proxy_outcome_corr:.4f}
        Proxy-Residual Correlation: {proxy_residual_corr:.4f}
        
        Predictive Power:
        R² without proxy: {np.mean(scores_without):.4f}
        R² with proxy: {np.mean(scores_with):.4f}
        Improvement: {improvement:.4f}
        
        Interpretation:
        - High proxy-outcome correlation is good
        - Low proxy-residual correlation indicates
          conditional independence
        - Positive improvement shows proxy adds
          information about confounding
        
        Validity Score: {'GOOD' if abs(proxy_residual_corr) < 0.1 and improvement > 0 else 'MODERATE' if abs(proxy_residual_corr) < 0.2 else 'POOR'}
        """
        ax.text(0.1, 0.5, validity_text, transform=ax.transAxes, 
                fontsize=11, verticalalignment='center', fontfamily='monospace')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('proxy_validity.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'proxy_outcome_corr': proxy_outcome_corr,
            'proxy_residual_corr': proxy_residual_corr,
            'predictive_improvement': improvement,
            'validity_score': 'GOOD' if abs(proxy_residual_corr) < 0.1 and improvement > 0 else 'MODERATE'
        }
    
    def _estimate_local_tangent_space(self, representations, center_idx, k=20):
        """Estimate local tangent space using neighboring points"""
        rep_numpy = representations.cpu().numpy()
        center = rep_numpy[center_idx]
        
        # Find k nearest neighbors
        distances = np.linalg.norm(rep_numpy - center, axis=1)
        neighbor_indices = np.argsort(distances)[1:k+1]  # Exclude self
        
        if len(neighbor_indices) < 3:
            return None
        
        # Center the neighbors
        neighbors = rep_numpy[neighbor_indices]
        centered = neighbors - center
        
        # PCA to find principal directions
        U, S, Vt = np.linalg.svd(centered.T, full_matrices=False)
        
        # Estimate local dimension using eigenvalue ratio
        eigenvalues = S**2
        if len(eigenvalues) > 1:
            ratios = eigenvalues[:-1] / eigenvalues[1:]
            local_dim = min(np.sum(ratios > 2), 5)  # Cap at 5
        else:
            local_dim = 1
        
        # Return top principal components as tangent basis
        tangent_basis = torch.tensor(U[:, :local_dim], dtype=torch.float32, device=self.device)
        return tangent_basis
    
    def run_all_diagnostics(self, test_loader, n_samples=1000):
        """Run all three diagnostic tests"""
        print("="*80)
        print("Running GCMP Diagnostic Analysis")
        print("="*80)
        
        # 1. Orthogonality Diagnostic
        ortho_scores = self.compute_orthogonality_diagnostic(test_loader, n_samples)
        
        # 2. Manifold Quality
        manifold_results = self.compute_manifold_quality(test_loader, n_samples)
        
        # 3. Proxy Validity
        validity_results = self.compute_proxy_validity(test_loader, n_samples)
        
        # Summary report
        print("\n" + "="*80)
        print("DIAGNOSTIC SUMMARY")
        print("="*80)
        print(f"\n1. Orthogonality Diagnostic:")
        print(f"   - Mean ρ⊥: {np.mean(ortho_scores):.4f}")
        print(f"   - Median ρ⊥: {np.median(ortho_scores):.4f}")
        print(f"   - Assessment: {'GOOD' if np.mean(ortho_scores) > 0.8 else 'MODERATE' if np.mean(ortho_scores) > 0.5 else 'POOR'}")
        
        print(f"\n2. Manifold Quality:")
        print(f"   - Mean Local Dimension: {np.mean(manifold_results['local_dimensions']):.2f}")
        print(f"   - Effective Dimension (95% var): {manifold_results['effective_dim_95']}")
        print(f"   - Assessment: {'GOOD' if np.std(manifold_results['local_dimensions']) < 1.0 else 'MODERATE'}")
        
        print(f"\n3. Proxy Validity:")
        print(f"   - Proxy-Outcome Correlation: {validity_results['proxy_outcome_corr']:.4f}")
        print(f"   - Proxy-Residual Correlation: {validity_results['proxy_residual_corr']:.4f}")
        print(f"   - Predictive Improvement: {validity_results['predictive_improvement']:.4f}")
        print(f"   - Assessment: {validity_results['validity_score']}")
        
        print("\n" + "="*80)
        
        return {
            'orthogonality': ortho_scores,
            'manifold': manifold_results,
            'validity': validity_results
        }


def main():
    """Main function to run diagnostic analysis"""
    
    # Set up configuration with best parameters
    config = get_default_config()
    
    # Apply best parameters for single continuous treatment
    best_params = {
        "model.dropout": 0.14312388724309716,
        "model.latent_dim": 8,
        "model.ssl_output_dim": 64,
        "model.ssl_hidden_dims": [256, 128],  # medium
        "model.vi_hidden_dims": [128, 64],     # medium
        "ssl.lr": 0.00396941409864625,
        "ssl.temperature": 0.8454270023172249,
        "ssl.lambda_preserve": 0.3619528075275261,
        "ssl.lambda_diverse": 0.01,  # Use default since not in best params
        "training.n_epochs_ssl": 50,
        "diffusion_training.lr": 3.7303775709894795e-05,
        "diffusion_training.lambda_orthogonal": 0.42933352488208476,
        "diffusion_training.lambda_outcome": 1.0,  # Use default
        "training.n_epochs_diffusion": 60,
        "diffusion_sampling.n_steps": 50,  # Use default
        "vi_training.lr": 0.0010076215005316367,
        "vi_training.lambda_entropy": 0.0062203607922536215,
        "vi_training.weight_decay": 7.02762428760389e-06,
        "training.n_epochs_vi": 40,
        "training.n_folds": 5  # Use default
    }
    
    # Update config with best parameters
    for key, value in best_params.items():
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            current = current[k]
        current[keys[-1]] = value
    
    # Set up for single continuous treatment with 10000 samples
    config['data']['n_samples'] = 10000
    config['data']['dataset_kwargs']['treatment_type'] = 'continuous'
    config['data']['dataset_kwargs']['orthogonality_violation'] = 0.1  # Mild violation for testing
    
    print("Setting up GCMP model with best parameters...")
    print(f"Dataset: Single Continuous Treatment, n=10000")
    
    # Initialize trainer
    trainer = GCMPTrainer(
        config=config,
        log_dir='./diagnostic_logs',
        checkpoint_dir='./diagnostic_checkpoints',
        use_wandb=False
    )
    
    # Train model
    print("\nTraining GCMP model...")
    n_epochs = {
        'ssl': config['training']['n_epochs_ssl'],
        'diffusion': config['training']['n_epochs_diffusion'],
        'vi': config['training']['n_epochs_vi']
    }
    
    trainer.train(n_epochs, n_folds=config['training']['n_folds'])
    
    # Get test data loader
    test_loader = trainer.data_module.get_dataloader('test')
    
    # Run diagnostics
    print("\nRunning diagnostic analysis...")
    diagnostics = GCMPDiagnostics(trainer.model, trainer.data_module)
    results = diagnostics.run_all_diagnostics(test_loader, n_samples=1000)
    
    # Evaluate model performance
    print("\nEvaluating model performance...")
    eval_results = trainer.evaluate()
    print(f"PEHE: {eval_results.get('pehe', 'N/A'):.4f}")
    print(f"ATE Error: {eval_results.get('ate_error', 'N/A'):.4f}")
    print(f"Coverage: {eval_results.get('coverage_95', 'N/A'):.4f}")
    
    return results


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run diagnostic analysis
    results = main()