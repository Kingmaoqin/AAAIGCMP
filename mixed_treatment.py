"""
Extension for handling mixed continuous and discrete treatments
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from data import CausalDataset, SyntheticCausalDataset
from torch.utils.data import Dataset    
@dataclass
class TreatmentSpec:
    """Specification for mixed treatment types"""
    continuous_dims: List[int]  # Indices of continuous treatments
    discrete_dims: List[int]    # Indices of discrete treatments
    discrete_levels: Dict[int, int]  # Number of levels for each discrete treatment
    
    @property
    def total_dim(self) -> int:
        return len(self.continuous_dims) + len(self.discrete_dims)
    
    def encode_treatment(self, t_continuous: np.ndarray, t_discrete: np.ndarray) -> np.ndarray:
        """Encode mixed treatments into single array"""
        n_samples = len(t_continuous)
        encoded = np.zeros((n_samples, self.total_dim))
        
        # Place continuous treatments
        for i, idx in enumerate(self.continuous_dims):
            encoded[:, idx] = t_continuous[:, i]
        
        # Place discrete treatments
        for i, idx in enumerate(self.discrete_dims):
            encoded[:, idx] = t_discrete[:, i]
        
        return encoded
    
    def decode_treatment(self, encoded: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode mixed treatments"""
        t_continuous = encoded[:, self.continuous_dims]
        t_discrete = encoded[:, self.discrete_dims].astype(int)
        return t_continuous, t_discrete


class MixedTreatmentDataset(CausalDataset):
    def __init__(self,
                 n_samples: int = 10000,
                 n_covariates: int = 50,
                 treatment_spec: TreatmentSpec = None,
                 treatment_type: str = 'mixed', 
                 confounder_dim: int = 3,
                 orthogonality_violation: float = 0.0,
                 noise_level: float = 0.1,
                 seed: Optional[int] = None):
        
        super().__init__(n_samples, n_covariates, treatment_type=treatment_type, seed=seed)
        if treatment_spec is None:
            raise ValueError("treatment_spec must be provided for mixed treatments")
        self.treatment_spec = treatment_spec
        self.confounder_dim = confounder_dim
        
        data = self.generate_data(orthogonality_violation, noise_level)
        self.U = data['U']
        self.X = data['X']
        self.T = data['T']
        self.Y = data['Y']
        self.true_params = data['true_params']
    
    def generate_data(self, delta: float, noise: float) -> Dict:
        """Generate data with mixed treatments"""
        # 1. Generate confounder
        U = np.random.randn(self.n_samples, self.confounder_dim)
        
        # 2. Generate covariates (same as before)
        W_X = np.random.randn(self.confounder_dim, self.n_covariates)
        X = U @ W_X + np.random.randn(self.n_samples, self.n_covariates) * noise
        
        # 3. Generate mixed treatments
        n_cont = len(self.treatment_spec.continuous_dims)
        if n_cont > 0:
            W_T_cont = np.random.randn(self.n_covariates, n_cont) * 0.5
            B_T_cont = np.random.randn(self.confounder_dim, n_cont) * 0.5
            T_cont = X @ W_T_cont + U @ B_T_cont + np.random.randn(self.n_samples, n_cont) * noise
            T_cont = (T_cont - T_cont.min(axis=0)) / (T_cont.max(axis=0) - T_cont.min(axis=0))
        else:
            T_cont = np.zeros((self.n_samples, 0))
        
        n_disc = len(self.treatment_spec.discrete_dims)
        T_disc_list = []
        if n_disc > 0:
            for i, (idx, n_levels) in enumerate(self.treatment_spec.discrete_levels.items()):
                W_T_disc = np.random.randn(self.n_covariates) * 0.5
                B_T_disc = np.random.randn(self.confounder_dim) * 0.5
                logits = X @ W_T_disc + U @ B_T_disc
                if n_levels == 2:
                    probs = 1 / (1 + np.exp(-logits))
                    T_disc_i = np.random.binomial(1, probs)
                else:
                    T_disc_i = np.clip(
                        np.round((n_levels - 1) * (logits - logits.min()) / (logits.max() - logits.min())), 
                        0, n_levels - 1
                    ).astype(int)
                T_disc_list.append(T_disc_i)
        
        T_disc = np.column_stack(T_disc_list) if T_disc_list else np.zeros((self.n_samples, 0))
        
        T = self.treatment_spec.encode_treatment(T_cont, T_disc)
        
        # 5. Generate outcome with interactions and store true parameters
        true_params = {}
        W_Y = np.random.randn(self.n_covariates)
        true_params['W_Y'] = W_Y
        Y = X @ W_Y
        
        true_params['cont_effects'] = []
        for i, t_idx in enumerate(self.treatment_spec.continuous_dims):
            beta_linear = np.random.randn()
            beta_quad = np.random.randn() * 0.1
            true_params['cont_effects'].append({'linear': beta_linear, 'quad': beta_quad})
            Y += beta_linear * T[:, t_idx] + beta_quad * T[:, t_idx]**2
        
        true_params['disc_effects'] = {}
        for i, t_idx in enumerate(self.treatment_spec.discrete_dims):
            n_levels = self.treatment_spec.discrete_levels[t_idx]
            effects = np.random.randn(n_levels)
            true_params['disc_effects'][t_idx] = effects
            for level in range(n_levels):
                mask = T[:, t_idx] == level
                Y[mask] += effects[level]
        
        true_params['interaction_effects'] = []
        if self.treatment_spec.total_dim >= 2:
            for i in range(self.treatment_spec.total_dim):
                for j in range(i+1, self.treatment_spec.total_dim):
                    interaction = np.random.randn() * 0.05
                    true_params['interaction_effects'].append({'i': i, 'j': j, 'val': interaction})
                    Y += interaction * T[:, i] * T[:, j]
        
        c_Y = np.random.randn(self.confounder_dim)
        true_params['c_Y'] = c_Y
        Y += U @ c_Y
        Y += np.random.randn(self.n_samples) * noise
        
        return {'U': U, 'X': X, 'T': T, 'Y': Y, 'true_params': true_params}

    def _compute_structural_outcome(self, X, T, U):
        """Helper to compute outcome from true parameters."""
        Y = X @ self.true_params['W_Y']
        
        for i, t_idx in enumerate(self.treatment_spec.continuous_dims):
            params = self.true_params['cont_effects'][i]
            Y += params['linear'] * T[:, t_idx] + params['quad'] * T[:, t_idx]**2
            
        for i, t_idx in enumerate(self.treatment_spec.discrete_dims):
            effects = self.true_params['disc_effects'][t_idx]
            for level in range(len(effects)):
                mask = T[:, t_idx] == level
                Y[mask] += effects[level]
        
        for params in self.true_params['interaction_effects']:
            Y += params['val'] * T[:, params['i']] * T[:, params['j']]
            
        if U is not None:
            Y += U @ self.true_params['c_Y']
            
        return Y

    def get_true_effect(self, x: np.ndarray, t: np.ndarray, t_cf: Optional[np.ndarray] = None) -> np.ndarray:
        if x.ndim == 1: x = x.reshape(1, -1)
        if t.ndim == 1: t = t.reshape(1, -1)
        if t_cf.ndim == 1: t_cf = t_cf.reshape(1, -1)
        u_zero = np.zeros((len(x), self.confounder_dim))
        
        y_t = self._compute_structural_outcome(x, t, u_zero)
        y_t_cf = self._compute_structural_outcome(x, t_cf, u_zero)
        
        return y_t - y_t_cf

    # __len__ is inherited from CausalDataset

    def __getitem__(self, idx):
        return {
            'covariates': torch.FloatTensor(self.X[idx]),
            'treatment': torch.FloatTensor(self.T[idx]),
            'outcome': torch.FloatTensor([self.Y[idx]]),
            'confounder': torch.FloatTensor(self.U[idx]),
            'idx': idx 
        }

class MixedTreatmentVIModel(nn.Module):
    """Extended VI model for mixed treatments"""
    
    def __init__(self,
                 covariate_dim: int,
                 treatment_spec: TreatmentSpec,
                 latent_dim: int = 10,
                 hidden_dims: List[int] = [128, 64]):
        super().__init__()
        
        self.treatment_spec = treatment_spec
        self.latent_dim = latent_dim
        
        # Outcome model handles mixed treatments
        treatment_dim = treatment_spec.total_dim
        self.outcome_model = nn.Sequential(
            nn.Linear(covariate_dim + treatment_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )
        
        # Treatment-specific embeddings for discrete treatments
        self.discrete_embeddings = nn.ModuleDict()
        for idx, n_levels in treatment_spec.discrete_levels.items():
            self.discrete_embeddings[str(idx)] = nn.Embedding(n_levels, 16)
        
        # Adjusted input dimension for encoder
        embed_dim = len(treatment_spec.discrete_dims) * 16
        encoder_input_dim = covariate_dim + len(treatment_spec.continuous_dims) + embed_dim + 2
        
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], latent_dim * 2)
        )
        
        # Learnable parameters
        self.delta = nn.Parameter(torch.randn(latent_dim))
        self.log_sigma_y = nn.Parameter(torch.zeros(1))
    
    def encode_treatment(self, treatment: torch.Tensor) -> torch.Tensor:
        """Encode mixed treatment with embeddings for discrete parts"""
        batch_size = treatment.shape[0]
        
        # Continuous parts
        cont_parts = []
        for idx in self.treatment_spec.continuous_dims:
            cont_parts.append(treatment[:, idx:idx+1])
        
        # Discrete parts with embeddings
        disc_parts = []
        for i, idx in enumerate(self.treatment_spec.discrete_dims):
            disc_vals = treatment[:, idx].long()
            embed = self.discrete_embeddings[str(idx)](disc_vals)
            disc_parts.append(embed)
        
        # Concatenate all parts
        if cont_parts and disc_parts:
            encoded = torch.cat(cont_parts + disc_parts, dim=-1)
        elif cont_parts:
            encoded = torch.cat(cont_parts, dim=-1)
        elif disc_parts:
            encoded = torch.cat(disc_parts, dim=-1)
        else:
            encoded = torch.zeros(batch_size, 0)
        
        return encoded
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        """Forward pass with mixed treatments"""
        # Encode treatment
        t_encoded = self.encode_treatment(t)
        
        # Encoder
        inputs = torch.cat([x, t_encoded, y, z], dim=-1)
        h = self.encoder(inputs)
        mu, log_var = torch.chunk(h, 2, dim=-1)
        
        # Sample latent
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        phi = mu + eps * std
        
        # Predict outcome
        xt = torch.cat([x, t], dim=-1)
        base_outcome = self.outcome_model(xt)
        y_pred = base_outcome + torch.matmul(phi, self.delta)
        
        # Compute ELBO terms
        sigma_y = torch.exp(self.log_sigma_y)
        log_p_y = -0.5 * torch.log(2 * np.pi * sigma_y**2) - \
                  0.5 * ((y - y_pred)**2) / sigma_y**2
        
        kl_div = -0.5 * torch.sum(1 + log_var - mu**2 - torch.exp(log_var), dim=-1)
        
        elbo = -(log_p_y.mean() - kl_div.mean())
        
        return {
            'y_pred': y_pred,
            'elbo': elbo,
            'mu': mu,
            'log_var': log_var
        }
    
    def estimate_effect_continuous_change(self, 
                                        x: torch.Tensor,
                                        t_base: torch.Tensor,
                                        treatment_idx: int,
                                        delta_t: float) -> torch.Tensor:
        """Estimate effect of changing one continuous treatment"""
        # Create modified treatment
        t_new = t_base.clone()
        t_new[:, treatment_idx] += delta_t
        
        # Predict outcomes
        xt_base = torch.cat([x, t_base], dim=-1)
        xt_new = torch.cat([x, t_new], dim=-1)
        
        y_base = self.outcome_model(xt_base)
        y_new = self.outcome_model(xt_new)
        
        return (y_new - y_base).squeeze()
    
    def estimate_effect_discrete_change(self,
                                      x: torch.Tensor,
                                      t_base: torch.Tensor,
                                      treatment_idx: int,
                                      new_level: int) -> torch.Tensor:
        """Estimate effect of changing one discrete treatment"""
        # Create modified treatment
        t_new = t_base.clone()
        t_new[:, treatment_idx] = new_level
        
        # Predict outcomes
        xt_base = torch.cat([x, t_base], dim=-1)
        xt_new = torch.cat([x, t_new], dim=-1)
        
        y_base = self.outcome_model(xt_base)
        y_new = self.outcome_model(xt_new)
        
        return (y_new - y_base).squeeze()


# Example usage
def example_mixed_treatment():
    """Example of using mixed treatments"""
    
    # Define treatment specification
    # Example: 2 continuous (drug dosages) + 2 discrete (surgery yes/no, therapy type 0/1/2)
    treatment_spec = TreatmentSpec(
        continuous_dims=[0, 1],    # First two are continuous
        discrete_dims=[2, 3],      # Last two are discrete
        discrete_levels={2: 2, 3: 3}  # Binary surgery, 3-level therapy
    )
    
    # Create dataset
    dataset = MixedTreatmentDataset(
        n_samples=5000,
        n_covariates=30,
        treatment_spec=treatment_spec,
        orthogonality_violation=0.1
    )
    
    # Create model
    model = MixedTreatmentVIModel(
        covariate_dim=30,
        treatment_spec=treatment_spec
    )
    
    # Example: Estimate effect of increasing drug 1 dosage
    sample = dataset[0]
    x = sample['covariates'].unsqueeze(0)
    t = sample['treatment'].unsqueeze(0)
    
    # Effect of increasing drug 1 by 0.1
    effect_drug1 = model.estimate_effect_continuous_change(x, t, 0, 0.1)
    print(f"Effect of increasing drug 1: {effect_drug1.item():.4f}")
    
    # Effect of switching to therapy type 2
    effect_therapy = model.estimate_effect_discrete_change(x, t, 3, 2)
    print(f"Effect of therapy type 2: {effect_therapy.item():.4f}")
    
    return dataset, model


if __name__ == "__main__":
    dataset, model = example_mixed_treatment()
    print("Mixed treatment model created successfully!")