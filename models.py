"""
Model module for GCMP framework
Contains all neural network architectures and model components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
import numpy as np
from typing import Dict, Tuple, Optional, List
import math
from abc import ABC, abstractmethod


class BaseNetwork(nn.Module, ABC):
    """Base class for all networks with common functionality"""
    
    def __init__(self):
        super().__init__()
        
    def init_weights(self, init_type='xavier'):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                if init_type == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif init_type == 'kaiming':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif init_type == 'normal':
                    nn.init.normal_(m.weight, 0, 0.02)
                    
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class CausalAwareSSLEncoder(BaseNetwork):
    """
    Causal-aware self-supervised learning encoder
    Preserves confounding signals while learning representations
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [256, 128],
                 output_dim: int = 64,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build encoder architecture
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        # Final projection layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        # Treatment predictor for causal preservation
        self.treatment_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[0] // 2, 1)  # Assuming scalar treatment for simplicity
        )
        
        self.init_weights()
    
    def _get_activation(self, activation: str):
        """Get activation function"""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning representations and auxiliary outputs"""
        # Main representation
        h = self.encoder(x)
        
        # Projection for contrastive loss
        z = self.projection_head(h)
        
        # Treatment prediction for preservation loss
        t_pred = self.treatment_predictor(x)
        
        return {
            'representation': h,
            'projection': z,
            'treatment_pred': t_pred
        }
    
    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get learned representation (inference mode)"""
        return self.encoder(x)


class ConditionalDiffusionModel(BaseNetwork):
    """
    Conditional diffusion model for generating causal perturbations
    Uses simplified U-Net architecture for tabular data
    """
    
    def __init__(self,
                 input_dim: int,
                 condition_dim: int,
                 hidden_dims: List[int] = [256, 512, 256],
                 time_embed_dim: int = 128,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.time_embed_dim = time_embed_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim)
        )
        
        # Condition embedding
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0])
        )
        
        # Combined embedding dimension
        emb_dim = time_embed_dim + hidden_dims[0]
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            block = ResidualBlock(prev_dim, hidden_dim, emb_dim, dropout=dropout)
            self.encoder_blocks.append(block)
            
            # Add attention at certain layers
            if i > 0 and i < len(hidden_dims) - 1:
                self.encoder_blocks.append(
                    MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
                )
            prev_dim = hidden_dim
        
        # Middle block
        self.middle_block = nn.Sequential(
            ResidualBlock(hidden_dims[-1], hidden_dims[-1], emb_dim, dropout=dropout),
            MultiHeadSelfAttention(hidden_dims[-1], num_heads, dropout),
            ResidualBlock(hidden_dims[-1], hidden_dims[-1], emb_dim, dropout=dropout)
        )
        
        # Decoder blocks - we'll use a simpler architecture without skip connections
        # to avoid dimension mismatches
        self.decoder_blocks = nn.ModuleList()
        
        # Reverse the hidden dimensions
        decoder_dims = list(reversed(hidden_dims[:-1])) + [input_dim]
        prev_dim = hidden_dims[-1]
        
        for i, hidden_dim in enumerate(decoder_dims):
            block = ResidualBlock(prev_dim, hidden_dim, emb_dim, dropout=dropout)
            self.decoder_blocks.append(block)
            
            # Add attention
            if i < len(decoder_dims) - 1:
                self.decoder_blocks.append(
                    MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
                )
            prev_dim = hidden_dim
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, input_dim)
        )
        
        self.init_weights()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of diffusion model
        Args:
            x: Noisy input [batch, input_dim]
            t: Timestep [batch]
            condition: Conditioning information [batch, condition_dim]
        Returns:
            Predicted noise [batch, input_dim]
        """
        # Embed time and condition
        t_emb = self.time_embed(t)
        c_emb = self.condition_embed(condition)
        emb = torch.cat([t_emb, c_emb], dim=-1)
        
        # Encoder
        h = x
        encoder_outputs = []
        
        for block in self.encoder_blocks:
            if isinstance(block, ResidualBlock):
                h = block(h, emb)
                encoder_outputs.append(h)
            else:  # Attention block
                h = block(h)
        
        # Middle
        for layer in self.middle_block:
            if isinstance(layer, ResidualBlock):
                h = layer(h, emb)
            else:
                h = layer(h)
        
        # Decoder - simplified without skip connections
        for block in self.decoder_blocks:
            if isinstance(block, ResidualBlock):
                h = block(h, emb)
            else:  # Attention block
                h = block(h)
        
        # Output
        return self.output_proj(h)


class ResidualBlock(nn.Module):
    """Residual block with time and condition modulation"""
    
    def __init__(self, in_dim: int, out_dim: int, emb_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Use LayerNorm which works better for varying dimensions
        self.norm1 = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, out_dim)
        
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_dim * 2)
        )
        
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(out_dim, out_dim)
        
        # Skip connection
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # First transformation
        h = self.norm1(x)
        h = F.silu(h)
        h = self.linear1(h)
        
        # Modulate with embedding
        emb_out = self.emb_proj(emb)
        scale, shift = torch.chunk(emb_out, 2, dim=-1)
        h = h * (1 + scale) + shift
        
        # Second transformation
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.linear2(h)
        
        # Skip connection
        return h + self.skip(x)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention module"""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Use LayerNorm for 1D data
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        H = self.num_heads
        
        # Normalize
        x_norm = self.norm(x)
        
        # Compute Q, K, V
        qkv = self.qkv(x_norm).reshape(B, 3, H, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.matmul(attn, v)
        out = out.reshape(B, D)
        out = self.proj(out)
        
        return out + x


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time steps"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class VariationalInferenceModel(BaseNetwork):
    """
    Variational inference model for final causal effect estimation
    Implements hierarchical Bayesian model with uncertainty quantification
    """
    
    def __init__(self,
                 covariate_dim: int,
                 treatment_dim: int,
                 latent_dim: int = 10,
                 hidden_dims: List[int] = [128, 64],
                 dropout: float = 0.1,
                 treatment_type: str = 'continuous'):
        super().__init__()
        
        self.covariate_dim = covariate_dim
        self.treatment_dim = treatment_dim
        self.latent_dim = latent_dim
        self.treatment_type = treatment_type
        
        # Outcome model f_θ(Φ(X), T)
        outcome_input_dim = covariate_dim + treatment_dim
        self.outcome_model = self._build_mlp(
            outcome_input_dim, 
            hidden_dims + [1],
            dropout,
            final_activation=None
        )
        
        # Encoder for variational posterior q(φ|X,Y,Z)
        encoder_input_dim = covariate_dim + 1 + 1  # X + Y + Z
        self.encoder = self._build_mlp(
            encoder_input_dim,
            hidden_dims + [latent_dim * 2],  # Mean and log variance
            dropout,
            final_activation=None
        )
        
        # Prior parameters
        self.register_buffer('prior_mean', torch.zeros(latent_dim))
        self.register_buffer('prior_cov', torch.eye(latent_dim))
        
        # Learnable parameters for confounding effect
        self.delta = nn.Parameter(torch.randn(latent_dim))
        self.gamma = nn.Parameter(torch.randn(latent_dim))
        
        # Noise parameters
        self.log_sigma_y = nn.Parameter(torch.zeros(1))
        self.log_sigma_z = nn.Parameter(torch.zeros(1))
        
        self.init_weights()
    
    def _build_mlp(self, 
                   input_dim: int, 
                   hidden_dims: List[int], 
                   dropout: float,
                   final_activation: Optional[str] = 'relu') -> nn.Sequential:
        """Build MLP with specified architecture"""
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims[:-1]):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final layer
        layers.append(nn.Linear(prev_dim, hidden_dims[-1]))
        if final_activation == 'relu':
            layers.append(nn.ReLU())
        elif final_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
            
        return nn.Sequential(*layers)
    
    def encode(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to variational posterior parameters"""
        inputs = torch.cat([x, y, z], dim=-1)
        h = self.encoder(inputs)
        
        # Split into mean and log variance
        mu, log_var = torch.chunk(h, 2, dim=-1)
        
        # Ensure numerical stability
        log_var = torch.clamp(log_var, min=-10, max=10)
        
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, 
                x: torch.Tensor, 
                t: torch.Tensor, 
                y: torch.Tensor, 
                z: torch.Tensor,
                n_samples: int = 1) -> Dict[str, torch.Tensor]:
        """
        Forward pass with variational inference
        Returns predictions and ELBO components
        """
        batch_size = x.shape[0]
        
        # Encode to posterior
        q_mu, q_log_var = self.encode(x, y, z)
        
        # Sample from posterior
        phi_samples = []
        for _ in range(n_samples):
            phi = self.reparameterize(q_mu, q_log_var)
            phi_samples.append(phi)
        phi_samples = torch.stack(phi_samples, dim=1)  # [batch, n_samples, latent_dim]
        
        # Compute outcome predictions
        xt = torch.cat([x, t], dim=-1)
        base_outcome = self.outcome_model(xt)  # [batch, 1]
        
        # Add confounding effect
        confound_effect = torch.matmul(phi_samples, self.delta)
        y_pred = base_outcome + confound_effect
        # Compute proxy predictions
        z_pred = torch.matmul(phi_samples, self.gamma)  # [batch, n_samples]
        
        # Compute ELBO components
        # 1. Reconstruction loss for Y
        sigma_y = torch.exp(self.log_sigma_y)
        log_p_y = -0.5 * torch.log(2 * np.pi * sigma_y**2) - \
                  0.5 * ((y.unsqueeze(1) - y_pred)**2) / sigma_y**2
        log_p_y = log_p_y.mean(dim=1)  # Average over samples
        
        # 2. Reconstruction loss for Z  
        sigma_z = torch.exp(self.log_sigma_z)
        log_p_z = -0.5 * torch.log(2 * np.pi * sigma_z**2) - \
                  0.5 * ((z.unsqueeze(1) - z_pred)**2) / sigma_z**2
        log_p_z = log_p_z.mean(dim=1)  # Average over samples
        
        # 3. KL divergence
        kl_div = -0.5 * torch.sum(
            1 + q_log_var - q_mu**2 - torch.exp(q_log_var), dim=-1
        )
        
        # Total ELBO (negative to minimize)
        elbo = -(log_p_y + log_p_z - kl_div).mean()
        
        return {
            'y_pred': y_pred.mean(dim=1),  # Average prediction
            'y_std': y_pred.std(dim=1),    # Prediction uncertainty
            'z_pred': z_pred.mean(dim=1),
            'phi_mean': q_mu,
            'phi_log_var': q_log_var,
            'elbo': elbo,
            'log_p_y': -log_p_y.mean(),
            'log_p_z': -log_p_z.mean(),
            'kl_div': kl_div.mean()
        }
    
    def predict_outcome(self, 
                       x: torch.Tensor, 
                       t: torch.Tensor,
                       phi: Optional[torch.Tensor] = None,
                       n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict outcome for new treatment values
        If phi is not provided, uses prior
        """
        batch_size = x.shape[0]
        
        # Sample latent variables
        if phi is None:
            # Sample from prior
            phi_samples = torch.randn(batch_size, n_samples, self.latent_dim).to(x.device)
        else:
            # Use provided phi
            if phi.dim() == 2:
                phi_samples = phi.unsqueeze(1).repeat(1, n_samples, 1)
            else:
                phi_samples = phi
        
        # Compute base outcome
        xt = torch.cat([x, t], dim=-1)
        base_outcome = self.outcome_model(xt)
        
        # Add confounding effect
        confound_effect = torch.matmul(phi_samples, self.delta)
        y_pred = base_outcome + confound_effect
        
        # Return mean and std as 1D tensors
        y_mean = y_pred.mean(dim=1).squeeze(-1)
        y_std = y_pred.std(dim=1).squeeze(-1)
        
        return y_mean, y_std
    
    def estimate_treatment_effect(self,
                                 x: torch.Tensor,
                                 t0: torch.Tensor,
                                 t1: torch.Tensor,
                                 phi: Optional[torch.Tensor] = None,
                                 n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate individual treatment effect τ(t1, t0 | x)"""
        # Get outcomes under both treatments
        y0_mean, y0_std = self.predict_outcome(x, t0, phi, n_samples)
        y1_mean, y1_std = self.predict_outcome(x, t1, phi, n_samples)
        
        # Treatment effect
        effect_mean = y1_mean - y0_mean
        effect_std = torch.sqrt(y0_std**2 + y1_std**2)  # Assuming independence
        
        # Ensure we return 1D tensors (squeeze any extra dimensions)
        effect_mean = effect_mean.squeeze(-1) if effect_mean.dim() > 1 else effect_mean
        effect_std = effect_std.squeeze(-1) if effect_std.dim() > 1 else effect_std
        
        return effect_mean, effect_std


class GradientOrthogonalityRegularizer(nn.Module):
    """
    Gradient orthogonality regularizer R_⊥
    Encourages orthogonality between outcome gradient and confounder manifold
    """
    
    def __init__(self, outcome_model: nn.Module, manifold_dim: int):
        super().__init__()
        self.outcome_model = outcome_model
        self.manifold_dim = manifold_dim
    
    def estimate_tangent_space(self, perturbations: torch.Tensor, k_neighbors: int = 50) -> torch.Tensor:
        """
        Estimate tangent space using local PCA on perturbations
        Args:
            perturbations: [batch, dim] generated perturbations
            k_neighbors: number of neighbors for local PCA
        Returns:
            tangent_basis: [batch, dim, manifold_dim] orthonormal basis
        """
        batch_size, dim = perturbations.shape
        k_neighbors = min(k_neighbors, batch_size)
        
        # For each point, find k nearest neighbors
        distances = torch.cdist(perturbations, perturbations)
        _, indices = distances.topk(k_neighbors, largest=False, dim=1)
        
        tangent_bases = []
        for i in range(batch_size):
            # Get local perturbations
            local_perturb = perturbations[indices[i]]
            
            # Center
            centered = local_perturb - local_perturb.mean(dim=0, keepdim=True)
            
            # SVD for PCA
            try:
                U, S, V = torch.svd(centered)
                # Top manifold_dim components
                basis = V[:, :self.manifold_dim]
            except:
                # If SVD fails, use random orthonormal basis
                basis = torch.qr(torch.randn(dim, self.manifold_dim).to(perturbations.device))[0]
                
            tangent_bases.append(basis)
        
        return torch.stack(tangent_bases)
    
    def forward(self, 
                x: torch.Tensor, 
                t: torch.Tensor, 
                perturbations: torch.Tensor) -> torch.Tensor:
        """
        Compute regularization loss
        Args:
            x: [batch, covariate_dim] covariates
            t: [batch, treatment_dim] treatments
            perturbations: [batch, covariate_dim] generated perturbations
        """
        # Enable gradient computation
        x.requires_grad_(True)
        
        # Compute outcome
        xt = torch.cat([x, t], dim=-1)
        y = self.outcome_model(xt)
        
        # Compute gradient w.r.t covariates
        grad_x = torch.autograd.grad(
            outputs=y.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Estimate tangent space
        tangent_basis = self.estimate_tangent_space(perturbations)
        
        # Project gradient onto tangent space
        # grad_proj = tangent_basis @ tangent_basis.T @ grad
        batch_size = x.shape[0]
        reg_loss = 0
        
        for i in range(batch_size):
            # Project gradient
            grad_i = grad_x[i].unsqueeze(1)  # [dim, 1]
            basis_i = tangent_basis[i]  # [dim, manifold_dim]
            
            # Projection: basis @ basis.T @ grad
            proj = basis_i @ (basis_i.t() @ grad_i)
            
            # Regularization: ||projection||^2
            reg_loss += torch.norm(proj)**2
        
        return reg_loss / batch_size


class GCMP(nn.Module):
    """
    Full GCMP model combining all components
    """
    
    def __init__(self,
                 covariate_dim: int,
                 treatment_dim: int,
                 treatment_type: str = 'continuous',
                 ssl_hidden_dims: List[int] = [256, 128],
                 ssl_output_dim: int = 64,
                 diffusion_hidden_dims: List[int] = [256, 512, 256],
                 vi_hidden_dims: List[int] = [128, 64],
                 latent_dim: int = 10,
                 manifold_dim: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        # Module 1: Causal-aware SSL encoder
        self.ssl_encoder = CausalAwareSSLEncoder(
            input_dim=covariate_dim,
            hidden_dims=ssl_hidden_dims,
            output_dim=ssl_output_dim,
            dropout=dropout
        )
        
        # Module 2: Conditional diffusion model
        # Condition on: Φ(X), T, Y, T'
        condition_dim = ssl_output_dim + treatment_dim + 1 + treatment_dim
        self.diffusion_model = ConditionalDiffusionModel(
            input_dim=ssl_output_dim,
            condition_dim=condition_dim,
            hidden_dims=diffusion_hidden_dims,
            dropout=dropout
        )
        
        # Module 3: Variational inference model
        self.vi_model = VariationalInferenceModel(
            covariate_dim=ssl_output_dim,
            treatment_dim=treatment_dim,
            latent_dim=latent_dim,
            hidden_dims=vi_hidden_dims,
            dropout=dropout,
            treatment_type=treatment_type
        )
        
        # Gradient orthogonality regularizer
        self.orthogonality_regularizer = GradientOrthogonalityRegularizer(
            self.vi_model.outcome_model,
            manifold_dim
        )
        
        # Proxy projection
        self.proxy_projection = nn.Linear(ssl_output_dim, 1)
        
    def encode_covariates(self, x: torch.Tensor) -> torch.Tensor:
        """Get SSL representations of covariates"""
        return self.ssl_encoder.get_representation(x)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Full forward pass through GCMP"""
        # This would be implemented in the training loop
        # as it requires multiple stages
        raise NotImplementedError("Use stage-specific forward methods")