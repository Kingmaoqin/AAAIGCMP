"""
Main training script for GCMP framework
Implements the three-stage training procedure with cross-fitting
FIXED VERSION with proper entropy regularization and orthogonality in diffusion
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.model_selection import KFold
import torch.nn.functional as F
from data import DataModule, TreatmentType
from models import GCMP, CausalAwareSSLEncoder, ConditionalDiffusionModel, VariationalInferenceModel
from utils import DiffusionSchedule, DiffusionTrainer, DiffusionSampler, CausalEvaluator

class GCMPTrainer:
    def __init__(self,
                config: Dict,
                device: str = 'cuda',
                log_dir: str = './logs',
                checkpoint_dir: str = './checkpoints',
                use_wandb: bool = False,
                data_module=None):
        
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.use_wandb = use_wandb
        
        self._setup_logging()
        
        if use_wandb:
            try:
                import wandb
                wandb.init(project="gcmp", config=config)
            except ImportError:
                self.logger.warning("wandb not installed, disabling wandb logging")
                self.use_wandb = False
        
        # Setup data

        if data_module:
            self.data_module = data_module
        else:
            self.data_module = self._setup_data()
        self.model = self._setup_model()        
           
        # Setup training components
        self.ssl_optimizer = None
        self.diffusion_optimizer = None
        self.vi_optimizer = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
    def _setup_logging(self):
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        (Path(self.log_dir) / 'tensorboard').mkdir(parents=True, exist_ok=True)

        log_file = Path(self.log_dir) / 'train.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ],
            force=True 
        )
        self.logger = logging.getLogger(__name__)

        # 3. TensorBoard
        self.writer = SummaryWriter(str(Path(self.log_dir) / 'tensorboard'))
    
    def _setup_data(self) -> DataModule:
        """Initialize data module"""
        return DataModule(
            dataset_name=self.config['data']['dataset_name'],
            n_samples=self.config['data']['n_samples'],
            n_covariates=self.config['data']['n_covariates'],
            batch_size=self.config['training']['batch_size'],
            train_ratio=self.config['data']['train_ratio'],
            val_ratio=self.config['data']['val_ratio'],
            normalize=self.config['data']['normalize'],
            **self.config['data'].get('dataset_kwargs', {})
        )
    
    def _setup_model(self) -> GCMP:
        """Initialize GCMP model"""
        dims = self.data_module.get_dims()
        
        return GCMP(
            covariate_dim=dims['n_covariates'],
            treatment_dim=dims['treatment_dim'],
            treatment_type=self.data_module.get_treatment_type(),
            **self.config['model']
        ).to(self.device)
    
    def train(self, n_epochs: Dict[str, int], n_folds: int = 5):
        """
        Main training procedure with cross-fitting
        Args:
            n_epochs: Dict with keys 'ssl', 'diffusion', 'vi' 
            n_folds: Number of cross-fitting folds
        """
        self.logger.info("Starting GCMP training")
        
        # Stage 1: Train SSL encoder
        self.logger.info("Stage 1: Training causal-aware SSL encoder")
        self.train_ssl(n_epochs['ssl'])
        
        # Stage 2 & 3: Cross-fitted diffusion and VI training
        self.logger.info(f"Stage 2 & 3: Cross-fitted training with {n_folds} folds")
        self.train_cross_fitted(n_epochs['diffusion'], n_epochs['vi'], n_folds)
        
        self.logger.info("Training completed!")
    
    def train_ssl(self, n_epochs: int):
        """Train causal-aware SSL encoder"""
        # Setup SSL-specific components
        ssl_trainer = CausalAwareSSLTrainer(
            model=self.model.ssl_encoder,
            device=self.device,
            config=self.config['ssl']
        )
        
        train_loader = self.data_module.get_dataloader('train')
        val_loader = self.data_module.get_dataloader('val')
        
        for epoch in range(n_epochs):
            # Training
            train_loss = ssl_trainer.train_epoch(train_loader)
            
            # Validation
            val_loss = ssl_trainer.validate(val_loader)
            
            # Logging
            self.logger.info(f"SSL Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            if self.use_wandb:
                import wandb
                wandb.log({
                    'ssl/train_loss': train_loss,
                    'ssl/val_loss': val_loss,
                    'ssl/epoch': epoch
                })
            
            self.writer.add_scalar('SSL/train_loss', train_loss, epoch)
            self.writer.add_scalar('SSL/val_loss', val_loss, epoch)
        
        # Save SSL encoder
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        torch.save(self.model.ssl_encoder.state_dict(), 
                self.checkpoint_dir / 'ssl_encoder.pt')
    
    def train_cross_fitted(self, n_epochs_diff: int, n_epochs_vi: int, n_folds: int):
        """Cross-fitted training of diffusion and VI models"""
        # Get all training data indices
        train_dataset = self.data_module.train_dataset
        n_train = len(train_dataset)
        indices = np.arange(n_train)
        
        # Handle special case when n_folds = 1 (no cross-fitting)
        if n_folds == 1:
            # Use all data for both training and validation
            fold_splits = [(indices, indices)]
        else:
            # Create folds
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.config.get('seed', 42))
            fold_splits = list(kf.split(indices))
        
        # Store cross-fitted models
        diffusion_models = []
        vi_models = []
        
        for fold, (train_idx, val_idx) in enumerate(fold_splits):
            self.logger.info(f"Training fold {fold + 1}/{n_folds}")
            
            # Map local indices to global indices
            train_global_idx = [train_dataset.dataset._train_indices[i] for i in train_idx]
            val_global_idx = [train_dataset.dataset._train_indices[i] for i in val_idx]
            
            # Create fold-specific data loaders
            fold_train_dataset = torch.utils.data.Subset(train_dataset.dataset, train_global_idx)
            fold_val_dataset = torch.utils.data.Subset(train_dataset.dataset, val_global_idx)
            
            fold_train_loader = torch.utils.data.DataLoader(
                fold_train_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=True,
                num_workers=4
            )
            
            fold_val_loader = torch.utils.data.DataLoader(
                fold_val_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=False,
                num_workers=4
            )
            
            # Train diffusion model for this fold
            diffusion_model = self.train_diffusion_fold(
                fold_train_loader, fold_val_loader, n_epochs_diff, fold
            )
            diffusion_models.append(diffusion_model)
            
            # Generate perturbations for validation fold
            perturbations = self.generate_perturbations(
                diffusion_model, fold_val_loader, val_global_idx
            )
            
            # Train VI model for this fold
            vi_model = self.train_vi_fold(
                fold_train_loader, perturbations, n_epochs_vi, fold
            )
            vi_models.append(vi_model)
        
        # Final VI training on all data using cross-fitted perturbations
        self.logger.info("Final VI training with cross-fitted perturbations")
        self.train_final_vi(diffusion_models, n_epochs_vi)
    
    def train_diffusion_fold(self, 
                           train_loader: torch.utils.data.DataLoader,
                           val_loader: torch.utils.data.DataLoader,
                           n_epochs: int,
                           fold: int) -> ConditionalDiffusionModel:
        """Train diffusion model for one fold"""
        # Create fold-specific diffusion model
        dims = self.data_module.get_dims()
        ssl_output_dim = self.config['model']['ssl_output_dim']
        condition_dim = ssl_output_dim + dims['treatment_dim'] + 1 + dims['treatment_dim']
        
        diffusion_model = ConditionalDiffusionModel(
            input_dim=ssl_output_dim,
            condition_dim=condition_dim,
            **self.config['diffusion']
        ).to(self.device)
        
        # Setup trainer with orthogonality regularizer
        diffusion_trainer = DiffusionTrainer(
            model=diffusion_model,
            device=self.device,
            config=self.config['diffusion_training'],
            orthogonality_regularizer=self.model.orthogonality_regularizer
        )
        
        # Training loop
        for epoch in range(n_epochs):
            train_loss = diffusion_trainer.train_epoch(
                train_loader, 
                self.model.ssl_encoder,
                self.model.vi_model.outcome_model
            )
            
            val_loss = diffusion_trainer.validate(
                val_loader,
                self.model.ssl_encoder,
                self.model.vi_model.outcome_model
            )
            
            self.logger.info(
                f"Diffusion Fold {fold} Epoch {epoch}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )
        
        # Save model
        torch.save(diffusion_model.state_dict(), 
                  self.checkpoint_dir / f'diffusion_fold_{fold}.pt')
        
        return diffusion_model
    
    def generate_perturbations(self,
                             diffusion_model: ConditionalDiffusionModel,
                             data_loader: torch.utils.data.DataLoader,
                             global_indices: List[int]) -> Dict[int, torch.Tensor]:
        """Generate perturbations for a dataset using trained diffusion model"""
        sampler = DiffusionSampler(
            model=diffusion_model,
            device=self.device,
            config=self.config['diffusion_sampling']
        )
        
        perturbations = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Generating perturbations")):
                # Move to device
                x = batch['covariates'].to(self.device)
                t = batch['treatment'].to(self.device)
                y = batch['outcome'].to(self.device)
                
                # Encode covariates
                phi_x = self.model.ssl_encoder.get_representation(x)
                
                # Generate counterfactual treatments
                if self.data_module.get_treatment_type() == TreatmentType.CONTINUOUS:
                    # Sample random counterfactual treatments
                    t_cf = torch.randn_like(t)
                else:
                    # Flip binary treatment
                    t_cf = 1 - t
                
                # Generate perturbations
                delta_phi = sampler.sample(
                    batch_size=x.shape[0],
                    condition=torch.cat([phi_x, t, y, t_cf], dim=-1)
                )
                
                # Store with global indices
                for i in range(len(x)):
                    global_idx = global_indices[batch_idx * data_loader.batch_size + i]
                    perturbations[global_idx] = delta_phi[i].cpu()
                        
        return perturbations
    
    def train_vi_fold(self,
                    train_loader: torch.utils.data.DataLoader,
                    perturbations: Dict[int, torch.Tensor],
                    n_epochs: int,
                    fold: int) -> VariationalInferenceModel:
        """Train VI model for one fold"""
        # This is primarily for generating outcome model f_θ for the next fold
        # Full VI training happens in final stage

        # Get dimensions and model config
        ssl_output_dim = self.config['model']['ssl_output_dim']
        dims = self.data_module.get_dims()
        model_config = self.config['model']
        
        # Create fold-specific VI model components
        vi_model = VariationalInferenceModel(
            covariate_dim=ssl_output_dim,
            treatment_dim=dims['treatment_dim'],
            latent_dim=model_config['latent_dim'],
            hidden_dims=model_config['vi_hidden_dims'],
            dropout=model_config['dropout'],
            treatment_type=self.data_module.get_treatment_type()
        ).to(self.device)

        optimizer = optim.Adam(vi_model.parameters(), lr=self.config['vi_training']['lr'])
        
        # Simple training loop for the outcome model part
        for epoch in range(n_epochs):
            vi_model.train()
            total_loss = 0
            
            for batch in train_loader:
                # Standard supervised training on outcome prediction
                x = batch['covariates'].to(self.device)
                t = batch['treatment'].to(self.device)
                y = batch['outcome'].to(self.device)
                
                # Get representations
                with torch.no_grad():
                    phi_x = self.model.ssl_encoder.get_representation(x)
                
                # Predict outcome using only the outcome_model part of VIModel
                xt = torch.cat([phi_x, t], dim=-1)
                y_pred = vi_model.outcome_model(xt)
                
                loss = nn.MSELoss()(y_pred, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            self.logger.info(f"VI Fold {fold} Epoch {epoch}: loss={total_loss/len(train_loader):.4f}")
        
        return vi_model
    
    def train_final_vi(self, diffusion_models: List[ConditionalDiffusionModel], n_epochs: int):
        """Final VI training with all cross-fitted perturbations"""
        # Generate perturbations for all training data
        all_perturbations = self._generate_all_perturbations(diffusion_models)
        
        # Setup VI training with entropy regularization
        vi_trainer = VITrainer(
            model=self.model.vi_model,
            proxy_projection=self.model.proxy_projection,
            orthogonality_regularizer=None,  # Already applied in diffusion
            device=self.device,
            config=self.config['vi_training']
        )
        
        # Create data loader with perturbations
        train_loader = self.data_module.get_dataloader('train')
        
        # Training loop
        for epoch in range(n_epochs):
            train_metrics = vi_trainer.train_epoch(
                train_loader, 
                all_perturbations,
                self.model.ssl_encoder
            )
            
            # Validation
            val_loader = self.data_module.get_dataloader('val')
            val_metrics = vi_trainer.validate(
                val_loader,
                all_perturbations,
                self.model.ssl_encoder
            )
            
            self.logger.info(
                f"Final VI Epoch {epoch}: "
                f"train_elbo={train_metrics['elbo']:.4f}, "
                f"val_elbo={val_metrics['elbo']:.4f}"
            )
            
            if self.use_wandb:
                import wandb
                wandb.log({
                    'vi/train_elbo': train_metrics['elbo'],
                    'vi/val_elbo': val_metrics['elbo'],
                    'vi/train_recon_y': train_metrics['recon_y'],
                    'vi/train_recon_z': train_metrics['recon_z'],
                    'vi/train_kl': train_metrics['kl_div'],
                    'vi/train_entropy': train_metrics.get('entropy', 0),
                    'vi/epoch': epoch
                })
        
        # Save final model
        torch.save(self.model.state_dict(), self.checkpoint_dir / 'gcmp_final.pt')
    
    def _generate_all_perturbations(self, diffusion_models: List[ConditionalDiffusionModel]) -> Dict[int, torch.Tensor]:
        """Generate perturbations for all data using cross-fitted models"""
        all_perturbations = {}
        
        # Get fold assignments
        train_dataset = self.data_module.train_dataset
        n_train = len(train_dataset)
        indices = np.arange(n_train)
        
        # Handle special case when only one model (no cross-fitting)
        if len(diffusion_models) == 1:
            fold_splits = [([], indices)]  # Use all data as validation
        else:
            kf = KFold(n_splits=len(diffusion_models), shuffle=True, 
                    random_state=self.config.get('seed', 42))
            fold_splits = list(kf.split(indices))
        
        for fold, (train_idx, val_idx) in enumerate(fold_splits):
            # Map to global indices
            val_global_idx = [train_dataset.dataset._train_indices[i] for i in val_idx]
            
            # Create validation dataset for this fold
            fold_dataset = torch.utils.data.Subset(train_dataset.dataset, val_global_idx)
            fold_loader = torch.utils.data.DataLoader(
                fold_dataset,
                batch_size=self.config['training']['batch_size'],
                shuffle=False,
                num_workers=4
            )
            
            # Use fold's diffusion model to generate perturbations
            fold_perturbations = self.generate_perturbations(
                diffusion_models[fold], fold_loader, val_global_idx
            )
            
            # Add to all perturbations
            all_perturbations.update(fold_perturbations)
        
        return all_perturbations
    
    def evaluate(self, test_loader: Optional[torch.utils.data.DataLoader] = None):
        """Evaluate trained model"""
        if test_loader is None:
            test_loader = self.data_module.get_dataloader('test')
        
        # For evaluation, we need to generate perturbations for test data if not available
        # This is a simplified approach - in practice, you might want to use the trained diffusion model
        self.logger.info("Generating evaluation perturbations...")
        
        evaluator = CausalEvaluator(
            model=self.model,
            device=self.device,
            config=self.config['evaluation']
        )
        
        # Pass test dataset for true effect computation
        test_dataset = self.data_module.test_dataset
        results = evaluator.evaluate(test_loader, test_dataset)
        
        self.logger.info("Evaluation Results:")
        for metric, value in results.items():
            if isinstance(value, (int, float)) and value >= 0:
                self.logger.info(f"  {metric}: {value:.4f}")
        
        if self.use_wandb:
            import wandb
            wandb.log({f'test/{k}': v for k, v in results.items() if isinstance(v, (int, float))})
        
        return results


class CausalAwareSSLTrainer:
    """Trainer for causal-aware SSL encoder"""
    
    def __init__(self, model: CausalAwareSSLEncoder, device: torch.device, config: Dict):
        self.model = model
        self.device = device
        self.config = config
        
        # Setup optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.get('n_epochs', 100)
        )
        
        # Loss weights
        self.lambda_preserve = config.get('lambda_preserve', 0.1)
        self.lambda_diverse = config.get('lambda_diverse', 0.01)
        
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="SSL Training"):
            x = batch['covariates'].to(self.device)
            t = batch['treatment'].to(self.device)
            
            # Create augmented views
            x1, x2 = self._augment(x)
            
            # Forward pass
            out1 = self.model(x1)
            out2 = self.model(x2)
            
            # Compute losses
            loss_contrast = self._contrastive_loss(out1['projection'], out2['projection'])
            loss_preserve = self._preservation_loss(
                out1['treatment_pred'], out2['treatment_pred'], 
                out1['representation'], out2['representation']
            )
            loss_diverse = self._diversity_loss(out1['representation'])
            
            # Total loss
            loss = loss_contrast + self.lambda_preserve * loss_preserve + \
                   self.lambda_diverse * loss_diverse
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        self.scheduler.step()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: torch.utils.data.DataLoader) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['covariates'].to(self.device)
                
                # Create augmented views
                x1, x2 = self._augment(x)
                
                # Forward pass
                out1 = self.model(x1)
                out2 = self.model(x2)
                
                # Compute losses
                loss_contrast = self._contrastive_loss(out1['projection'], out2['projection'])
                loss_preserve = self._preservation_loss(
                    out1['treatment_pred'], out2['treatment_pred'],
                    out1['representation'], out2['representation']
                )
                loss_diverse = self._diversity_loss(out1['representation'])
                
                # Total loss
                loss = loss_contrast + self.lambda_preserve * loss_preserve + \
                       self.lambda_diverse * loss_diverse
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def _augment(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply causal-aware augmentations"""
        # Mild augmentations that preserve treatment-relevant information
        noise_scale = self.config.get('augment_noise', 0.1)
        
        x1 = x + torch.randn_like(x) * noise_scale
        x2 = x + torch.randn_like(x) * noise_scale
        
        # Mild scaling
        scale = 1 + (torch.rand(x.shape[0], 1, device=x.device) - 0.5) * 0.1
        x2 = x2 * scale
        
        return x1, x2
    
    def _contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """SimCLR-style contrastive loss"""
        batch_size = z1.shape[0]
        temperature = self.config.get('temperature', 0.5)
        
        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Concatenate
        representations = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(representations, representations.T)
        
        # Create positive mask
        mask = torch.eye(batch_size * 2, dtype=torch.bool, device=self.device)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 1
            mask[batch_size + i, i] = 1
        
        # Compute loss
        positives = similarity_matrix[mask].view(batch_size * 2, 1)
        negatives = similarity_matrix[~mask].view(batch_size * 2, -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(batch_size * 2, dtype=torch.long, device=self.device)
        
        loss = F.cross_entropy(logits / temperature, labels)
        
        return loss
    
    def _preservation_loss(self, t_pred1: torch.Tensor, t_pred2: torch.Tensor,
                          h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        """Loss to preserve treatment-relevant variation"""
        # Treatment similarity
        t_sim = torch.exp(-self.config.get('gamma', 1.0) * (t_pred1 - t_pred2)**2)
        
        # Representation difference
        h_diff = torch.norm(h1 - h2, dim=1)
        
        # Loss: high treatment similarity should mean low representation difference
        loss = (t_sim * h_diff).mean()
        
        return loss
    
    def _diversity_loss(self, h: torch.Tensor) -> torch.Tensor:
        """Encourage diverse representations"""
        if h.shape[0] < 2:
            return torch.tensor(0.0, device=h.device)
            
        # Compute covariance matrix
        h_centered = h - h.mean(dim=0, keepdim=True)
        cov = torch.matmul(h_centered.T, h_centered) / (h.shape[0] - 1)
        
        # Use slogdet for better numerical stability
        # Add a small value to the diagonal to prevent the matrix from being singular
        cov_stable = cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device)
        sign, logdet = torch.slogdet(cov_stable)
        
        # We expect the sign of the determinant of a covariance matrix to be positive.
        # If it's not, it's a sign of numerical issues, but we can handle it gracefully.
        # Return -logdet if sign is 1, and a large penalty if not (or just -logdet).
        return -logdet


class VITrainer:
    """Trainer for variational inference model with entropy regularization"""
    
    def __init__(self, 
                 model: VariationalInferenceModel,
                 proxy_projection: nn.Module,
                 orthogonality_regularizer: Optional[nn.Module],
                 device: torch.device,
                 config: Dict):
        self.model = model
        self.proxy_projection = proxy_projection
        self.orthogonality_regularizer = orthogonality_regularizer
        self.device = device
        self.config = config
        
        # Setup optimizer
        params = list(model.parameters()) + list(proxy_projection.parameters())
        self.optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config.get('weight_decay', 0))
        
        # Regularization weights
        self.lambda_orthogonal = config.get('lambda_orthogonal', 0.1)
        self.lambda_entropy = config.get('lambda_entropy', 0.01)
    
    def train_epoch(self, 
                   train_loader: torch.utils.data.DataLoader,
                   perturbations: Dict[int, torch.Tensor],
                   ssl_encoder: CausalAwareSSLEncoder) -> Dict[str, float]:
        """Train for one epoch with entropy regularization"""
        self.model.train()
        metrics = {
            'elbo': 0,
            'recon_y': 0,
            'recon_z': 0,
            'kl_div': 0,
            'entropy': 0,
            'orthogonality_reg': 0
        }
        
        n_batches = 0
        
        for batch in tqdm(train_loader, desc="VI Training"):
            # Get data
            x = batch['covariates'].to(self.device)
            t = batch['treatment'].to(self.device)
            y = batch['outcome'].to(self.device)
            indices = batch['idx']
            
            # Get SSL representations
            with torch.no_grad():
                phi_x = ssl_encoder.get_representation(x)
            
            # Get perturbations and compute proxy
            batch_perturbations = []
            valid_mask = []
            
            for i, idx in enumerate(indices):
                if idx.item() in perturbations:
                    batch_perturbations.append(perturbations[idx.item()].to(self.device))
                    valid_mask.append(i)
            
            if len(batch_perturbations) == 0:
                continue
                
            # Stack perturbations and filter batch
            delta_phi = torch.stack(batch_perturbations)
            valid_mask = torch.tensor(valid_mask, device=self.device)
            
            phi_x = phi_x[valid_mask]
            t = t[valid_mask]
            y = y[valid_mask]
            
            # Compute proxy
            z = self.proxy_projection(delta_phi)
            
            # Forward pass
            outputs = self.model(phi_x, t, y, z)
            
            # FIXED: Add entropy regularization
            # Entropy of variational posterior
            q_entropy = 0.5 * (1 + outputs['phi_log_var'] + np.log(2 * np.pi)).sum(dim=-1).mean()
            
            # Total loss with entropy regularization
            loss = outputs['elbo'] - self.lambda_entropy * q_entropy
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Update metrics
            metrics['elbo'] += outputs['elbo'].item()
            metrics['recon_y'] += outputs['log_p_y'].item()
            metrics['recon_z'] += outputs['log_p_z'].item()
            metrics['kl_div'] += outputs['kl_div'].item()
            metrics['entropy'] += q_entropy.item()
            
            n_batches += 1
        
        # Average metrics
        for key in metrics:
            metrics[key] /= max(n_batches, 1)
        
        return metrics
    
    def validate(self,
                val_loader: torch.utils.data.DataLoader,
                perturbations: Dict[int, torch.Tensor],
                ssl_encoder: CausalAwareSSLEncoder) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        metrics = {
            'elbo': 0,
            'recon_y': 0,
            'recon_z': 0,
            'kl_div': 0,
            'entropy': 0
        }
        
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Similar to train_epoch but without gradients
                x = batch['covariates'].to(self.device)
                t = batch['treatment'].to(self.device)
                y = batch['outcome'].to(self.device)
                indices = batch['idx']
                
                phi_x = ssl_encoder.get_representation(x)
                
                # Get perturbations
                batch_perturbations = []
                valid_mask = []
                
                for i, idx in enumerate(indices):
                    if idx.item() in perturbations:
                        batch_perturbations.append(perturbations[idx.item()].to(self.device))
                        valid_mask.append(i)
                
                if len(batch_perturbations) == 0:
                    continue
                
                delta_phi = torch.stack(batch_perturbations)
                valid_mask = torch.tensor(valid_mask, device=self.device)
                
                phi_x = phi_x[valid_mask]
                t = t[valid_mask]
                y = y[valid_mask]
                
                z = self.proxy_projection(delta_phi)
                
                outputs = self.model(phi_x, t, y, z)
                
                # Compute entropy
                q_entropy = 0.5 * (1 + outputs['phi_log_var'] + np.log(2 * np.pi)).sum(dim=-1).mean()
                
                metrics['elbo'] += outputs['elbo'].item()
                metrics['recon_y'] += outputs['log_p_y'].item()
                metrics['recon_z'] += outputs['log_p_z'].item()
                metrics['kl_div'] += outputs['kl_div'].item()
                metrics['entropy'] += q_entropy.item()
                
                n_batches += 1
        
        # Average metrics
        for key in metrics:
            metrics[key] /= max(n_batches, 1)
        
        return metrics


def main(config_path: str):
    """Main entry point"""
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Set random seeds
    torch.manual_seed(config.get('seed', 42))
    np.random.seed(config.get('seed', 42))
    
    # Create trainer
    trainer = GCMPTrainer(config)
    
    # Train
    n_epochs = {
        'ssl': config['training']['n_epochs_ssl'],
        'diffusion': config['training']['n_epochs_diffusion'],
        'vi': config['training']['n_epochs_vi']
    }
    
    trainer.train(n_epochs, n_folds=config['training'].get('n_folds', 5))
    
    # Evaluate
    results = trainer.evaluate()
    
    # Save results
    with open(trainer.log_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/gcmp_default.json')
    args = parser.parse_args()
    
    main(args.config)