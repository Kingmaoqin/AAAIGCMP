"""
Data module for GCMP framework
Handles data generation, loading, and preprocessing
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, Union, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pathlib import Path
import pandas as pd
logger = logging.getLogger(__name__)


@dataclass
class TreatmentType:
    """Enum-like class for treatment types"""
    CONTINUOUS = "continuous"
    BINARY = "binary"
    MULTILABEL = "multilabel"


class CausalDataset(Dataset, ABC):
    """Abstract base class for causal datasets"""
    
    def __init__(self, 
                 n_samples: int,
                 n_covariates: int,
                 treatment_type: str,
                 seed: Optional[int] = None):
        self.n_samples = n_samples
        self.n_covariates = n_covariates
        self.treatment_type = treatment_type
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    @abstractmethod
    def generate_data(self) -> Dict[str, np.ndarray]:
        """Generate dataset"""
        pass
    
    @abstractmethod
    def get_true_effect(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Get true causal effect for evaluation"""
        pass
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        if not hasattr(self, '_index_mapping'):
            self._index_mapping = np.arange(self.n_samples)
        
        actual_idx = self._index_mapping[idx]
        
        item = {
            'covariates': torch.FloatTensor(self.X[actual_idx]),
            'treatment': torch.FloatTensor(self.T[actual_idx]) if self.T.ndim > 1 else torch.FloatTensor([self.T[actual_idx]]),
            'outcome': torch.FloatTensor([self.Y[actual_idx]]),
            'idx': actual_idx
        }
        
        if hasattr(self, 'U'):
            item['confounder'] = torch.FloatTensor(self.U[actual_idx])
        
        return item


class SyntheticCausalDataset(CausalDataset):
    """
    Synthetic dataset with controlled manifold structure and orthogonality violations
    Following the SCM in the proposal
    """
    
    def __init__(self,
                 n_samples: int = 10000,
                 n_covariates: int = 50,
                 confounder_dim: int = 3,
                 treatment_type: str = TreatmentType.CONTINUOUS,
                 orthogonality_violation: float = 0.0,  # δ parameter
                 nonlinearity: str = "neural",  # "linear", "polynomial", "neural"
                 noise_level: float = 0.1,
                 treatment_dim: Optional[int] = None,  # for multilabel
                 seed: Optional[int] = None):
        
        super().__init__(n_samples, n_covariates, treatment_type, seed)
        
        self.confounder_dim = confounder_dim
        self.orthogonality_violation = orthogonality_violation
        self.nonlinearity = nonlinearity
        self.noise_level = noise_level
        self.treatment_dim = treatment_dim if treatment_type == TreatmentType.MULTILABEL else 1
        
        # Generate weight matrices with controlled orthogonality
        self._setup_weight_matrices()
        
        # Generate data
        data = self.generate_data()
        self.U = data['U']
        self.X = data['X']
        self.T = data['T']
        self.Y = data['Y']
        
        # Store true parameters for evaluation
        self.true_params = self._get_true_params()
        
    def _setup_weight_matrices(self):
        """Setup weight matrices with controlled orthogonality violation"""
        # Confounder to covariates mapping
        if self.nonlinearity == "neural":
            self.W_X0 = np.random.randn(self.confounder_dim, 20) * 0.5
            self.b_X0 = np.random.randn(20) * 0.1
            self.W_X1_perp = np.random.randn(20, self.n_covariates)
            self.W_X1_parallel = np.random.randn(20, self.n_covariates)
            
            # Orthogonalize W_X1_perp w.r.t outcome gradient direction
            # This ensures orthogonality when δ=0
            self.W_Y = np.random.randn(self.n_covariates, 1)
            for i in range(20):
                self.W_X1_perp[i] -= np.dot(self.W_X1_perp[i], self.W_Y.flatten()) * self.W_Y.flatten() / np.linalg.norm(self.W_Y)**2
            
            # Mix perpendicular and parallel components based on δ
            self.W_X1 = (1 - self.orthogonality_violation) * self.W_X1_perp + \
                        self.orthogonality_violation * self.W_X1_parallel
            self.b_X1 = np.random.randn(self.n_covariates) * 0.1
        else:
            # Simpler linear mapping
            self.W_X = np.random.randn(self.confounder_dim, self.n_covariates)
            self.W_Y = np.random.randn(self.n_covariates, 1)
            
        # Treatment assignment parameters
        self.w_T = np.random.randn(self.n_covariates, self.treatment_dim) * 0.5
        self.b_T = np.random.randn(self.confounder_dim, self.treatment_dim) * 0.5
        
        # Outcome parameters
        self.c_Y = np.random.randn(self.confounder_dim) * 0.5
        
        # Interaction matrices for outcome
        self.A = np.random.randn(self.n_covariates, self.n_covariates) * 0.01
        self.A = (self.A + self.A.T) / 2  # Symmetric
        self.B = np.random.randn(self.n_covariates, self.treatment_dim) * 0.1
        self.C = np.random.randn(self.treatment_dim, self.treatment_dim) * 0.1
        self.C = (self.C + self.C.T) / 2  # Symmetric
    
    def _get_true_params(self):
        """Get true parameters for storage"""
        params = {
            'W_Y': self.W_Y,
            'c_Y': self.c_Y,
            'A': self.A,
            'B': self.B,
            'C': self.C
        }
        
        if self.nonlinearity == "neural":
            params.update({
                'W_X0': self.W_X0,
                'W_X1': self.W_X1,
                'b_X0': self.b_X0,
                'b_X1': self.b_X1
            })
        else:
            params['W_X'] = self.W_X
            
        return params
    
    def generate_data(self) -> Dict[str, np.ndarray]:
        """Generate data following the SCM"""
        # 1. Generate unobserved confounder
        U = np.random.randn(self.n_samples, self.confounder_dim)
        
        # 2. Generate covariates X = g_X(U) + ε_X
        if self.nonlinearity == "neural":
            hidden = np.tanh(U @ self.W_X0 + self.b_X0)
            X = hidden @ self.W_X1 + self.b_X1
        else:
            X = U @ self.W_X
        
        X += np.random.randn(self.n_samples, self.n_covariates) * self.noise_level
        
        # 3. Generate treatment T = h(w_T^T X + b_T^T U) + ε_T
        treatment_linear = X @ self.w_T + U @ self.b_T
        
        if self.treatment_type == TreatmentType.CONTINUOUS:
            T = treatment_linear.flatten() + np.random.randn(self.n_samples) * self.noise_level
        elif self.treatment_type == TreatmentType.BINARY:
            T_prob = 1 / (1 + np.exp(-treatment_linear.flatten()))
            T = np.random.binomial(1, T_prob).astype(np.float32)
        elif self.treatment_type == TreatmentType.MULTILABEL:
            T_prob = 1 / (1 + np.exp(-treatment_linear))
            T = np.random.binomial(1, T_prob).astype(np.float32)
        
        # 4. Generate outcome Y = f_Y(X, T) + c_Y^T U + ε_Y
        Y = self._compute_structural_outcome(X, T, U)
        Y += np.random.randn(self.n_samples) * self.noise_level
        
        return {'U': U, 'X': X, 'T': T, 'Y': Y.flatten()}
    
    def _compute_structural_outcome(self, X: np.ndarray, T: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Compute structural outcome f_Y(X, T) + c_Y^T U"""
        # Linear terms
        Y = X @ self.W_Y
        
        # Confounder effect
        Y = Y.flatten() + U @ self.c_Y
        
        # Treatment effect with interactions
        if self.treatment_type == TreatmentType.CONTINUOUS:
            T_expanded = T.reshape(-1, 1)
        else:
            T_expanded = T.reshape(len(T), -1) if T.ndim == 1 else T
            
        # Quadratic terms: X^T A X
        Y += np.sum(X * (X @ self.A), axis=1) * 0.1
        
        # Cross terms: X^T B T
        Y += np.sum(X * (T_expanded @ self.B.T), axis=1)
        
        # Treatment quadratic: T^T C T
        if self.treatment_type == TreatmentType.CONTINUOUS:
            # For continuous treatment, handle both scalar and vector cases
            if T.ndim == 1:
                Y += T * T * self.C[0, 0]
            else:
                # T is already 2D
                Y += (T_expanded * (T_expanded @ self.C)).sum(axis=1)
        else:
            Y += np.sum(T_expanded * (T_expanded @ self.C), axis=1)
            
        return Y
    
    def get_true_effect(self, x: np.ndarray, t: np.ndarray, t_cf: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get true causal effect τ(t, t_cf | x)
        For continuous treatment, returns dose-response at t
        """
        # Ensure x is 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Ensure t has proper shape
        if t.ndim == 1:
            t = t.reshape(-1, self.treatment_dim)
        
        if t_cf is None:
            # For continuous, compare to baseline t=0
            if self.treatment_type == TreatmentType.CONTINUOUS:
                t_cf = np.zeros_like(t)
            else:
                t_cf = 1 - t  # Binary flip
        else:
            # Ensure t_cf has proper shape
            if t_cf.ndim == 1:
                t_cf = t_cf.reshape(-1, self.treatment_dim)
        
        # Compute Y(t) - Y(t_cf) marginalizing over U
        # Since U ~ N(0, I), E[U] = 0, so confounder terms cancel
        y_t = self._compute_structural_outcome(x, t, np.zeros((len(x), self.confounder_dim)))
        y_t_cf = self._compute_structural_outcome(x, t_cf, np.zeros((len(x), self.confounder_dim)))
        
        return y_t - y_t_cf
    
    def get_orthogonality_measure(self, x: np.ndarray) -> float:
        """Compute actual orthogonality measure at point x"""
        # Gradient of outcome w.r.t X
        grad_f = self.W_Y.flatten() + 2 * x @ self.A
        
        # Tangent space of confounder manifold (columns of Jacobian of g_X)
        if self.nonlinearity == "neural":
            # Approximate tangent space
            tangent_basis = self.W_X1.T @ self.W_X0.T
        else:
            tangent_basis = self.W_X.T
            
        # Project gradient onto tangent space
        proj_grad = tangent_basis @ np.linalg.pinv(tangent_basis) @ grad_f
        
        # Compute orthogonality measure
        rho = 1 - np.linalg.norm(proj_grad)**2 / np.linalg.norm(grad_f)**2
        
        return rho

class TwinsDataset(CausalDataset):
    """
    加载和处理 "Twins" 数据集的类。
    这个数据集的特殊之处在于每行代表一对双胞胎，我们需要将其展开为每个个体的数据。
    治疗变量 T 定义为：1 表示双胞胎中较轻的一个（高风险），0 表示较重的一个。
    结果变量 Y 是死亡率。
    """
    def __init__(self, data_path: str, normalize: bool = True):
        self.data_path = Path(data_path)
        self.treatment_type = TreatmentType.BINARY

        df_X = pd.read_csv(self.data_path / 'twin_X.csv', index_col=0)
        df_T = pd.read_csv(self.data_path / 'twin_T.csv', index_col=0) 
        df_Y = pd.read_csv(self.data_path / 'twin_Y.csv', index_col=0) 

        x_cols = [c for c in df_X.columns if c not in ['Unnamed: 0', 'infant_id_0', 'infant_id_1']]
        X_raw = df_X[x_cols].values

        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X_raw)

        X_list, T_list, Y_list, ITE_list = [], [], [], []

        for i in range(len(df_X)):
            covariates = X_imputed[i]

            weight_0 = df_T.loc[i, 'dbirwt_0']
            weight_1 = df_T.loc[i, 'dbirwt_1']

            mort_0 = df_Y.loc[i, 'mort_0']
            mort_1 = df_Y.loc[i, 'mort_1']

            if weight_0 == weight_1:
                continue

            treat_0 = 1.0 if weight_0 < weight_1 else 0.0
            treat_1 = 1.0 if weight_1 < weight_0 else 0.0

            if treat_0 == 1: 
                ite_0 = mort_0 - mort_1 
            else: 
                ite_0 = mort_1 - mort_0 

            ite_1 = ite_0

            X_list.extend([covariates, covariates])
            T_list.extend([treat_0, treat_1])
            Y_list.extend([mort_0, mort_1])
            ITE_list.extend([ite_0, ite_1])

        self.X = np.array(X_list, dtype=np.float32)
        self.T = np.array(T_list, dtype=np.float32)
        self.Y = np.array(Y_list, dtype=np.float32)
        self.true_ite = np.array(ITE_list, dtype=np.float32)

        self.n_samples = self.X.shape[0]
        self.n_covariates = self.X.shape[1]

        super().__init__(self.n_samples, self.n_covariates, self.treatment_type)
    def generate_data(self) -> dict[str, np.ndarray]:
        """
        This method is required by the abstract parent class CausalDataset.
        Since TwinsDataset loads data from files, this method does nothing.
        """
        pass
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            'covariates': torch.tensor(self.X[idx]),
            'treatment': torch.tensor([self.T[idx]]),
            'outcome': torch.tensor([self.Y[idx]]),
            'true_ite': torch.tensor(self.true_ite[idx]),
            'idx': torch.tensor(idx, dtype=torch.long)
        }

    def get_true_effect(self, x: np.ndarray, t: np.ndarray, t_cf: Optional[np.ndarray] = None) -> np.ndarray:
        logger.warning("get_true_effect for TwinsDataset returns a proxy ITE and might not be accurate for all evaluations.")
        return np.zeros(len(x))

class DataModule:
    """High-level data module for managing datasets and dataloaders"""
    
    def __init__(self,
                 dataset_name: str = "synthetic",
                 n_samples: int = 10000,
                 n_covariates: int = 50,
                 batch_size: int = 256,
                 num_workers: int = 4,
                 train_ratio: float = 0.6,
                 val_ratio: float = 0.2,
                 normalize: bool = True,
                 **dataset_kwargs):
        
        self.dataset_name = dataset_name
        self.n_samples = n_samples
        self.n_covariates = n_covariates
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.normalize = normalize
        if 'dataset_kwargs' in dataset_kwargs:
            self.dataset_kwargs = dataset_kwargs['dataset_kwargs']
        else:
            self.dataset_kwargs = dataset_kwargs        
        
        # Create dataset
        self.dataset = self._create_dataset()
        
        # Split data
        self.train_dataset, self.val_dataset, self.test_dataset = self._split_dataset()
        
        # Fit normalizer on training data if needed
        if self.normalize:
            self._fit_normalizer()
    
    def _create_dataset(self) -> CausalDataset:
        """Factory method for creating datasets"""
        if self.dataset_name == "synthetic":
            return SyntheticCausalDataset(
                n_samples=self.n_samples,
                n_covariates=self.n_covariates,
                **self.dataset_kwargs
            )
        elif self.dataset_name == "mixed_synthetic":
            from mixed_treatment import MixedTreatmentDataset 
            return MixedTreatmentDataset(
                n_samples=self.n_samples,
                n_covariates=self.n_covariates,
                **self.dataset_kwargs
            )            
        elif self.dataset_name == "ihdp":
            # Import here to avoid circular dependency
            from IHDPnew import IHDPDataset
            return IHDPDataset(
                data_path=self.dataset_kwargs.get('data_path', '/home/xqin5/Multitreatment_adjusted_0707/IHDP/ihdp_full.csv'),
                normalize_covariates=self.normalize
            )
        elif self.dataset_name == "twins":
            dataset = TwinsDataset(
                data_path=self.dataset_kwargs.get('data_path'),
                normalize=False 
            )
            self.n_samples = dataset.n_samples
            return dataset            
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _split_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Split dataset into train/val/test"""
        n_train = int(self.n_samples * self.train_ratio)
        n_val = int(self.n_samples * self.val_ratio)
        n_test = self.n_samples - n_train - n_val
        
        indices = np.random.permutation(self.n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        # Create subset datasets
        train_dataset = torch.utils.data.Subset(self.dataset, train_idx)
        val_dataset = torch.utils.data.Subset(self.dataset, val_idx)
        test_dataset = torch.utils.data.Subset(self.dataset, test_idx)
        
        # Store indices for later use
        self.dataset._train_indices = train_idx
        self.dataset._val_indices = val_idx
        self.dataset._test_indices = test_idx
        
        return train_dataset, val_dataset, test_dataset
    
    def _fit_normalizer(self):
        """Fit normalizer on training data"""
        # Collect training data
        X_train = []
        for i in range(len(self.train_dataset)):
            X_train.append(self.train_dataset[i]['covariates'].numpy())
        X_train = np.array(X_train)
        
        # Fit scaler
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        
        # Apply normalization to dataset
        if hasattr(self.dataset, 'X'):
            self.dataset.X = self.scaler.transform(self.dataset.X)
    
    def get_dataloader(self, split: str = "train", shuffle: Optional[bool] = None) -> DataLoader:
        """Get dataloader for specified split"""
        if split == "train":
            dataset = self.train_dataset
            shuffle = shuffle if shuffle is not None else True
        elif split == "val":
            dataset = self.val_dataset
            shuffle = shuffle if shuffle is not None else False
        elif split == "test":
            dataset = self.test_dataset
            shuffle = shuffle if shuffle is not None else False
        else:
            raise ValueError(f"Unknown split: {split}")
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_treatment_type(self) -> str:
        """Get treatment type of the dataset"""
        # Handle Subset case
        if hasattr(self.dataset, 'dataset'):
            # This is a Subset, get the underlying dataset
            actual_dataset = self.dataset.dataset
            while hasattr(actual_dataset, 'dataset'):
                actual_dataset = actual_dataset.dataset
            return actual_dataset.treatment_type
        else:
            return self.dataset.treatment_type
    
    def get_dims(self) -> Dict[str, int]:
        """Get dimensionality information"""
        # Get actual dataset if wrapped in Subset
        actual_dataset = self.dataset
        while hasattr(actual_dataset, 'dataset'):
            actual_dataset = actual_dataset.dataset
        
        # Use the actual dataset to get a sample
        if hasattr(actual_dataset, '__getitem__'):
            sample = actual_dataset[0]
        else:
            sample = self.dataset[0]
        
        return {
            'n_covariates': sample['covariates'].shape[0],
            'treatment_dim': sample['treatment'].shape[0],
            'outcome_dim': sample['outcome'].shape[0],
            'confounder_dim': sample.get('confounder').shape[0] if sample.get('confounder') is not None else 0
        }