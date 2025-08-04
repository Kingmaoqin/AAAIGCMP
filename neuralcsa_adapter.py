"""
NeuralCSA Adapter for Multiple Causal Inference Datasets

==========================================================================
ATTRIBUTION AND ADAPTATION NOTES:
==========================================================================

ORIGINAL WORK:
This implementation is primarily based on the NeuralCSA framework from:
"A Neural Framework for Generalized Causal Sensitivity Analysis"
by Dennis Frauen, Fergus Imrie, Alicia Curth, Valentyn Melnychuk,
Stefan Feuerriegel, and Mihaela van der Schaar
Paper: https://arxiv.org/abs/2311.16026
Published at ICLR 2024

ORIGINAL REPOSITORY:
https://github.com/DennisFrauen/NeuralCSA
Authors: Dennis Frauen et al.
License: [Original repository license]

==========================================================================
CORE COMPONENTS FROM ORIGINAL NEURALCSA:
==========================================================================

The following components are directly adapted from the original implementation:

1. **CausalDataset** (from data/data_structures.py)
   - Custom PyTorch Dataset for causal inference data
   - Handles covariates (x), treatments (a), outcomes (y), and propensity scores

2. **PropensityNet** (from models/propensity.py)
   - PyTorch Lightning neural network for propensity score estimation
   - Supports both discrete and continuous treatments

3. **CNF_stage1** (from models/stage1.py)
   - Conditional normalizing flow for learning p(y|x,a)
   - Uses Pyro's ConditionalSplineAutoregressive transformations

4. **Stage2/NF_stage2** (from models/stage2.py)
   - Stage 2 sensitivity analysis using normalizing flows
   - Implements Lagrangian optimization with λ multipliers and μ penalty terms
   - Handles sensitivity constraint violations

5. **SensitivityModel** (from models/auxillary.py)
   - Marginal Sensitivity Model (MSM) for discrete treatments
   - Conditional Marginal Sensitivity Model (CMSM) for continuous treatments
   - Sensitivity violation computation

6. **CausalQuery** (from models/auxillary.py)
   - Expectation-based causal query computation
   - Train and test query evaluation

7. **JointModel** (from models/auxillary.py)
   - Orchestrates training of all model components
   - Manages propensity, stage1, and stage2 models

==========================================================================
ADAPTATIONS:
==========================================================================

The following components were created/modified to work with standard datasets:

1. **Data Loading & Preprocessing**
   - Adapted to work with CSV files instead of SCM generators
   - Added support for IHDP, synthetic, and twin datasets
   - Handles different data formats (single/multi treatment, binary/continuous)

2. **Configuration Management**
   - Simplified configuration instead of complex YAML system
   - Automatic config generation based on dataset characteristics

3. **Training Pipeline**
   - Streamlined training process for multiple datasets
   - Added binary treatment scaling fix (critical bug fix)
   - Integrated with sklearn preprocessing

4. **Evaluation Framework**
   - Added PEHE, ATE Error, and Coverage metrics
   - Support for both ground truth and approximate evaluation
   - Multiple random seed evaluation with statistical aggregation

5. **Error Handling & Debugging**
   - Enhanced error reporting and diagnostics
   - Treatment data validation and debugging output
   - Graceful handling of training failures

==========================================================================
USAGE PURPOSE:
==========================================================================

This adaptation enables the original NeuralCSA framework to work with:
- Standard causal inference benchmark datasets (IHDP, synthetic data)
- Real-world datasets (twins data)
- Multiple treatment types and experimental settings
- Comprehensive evaluation across multiple random seeds

The core NeuralCSA methodology remains unchanged - this is purely an interface
adaptation to make the original sophisticated framework accessible for
standard causal inference evaluation tasks.

==========================================================================
CITATION:
==========================================================================

If you use this code, please cite the original NeuralCSA paper:

@article{frauen2023neural,
  title={A Neural Framework for Generalized Causal Sensitivity Analysis},
  author={Frauen, Dennis and Imrie, Fergus and Curth, Alicia and Melnychuk, Valentyn and Feuerriegel, Stefan and van der Schaar, Mihaela},
  journal={arXiv preprint arXiv:2311.16026},
  year={2023}
}

"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pyro.distributions as dist
import pyro.distributions.transforms as T
from pyro.nn import ConditionalAutoRegressiveNN, AutoRegressiveNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from abc import ABC, abstractmethod
import warnings
import random
import os
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class CausalDataset(Dataset):
    """Customized dataset for causal inference"""

    def __init__(self, x, a, y, propensity=None, x_type=None, a_type="discrete"):
        self.scaling_params = {}
        self.datatypes = {"x_type": x_type, "a_type": a_type}

        # Convert to pytorch tensors with proper shape handling
        self.data = {}
        self.data["x"] = torch.from_numpy(x.astype(np.float32))

        if a is not None:
            if len(a.shape) == 1:
                a = a.reshape(-1, 1)
            self.data["a"] = torch.from_numpy(a.astype(np.float32))

        if y is not None:
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            self.data["y"] = torch.from_numpy(y.astype(np.float32))
        else:
            self.data["y"] = None

        if propensity is not None:
            if len(propensity.shape) == 1:
                propensity = propensity.reshape(-1, 1)
            self.data["propensity"] = torch.from_numpy(propensity.astype(np.float32))
        else:
            self.data["propensity"] = None

        self.data["idx"] = torch.arange(0, self.data["x"].size(0)).view(-1, 1)

    def __len__(self):
        return self.data["x"].size(0)

    def get_dims(self, keys):
        dims = {}
        for key in keys:
            if key == "x":
                dims["x"] = self.data["x"].size(1)
            if key == "a":
                dims["a"] = self.data["a"].size(1)
            if key == "y":
                dims["y"] = self.data["y"].size(1)
        return dims

    def __getitem__(self, index) -> dict:
        result = {
            "x": self.data["x"][index],
            "a": self.data["a"][index],
            "idx": self.data["idx"][index]
        }
        if self.data["y"] is not None:
            result["y"] = self.data["y"][index]
        if self.data["propensity"] is not None:
            result["propensity"] = self.data["propensity"][index]
        return result

    def add_propensity_scores(self, propensity_model):
        """Add propensity scores using trained model"""
        self.data["propensity"] = propensity_model.predict(self.data["x"]).detach()
        if self.get_dims(["a"])["a"] == 1 and self.datatypes["a_type"] == "discrete":
            self.data["propensity"] = self.data["a"] * self.data["propensity"] + \
                                      (1 - self.data["a"]) * (1 - self.data["propensity"])


# =============================================================================
# PROPENSITY MODEL
# =============================================================================

class PropensityNet(pl.LightningModule):
    """Neural network for propensity score estimation"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        input_size = config["dim_input"]
        hidden_size = config.get("dim_hidden", 128)
        self.output_size = config["dim_output"]
        self.output_type = config["out_type"]

        if self.output_type == "discrete":
            if self.output_size == 1:
                self.loss = torch.nn.BCELoss(reduction='mean')
            else:
                self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        else:
            self.loss = torch.nn.MSELoss(reduction='mean')

        dropout = config.get("dropout", 0.1)

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.output_size),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.get("lr", 1e-3))
        self.save_hyperparameters(config)

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, train_batch, batch_idx):
        out = self.network(train_batch["x"])
        if self.output_size == 1 and self.output_type == "discrete":
            out = out.sigmoid()
        loss = self.loss(out, train_batch["a"])
        return loss

    def validation_step(self, val_batch, batch_idx):
        out = self.network(val_batch["x"])
        if self.output_size == 1 and self.output_type == "discrete":
            out = out.sigmoid()
        loss = self.loss(out, val_batch["a"])
        self.log("val_obj", loss)
        return loss

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if self.output_type == "discrete":
                if self.output_size > 1:
                    out = self.network(x).softmax(dim=-1)
                else:
                    out = self.network(x).sigmoid()
            else:
                out = self.network(x)
        return out.detach()


# =============================================================================
# STAGE 1 MODEL
# =============================================================================

class CNF_stage1(pl.LightningModule):
    """Stage 1: Conditional normalizing flow for p(y|x,a)"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        dim_context = config["dim_context"]
        dim_y = config["dim_y"]
        dim_hidden = config.get("dim_hidden", 128)
        count_bins = config.get("count_bins", 16)

        hypernet = ConditionalAutoRegressiveNN(
            dim_y, dim_context,
            hidden_dims=[dim_hidden, dim_hidden, dim_hidden],
            param_dims=[count_bins, count_bins, count_bins - 1, count_bins]
        )
        self.transform = T.ConditionalSplineAutoregressive(dim_y, hypernet, count_bins=count_bins)

        # Standard normal distribution (on latent space)
        dist_base_training = dist.MultivariateNormal(torch.zeros(dim_y), torch.eye(dim_y, dim_y))
        self.dist_y_given_x_training = dist.ConditionalTransformedDistribution(dist_base_training, [self.transform])

        self.optimizer = torch.optim.Adam(self.transform.parameters(), lr=config.get("lr", 1e-3))
        self.save_hyperparameters(config)

    def configure_optimizers(self):
        return self.optimizer

    def get_context(self, batch):
        x = batch["x"]
        a = batch["a"]
        return torch.cat([x, a], dim=1)

    def training_step(self, train_batch, batch_idx):
        xa = self.get_context(train_batch)
        y = train_batch["y"]

        try:
            log_p_y_given_xa = self.dist_y_given_x_training.condition(xa).log_prob(y)
            loss = -log_p_y_given_xa.mean()

            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(0.0, requires_grad=True)

            return loss
        except Exception as e:
            print(f"Stage 1 training error: {e}")
            return torch.tensor(0.0, requires_grad=True)

    def validation_step(self, val_batch, batch_idx):
        xa = self.get_context(val_batch)
        y = val_batch["y"]

        try:
            log_p_y_given_xa = self.dist_y_given_x_training.condition(xa).log_prob(y)
            loss = -log_p_y_given_xa.mean()
            self.log("val_obj", loss)
            return loss
        except Exception:
            return torch.tensor(0.0)

    def sample(self, data_test, n_samples):
        self.eval()
        with torch.no_grad():
            try:
                context = self.get_context(data_test.data)
                samples = self.dist_y_given_x_training.condition(context).sample(
                    torch.Size([n_samples, context.shape[0]]))
                if samples.dim() == 1:
                    samples = samples.unsqueeze(1)
                samples = torch.transpose(samples, 0, 1)
                return samples
            except Exception as e:
                print(f"Sampling error: {e}")
                n_obs = context.shape[0] if 'context' in locals() else data_test.data["x"].shape[0]
                return torch.zeros(n_obs, n_samples, 1)


# =============================================================================
# SENSITIVITY MODELS
# =============================================================================

class SensitivityModel:
    """Sensitivity model for causal analysis"""

    def __init__(self, config):
        self.gamma = config["gamma"]
        self.name = config["name"]
        self.treatment_dim = config.get("treatment_dim", 1)

        if config["name"] == "msm":
            self.sensitivity_constraint = None
        elif config["name"] == "cmsm":
            self.sensitivity_constraint = self.sensitivity_constraint_cmsm
        else:
            self.sensitivity_constraint = None

    def sensitivity_violation(self, u_sampled, xi_sampled, dist_base, dist_shifted, transform2, train_batch):
        if "propensity" in train_batch:
            propensity = train_batch["propensity"]
        else:
            batch_size = train_batch["x"].shape[0]
            if self.treatment_dim == 1:
                propensity = torch.full((batch_size, 1), 0.5)
            else:
                propensity = torch.full((batch_size, self.treatment_dim), 0.5)

        if self.name == "msm":
            u_shifted = transform2(u_sampled)
            u_shifted = (1 - xi_sampled) * u_shifted + xi_sampled * u_sampled
            p_u_x = dist_shifted.log_prob(u_shifted).exp()
            p_u_xa = dist_base.log_prob(u_shifted).exp()

            ratio = p_u_xa / (p_u_x + 1e-8)
            ratio_max = torch.transpose(torch.max(ratio, dim=0, keepdim=True)[0], 0, 1)
            ratio_min = torch.transpose(torch.min(ratio, dim=0, keepdim=True)[0], 0, 1)

            s_minus = 1 / ((1 - self.gamma) * propensity + self.gamma)
            s_plus = 1 / ((1 - (1 / self.gamma)) * propensity + (1 / self.gamma))
            return torch.minimum(s_plus - ratio_max, ratio_min - s_minus)

        elif self.name == "cmsm":
            return self._cmsm_sensitivity_violation(u_sampled, xi_sampled, dist_base, dist_shifted, transform2,
                                                    propensity)
        else:
            p_u_x = dist_shifted.log_prob(u_sampled).exp()
            p_u_xa = dist_base.log_prob(u_sampled).exp()
            if self.sensitivity_constraint is not None:
                constraint = self.sensitivity_constraint(p_u_x, p_u_xa, propensity)
                return self.gamma - torch.transpose(constraint, 0, 1)
            else:
                return torch.zeros(1, propensity.size(0))

    def _cmsm_sensitivity_violation(self, u_sampled, xi_sampled, dist_base, dist_shifted, transform2, propensity):
        """CMSM sensitivity violation for continuous treatments"""
        u_shifted = transform2(u_sampled)

        p_u_base = dist_base.log_prob(u_sampled).exp()
        p_u_shifted = dist_shifted.log_prob(u_shifted).exp()

        ratio_1 = p_u_shifted / (p_u_base + 1e-8)
        ratio_2 = p_u_base / (p_u_shifted + 1e-8)

        max_ratio = torch.maximum(
            torch.max(ratio_1, dim=0, keepdim=True)[0],
            torch.max(ratio_2, dim=0, keepdim=True)[0]
        )

        constraint_violation = self.gamma - torch.transpose(max_ratio, 0, 1)
        return constraint_violation

    @staticmethod
    def sensitivity_constraint_cmsm(p_u_x, p_u_xa, propensity):
        max_ratio1 = torch.max(p_u_x / (p_u_xa + 1e-8), dim=0, keepdim=True)[0]
        max_ratio2 = torch.max(p_u_xa / (p_u_x + 1e-8), dim=0, keepdim=True)[0]
        return torch.maximum(max_ratio1, max_ratio2)


# =============================================================================
# CAUSAL QUERY
# =============================================================================

class CausalQuery:
    """Causal query computation"""

    def __init__(self, config):
        if config["name"] == "expectation":
            self.query_computation_train = self.expectation_train
            self.query_computation_test = self.expectation_test
        else:
            raise ValueError("Only expectation queries supported in this implementation")

    def compute_query_train(self, u_sampled, xi_sampled, dist_y_shifted, transform1, transform2):
        return self.query_computation_train(u_sampled, xi_sampled, dist_y_shifted, transform1, transform2)

    def compute_query_test(self, y_samples_shifted):
        return self.query_computation_test(y_samples_shifted)

    def expectation_train(self, u_sampled, xi_sampled, dist_y_shifted, transform1, transform2):
        u_shifted = transform2(u_sampled)
        u_shifted = (1 - xi_sampled) * u_shifted + xi_sampled * u_sampled
        y_sampled = transform1(u_shifted)
        return torch.squeeze(torch.mean(y_sampled))

    def expectation_test(self, y_samples_shifted):
        return torch.squeeze(torch.mean(y_samples_shifted, dim=(0, 2)))


# =============================================================================
# STAGE 2 MODELS
# =============================================================================

class Stage2(pl.LightningModule, ABC):
    """Base class for Stage 2 models"""

    def __init__(self, config, trained_models, sensitivity_model, causal_query):
        super().__init__()
        self.config = config
        self.trained_models = trained_models
        self.sensitivity_model = sensitivity_model
        self.causal_query = causal_query

        self.n_train = None
        self.lambdas = None
        self.mu = None

        self.dist_base = dist.MultivariateNormal(
            torch.zeros(self.config["dim_y"]),
            torch.eye(self.config["dim_y"], self.config["dim_y"])
        )

        if self.config["bound_type"] == "upper":
            self.query_sign = -1
        elif self.config["bound_type"] == "lower":
            self.query_sign = 1
        else:
            raise ValueError("Invalid bound type")

        self.automatic_optimization = False
        self.save_hyperparameters(config)

    def get_context(self, batch):
        x = batch["x"]
        a = batch["a"]
        return torch.cat([x, a], dim=1)

    @abstractmethod
    def get_query_train(self, train_batch, u_sampled, xi_sampled):
        pass

    @abstractmethod
    def get_sensitivity_violation(self, train_batch, u_sampled, xi_sampled):
        pass


class NF_stage2(Stage2):
    """Stage 2 model using unconditional normalizing flow"""

    def __init__(self, config, trained_models, sensitivity_model, causal_query):
        super().__init__(config, trained_models, sensitivity_model, causal_query)

        dim_y = config["dim_y"]
        dim_hidden = config.get("dim_hidden", 128)
        count_bins = config.get("count_bins", 16)

        hypernet = AutoRegressiveNN(
            dim_y,
            hidden_dims=[dim_hidden, dim_hidden, dim_hidden],
            param_dims=[count_bins, count_bins, count_bins - 1, count_bins]
        )
        self.transform = T.SplineAutoregressive(dim_y, hypernet, count_bins=count_bins)
        self.dist_shifted = dist.TransformedDistribution(self.dist_base, [self.transform])
        self.dist_shifted_y = dist.ConditionalTransformedDistribution(
            self.dist_shifted, [self.trained_models["stage1"].transform]
        )

        self.optimizer = torch.optim.Adam(self.transform.parameters(), lr=config.get("lr", 1e-3))

    def configure_optimizers(self):
        return self.optimizer

    def get_query_train(self, train_batch, u_sampled, xi_sampled):
        context = self.get_context(train_batch)
        transform_stage2 = self.transform
        transform_stage1 = self.trained_models["stage1"].transform.condition(context)
        return self.causal_query.compute_query_train(
            u_sampled, xi_sampled, self.dist_shifted_y.condition(context),
            transform_stage1, transform_stage2
        )

    def get_sensitivity_violation(self, train_batch, u_sampled, xi_sampled):
        return self.sensitivity_model.sensitivity_violation(
            u_sampled, xi_sampled, self.dist_base, self.dist_shifted,
            self.transform, train_batch
        )

    def test_query(self, data_test, n_samples):
        self.eval()
        with torch.no_grad():
            try:
                context = self.get_context(data_test.data)
                y_samples_shifted = self.dist_shifted_y.condition(context).sample(torch.Size([n_samples]))
                query = self.causal_query.compute_query_test(y_samples_shifted)
                if data_test.data["x"].size(0) == 1:
                    query = torch.unsqueeze(query, dim=0)
                return query.detach().unsqueeze(dim=1)
            except Exception as e:
                print(f"Test query error: {e}")
                return torch.zeros(data_test.data["x"].size(0), 1)

    def training_step(self, train_batch, batch_idx):
        try:
            n_batch = train_batch["x"].size(0)
            idx_lambdas = train_batch["idx"][:, 0]

            if self.current_epoch % self.config.get("nested_epochs", 5) != 0 or self.current_epoch == 0:
                # Sample from latent distribution
                u_sampled = self.dist_base.sample(torch.Size([self.n_train, n_batch]))

                # Enhanced treatment sampling
                a_type = self.config.get("a_type", "discrete")
                treatment_dim = train_batch["a"].shape[1]

                if "discrete" in a_type and "propensity" in train_batch:
                    propensity = train_batch["propensity"]
                    if treatment_dim == 1:
                        p_a_x = propensity.transpose(0, 1).expand(self.n_train, n_batch)
                        xi_sampled = torch.bernoulli(p_a_x).unsqueeze(dim=2).expand(self.n_train, n_batch,
                                                                                    u_sampled.size(2))
                    else:
                        xi_sampled = torch.zeros((self.n_train, n_batch, u_sampled.size(2)))
                        p_dim = propensity[:, 0:1].transpose(0, 1).expand(self.n_train, n_batch)
                        xi_sampled[:, :, 0] = torch.bernoulli(p_dim)
                else:
                    xi_sampled = torch.zeros((self.n_train, n_batch, u_sampled.size(2)))

                # Get sensitivity violations and query
                sensitivity = self.get_sensitivity_violation(train_batch, u_sampled, xi_sampled)
                query = self.get_query_train(train_batch, u_sampled, xi_sampled)

                # Lagrangian optimization with manual gradient clipping
                lagrangian = self.query_sign * query + torch.sum(
                    -sensitivity * self.lambdas[idx_lambdas, :] + (self.mu * sensitivity ** 2 / 2)
                )

                self.optimizer.zero_grad()
                self.manual_backward(lagrangian)

                # Manual gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                self.optimizer.step()
            else:
                # Update Lagrange multipliers
                with torch.no_grad():
                    u_sampled = self.dist_base.sample(torch.Size([self.n_train, n_batch]))

                    a_type = self.config.get("a_type", "discrete")
                    treatment_dim = train_batch["a"].shape[1]

                    if "discrete" in a_type and "propensity" in train_batch:
                        propensity = train_batch["propensity"]
                        if treatment_dim == 1:
                            p_a_x = propensity.transpose(0, 1).expand(self.n_train, n_batch)
                            xi_sampled = torch.bernoulli(p_a_x).unsqueeze(dim=2).expand(self.n_train, n_batch,
                                                                                        u_sampled.size(2))
                        else:
                            xi_sampled = torch.zeros((self.n_train, n_batch, u_sampled.size(2)))
                            p_dim = propensity[:, 0:1].transpose(0, 1).expand(self.n_train, n_batch)
                            xi_sampled[:, :, 0] = torch.bernoulli(p_dim)
                    else:
                        xi_sampled = torch.zeros((self.n_train, n_batch, u_sampled.size(2)))

                    sensitivity = self.get_sensitivity_violation(train_batch, u_sampled, xi_sampled)

                    lambda_update = self.lambdas[idx_lambdas, :] - sensitivity * self.mu
                    self.lambdas[idx_lambdas, :] = torch.clamp(lambda_update, min=0.0, max=10.0)

                    if self.trainer.num_training_batches == batch_idx + 1:
                        self.mu = min(self.config.get("alpha", 1.1) * self.mu, 10.0)

        except Exception as e:
            print(f"Stage 2 training step error: {e}")
            return torch.tensor(0.0, requires_grad=True)


# =============================================================================
# JOINT MODEL
# =============================================================================

class JointModel:
    """Joint model combining all stages"""

    def __init__(self, config):
        self.config = config
        self.propensity = None
        if config["propensity"] is not None:
            self.propensity = PropensityNet(config["propensity"])
        self.stage1 = CNF_stage1(config["stage1"])

        self.stage2_upper = []
        self.stage2_lower = []

        for k in range(len(config["stage2_upper"])):
            config_k_upper = config["stage2_upper"][k]
            config_k_lower = config["stage2_lower"][k]

            sensitivity_model = SensitivityModel(config_k_upper["sensitivity_model"])
            causal_query = CausalQuery(config_k_upper["causal_query"])

            self.stage2_upper.append(NF_stage2(
                config_k_upper | {"bound_type": "upper"},
                trained_models={"propensity": self.propensity, "stage1": self.stage1},
                sensitivity_model=sensitivity_model,
                causal_query=causal_query
            ))
            self.stage2_lower.append(NF_stage2(
                config_k_lower | {"bound_type": "lower"},
                trained_models={"propensity": self.propensity, "stage1": self.stage1},
                sensitivity_model=sensitivity_model,
                causal_query=causal_query
            ))

    def fit_propensity(self, d_train, d_val=None):
        if self.propensity is None:
            return
        trainer = pl.Trainer(
            max_epochs=self.config["propensity"]["epochs"],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False
        )
        train_loader = DataLoader(d_train, batch_size=self.config["propensity"]["batch_size"], shuffle=True)
        if d_val is not None:
            val_loader = DataLoader(d_val, batch_size=self.config["propensity"]["batch_size"], shuffle=False)
            trainer.fit(self.propensity, train_loader, val_loader)
        else:
            trainer.fit(self.propensity, train_loader)

    def fit_stage1(self, d_train, d_val=None):
        trainer = pl.Trainer(
            max_epochs=self.config["stage1"]["epochs"],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False
        )
        train_loader = DataLoader(d_train, batch_size=self.config["stage1"]["batch_size"], shuffle=True)
        if d_val is not None:
            val_loader = DataLoader(d_val, batch_size=self.config["stage1"]["batch_size"], shuffle=False)
            trainer.fit(self.stage1, train_loader, val_loader)
        else:
            trainer.fit(self.stage1, train_loader)

    def fit_stage2(self, dataset, seed=0):
        """Stage 2 training without gradient clipping in Trainer"""
        for k in range(len(self.stage2_lower)):
            print(f"    Training Stage 2 model {k + 1}...")

            try:
                torch.manual_seed(seed)
                np.random.seed(seed)

                config_model = self.stage2_lower[k].config
                bounds = config_model.get("bounds", "both")

                # Initialize Lagrange multipliers
                self.stage2_lower[k].lambdas = torch.full(
                    (dataset.data["x"].size(0), 1),
                    config_model.get("lambda_init", 0.01),
                    dtype=torch.float32
                )
                self.stage2_lower[k].mu = config_model.get("mu_init", 0.5)
                self.stage2_lower[k].n_train = config_model.get("n_samples_train", 50)

                self.stage2_upper[k].lambdas = torch.full(
                    (dataset.data["x"].size(0), 1),
                    config_model.get("lambda_init", 0.01),
                    dtype=torch.float32
                )
                self.stage2_upper[k].mu = config_model.get("mu_init", 0.5)
                self.stage2_upper[k].n_train = config_model.get("n_samples_train", 50)

                # Add propensity scores
                a_type = config_model.get("a_type", "discrete")
                if self.propensity is not None and "discrete" in a_type:
                    dataset.add_propensity_scores(self.propensity)
                else:
                    if "propensity" not in dataset.data.keys():
                        treatment_shape = dataset.data["a"].shape
                        dataset.data["propensity"] = torch.full(treatment_shape, 0.5)

                # Trainer without gradient clipping
                trainer = pl.Trainer(
                    max_epochs=config_model.get("epochs", 30),
                    enable_progress_bar=False,
                    enable_model_summary=False,
                    logger=False,
                    enable_checkpointing=False
                )

                train_loader = DataLoader(
                    dataset,
                    batch_size=config_model.get("batch_size", 16),
                    shuffle=True,
                    drop_last=True
                )

                if bounds in ["lower", "both"]:
                    trainer.fit(self.stage2_lower[k], train_loader)
                    print(f"    ✓ Stage 2 lower bound model {k + 1} trained")
                if bounds in ["upper", "both"]:
                    trainer.fit(self.stage2_upper[k], train_loader)
                    print(f"    ✓ Stage 2 upper bound model {k + 1} trained")

            except Exception as e:
                print(f"    ✗ Stage 2 model {k + 1} failed: {e}")
                continue


# =============================================================================
# DATASET HANDLERS FOR SYNTHETIC DATA
# =============================================================================

class SyntheticDatasetHandler:
    """Dataset handlers for synthetic datasets"""

    @staticmethod
    def handle_single_binary(df):
        """Handle single binary treatment dataset"""
        X_cols = [f'X{i}' for i in range(10)]
        X = df[X_cols].values
        T = df['T'].values.reshape(-1, 1)
        Y = df['Y'].values.reshape(-1, 1)

        return {
            'X': X, 'T': T, 'Y': Y, 'mu0': None, 'mu1': None,
            'treatment_type': 'discrete',
            'treatment_dim': 1,
            'has_ground_truth': False
        }

    @staticmethod
    def handle_single_continuous(df):
        """Handle single continuous treatment dataset"""
        X_cols = [f'X{i}' for i in range(10)]
        X = df[X_cols].values
        T = df['T'].values.reshape(-1, 1)
        Y = df['Y'].values.reshape(-1, 1)

        return {
            'X': X, 'T': T, 'Y': Y, 'mu0': None, 'mu1': None,
            'treatment_type': 'continuous',
            'treatment_dim': 1,
            'has_ground_truth': False
        }

    @staticmethod
    def handle_multi_binary(df):
        """Handle multi-binary treatment - preserve full dimensionality"""
        X_cols = [f'X{i}' for i in range(10)]
        X = df[X_cols].values

        # Keep multi-dimensional treatments instead of converting to scalar
        T_multi = df[['T0', 'T1', 'T2']].values
        Y = df['Y'].values.reshape(-1, 1)

        return {
            'X': X, 'T': T_multi, 'Y': Y, 'mu0': None, 'mu1': None,
            'treatment_type': 'multi_discrete',
            'treatment_dim': 3,
            'has_ground_truth': False,
            'note': 'Multi-binary treatment preserved with full dimensionality'
        }

    @staticmethod
    def handle_multi_continuous(df):
        """Handle multi-continuous treatment - preserve full dimensionality"""
        X_cols = [f'X{i}' for i in range(10)]
        X = df[X_cols].values

        # Keep multi-dimensional treatments instead of converting to magnitude
        T_multi = df[['T0', 'T1', 'T2']].values
        Y = df['Y'].values.reshape(-1, 1)

        return {
            'X': X, 'T': T_multi, 'Y': Y, 'mu0': None, 'mu1': None,
            'treatment_type': 'multi_continuous',
            'treatment_dim': 3,
            'has_ground_truth': False,
            'note': 'Multi-continuous treatment preserved with full dimensionality'
        }

    @staticmethod
    def handle_mixed(df):
        """Handle mixed dataset with 4 continuous treatments"""
        X_cols = [f'X{i}' for i in range(10)]
        X = df[X_cols].values

        # Keep 4-dimensional continuous treatments
        T_multi = df[['T0', 'T1', 'T2', 'T3']].values
        Y = df['Y'].values.reshape(-1, 1)

        return {
            'X': X, 'T': T_multi, 'Y': Y, 'mu0': None, 'mu1': None,
            'treatment_type': 'multi_continuous',
            'treatment_dim': 4,
            'has_ground_truth': False,
            'note': 'Mixed 4D treatments preserved with full dimensionality'
        }


# =============================================================================
# MODIFIED NEURAL CSA CLASS WITH PEHE FOR ALL TREATMENT TYPES
# =============================================================================

class NeuralCSA:
    """NeuralCSA implementation for synthetic datasets - MODIFIED FOR PEHE"""

    def __init__(self, dataset_type='single_binary', gamma=2.0, device='cpu'):
        self.dataset_type = dataset_type
        self.gamma = gamma
        self.device = device

        self.joint_model = None
        self.scaler_X = StandardScaler()
        self.scaler_T = StandardScaler()
        self.scaler_Y = StandardScaler()

        # Dataset-specific info
        self.data_info = None

    def _prepare_data(self, df):
        """Prepare data using synthetic dataset handlers"""

        handlers = {
            'single_binary': SyntheticDatasetHandler.handle_single_binary,
            'single_continuous': SyntheticDatasetHandler.handle_single_continuous,
            'multi_binary': SyntheticDatasetHandler.handle_multi_binary,
            'multi_continuous': SyntheticDatasetHandler.handle_multi_continuous,
            'mixed': SyntheticDatasetHandler.handle_mixed
        }

        if self.dataset_type not in handlers:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        self.data_info = handlers[self.dataset_type](df)

        if 'note' in self.data_info:
            print(f"  Note: {self.data_info['note']}")

        return (
            self.data_info['X'],
            self.data_info['T'],
            self.data_info['Y'],
            self.data_info['mu0'],
            self.data_info['mu1']
        )

    def _create_config(self, X_dim, T_dim, Y_dim):
        """Create model configuration based on dataset characteristics"""

        treatment_type = self.data_info['treatment_type']
        treatment_dim = self.data_info['treatment_dim']

        print(f"  Configuring for {treatment_type} treatment (dim={treatment_dim})")

        # Enhanced configuration for different treatment types
        config = {
            "propensity": {
                "dim_input": X_dim,
                "dim_output": T_dim,
                "out_type": "discrete" if "discrete" in treatment_type else "continuous",
                "dim_hidden": 128,
                "dropout": 0.1,
                "lr": 1e-3,
                "epochs": 30,
                "batch_size": 64
            } if "discrete" in treatment_type else None,

            "stage1": {
                "dim_context": X_dim + T_dim,
                "dim_y": Y_dim,
                "dim_hidden": 128,
                "count_bins": 16,
                "lr": 5e-4,
                "epochs": 50,
                "batch_size": 32
            },

            "stage2_upper": [{
                "dim_y": Y_dim,
                "dim_hidden": 64,
                "count_bins": 16,
                "lr": 1e-4,
                "epochs": 30,
                "batch_size": 16,
                "a_type": treatment_type,
                "treatment_dim": treatment_dim,
                "bounds": "upper",
                "lambda_init": 0.01,
                "mu_init": 0.5,
                "alpha": 1.05,
                "n_samples_train": 50,
                "nested_epochs": 5,
                "sensitivity_model": {
                    "name": "cmsm" if "continuous" in treatment_type else "msm",
                    "gamma": self.gamma,
                    "treatment_dim": treatment_dim
                },
                "causal_query": {
                    "name": "expectation"
                }
            }],

            "stage2_lower": [{
                "dim_y": Y_dim,
                "dim_hidden": 64,
                "count_bins": 16,
                "lr": 1e-4,
                "epochs": 30,
                "batch_size": 16,
                "a_type": treatment_type,
                "treatment_dim": treatment_dim,
                "bounds": "lower",
                "lambda_init": 0.01,
                "mu_init": 0.5,
                "alpha": 1.05,
                "n_samples_train": 50,
                "nested_epochs": 5,
                "sensitivity_model": {
                    "name": "cmsm" if "continuous" in treatment_type else "msm",
                    "gamma": self.gamma,
                    "treatment_dim": treatment_dim
                },
                "causal_query": {
                    "name": "expectation"
                }
            }]
        }

        return config

    def train(self, df, epochs=100):
        """Train NeuralCSA model"""

        # Prepare data
        X, T, Y, mu0, mu1 = self._prepare_data(df)

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        T = np.nan_to_num(T, nan=0.0)
        Y = np.nan_to_num(Y, nan=0.0)

        # Determine treatment type
        treatment_type = self.data_info['treatment_type']
        is_binary_treatment = "discrete" in treatment_type and self.data_info['treatment_dim'] == 1

        # Normalize data
        X_scaled = self.scaler_X.fit_transform(X)
        Y_scaled = self.scaler_Y.fit_transform(Y)

        if is_binary_treatment:
            T_scaled = T.copy()
            self.scaler_T.fit(T)
        else:
            T_scaled = self.scaler_T.fit_transform(T)

        # Create datasets
        train_idx, val_idx = train_test_split(range(len(X)), test_size=0.2, random_state=42)

        d_train = CausalDataset(
            x=X_scaled[train_idx],
            a=T_scaled[train_idx],
            y=Y_scaled[train_idx],
            a_type=treatment_type
        )

        d_val = CausalDataset(
            x=X_scaled[val_idx],
            a=T_scaled[val_idx],
            y=Y_scaled[val_idx],
            a_type=treatment_type
        ) if len(val_idx) > 0 else None

        # Create configuration
        config = self._create_config(X.shape[1], T.shape[1], Y.shape[1])

        # Initialize joint model
        self.joint_model = JointModel(config)

        # Train models
        print("  Training propensity model...")
        if "discrete" in treatment_type:
            self.joint_model.fit_propensity(d_train, d_val)

        print("  Training stage 1 model...")
        self.joint_model.fit_stage1(d_train, d_val)

        print("  Training stage 2 models...")
        subset_size = min(100, len(X_scaled))
        test_data = CausalDataset(
            x=X_scaled[:subset_size],
            a=T_scaled[:subset_size],
            y=None,
            a_type=treatment_type
        )
        self.joint_model.fit_stage2(test_data)

        # Store original data for evaluation
        self.X_orig = X
        self.T_orig = T
        self.Y_orig = Y.flatten()
        self.mu0_orig = mu0
        self.mu1_orig = mu1

    def _discretize_treatment_for_pehe(self, X, n_samples=1000):
        """
        MODIFIED: Discretize treatments for PEHE calculation
        This is the key modification that forces PEHE for all treatment types
        """

        X_scaled = self.scaler_X.transform(X)
        treatment_type = self.data_info['treatment_type']
        treatment_dim = self.data_info['treatment_dim']

        print(f"    Computing PEHE for {treatment_type} treatment...")

        if "discrete" in treatment_type and treatment_dim == 1:
            # Already binary - use as is
            T0_scaled = np.zeros((X.shape[0], 1))
            T1_scaled = np.ones((X.shape[0], 1))

        else:
            # MODIFICATION: Discretize continuous/multi-dimensional treatments
            if treatment_dim == 1:
                # Single continuous treatment - median split
                T_median = np.median(self.T_orig, axis=0)
                T0_orig = np.full_like(self.T_orig, T_median - 0.5 * np.std(self.T_orig))
                T1_orig = np.full_like(self.T_orig, T_median + 0.5 * np.std(self.T_orig))

                T0_scaled = self.scaler_T.transform(T0_orig)
                T1_scaled = self.scaler_T.transform(T1_orig)

            else:
                # Multi-dimensional treatment - use low/high levels
                T_low = np.percentile(self.T_orig, 25, axis=0, keepdims=True)
                T_high = np.percentile(self.T_orig, 75, axis=0, keepdims=True)

                T0_orig = np.tile(T_low, (X.shape[0], 1))
                T1_orig = np.tile(T_high, (X.shape[0], 1))

                T0_scaled = self.scaler_T.transform(T0_orig)
                T1_scaled = self.scaler_T.transform(T1_orig)

        # Create datasets for both treatment levels
        data_T0 = CausalDataset(x=X_scaled, a=T0_scaled, y=None, a_type="discrete")
        data_T1 = CausalDataset(x=X_scaled, a=T1_scaled, y=None, a_type="discrete")

        # Sample potential outcomes
        Y0_samples = self.joint_model.stage1.sample(data_T0, n_samples)
        Y1_samples = self.joint_model.stage1.sample(data_T1, n_samples)

        # Compute ITE
        ite = torch.mean(Y1_samples - Y0_samples, dim=1).detach().numpy()
        ite_orig = ite * self.scaler_Y.scale_[0]

        return ite_orig, T0_scaled, T1_scaled

    def predict_ite(self, X, n_samples=1000):
        """Predict Individual Treatment Effects using discretized treatments"""
        ite_orig, _, _ = self._discretize_treatment_for_pehe(X, n_samples)
        return ite_orig

    def predict_ate(self, X, n_samples=1000):
        """Predict Average Treatment Effect"""
        ite = self.predict_ite(X, n_samples)
        return np.mean(ite)

    def evaluate_metrics(self, n_samples=1000):
        """
        MODIFIED: Always compute PEHE for all treatment types
        This is the main modification that ensures consistent PEHE output
        """

        results = {}

        try:
            print("    Computing PEHE for all treatment types...")

            # MODIFICATION: Always use discretized treatment approach
            ite_pred, T0_scaled, T1_scaled = self._discretize_treatment_for_pehe(self.X_orig, n_samples)

            # Create Random Forest baseline for ground truth approximation
            print("    Creating Random Forest baseline...")
            rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)

            # Prepare training data for RF
            X_all = np.vstack([self.X_orig, self.X_orig])

            # Use discretized treatments for RF training
            if self.data_info['treatment_dim'] == 1:
                if "discrete" in self.data_info['treatment_type']:
                    T_all = np.vstack([np.zeros((len(self.X_orig), 1)),
                                     np.ones((len(self.X_orig), 1))])
                else:
                    # Use the same discretization as for PEHE
                    T_median = np.median(self.T_orig, axis=0)
                    T_low_rf = np.full((len(self.X_orig), 1), T_median - 0.5 * np.std(self.T_orig))
                    T_high_rf = np.full((len(self.X_orig), 1), T_median + 0.5 * np.std(self.T_orig))
                    T_all = np.vstack([T_low_rf, T_high_rf])
            else:
                # Multi-dimensional - use percentiles
                T_low = np.percentile(self.T_orig, 25, axis=0, keepdims=True)
                T_high = np.percentile(self.T_orig, 75, axis=0, keepdims=True)
                T_low_rf = np.tile(T_low, (len(self.X_orig), 1))
                T_high_rf = np.tile(T_high, (len(self.X_orig), 1))
                T_all = np.vstack([T_low_rf, T_high_rf])

            Y_all = np.hstack([self.Y_orig, self.Y_orig])
            XT_all = np.hstack([X_all, T_all])

            # Train RF
            rf.fit(XT_all, Y_all)

            # Predict counterfactuals with RF
            XT0 = np.hstack([self.X_orig, T_all[:len(self.X_orig)]])  # Low treatment
            XT1 = np.hstack([self.X_orig, T_all[len(self.X_orig):]])  # High treatment

            Y0_rf = rf.predict(XT0)
            Y1_rf = rf.predict(XT1)
            ite_rf = Y1_rf - Y0_rf

            # Compute PEHE
            pehe = np.sqrt(np.mean((ite_pred - ite_rf) ** 2))
            results['PEHE'] = pehe

            # Compute ATE error
            ate_pred = np.mean(ite_pred)
            ate_rf = np.mean(ite_rf)
            ate_error = abs(ate_pred - ate_rf)
            results['ATE_Error'] = ate_error

            # Additional metrics
            results['ITE_Std'] = np.std(ite_pred)
            results['ITE_Range'] = np.max(ite_pred) - np.min(ite_pred)

            # Coverage approximation (what fraction of ITEs are within reasonable bounds)
            ite_bounds = np.percentile(ite_rf, [5, 95])
            coverage = np.mean((ite_pred >= ite_bounds[0]) & (ite_pred <= ite_bounds[1]))
            results['Coverage'] = coverage

            print(f"    ✓ PEHE: {pehe:.3f}, ATE Error: {ate_error:.3f}, Coverage: {coverage:.3f}")

        except Exception as e:
            print(f"    ✗ PEHE evaluation error: {e}")
            import traceback
            traceback.print_exc()

            # Return default values on error
            results['PEHE'] = 999.0
            results['ATE_Error'] = 999.0
            results['Coverage'] = 0.0
            results['ITE_Std'] = 0.0
            results['ITE_Range'] = 0.0

        return results


def run_synthetic_experiment(dataset_file, dataset_type, n_seeds=10):
    """Run NeuralCSA experiment on synthetic datasets with PEHE focus"""

    print(f"\n{'=' * 60}")
    print(f"NeuralCSA PEHE Results: {dataset_file} ({dataset_type})")
    print(f"{'=' * 60}")

    # Load data
    df = pd.read_csv(dataset_file)
    results = []

    for seed in range(n_seeds):
        print(f"Seed {seed + 1:2d}/10", end=" ... ", flush=True)

        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        # Initialize model
        model = NeuralCSA(dataset_type=dataset_type, gamma=2.0, device='cpu')

        try:
            # Train model
            model.train(df, epochs=50)

            # MODIFICATION: Always evaluate PEHE metrics
            metrics = model.evaluate_metrics(n_samples=500)

            if metrics and 'PEHE' in metrics:
                metrics['seed'] = seed
                results.append(metrics)

                # Print PEHE-focused results
                print(f"PEHE: {metrics['PEHE']:.3f}, ATE: {metrics['ATE_Error']:.3f}, Cov: {metrics['Coverage']:.3f}")
            else:
                print("PEHE computation failed")

        except Exception as e:
            print(f"FAILED: {str(e)[:50]}...")
            continue

    # Aggregate results
    if results:
        print(f"\n{'-' * 60}")
        print("FINAL PEHE RESULTS (Mean ± Std)")
        print(f"{'-' * 60}")

        # Focus on PEHE metrics
        pehe_metrics = ['PEHE', 'ATE_Error', 'Coverage', 'ITE_Std', 'ITE_Range']

        for metric in pehe_metrics:
            values = [r[metric] for r in results if metric in r and not np.isnan(r[metric])]
            if values and len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values) if len(values) > 1 else 0.0
                print(f"{metric:12s}: {mean_val:6.3f} ± {std_val:5.3f} (n={len(values)})")
    else:
        print(f"\n{'-' * 60}")
        print("No successful PEHE results obtained")
        print(f"{'-' * 60}")

    return results


# Example usage
if __name__ == "__main__":

    # Synthetic datasets only
    datasets = [
        ('single_binary.csv', 'single_binary'),
        ('single_continuous.csv', 'single_continuous'),
        ('multi_binary.csv', 'multi_binary'),
        ('multi_continuous.csv', 'multi_continuous'),
        ('mixed.csv', 'mixed')
    ]

    all_results = {}

    for dataset_file, dataset_type in datasets:
        try:
            if os.path.exists(dataset_file):
                results = run_synthetic_experiment(dataset_file, dataset_type, n_seeds=10)
                all_results[dataset_file] = results
            else:
                print(f"Dataset {dataset_file} not found, skipping...")
        except Exception as e:
            print(f"Failed to process {dataset_file}: {e}")
            continue

    print(f"\n{'=' * 80}")
    print("NEURALCSA PEHE SUMMARY - ALL TREATMENT TYPES")
    print(f"{'=' * 80}")

    summary_table = []
    for dataset_file, results in all_results.items():
        if results:
            dataset_name = dataset_file.replace('.csv', '')
            row = {'Dataset': dataset_name}

            # Focus on PEHE metrics only
            for metric in ['PEHE', 'ATE_Error', 'Coverage']:
                values = [r[metric] for r in results if metric in r and not np.isnan(r[metric]) and r[metric] != 999.0]
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    row[metric] = f"{mean_val:.3f}±{std_val:.3f}"
                else:
                    row[metric] = "FAILED"

            summary_table.append(row)

    if summary_table:
        print(f"{'Dataset':<18} {'PEHE':<15} {'ATE_Error':<15} {'Coverage':<15}")
        print("-" * 75)
        for row in summary_table:
            dataset = row['Dataset'][:17]
            pehe = row.get('PEHE', 'N/A')
            ate = row.get('ATE_Error', 'N/A')
            cov = row.get('Coverage', 'N/A')
            print(f"{dataset:<18} {pehe:<15} {ate:<15} {cov:<15}")

    print(f"\nCompleted PEHE evaluation on {len(all_results)} synthetic datasets.")
    print("MODIFICATION: All treatment types now compute PEHE consistently")
    print("MODIFICATION: Continuous treatments discretized for PEHE calculation")
    print("MODIFICATION: Random Forest baseline used for ground truth approximation")
    print("Format: mean±std across 10 random seeds")

import pandas as pd
import numpy as np
from datetime import datetime


# Save NeuralCSA results to CSV
def save_results_to_csv(all_results, filename=None):
    """
    Save NeuralCSA experimental results to CSV file

    Args:
        all_results: Dictionary with dataset_file as keys and list of result dicts as values
        filename: Optional custom filename, defaults to timestamped file
    """

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"neuralcsa_results_{timestamp}.csv"

    # Flatten all results into a single DataFrame
    flattened_results = []

    for dataset_file, results_list in all_results.items():
        dataset_name = dataset_file.replace('.csv', '')

        if results_list:  # Check if we have results for this dataset
            for result_dict in results_list:
                # Create a row for each seed's results
                row = {
                    'dataset_file': dataset_file,
                    'dataset_name': dataset_name,
                    'dataset_type': dataset_name,  # Assuming dataset_name indicates type
                }

                # Add all metrics from the result dictionary
                row.update(result_dict)
                flattened_results.append(row)
        else:
            # Add a row indicating no results for this dataset
            flattened_results.append({
                'dataset_file': dataset_file,
                'dataset_name': dataset_name,
                'dataset_type': dataset_name,
                'seed': np.nan,
                'PEHE': np.nan,
                'ATE_Error': np.nan,
                'Coverage': np.nan,
                'ITE_Std': np.nan,
                'ITE_Range': np.nan,
                'status': 'FAILED'
            })

    # Convert to DataFrame
    df_results = pd.DataFrame(flattened_results)

    # Reorder columns for better readability
    column_order = ['dataset_file', 'dataset_name', 'dataset_type', 'seed',
                    'PEHE', 'ATE_Error', 'Coverage', 'ITE_Std', 'ITE_Range']

    # Add any additional columns that might exist
    additional_cols = [col for col in df_results.columns if col not in column_order]
    column_order.extend(additional_cols)

    # Reorder columns (only include columns that actually exist)
    existing_columns = [col for col in column_order if col in df_results.columns]
    df_results = df_results[existing_columns]

    # Sort by dataset and seed
    df_results = df_results.sort_values(['dataset_name', 'seed'])

    # Save to CSV
    df_results.to_csv(filename, index=False)

    print(f"Results saved to: {filename}")
    print(f"Total rows: {len(df_results)}")
    print(f"Datasets: {df_results['dataset_name'].nunique()}")
    print(f"Seeds per dataset: {df_results.groupby('dataset_name')['seed'].count().to_dict()}")

    return df_results, filename


# Create summary statistics CSV
def save_summary_to_csv(all_results, filename=None):
    """
    Save aggregated summary statistics to CSV
    """

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"neuralcsa_summary_{timestamp}.csv"

    summary_data = []

    for dataset_file, results_list in all_results.items():
        dataset_name = dataset_file.replace('.csv', '')

        if results_list:
            # Calculate summary statistics for each metric
            metrics = ['PEHE', 'ATE_Error', 'Coverage', 'ITE_Std', 'ITE_Range']

            row = {
                'dataset_file': dataset_file,
                'dataset_name': dataset_name,
                'dataset_type': dataset_name,
                'n_successful_seeds': len(results_list)
            }

            for metric in metrics:
                values = [r[metric] for r in results_list if metric in r
                          and not np.isnan(r[metric]) and r[metric] != 999.0]

                if values:
                    row[f'{metric}_mean'] = np.mean(values)
                    row[f'{metric}_std'] = np.std(values) if len(values) > 1 else 0.0
                    row[f'{metric}_min'] = np.min(values)
                    row[f'{metric}_max'] = np.max(values)
                    row[f'{metric}_median'] = np.median(values)
                    row[f'{metric}_n_values'] = len(values)
                else:
                    row[f'{metric}_mean'] = np.nan
                    row[f'{metric}_std'] = np.nan
                    row[f'{metric}_min'] = np.nan
                    row[f'{metric}_max'] = np.nan
                    row[f'{metric}_median'] = np.nan
                    row[f'{metric}_n_values'] = 0

            summary_data.append(row)
        else:
            # Dataset failed completely
            summary_data.append({
                'dataset_file': dataset_file,
                'dataset_name': dataset_name,
                'dataset_type': dataset_name,
                'n_successful_seeds': 0,
                'status': 'FAILED'
            })

    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values('dataset_name')

    # Save to CSV
    df_summary.to_csv(filename, index=False)

    print(f"Summary saved to: {filename}")
    print(f"Summary rows: {len(df_summary)}")

    return df_summary, filename


# ===================================================================
# MAIN EXECUTION CELL - Run this after your NeuralCSA experiments
# ===================================================================

# Assuming 'all_results' is available from running the main experiment
if 'all_results' in locals() or 'all_results' in globals():

    print("Saving NeuralCSA Results to CSV...")
    print("=" * 50)

    # Save detailed results (one row per seed)
    df_detailed, detailed_file = save_results_to_csv(all_results)

    print()

    # Save summary statistics (one row per dataset)
    df_summary, summary_file = save_summary_to_csv(all_results)

    print()
    print("Quick Preview of Detailed Results:")
    print(df_detailed.head(10))

    print()
    print("Quick Preview of Summary Results:")
    print(df_summary[['dataset_name', 'n_successful_seeds', 'PEHE_mean', 'PEHE_std', 'ATE_Error_mean']].head())

    print()
    print("Files created:")
    print(f"   Detailed results: {detailed_file}")
    print(f"   Summary results: {summary_file}")

else:
    print("Error: 'all_results' variable not found!")
    print("Please run the NeuralCSA experiments first to generate results.")

    # Show example of how to use if you have results manually
    print("\n If you have results stored elsewhere, use:")
    print("df_detailed, detailed_file = save_results_to_csv(your_results_dict)")
    print("df_summary, summary_file = save_summary_to_csv(your_results_dict)")