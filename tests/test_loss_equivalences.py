"""
Unit tests to verify mathematical equivalences between loss implementations.

Tests the following claims:
1. VICRegLoss std computation (without centering) equals HingeStdLoss (with centering)
2. VICRegLoss cov computation equals CovarianceLoss
3. VICRegLoss decomposes into HingeStdLoss + CovarianceLoss + MSE invariance
4. Centering before torch.var() is redundant
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from eb_jepa.losses import (
    CovarianceLoss,
    HingeStdLoss,
    VCLoss,
    VICRegLoss,
)


class TestStdLossEquivalence:
    """Test equivalences for standard deviation / variance loss."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        torch.manual_seed(42)
        return torch.randn(64, 128)  # batch_size=64, features=128

    def test_vicreg_std_vs_hinge_std(self, sample_data):
        """
        Test if VICRegLoss std computation (without explicit centering)
        equals HingeStdLoss (with centering).

        This tests the claim that centering before var() is redundant.
        """
        x = sample_data

        # VICRegLoss style: no explicit centering, uses 1e-4 epsilon
        z_std_vicreg = torch.sqrt(x.var(dim=0) + 1e-4)
        std_loss_vicreg = torch.mean(F.relu(1 - z_std_vicreg))

        # HingeStdLoss style: explicit centering, uses 0.0001 epsilon
        x_centered = x - x.mean(dim=0, keepdim=True)
        z_std_hinge = torch.sqrt(x_centered.var(dim=0) + 0.0001)
        std_loss_hinge = torch.mean(F.relu(1 - z_std_hinge))

        # Should be equal because:
        # 1. torch.var() computes variance around the mean anyway
        # 2. 1e-4 == 0.0001
        assert torch.allclose(std_loss_vicreg, std_loss_hinge, atol=1e-7), (
            f"VICReg std: {std_loss_vicreg.item():.8f} vs "
            f"HingeStd: {std_loss_hinge.item():.8f}"
        )

    def test_centering_before_var_is_redundant(self, sample_data):
        """
        Directly test that centering before torch.var() is redundant.
        """
        x = sample_data

        # Without centering
        var_no_center = x.var(dim=0)

        # With centering
        x_centered = x - x.mean(dim=0, keepdim=True)
        var_with_center = x_centered.var(dim=0)

        # Should be exactly equal
        assert torch.allclose(var_no_center, var_with_center, atol=1e-7), (
            f"Var without centering and var with centering should be equal. "
            f"Max diff: {(var_no_center - var_with_center).abs().max().item():.8e}"
        )


class TestCovLossEquivalence:
    """Test equivalences for covariance loss."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        torch.manual_seed(42)
        return torch.randn(64, 128)  # batch_size=64, features=128

    def test_vicreg_cov_vs_covariance_loss(self, sample_data):
        """
        Verify that VICRegLoss cov computation equals CovarianceLoss.
        """
        x = sample_data
        batch_size = x.size(0)

        # VICRegLoss style computation
        z_centered = x - x.mean(dim=0)
        z_cov = torch.mm(z_centered.T, z_centered) / (batch_size - 1)
        cov_loss_vicreg = (z_cov.pow(2).sum() - z_cov.diagonal().pow(2).sum()) / (
            z_cov.size(0) ** 2 - z_cov.size(0)
        )

        # CovarianceLoss style
        cov_loss_fn = CovarianceLoss()
        cov_loss_class = cov_loss_fn(x)

        assert torch.allclose(cov_loss_vicreg, cov_loss_class, atol=1e-6), (
            f"VICReg cov: {cov_loss_vicreg.item():.8f} vs "
            f"CovarianceLoss: {cov_loss_class.item():.8f}"
        )


class TestFullLossEquivalence:
    """Test full loss function equivalences."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        torch.manual_seed(42)
        return torch.randn(64, 128)

    def test_vicreg_decomposition(self, sample_data):
        """
        Test that VICRegLoss can be decomposed into HingeStdLoss + CovarianceLoss
        when applied to both views and summed.
        """
        torch.manual_seed(42)
        z1 = torch.randn(64, 128)
        z2 = torch.randn(64, 128)

        std_coeff = 25.0
        cov_coeff = 1.0

        # Using VICRegLoss
        vicreg = VICRegLoss(std_coeff=std_coeff, cov_coeff=cov_coeff)
        result = vicreg(z1, z2)

        # Manual decomposition using primitives
        std_loss_fn = HingeStdLoss(std_margin=1.0)
        cov_loss_fn = CovarianceLoss()

        sim_loss = F.mse_loss(z1, z2)
        std_loss = std_loss_fn(z1) + std_loss_fn(z2)
        cov_loss = cov_loss_fn(z1) + cov_loss_fn(z2)

        expected_total = sim_loss + std_coeff * std_loss + cov_coeff * cov_loss

        # Compare
        assert torch.allclose(
            result["invariance_loss"], sim_loss, atol=1e-6
        ), f"Sim loss: {result['invariance_loss'].item():.8f} vs {sim_loss.item():.8f}"
        assert torch.allclose(
            result["var_loss"], std_loss, atol=1e-6
        ), f"Var loss: {result['var_loss'].item():.8f} vs {std_loss.item():.8f}"
        assert torch.allclose(
            result["cov_loss"], cov_loss, atol=1e-6
        ), f"Cov loss: {result['cov_loss'].item():.8f} vs {cov_loss.item():.8f}"
        assert torch.allclose(
            result["loss"], expected_total, atol=1e-6
        ), f"Total loss: {result['loss'].item():.8f} vs {expected_total.item():.8f}"


class TestVCLoss:
    """Test VCLoss functionality."""

    def test_vc_loss_output_structure(self):
        """Test that VCLoss produces correct output structure."""
        torch.manual_seed(42)
        x_5d = torch.randn(8, 16, 2, 4, 4)  # B=8, F=16, T=2, H=4, W=4

        std_coeff = 10.0
        cov_coeff = 5.0

        vc_loss = VCLoss(std_coeff=std_coeff, cov_coeff=cov_coeff)
        loss, unweighted, loss_dict = vc_loss(x_5d)

        # Verify outputs are valid
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isnan(unweighted), "Unweighted loss should not be NaN"
        assert loss.item() >= 0, "Loss should be non-negative"
        assert "std_loss" in loss_dict
        assert "cov_loss" in loss_dict

    def test_vc_loss_with_projector(self):
        """Test that VCLoss works correctly with a projector."""
        torch.manual_seed(42)
        x_5d = torch.randn(8, 16, 2, 4, 4)  # B=8, F=16, T=2, H=4, W=4

        # Create a simple projector
        projector = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        std_coeff = 10.0
        cov_coeff = 5.0

        # Using VCLoss with projector
        vc_loss = VCLoss(std_coeff=std_coeff, cov_coeff=cov_coeff, proj=projector)
        loss, unweighted, loss_dict = vc_loss(x_5d)

        # Verify outputs are valid
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isnan(unweighted), "Unweighted loss should not be NaN"
        assert loss.item() >= 0, "Loss should be non-negative"
        assert "std_loss" in loss_dict
        assert "cov_loss" in loss_dict

    def test_vc_loss_coefficient_weighting(self):
        """Test that VCLoss correctly applies coefficient weighting."""
        torch.manual_seed(42)
        x_5d = torch.randn(8, 16, 2, 4, 4)

        std_coeff = 10.0
        cov_coeff = 5.0

        vc_loss = VCLoss(std_coeff=std_coeff, cov_coeff=cov_coeff)
        loss, unweighted, loss_dict = vc_loss(x_5d)

        # Verify that weighted loss equals coefficient-weighted components
        expected_loss = (
            std_coeff * loss_dict["std_loss"] + cov_coeff * loss_dict["cov_loss"]
        )
        assert torch.allclose(loss, torch.tensor(expected_loss), atol=1e-5), (
            f"Weighted loss {loss.item():.6f} should equal "
            f"{std_coeff}*{loss_dict['std_loss']:.6f} + {cov_coeff}*{loss_dict['cov_loss']:.6f} = {expected_loss:.6f}"
        )

    def test_vc_loss_consistency_across_seeds(self):
        """Test that VCLoss is deterministic given same seed."""
        std_coeff = 10.0
        cov_coeff = 5.0

        for seed in [1, 42, 100, 1000]:
            # First run
            torch.manual_seed(seed)
            x_5d = torch.randn(8, 16, 2, 4, 4)
            vc_loss = VCLoss(std_coeff=std_coeff, cov_coeff=cov_coeff)
            loss1, _, _ = vc_loss(x_5d)

            # Second run with same seed
            torch.manual_seed(seed)
            x_5d = torch.randn(8, 16, 2, 4, 4)
            vc_loss = VCLoss(std_coeff=std_coeff, cov_coeff=cov_coeff)
            loss2, _, _ = vc_loss(x_5d)

            assert torch.allclose(
                loss1, loss2, atol=1e-7
            ), f"Seed {seed}: loss should be deterministic, got {loss1.item():.6f} vs {loss2.item():.6f}"


class TestVICRegLossRegression:
    """
    Regression tests to ensure VICRegLoss using HingeStdLoss + CovarianceLoss
    produces identical results to the original inline implementation.
    """

    def test_refactored_vicreg_vs_original_implementation(self):
        """
        Regression test: Refactored VICRegLoss using HingeStdLoss + CovarianceLoss
        produces identical results to the original inline implementation.
        """
        torch.manual_seed(42)
        z1 = torch.randn(64, 128)
        z2 = torch.randn(64, 128)

        std_coeff = 25.0
        cov_coeff = 1.0

        # Using refactored VICRegLoss
        vicreg = VICRegLoss(std_coeff=std_coeff, cov_coeff=cov_coeff)
        result = vicreg(z1, z2)

        # Original VICRegLoss implementation (inlined for comparison)
        batch_size = z1.size(0)

        # Original invariance loss
        sim_loss_orig = F.mse_loss(z1, z2)

        # Original variance loss
        z1_std = torch.sqrt(z1.var(dim=0) + 1e-4)
        z2_std = torch.sqrt(z2.var(dim=0) + 1e-4)
        var_loss_orig = torch.mean(F.relu(1 - z1_std)) + torch.mean(F.relu(1 - z2_std))

        # Original covariance loss
        z1_centered = z1 - z1.mean(dim=0)
        z2_centered = z2 - z2.mean(dim=0)
        z1_cov = torch.mm(z1_centered.T, z1_centered) / (batch_size - 1)
        z2_cov = torch.mm(z2_centered.T, z2_centered) / (batch_size - 1)
        cov_loss_orig = (z1_cov.pow(2).sum() - z1_cov.diagonal().pow(2).sum()) / (
            z1_cov.size(0) ** 2 - z1_cov.size(0)
        ) + (z2_cov.pow(2).sum() - z2_cov.diagonal().pow(2).sum()) / (
            z2_cov.size(0) ** 2 - z2_cov.size(0)
        )

        total_loss_orig = (
            sim_loss_orig + std_coeff * var_loss_orig + cov_coeff * cov_loss_orig
        )

        # Verify all components match
        assert torch.allclose(result["invariance_loss"], sim_loss_orig, atol=1e-6), (
            f"Refactored sim: {result['invariance_loss'].item():.8f} vs "
            f"Original sim: {sim_loss_orig.item():.8f}"
        )
        assert torch.allclose(result["var_loss"], var_loss_orig, atol=1e-6), (
            f"Refactored var: {result['var_loss'].item():.8f} vs "
            f"Original var: {var_loss_orig.item():.8f}"
        )
        assert torch.allclose(result["cov_loss"], cov_loss_orig, atol=1e-6), (
            f"Refactored cov: {result['cov_loss'].item():.8f} vs "
            f"Original cov: {cov_loss_orig.item():.8f}"
        )
        assert torch.allclose(result["loss"], total_loss_orig, atol=1e-6), (
            f"Refactored total: {result['loss'].item():.8f} vs "
            f"Original total: {total_loss_orig.item():.8f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
