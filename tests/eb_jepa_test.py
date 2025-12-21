import pytest
import torch

from eb_jepa.losses import ContrastiveLoss, contrastive_cost, vc_cost_chunked


@pytest.mark.skip(reason="Not implemented yet")
def test_forwardn():
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_infern():
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_vc_cost():
    pass


def test_vc_cost_chunked():
    """Test the vc_cost_chunked function with various scenarios."""

    # Test parameters
    batch_size = 8
    feature_size = 4
    time_steps = 3
    height = 2
    width = 2

    # Create test input tensor in BFTHW format
    x = torch.randn(batch_size, feature_size, time_steps, height, width)

    # Create coefficient matrix for covariance regularization
    ccoeff = torch.ones(feature_size, feature_size) * 0.1

    # Calculate total flattened samples
    total_samples = batch_size * time_steps * height * width  # 8 * 3 * 2 * 2 = 96

    # Test 1: Basic functionality with default parameters
    result = vc_cost_chunked(x, ccoeff)

    # Check return type is tuple
    assert isinstance(result, tuple), "Function should return a tuple"
    assert len(result) == 2, "Function should return a tuple of length 2"

    mean_cost, individual_costs = result

    # Check that mean_cost is a scalar tensor
    assert isinstance(mean_cost, torch.Tensor), "Mean cost should be a tensor"
    assert mean_cost.dim() == 0, "Mean cost should be a scalar (0-dim tensor)"

    # Check that individual_costs is a 1D tensor
    assert isinstance(
        individual_costs, torch.Tensor
    ), "Individual costs should be a tensor"
    assert individual_costs.dim() == 1, "Individual costs should be 1D tensor"

    # When bdim=0 (default), it uses feature_size=4, so chunks = 96//4 = 24
    expected_chunks = total_samples // feature_size
    assert (
        individual_costs.size(0) == expected_chunks
    ), f"Expected {expected_chunks} chunks, got {individual_costs.size(0)}"

    # Check that mean equals the mean of individual costs
    assert torch.allclose(
        mean_cost, individual_costs.mean()
    ), "Mean cost should equal mean of individual costs"

    # Test 2: Explicit chunk size
    bdim = 2
    result2 = vc_cost_chunked(x, ccoeff, batch_dim=bdim)
    mean_cost2, individual_costs2 = result2

    expected_chunks2 = total_samples // bdim  # 96 // 2 = 48
    assert (
        individual_costs2.size(0) == expected_chunks2
    ), f"Expected {expected_chunks2} chunks with bdim={bdim}"

    # Test 3: Different coefficients
    mcoeff = 0.5
    c = 2.0
    result3 = vc_cost_chunked(x, ccoeff, mcoeff=mcoeff, c=c, batch_dim=bdim)
    mean_cost3, individual_costs3 = result3

    # Should have same number of chunks but different costs
    assert individual_costs3.size(0) == expected_chunks2
    # Costs should be different due to different parameters
    assert not torch.allclose(
        mean_cost2, mean_cost3
    ), "Different parameters should produce different costs"

    # Test 4: Edge case - bdim that divides total_samples evenly
    bdim4 = 8
    result4 = vc_cost_chunked(x, ccoeff, batch_dim=bdim4)
    mean_cost4, individual_costs4 = result4

    expected_chunks4 = total_samples // bdim4  # 96 // 8 = 12
    assert (
        individual_costs4.size(0) == expected_chunks4
    ), f"Should have {expected_chunks4} chunks when bdim={bdim4}"

    # Test 5: Check that all costs are non-negative (reasonable for a loss function)
    assert torch.all(
        individual_costs >= 0
    ), "All individual costs should be non-negative"
    assert mean_cost >= 0, "Mean cost should be non-negative"

    # Test 6: Edge case - large bdim (fewer chunks)
    bdim_large = 32
    result5 = vc_cost_chunked(x, ccoeff, batch_dim=bdim_large)
    mean_cost5, individual_costs5 = result5

    expected_chunks5 = total_samples // bdim_large  # 96 // 32 = 3
    assert (
        individual_costs5.size(0) == expected_chunks5
    ), f"Should have {expected_chunks5} chunks when bdim={bdim_large}"


def test_contrastive_loss():
    """Test the ContrastiveLoss class and contrastive_cost function with various scenarios."""

    # Test parameters
    batch_size = 4
    feature_size = 8
    time_steps = 2
    height = 3
    width = 3

    # Create test input tensor in BFTHW format
    x = torch.randn(batch_size, feature_size, time_steps, height, width)

    # Test 1: Basic functionality with ContrastiveLoss class
    contrastive_loss = ContrastiveLoss(temperature=0.1, negative_weight=1.0)
    result = contrastive_loss(x)

    # Check return type is scalar tensor
    assert isinstance(result, torch.Tensor), "ContrastiveLoss should return a tensor"
    assert result.dim() == 0, "ContrastiveLoss should return a scalar (0-dim tensor)"
    assert result >= 0, "Contrastive loss should be non-negative"

    # Test 2: Different temperature values
    temp_low = ContrastiveLoss(temperature=0.01, negative_weight=1.0)
    temp_high = ContrastiveLoss(temperature=1.0, negative_weight=1.0)

    result_low_temp = temp_low(x)
    result_high_temp = temp_high(x)

    # Both should be non-negative
    assert (
        result_low_temp >= 0 and result_high_temp >= 0
    ), "Both losses should be non-negative"

    # Test 3: Different negative weights
    weight_small = ContrastiveLoss(temperature=0.1, negative_weight=0.5)
    weight_large = ContrastiveLoss(temperature=0.1, negative_weight=2.0)

    result_small_weight = weight_small(x)
    result_large_weight = weight_large(x)

    # Larger weight should give proportionally larger loss
    assert torch.allclose(
        result_large_weight, result_small_weight * 4.0
    ), "Loss should scale proportionally with negative_weight"

    # Test 4: Test contrastive_cost function directly
    cost_result = contrastive_cost(x, temperature=0.1, negative_weight=1.0)

    assert isinstance(
        cost_result, torch.Tensor
    ), "contrastive_cost should return a tensor"
    assert cost_result.dim() == 0, "contrastive_cost should return a scalar"
    assert cost_result >= 0, "Contrastive cost should be non-negative"

    # Test 4b: Test subset sampling
    total_samples = batch_size * time_steps * height * width  # 4 * 2 * 3 * 3 = 72
    subset_size = 16
    num_subsets = 3

    cost_subset = contrastive_cost(
        x,
        temperature=0.1,
        negative_weight=1.0,
        subset_size=subset_size,
        num_subsets=num_subsets,
    )

    assert isinstance(
        cost_subset, torch.Tensor
    ), "Subset contrastive_cost should return a tensor"
    assert cost_subset.dim() == 0, "Subset contrastive_cost should return a scalar"
    assert cost_subset >= 0, "Subset contrastive cost should be non-negative"

    # Test 5: With projection layer
    proj_layer = torch.nn.Linear(feature_size, feature_size // 2)
    contrastive_with_proj = ContrastiveLoss(
        temperature=0.1, negative_weight=1.0, proj=proj_layer
    )

    result_with_proj = contrastive_with_proj(x)
    assert isinstance(
        result_with_proj, torch.Tensor
    ), "Loss with projection should return a tensor"
    assert result_with_proj.dim() == 0, "Loss with projection should return a scalar"
    assert result_with_proj >= 0, "Loss with projection should be non-negative"

    # Test 5b: ContrastiveLoss with subset sampling
    contrastive_subset = ContrastiveLoss(
        temperature=0.1,
        negative_weight=1.0,
        subset_size=subset_size,
        num_subsets=num_subsets,
    )
    result_subset = contrastive_subset(x)

    assert isinstance(
        result_subset, torch.Tensor
    ), "Subset ContrastiveLoss should return a tensor"
    assert result_subset.dim() == 0, "Subset ContrastiveLoss should return a scalar"
    assert result_subset >= 0, "Subset contrastive loss should be non-negative"

    # Test 6: Edge case - identical samples should give high loss
    x_identical = torch.ones(batch_size, feature_size, time_steps, height, width)
    identical_loss = contrastive_loss(x_identical)

    # Identical samples should result in high similarity and thus high contrastive loss
    assert (
        identical_loss > 0
    ), "Identical samples should produce positive contrastive loss"

    # Test 7: Edge case - very different samples
    x_diverse = torch.randn(batch_size, feature_size, time_steps, height, width) * 10
    # Make samples very different by scaling different samples differently
    for i in range(batch_size):
        x_diverse[i] = x_diverse[i] * (i + 1)

    diverse_loss = contrastive_loss(x_diverse)
    assert diverse_loss >= 0, "Diverse samples should still produce non-negative loss"

    # Test 8: Gradient flow (ensure loss is differentiable)
    x_grad = torch.randn(
        batch_size, feature_size, time_steps, height, width, requires_grad=True
    )
    loss_grad = contrastive_loss(x_grad)
    loss_grad.backward()

    assert x_grad.grad is not None, "Gradients should flow through the contrastive loss"
    assert not torch.isnan(x_grad.grad).any(), "Gradients should not contain NaN values"

    # Test 9: Consistency between class and function
    class_result = contrastive_loss(x)
    function_result = contrastive_cost(x, temperature=0.1, negative_weight=1.0)

    assert torch.allclose(
        class_result, function_result
    ), "ContrastiveLoss class and contrastive_cost function should give same result"

    # Test 10: Subset edge cases
    # Test with subset_size larger than total samples
    large_subset = ContrastiveLoss(
        temperature=0.1, negative_weight=1.0, subset_size=1000, num_subsets=2
    )
    result_large = large_subset(x)
    assert result_large >= 0, "Large subset size should still work"

    # Test with single subset
    single_subset = ContrastiveLoss(
        temperature=0.1, negative_weight=1.0, subset_size=8, num_subsets=1
    )
    result_single = single_subset(x)
    assert result_single >= 0, "Single subset should work"

    # Test efficiency: subset sampling should be faster for large inputs
    # (This is more of a conceptual test - in practice we'd need larger tensors to see the difference)
    efficient_contrastive = ContrastiveLoss(
        temperature=0.1,
        negative_weight=1.0,
        subset_size=min(16, total_samples // 2),
        num_subsets=3,
    )
    result_efficient = efficient_contrastive(x)
    assert result_efficient >= 0, "Efficient subset sampling should work"
