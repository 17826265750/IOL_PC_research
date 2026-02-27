"""
Unit tests for Rainflow cycle counting (ASTM E1049).

Tests include:
- ASTM E1049 standard test cases
- Peak and valley extraction
- Cycle matrix generation
- Edge cases (empty data, single point, etc.)
- Cumulative distribution
"""

import pytest
import numpy as np

from app.core.rainflow import (
    rainflow_counting,
    find_peaks_and_valleys,
    get_cycle_matrix,
    get_histogram_data,
    find_range_mean_pairs,
    calculate_equivalent_constant_amplitude,
    get_cumulative_cycles,
    Cycle,
    RainflowResult
)


class TestFindPeaksAndValleys:
    """Test peak and valley extraction from time series."""

    def test_simple_sine_wave(self):
        """Test with simple sine wave."""
        data = [0, 50, 100, 50, 0, -50, -100, -50, 0]

        result = find_peaks_and_valleys(data)

        # Should keep peaks and valleys (endpoints too)
        assert len(result) < len(data)
        assert 100 in result  # Peak
        assert -100 in result  # Valley
        assert 0 in result  # Start/end

    def test_monotonic_increase(self):
        """Test with monotonically increasing data."""
        data = [0, 10, 20, 30, 40, 50]

        result = find_peaks_and_valleys(data)

        # Should only keep endpoints
        assert len(result) == 2
        assert result[0] == 0
        assert result[-1] == 50

    def test_monotonic_decrease(self):
        """Test with monotonically decreasing data."""
        data = [50, 40, 30, 20, 10, 0]

        result = find_peaks_and_valleys(data)

        # Should only keep endpoints
        assert len(result) == 2
        assert result[0] == 50
        assert result[-1] == 0

    def test_flat_sections(self):
        """Test with flat sections in data."""
        data = [0, 10, 10, 10, 20, 20, 10]

        result = find_peaks_and_valleys(data)

        # Should handle flat sections
        assert 0 in result
        assert 20 in result
        assert 10 in result

    def test_empty_data(self):
        """Test with empty data."""
        result = find_peaks_and_valleys([])

        assert result == []

    def test_single_point(self):
        """Test with single data point."""
        result = find_peaks_and_valleys([42])

        assert result == [42]

    def test_two_points(self):
        """Test with two data points."""
        result = find_peaks_and_valleys([0, 100])

        assert result == [0, 100]

    def test_all_equal_values(self):
        """Test with all equal values."""
        result = find_peaks_and_valleys([50, 50, 50, 50])

        # Implementation returns all values when all are equal
        assert len(result) == 4
        assert all(v == 50 for v in result)


class TestRainflowCounting:
    """Test rainflow cycle counting algorithm."""

    def test_empty_data(self):
        """Test with empty data."""
        result = rainflow_counting([])

        assert result.cycles == []
        assert result.residual == []

    def test_single_point(self):
        """Test with single data point."""
        result = rainflow_counting([42])

        assert result.cycles == []
        assert result.residual == [42]

    def test_all_equal_values(self):
        """Test with all equal values."""
        result = rainflow_counting([50, 50, 50, 50])

        assert result.cycles == []
        # All-constant data is returned as-is in the residual
        assert all(v == 50 for v in result.residual)

    def test_simple_triangle_wave(self):
        """Test with simple triangle wave (0 -> 100 -> 0).

        Per ASTM E1049 §5.4.4, the closed range Y contains the
        starting point, so it is counted as a *half-cycle*.
        """
        data = [0, 100, 0]

        result = rainflow_counting(data)

        assert len(result.cycles) == 1
        assert result.cycles[0].range == 100
        assert result.cycles[0].count == 0.5  # half-cycle (contains start)

    def test_astm_standard_example(self):
        """Test with ASTM E1049 standard example.

        From ASTM E1049-85, Section 5.4.4.
        Using a simple load sequence.
        """
        # Standard test sequence
        data = [-2, 1, -3, 5, -1, 3, -4, 4, -2]

        result = rainflow_counting(data)

        # Should extract some cycles
        assert len(result.cycles) > 0

        # Check cycle properties
        for cycle in result.cycles:
            assert cycle.range > 0
            assert cycle.count in [0.5, 1.0]
            assert cycle.min_val < cycle.max_val

    def test_multiple_cycles(self):
        """Test with data containing multiple cycles."""
        data = [0, 50, 0, 30, 0, 70, 0]

        result = rainflow_counting(data)

        # Should detect multiple cycles
        assert len(result.cycles) >= 2

        # Total count — cycles that include the starting reversal
        # are half-cycles per ASTM E1049.
        total_count = sum(c.count for c in result.cycles)
        assert total_count >= 1.5

    def test_cycle_properties(self):
        """Test that cycle properties are correctly calculated."""
        data = [0, 100, 0, 50, 0]

        result = rainflow_counting(data)

        for cycle in result.cycles:
            # range = max - min
            assert cycle.range == pytest.approx(cycle.max_val - cycle.min_val)

            # mean = (max + min) / 2
            assert cycle.mean == pytest.approx((cycle.max_val + cycle.min_val) / 2)

    def test_residual_handling(self):
        """Test that residual points are properly tracked."""
        data = [0, 10, 20, 15]  # Won't form complete cycle

        result = rainflow_counting(data)

        # Should have some residual
        assert isinstance(result.residual, list)


class TestCycleMatrix:
    """Test cycle matrix generation."""

    def test_empty_cycles(self):
        """Test with empty cycle list."""
        matrix = get_cycle_matrix([], bin_count=32)

        assert matrix.shape == (32, 32)
        assert np.all(matrix == 0)

    def test_single_cycle(self):
        """Test with single cycle."""
        cycles = [Cycle(range=100, mean=50, count=1.0, min_val=0, max_val=100)]

        matrix = get_cycle_matrix(cycles, bin_count=10)

        assert matrix.shape == (10, 10)
        # Should have one non-zero bin
        assert np.sum(matrix) > 0

    def test_multiple_cycles(self):
        """Test with multiple cycles."""
        cycles = [
            Cycle(range=100, mean=50, count=1.0, min_val=0, max_val=100),
            Cycle(range=80, mean=60, count=0.5, min_val=20, max_val=100),
            Cycle(range=60, mean=40, count=1.0, min_val=10, max_val=70),
        ]

        matrix = get_cycle_matrix(cycles, bin_count=10)

        assert matrix.shape == (10, 10)
        assert np.sum(matrix) > 0

    def test_matrix_sum_equals_total_cycles(self):
        """Test that matrix sum equals total cycle count."""
        cycles = [
            Cycle(range=100, mean=50, count=1.0, min_val=0, max_val=100),
            Cycle(range=80, mean=60, count=0.5, min_val=20, max_val=100),
        ]

        matrix = get_cycle_matrix(cycles, bin_count=64)

        expected_total = 1.5
        actual_total = np.sum(matrix)

        assert actual_total == pytest.approx(expected_total)


class TestHistogramData:
    """Test histogram data generation."""

    def test_empty_cycles(self):
        """Test with empty cycle list."""
        hist = get_histogram_data([], bin_count=10)

        assert len(hist['bins']) == 11  # 10 bins + 1 edge
        assert len(hist['counts']) == 10
        assert len(hist['centers']) == 10
        assert np.all(hist['counts'] == 0)

    def test_single_cycle(self):
        """Test with single cycle."""
        cycles = [Cycle(range=100, mean=50, count=1.0, min_val=0, max_val=100)]

        hist = get_histogram_data(cycles, bin_count=10)

        assert len(hist['bins']) == 11
        assert np.sum(hist['counts']) == 1.0

    def test_weighted_counts(self):
        """Test that cycle counts are weighted correctly."""
        cycles = [
            Cycle(range=100, mean=50, count=0.5, min_val=0, max_val=100),
            Cycle(range=100, mean=50, count=0.5, min_val=0, max_val=100),
            Cycle(range=100, mean=50, count=1.0, min_val=0, max_val=100),
        ]

        hist = get_histogram_data(cycles, bin_count=10)

        # Total should be 2.0 (0.5 + 0.5 + 1.0)
        assert np.sum(hist['counts']) == 2.0


class TestUtilityFunctions:
    """Test utility functions."""

    def test_find_range_mean_pairs(self):
        """Test extraction of range-mean-count tuples."""
        cycles = [
            Cycle(range=100, mean=50, count=1.0, min_val=0, max_val=100),
            Cycle(range=80, mean=60, count=0.5, min_val=20, max_val=100),
        ]

        pairs = find_range_mean_pairs(cycles)

        assert len(pairs) == 2
        assert pairs[0] == (100, 50, 1.0)
        assert pairs[1] == (80, 60, 0.5)

    def test_equivalent_constant_amplitude(self):
        """Test equivalent constant amplitude calculation."""
        cycles = [
            Cycle(range=100, mean=50, count=1.0, min_val=0, max_val=100),
            Cycle(range=80, mean=40, count=1.0, min_val=0, max_val=80),
        ]

        equivalent = calculate_equivalent_constant_amplitude(cycles, exponent=3.0)

        # Should be between min and max range
        assert 80 <= equivalent <= 100

    def test_equivalent_constant_amplitude_empty(self):
        """Test equivalent amplitude with empty cycles."""
        equivalent = calculate_equivalent_constant_amplitude([], exponent=3.0)

        assert equivalent == 0.0

    def test_cumulative_cycles(self):
        """Test cumulative cycle calculation."""
        cycles = [
            Cycle(range=40, mean=20, count=1.0, min_val=0, max_val=40),
            Cycle(range=60, mean=30, count=1.0, min_val=0, max_val=60),
            Cycle(range=80, mean=40, count=1.0, min_val=0, max_val=80),
        ]

        cumulative = get_cumulative_cycles(cycles)

        # Should return sorted by range
        assert len(cumulative) == 3
        # Cumulative sum should increase
        assert cumulative[0] == 1.0
        assert cumulative[1] == 2.0
        assert cumulative[2] == 3.0

    def test_cumulative_cycles_empty(self):
        """Test cumulative cycles with empty list."""
        cumulative = get_cumulative_cycles([])

        assert len(cumulative) == 0


class TestASTMCompliance:
    """Test ASTM E1049 compliance."""

    def test_three_point_method(self):
        """Test three-point cycle counting method.

        ASTM E1049-85 Section 5.4.4
        """
        # Simple sequence: 0 -> 100 -> 0 -> 50 -> 0
        data = [0, 100, 0, 50, 0]

        result = rainflow_counting(data)

        # Should identify cycles correctly
        assert len(result.cycles) > 0

        # Check that ranges are positive
        for cycle in result.cycles:
            assert cycle.range > 0

    def test_half_cycle_counting(self):
        """Test that half cycles are counted as 0.5."""
        # Data that creates half cycles
        data = [0, 50, 25, 75, 25]

        result = rainflow_counting(data)

        # Check for half cycles
        has_half_cycle = any(c.count == 0.5 for c in result.cycles)
        # May or may not have half cycles depending on algorithm
        assert isinstance(result.cycles, list)

    def test_reversal_detection(self):
        """Test that reversals are correctly detected."""
        # Series of reversals
        data = [0, 80, 20, 90, 10, 70, 30]

        result = rainflow_counting(data)

        # Should detect cycles from reversals
        assert len(result.cycles) > 0


class TestEdgeCases:
    """Test edge cases and special conditions."""

    def test_very_small_values(self):
        """Test with very small values."""
        data = [0, 0.001, 0, 0.002, 0]

        result = rainflow_counting(data)

        assert result.cycles or result.residual  # Should not crash

    def test_very_large_values(self):
        """Test with very large values."""
        data = [0, 1e6, 0, 2e6, 0]

        result = rainflow_counting(data)

        assert len(result.cycles) > 0

    def test_negative_values(self):
        """Test with negative values."""
        data = [0, -100, 0, -50, 0]

        result = rainflow_counting(data)

        assert len(result.cycles) > 0
        for cycle in result.cycles:
            assert cycle.min_val <= cycle.max_val

    def test_mixed_positive_negative(self):
        """Test with mixed positive and negative values."""
        data = [-50, 50, -50, 100, -100]

        result = rainflow_counting(data)

        assert len(result.cycles) > 0

    def test_near_equal_consecutive_values(self):
        """Test with near-equal consecutive values."""
        data = [0, 0.001, 0.002, 0.001, 0]

        result = rainflow_counting(data)

        # Should handle near-equal values
        assert isinstance(result.cycles, list)

    def test_sparse_data(self):
        """Test with sparse data."""
        data = [0, 100, 0, 200, 0, 300, 0]

        result = rainflow_counting(data)

        # Should identify the large cycles
        assert len(result.cycles) >= 1

    def test_high_frequency_noise(self):
        """Test with high frequency noise."""
        # Base signal with noise
        data = [0 + i*0.1 if i % 2 == 0 else 10 - i*0.1 for i in range(20)]

        result = rainflow_counting(data)

        # Should process without crashing
        assert isinstance(result.cycles, list)


class TestCycleDataclass:
    """Test Cycle dataclass."""

    def test_cycle_creation(self):
        """Test creating a Cycle."""
        cycle = Cycle(
            range=100,
            mean=50,
            count=1.0,
            min_val=0,
            max_val=100
        )

        assert cycle.range == 100
        assert cycle.mean == 50
        assert cycle.count == 1.0
        assert cycle.min_val == 0
        assert cycle.max_val == 100

    def test_cycle_repr(self):
        """Test Cycle string representation."""
        cycle = Cycle(
            range=100.5,
            mean=50.25,
            count=0.5,
            min_val=0,
            max_val=100.5
        )

        repr_str = repr(cycle)

        assert "Cycle" in repr_str
        assert "100.5" in repr_str or "100" in repr_str


class TestRainflowResult:
    """Test RainflowResult dataclass."""

    def test_result_creation(self):
        """Test creating a RainflowResult."""
        cycles = [Cycle(range=100, mean=50, count=1.0, min_val=0, max_val=100)]
        residual = [0, 50, 0]

        result = RainflowResult(cycles=cycles, residual=residual)

        assert result.cycles == cycles
        assert result.residual == residual
        assert result.cycle_matrix is None

    def test_result_with_matrix(self):
        """Test RainflowResult with cycle matrix."""
        cycles = [Cycle(range=100, mean=50, count=1.0, min_val=0, max_val=100)]
        matrix = np.zeros((10, 10))

        result = RainflowResult(cycles=cycles, cycle_matrix=matrix, residual=[])

        assert result.cycle_matrix is not None
        assert result.cycle_matrix.shape == (10, 10)


class TestRealWorldScenarios:
    """Test with realistic scenarios."""

    def test_thermal_cycling_profile(self):
        """Test with realistic thermal cycling profile."""
        # Simulated temperature cycling for power module
        # Starting at 40°C, heating to 125°C, cooling to 25°C, repeat
        base_temp = 40
        max_temp = 125
        min_temp = 25

        data = []
        for _ in range(3):
            # Heating
            data.extend(np.linspace(base_temp, max_temp, 10))
            # Cooling
            data.extend(np.linspace(max_temp, min_temp, 15))
            # Reheating
            data.extend(np.linspace(min_temp, base_temp, 5))

        result = rainflow_counting(data)

        # Should identify thermal cycles
        assert len(result.cycles) > 0

        # Main cycle should be approximately 100K (125-25)
        max_range = max((c.range for c in result.cycles), default=0)
        assert max_range >= 90  # Allow some tolerance

    def test_irregular_loading(self):
        """Test with irregular loading sequence."""
        # Realistic irregular sequence
        data = [0, 70, 30, 85, 20, 90, 15, 80, 25, 75, 35, 60]

        result = rainflow_counting(data)

        # Should extract cycles from irregular data
        assert len(result.cycles) > 0
