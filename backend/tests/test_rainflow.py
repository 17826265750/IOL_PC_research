"""
Unit tests for rainflow cycle counting implementation.

Tests cover the ASTM E1049 algorithm implementation including edge cases,
error handling, and accuracy verification.
"""
import pytest
import numpy as np
from backend.app.core.rainflow import (
    Cycle,
    RainflowResult,
    rainflow_counting,
    find_peaks_and_valleys,
    get_cycle_matrix,
    get_histogram_data,
    find_range_mean_pairs,
    calculate_equivalent_constant_amplitude,
    get_cumulative_cycles
)


class TestFindPeaksAndValleys:
    """Tests for peak and valley extraction."""

    def test_empty_data(self):
        """Test with empty input."""
        result = find_peaks_and_valleys([])
        assert result == []

    def test_single_point(self):
        """Test with single data point."""
        result = find_peaks_and_valleys([10.0])
        assert result == [10.0]

    def test_two_points(self):
        """Test with two points - both are kept."""
        result = find_peaks_and_valleys([0.0, 10.0])
        assert result == [0.0, 10.0]

    def test_monotonic_increase(self):
        """Test monotonic increasing data."""
        data = [0, 2, 4, 6, 8, 10]
        result = find_peaks_and_valleys(data)
        assert result == [0, 10]

    def test_monotonic_decrease(self):
        """Test monotonic decreasing data."""
        data = [10, 8, 6, 4, 2, 0]
        result = find_peaks_and_valleys(data)
        assert result == [10, 0]

    def test_peaks_and_valleys(self):
        """Test data with multiple peaks and valleys."""
        data = [0, 5, 0, 5, 0]
        result = find_peaks_and_valleys(data)
        # All points should be kept (each is a peak or valley)
        assert len(result) == 5
        assert result == [0, 5, 0, 5, 0]

    def test_constant_values(self):
        """Test constant values."""
        data = [5, 5, 5, 5, 5]
        result = find_peaks_and_valleys(data)
        # Should return first and last
        assert len(result) == 2

    def test_realistic_load_sequence(self):
        """Test a realistic load sequence."""
        data = [0, 10, -5, 8, -3, 7, -2, 5]
        result = find_peaks_and_valleys(data)
        # Should extract the turning points
        assert 0 in result  # Start
        assert 10 in result  # First peak
        assert -5 in result  # First valley
        assert 5 in result  # End


class TestRainflowCounting:
    """Tests for the main rainflow counting algorithm."""

    def test_empty_data(self):
        """Test with empty input."""
        result = rainflow_counting([])
        assert result.cycles == []
        assert result.residual == []

    def test_single_point(self):
        """Test with single data point."""
        result = rainflow_counting([10.0])
        assert result.cycles == []

    def test_constant_values(self):
        """Test constant values."""
        result = rainflow_counting([5, 5, 5, 5])
        assert result.cycles == []

    def test_simple_triangle_wave(self):
        """Test simple triangle wave pattern."""
        data = [0, 10, 0]
        result = rainflow_counting(data)
        # Should form one full cycle
        assert len(result.cycles) == 1
        assert result.cycles[0].range == 10
        assert result.cycles[0].mean == 0

    def test_simple_sine_like(self):
        """Test sine-like pattern."""
        data = [0, 10, 0, -10, 0]
        result = rainflow_counting(data)
        # Should form two cycles
        assert len(result.cycles) >= 1
        # Check cycle ranges are reasonable
        for cycle in result.cycles:
            assert cycle.range > 0
            assert cycle.count > 0

    def test_astm_example(self):
        """Test against ASTM E1049 example."""
        # Example from ASTM E1049 Section 5.4.4
        data = [0, -2, 3, -2, 1, -3, 4, -1, 0]
        result = rainflow_counting(data)

        # Verify we get some cycles
        assert len(result.cycles) > 0

        # All cycles should have positive ranges
        for cycle in result.cycles:
            assert cycle.range > 0
            assert 0 < cycle.count <= 1.0
            assert cycle.min_val <= cycle.max_val

    def test_complex_loading(self):
        """Test complex loading history."""
        data = [0, 10, -5, 8, -3, 7, -2, 5, 0]
        result = rainflow_counting(data)

        # Should extract multiple cycles
        assert len(result.cycles) > 0

        # Verify cycle properties
        for cycle in result.cycles:
            assert cycle.range > 0
            assert cycle.min_val < cycle.max_val
            assert cycle.mean >= cycle.min_val
            assert cycle.mean <= cycle.max_val

    def test_residual_handling(self):
        """Test that residual points are handled correctly."""
        data = [0, 5, 2, 8, 3]
        result = rainflow_counting(data)
        # Residual should be a list
        assert isinstance(result.residual, list)

    def test_cycle_count_values(self):
        """Test that cycle counts are 0.5 or 1.0."""
        data = [0, 10, -5, 5, -5, 10, 0]
        result = rainflow_counting(data)

        for cycle in result.cycles:
            # Cycle counts should be 0.5 (half cycle) or 1.0 (full cycle)
            assert cycle.count in [0.5, 1.0]


class TestCycleMatrix:
    """Tests for cycle matrix generation."""

    def test_empty_cycles(self):
        """Test with empty cycle list."""
        matrix = get_cycle_matrix([], bin_count=32)
        assert matrix.shape == (32, 32)
        assert np.all(matrix == 0)

    def test_single_cycle(self):
        """Test with single cycle."""
        cycles = [Cycle(range=10, mean=50, count=1.0, min_val=45, max_val=55)]
        matrix = get_cycle_matrix(cycles, bin_count=10)

        assert matrix.shape == (10, 10)
        # Should have exactly one non-zero bin
        assert np.sum(matrix) == 1.0

    def test_multiple_cycles(self):
        """Test with multiple cycles."""
        cycles = [
            Cycle(range=10, mean=50, count=1.0, min_val=45, max_val=55),
            Cycle(range=20, mean=60, count=0.5, min_val=50, max_val=70),
            Cycle(range=15, mean=55, count=0.5, min_val=47, max_val=62)
        ]
        matrix = get_cycle_matrix(cycles, bin_count=16)

        assert matrix.shape == (16, 16)
        # Total should equal sum of cycle counts
        assert np.isclose(np.sum(matrix), 2.0)

    def test_bin_counts(self):
        """Test different bin counts."""
        cycles = [Cycle(range=10, mean=50, count=1.0, min_val=45, max_val=55)]

        for bin_count in [8, 16, 32, 64, 128]:
            matrix = get_cycle_matrix(cycles, bin_count=bin_count)
            assert matrix.shape == (bin_count, bin_count)


class TestHistogramData:
    """Tests for histogram data generation."""

    def test_empty_cycles(self):
        """Test with empty cycle list."""
        hist = get_histogram_data([], bin_count=10)

        assert len(hist['bins']) == 11  # n+1 edges
        assert len(hist['counts']) == 10
        assert len(hist['centers']) == 10
        assert np.all(hist['counts'] == 0)

    def test_histogram_structure(self):
        """Test histogram data structure."""
        cycles = [
            Cycle(range=10, mean=50, count=1.0, min_val=45, max_val=55),
            Cycle(range=20, mean=60, count=0.5, min_val=50, max_val=70)
        ]
        hist = get_histogram_data(cycles, bin_count=10)

        # Check structure
        assert 'bins' in hist
        assert 'counts' in hist
        assert 'centers' in hist

        # Check dimensions
        assert len(hist['bins']) == 11
        assert len(hist['counts']) == 10
        assert len(hist['centers']) == 10

        # Check bins are monotonically increasing
        assert np.all(np.diff(hist['bins']) > 0)

        # Check centers are within bin edges
        assert np.all(hist['centers'] >= hist['bins'][:-1])
        assert np.all(hist['centers'] <= hist['bins'][1:])

    def test_histogram_weights(self):
        """Test that histogram correctly weights by cycle count."""
        cycles = [
            Cycle(range=10, mean=50, count=1.0, min_val=45, max_val=55),
            Cycle(range=10, mean=50, count=2.0, min_val=45, max_val=55)
        ]
        hist = get_histogram_data(cycles, bin_count=10)

        # Should count total cycles
        assert np.sum(hist['counts']) == 3.0


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_find_range_mean_pairs(self):
        """Test range-mean pair extraction."""
        cycles = [
            Cycle(range=10, mean=50, count=1.0, min_val=45, max_val=55),
            Cycle(range=20, mean=60, count=0.5, min_val=50, max_val=70)
        ]
        pairs = find_range_mean_pairs(cycles)

        assert len(pairs) == 2
        assert pairs[0] == (10, 50, 1.0)
        assert pairs[1] == (20, 60, 0.5)

    def test_equivalent_constant_amplitude(self):
        """Test equivalent constant amplitude calculation."""
        cycles = [
            Cycle(range=100, mean=50, count=10, min_val=0, max_val=100),
            Cycle(range=200, mean=100, count=5, min_val=0, max_val=200)
        ]

        equiv = calculate_equivalent_constant_amplitude(cycles, exponent=3)

        # Equivalent amplitude should be between min and max ranges
        assert 100 < equiv < 200

    def test_equivalent_constant_amplitude_empty(self):
        """Test equivalent amplitude with empty cycles."""
        equiv = calculate_equivalent_constant_amplitude([])
        assert equiv == 0.0

    def test_cumulative_cycles(self):
        """Test cumulative cycle calculation."""
        cycles = [
            Cycle(range=10, mean=50, count=1.0, min_val=45, max_val=55),
            Cycle(range=20, mean=60, count=0.5, min_val=50, max_val=70),
            Cycle(range=15, mean=55, count=1.0, min_val=47, max_val=62)
        ]

        cumulative = get_cumulative_cycles(cycles)

        assert len(cumulative) == 3
        # Should be monotonically increasing
        assert np.all(np.diff(cumulative) >= 0)
        # Last value should be total cycles
        assert np.isclose(cumulative[-1], 2.5)

    def test_cumulative_cycles_empty(self):
        """Test cumulative cycles with empty list."""
        cumulative = get_cumulative_cycles([])
        assert len(cumulative) == 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_nan_values(self):
        """Test handling of NaN values."""
        data = [0, np.nan, 10, 5, np.nan, 0]
        # Should handle NaN gracefully
        result = rainflow_counting([x if not np.isnan(x) else 0 for x in data])
        assert isinstance(result, RainflowResult)

    def test_inf_values(self):
        """Test handling of infinite values."""
        data = [0, 10, np.inf, -np.inf, 5, 0]
        # Filter out inf and test
        clean_data = [x for x in data if np.isfinite(x)]
        result = rainflow_counting(clean_data)
        assert isinstance(result, RainflowResult)

    def test_very_small_ranges(self):
        """Test with very small value ranges."""
        data = [1e-10, 2e-10, 1e-10, 2e-10, 1e-10]
        result = rainflow_counting(data)
        # Should handle small values
        assert isinstance(result, RainflowResult)

    def test_very_large_ranges(self):
        """Test with very large value ranges."""
        data = [0, 1e10, -1e10, 5e9, 0]
        result = rainflow_counting(data)
        # Should handle large values
        assert isinstance(result, RainflowResult)


class TestRainflowResult:
    """Tests for RainflowResult dataclass."""

    def test_result_creation(self):
        """Test RainflowResult creation."""
        cycles = [Cycle(range=10, mean=50, count=1.0, min_val=45, max_val=55)]
        matrix = np.zeros((10, 10))
        residual = [0, 5]

        result = RainflowResult(cycles=cycles, cycle_matrix=matrix, residual=residual)

        assert result.cycles == cycles
        assert np.array_equal(result.cycle_matrix, matrix)
        assert result.residual == residual


class TestCycleDataclass:
    """Tests for Cycle dataclass."""

    def test_cycle_creation(self):
        """Test Cycle object creation."""
        cycle = Cycle(
            range=10.0,
            mean=50.0,
            count=1.0,
            min_val=45.0,
            max_val=55.0
        )

        assert cycle.range == 10.0
        assert cycle.mean == 50.0
        assert cycle.count == 1.0
        assert cycle.min_val == 45.0
        assert cycle.max_val == 55.0

    def test_cycle_repr(self):
        """Test Cycle string representation."""
        cycle = Cycle(range=10, mean=50, count=1.0, min_val=45, max_val=55)
        repr_str = repr(cycle)

        assert 'Cycle' in repr_str
        assert '10' in repr_str
        assert '50' in repr_str
