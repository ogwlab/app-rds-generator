"""
Tests for Streamlit app helper functions
"""

import pytest

from rds_generator.app import (
    MAX_BATCH_IMAGES,
    get_batch_disparities,
    validate_batch_config,
)
from rds_generator.config import RDSConfig


class TestBatchValidation:
    """Test batch generation safety limits"""

    def test_get_batch_disparities_accepts_bounded_range(self):
        """Valid batch ranges should produce inclusive disparity values"""
        disparities = get_batch_disparities(-20, 20, 20)

        assert disparities.tolist() == [-20, 0, 20]

    def test_get_batch_disparities_rejects_too_many_images(self):
        """Batch generation should reject excessive condition counts"""
        end = MAX_BATCH_IMAGES + 1

        with pytest.raises(ValueError, match="最大"):
            get_batch_disparities(0, end, 1)

    def test_get_batch_disparities_rejects_out_of_range_disparity(self):
        """Batch disparity values should stay within RDSConfig limits"""
        with pytest.raises(ValueError, match="-600"):
            get_batch_disparities(-700, 0, 10)

    def test_validate_batch_config_rejects_large_images(self):
        """Batch generation should reject expensive image sizes"""
        config = RDSConfig(width=1024, height=1024)

        with pytest.raises(ValueError, match="512"):
            validate_batch_config(config)
