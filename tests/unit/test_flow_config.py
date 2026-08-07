"""
Unit tests for FlowConfig (flow-based track).

Covers the three deliberate departures from the TFA's config: no global
singleton, lowercase field names, and a validated `split_strategy`.
"""

from pathlib import Path

import pytest

import ml.flow.config as flow_config_module
from ml.flow.config import (
    SPLIT_STRATEGIES,
    FlowConfig,
    IsolationForestConfig,
    NeuralNetworkConfig,
    XGBoostConfig,
)

# ============================================================================
# Defaults
# ============================================================================


class TestFlowConfigDefaults:
    """Default values and derived paths."""

    def test_ratios_sum_to_one(self):
        config = FlowConfig()
        assert config.train_ratio + config.val_ratio + config.test_ratio == pytest.approx(1.0)

    def test_default_split_strategy_is_temporal(self):
        """The random split is the leaky one; temporal must be the default."""
        assert FlowConfig().split_strategy == "temporal"

    def test_default_paths_are_path_objects(self):
        config = FlowConfig()
        assert isinstance(config.data_path, Path)
        assert isinstance(config.output_path, Path)

    def test_string_paths_are_coerced(self):
        config = FlowConfig(data_path="some/data", output_path="some/out")
        assert config.data_path == Path("some/data")
        assert config.output_path == Path("some/out")

    def test_sub_configs_are_instantiated(self):
        config = FlowConfig()
        assert isinstance(config.isolation_forest, IsolationForestConfig)
        assert isinstance(config.xgboost, XGBoostConfig)
        assert isinstance(config.neural_network, NeuralNetworkConfig)

    def test_sub_configs_are_not_shared_between_instances(self):
        """A shared mutable default would make runs order-dependent."""
        a = FlowConfig()
        b = FlowConfig()
        assert a.isolation_forest is not b.isolation_forest
        assert a.xgboost.search_space is not b.xgboost.search_space
        assert a.neural_network.hidden_layers is not b.neural_network.hidden_layers

    def test_beta_favours_recall(self):
        assert FlowConfig().beta > 1.0


# ============================================================================
# No global singleton
# ============================================================================


class TestNoGlobalSingleton:
    """The TFA's module-level `cfg` was dropped on purpose."""

    def test_module_exposes_no_cfg(self):
        assert not hasattr(flow_config_module, "cfg")

    def test_instances_are_independent(self):
        a = FlowConfig()
        b = FlowConfig(sample_size=123)
        assert a.sample_size != 123
        assert b.sample_size == 123


# ============================================================================
# Field naming
# ============================================================================


class TestFieldNaming:
    """Fields were lowercased when porting; the old names must be gone."""

    @pytest.mark.parametrize(
        "old_name",
        [
            "DATA_PATH",
            "OUTPUT_PATH",
            "RANDOM_SEED",
            "SAMPLE_SIZE",
            "TRAIN_RATIO",
            "VAL_RATIO",
            "TEST_RATIO",
            "MIN_CLASS_SAMPLES",
            "MIN_CLASS_RATIO",
            "BETA",
        ],
    )
    def test_uppercase_field_names_are_gone(self, old_name):
        assert not hasattr(FlowConfig(), old_name)

    @pytest.mark.parametrize(
        "new_name",
        [
            "data_path",
            "output_path",
            "random_seed",
            "sample_size",
            "train_ratio",
            "val_ratio",
            "test_ratio",
            "min_class_samples",
            "min_class_ratio",
            "beta",
            "split_strategy",
        ],
    )
    def test_lowercase_field_names_exist(self, new_name):
        assert hasattr(FlowConfig(), new_name)


# ============================================================================
# Validation
# ============================================================================


class TestValidation:
    """__post_init__ guards."""

    def test_rejects_ratios_that_do_not_sum_to_one(self):
        with pytest.raises(ValueError, match="must be 1.0"):
            FlowConfig(train_ratio=0.8, val_ratio=0.15, test_ratio=0.15)

    def test_accepts_custom_ratios_that_do_sum_to_one(self):
        config = FlowConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        assert config.train_ratio == 0.6

    def test_tolerates_floating_point_noise(self):
        config = FlowConfig(train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)
        assert config.val_ratio == 0.1

    @pytest.mark.parametrize("strategy", SPLIT_STRATEGIES)
    def test_accepts_every_declared_split_strategy(self, strategy):
        assert FlowConfig(split_strategy=strategy).split_strategy == strategy

    @pytest.mark.parametrize("strategy", ["stratified", "TEMPORAL", "day_holdout", ""])
    def test_rejects_unknown_split_strategy(self, strategy):
        with pytest.raises(ValueError, match="Unknown split_strategy"):
            FlowConfig(split_strategy=strategy)

    def test_split_strategies_are_the_three_documented_protocols(self):
        assert set(SPLIT_STRATEGIES) == {"temporal", "random", "day-holdout"}


# ============================================================================
# Output layout
# ============================================================================


class TestOutputPaths:
    """Derived paths and directory creation."""

    def test_derived_paths_live_under_output_path(self, tmp_path):
        config = FlowConfig(output_path=tmp_path)
        for path in (
            config.eda_path,
            config.cascade_xgb_path,
            config.cascade_mlp_path,
            config.comparison_path,
            config.results_file,
        ):
            assert path.parent == tmp_path

    def test_derived_paths_are_distinct(self, tmp_path):
        config = FlowConfig(output_path=tmp_path)
        paths = [
            config.eda_path,
            config.cascade_xgb_path,
            config.cascade_mlp_path,
            config.comparison_path,
        ]
        assert len(set(paths)) == len(paths)

    def test_create_output_dirs_creates_the_tree(self, tmp_path):
        config = FlowConfig(output_path=tmp_path / "run")
        config.create_output_dirs()
        assert config.eda_path.is_dir()
        assert config.cascade_xgb_path.is_dir()
        assert config.cascade_mlp_path.is_dir()
        assert config.comparison_path.is_dir()

    def test_create_output_dirs_is_idempotent(self, tmp_path):
        config = FlowConfig(output_path=tmp_path / "run")
        config.create_output_dirs()
        config.create_output_dirs()
        assert config.eda_path.is_dir()


# ============================================================================
# Sub-configurations
# ============================================================================


class TestSubConfigs:
    """Hyperparameter defaults carried over from the TFA."""

    def test_isolation_forest_contamination_matches_dataset(self):
        assert IsolationForestConfig().contamination == pytest.approx(0.15)

    def test_xgboost_search_space_covers_the_tuned_hyperparameters(self):
        space = XGBoostConfig().search_space
        assert set(space) == {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
        }

    def test_xgboost_defaults_are_inside_their_search_ranges(self):
        config = XGBoostConfig()
        assert config.n_estimators in config.search_space["n_estimators"]
        assert config.max_depth in config.search_space["max_depth"]
        assert config.learning_rate in config.search_space["learning_rate"]

    def test_neural_network_layer_and_dropout_lists_align(self):
        config = NeuralNetworkConfig()
        assert len(config.hidden_layers) == len(config.dropout_rates)

    def test_neural_network_overrides_are_honoured(self):
        config = NeuralNetworkConfig(epochs=3, batch_size=32)
        assert (config.epochs, config.batch_size) == (3, 32)
