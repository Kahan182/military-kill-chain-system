# test_pipeline.py - simple tests for every stage
import os
import sys
import unittest
import numpy as np

# so that "core" and "data" can be imported from inside the tests folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.stage1_rough_sets import RoughSetTheoryEngine
from core.stage2_neural_net import ThreatIntelligenceEngine, TrainableTargetANN
from core.stage3_moga import MOGAWeaponAllocator
from core.stage4_pso import PSOImpactOptimizer
from core.stage5_salvo import salvo_size
from data.scenario_data import (STRUCTURAL_BOUNDS_CATALOG, WEAPONS_STOCKPILE,
                                load_targets_dataframe)
from main import run_pipeline


class TestPipeline(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.df = load_targets_dataframe()

    def make_rough_set_engine(self):
        return RoughSetTheoryEngine(self.df, "target_id", [], "is_valid_military_target")

    def test_rough_sets_boundary(self):
        # T_03 and T_05 look identical but have different labels
        result = self.make_rough_set_engine().compute_approximations(1)
        self.assertEqual(result["boundary_region"], {"T_03", "T_05"})
        for t in ["T_01", "T_02", "T_04"]:
            self.assertIn(t, result["lower_approximation"])

    def test_reducts_not_empty(self):
        core, reducts = self.make_rough_set_engine().compute_core_and_reducts()
        self.assertTrue(len(reducts) > 0)

    def test_threat_tiers(self):
        tier = ThreatIntelligenceEngine.calculate_air_defense_tier
        self.assertEqual(tier(9.4, 0.1, 0)["sam_tier"], 3)
        self.assertEqual(tier(3.1, 12, 60)["sam_tier"], 2)
        self.assertEqual(tier(0, 18, 45)["sam_tier"], 1)
        self.assertEqual(tier(0, 0.05, 0)["sam_tier"], 0)

    def test_ann_output(self):
        result = TrainableTargetANN().evaluate(self.df.iloc[0])
        self.assertTrue(0.0 <= result["vulnerability_index"] <= 1.0)
        self.assertIn(result["predicted_class"], TrainableTargetANN.CLASS_NAMES)

    def test_lead_point_static_target(self):
        # a target that is not moving should not change position
        point = TrainableTargetANN.calculate_lead_point(10, 20, 0, 900)
        self.assertEqual(point, (10, 20))

    def test_moga_gives_one_weapon_per_target(self):
        result = MOGAWeaponAllocator(self.df, WEAPONS_STOCKPILE).solve()
        self.assertEqual(len(result), len(self.df))

    def test_pso_gets_close_to_weak_point(self):
        bounds = STRUCTURAL_BOUNDS_CATALOG["T_01"]
        result = PSOImpactOptimizer(bounds, seed=1).optimize()
        self.assertLess(result["miss_distance"], 1.0)
        history = result["convergence_history"]
        self.assertLessEqual(history[-1], history[0])   # score got better

    def test_salvo_reaches_desired_chance(self):
        target = self.df.iloc[0]
        weapon = WEAPONS_STOCKPILE["W_BUNKER_BUSTER"]
        result = salvo_size(target, weapon, desired_pk=0.90)
        self.assertGreaterEqual(result["achieved_pk"], 0.90)
        self.assertGreaterEqual(result["rounds_needed"], 1)

    def test_full_pipeline_holds_uncertain_targets(self):
        out = run_pipeline(seed=1)
        actions = {}
        for entry in out["advisories"]:
            actions[entry.target_id] = entry.action
        self.assertIn("HUMAN VERIFICATION", actions["T_03"])
        self.assertIn("HUMAN VERIFICATION", actions["T_05"])


if __name__ == "__main__":
    unittest.main()