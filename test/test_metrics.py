#!/usr/bin/env python
"""
unit test for metrics module
author: Tommaso Faraci

This file is part of evo (github.com/MichaelGrupp/evo).

evo is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

evo is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with evo.  If not, see <http://www.gnu.org/licenses/>.
"""

import unittest

import numpy as np
from scipy import stats

import helpers
from evo.core import lie_algebra as lie
from evo.core import metrics
from evo.core.metrics import NEES, MetricsException, PoseRelation


class TestNEES(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.traj_est = helpers.fake_trajectory_with_covariances(
            2000, 0.1, self.rng
        )

    def process(
        self, traj_ref, pose_relation=PoseRelation.full_transformation
    ):
        nees = NEES(pose_relation)
        nees.process_data((traj_ref, self.traj_est))
        return nees

    def test_consistent_covariances(self):
        traj_ref = helpers.perturbed_reference(self.traj_est, self.rng)
        statistics = self.process(traj_ref).get_all_statistics()
        self.assertEqual(statistics["samples"], 2000)
        self.assertEqual(statistics["rejected"], 0)
        self.assertEqual(statistics["anees"], statistics["mean"])
        self.assertLess(statistics["anees_95_low"], statistics["anees"])
        self.assertLess(statistics["anees"], statistics["anees_95_high"])
        for level in (50, 90, 95, 99):
            self.assertLess(
                statistics[f"coverage_{level}_95_low"], level / 100
            )
            self.assertLess(
                level / 100, statistics[f"coverage_{level}_95_high"]
            )
        self.assertLess(statistics["max_abs_whitened_mean"], 0.1)
        self.assertLess(statistics["max_abs_whitened_cov_error"], 0.15)

    def test_inconsistent_covariances(self):
        # Same errors, but too small (overconfident) or too large
        # (underconfident) covariances reported by the estimate.
        traj_ref = helpers.perturbed_reference(self.traj_est, self.rng)
        covariances = self.traj_est.covariances.copy()
        statistics = {}
        for name, scale in (
            ("overconfident", 0.25),
            ("underconfident", 4.0),
            ("consistent", 1.0),
        ):
            self.traj_est.covariances = scale * covariances
            statistics[name] = self.process(traj_ref).get_all_statistics()
        over = statistics["overconfident"]
        under = statistics["underconfident"]
        self.assertGreater(over["anees"], over["anees_95_high"])
        self.assertLess(under["anees"], under["anees_95_low"])
        self.assertLess(over["coverage_95"], 0.95)
        self.assertGreater(under["coverage_95"], 0.95)
        nll = statistics["consistent"]["mean_nll"]
        self.assertLess(nll, over["mean_nll"])
        self.assertLess(nll, under["mean_nll"])

    def test_matches_direct_computation(self):
        traj_ref = helpers.perturbed_reference(self.traj_est, self.rng)
        for pose_relation, part in metrics.NEES_PARTS.items():
            nees = self.process(traj_ref, pose_relation)
            self.assertEqual(nees.dof, part.stop - part.start)
            for i in (0, 1, 1999):
                xi = lie.se3_log(
                    lie.relative_se3(
                        self.traj_est.poses_se3[i], traj_ref.poses_se3[i]
                    )
                )[part]
                covariance = self.traj_est.covariances[i][part, part]
                self.assertAlmostEqual(
                    nees.error[i], xi.dot(np.linalg.solve(covariance, xi))
                )
                self.assertAlmostEqual(
                    nees.nll[i],
                    -stats.multivariate_normal.logpdf(xi, cov=covariance),
                )

    def test_skips_invalid_covariances(self):
        traj_ref = helpers.perturbed_reference(self.traj_est, self.rng)
        expected = self.process(traj_ref).error
        self.traj_est.covariances[3] = -np.eye(6)
        self.traj_est.covariances[5, 0, 0] = np.nan
        nees = self.process(traj_ref)
        self.assertEqual(nees.num_rejected, 2)
        self.assertEqual(len(nees.valid_ids), 1998)
        self.assertNotIn(3, nees.valid_ids)
        self.assertNotIn(5, nees.valid_ids)
        self.assertTrue(np.allclose(nees.error, expected[nees.valid_ids]))
        self.assertEqual(nees.get_all_statistics()["rejected"], 2)

    def test_marginal_ignores_invalid_other_part(self):
        # A broken translation block must not affect the rotation NEES.
        traj_ref = helpers.perturbed_reference(self.traj_est, self.rng)
        self.traj_est.covariances[3, 3:, 3:] = -np.eye(3)
        nees = self.process(traj_ref, PoseRelation.rotation_part)
        self.assertEqual(nees.num_rejected, 0)

    def test_result(self):
        traj_ref = helpers.perturbed_reference(self.traj_est, self.rng)
        result = self.process(traj_ref).get_result()
        self.assertIn("anees", result.stats)
        self.assertIn("rmse", result.stats)
        self.assertEqual(len(result.np_arrays["error_array"]), 2000)
        self.assertEqual(len(result.np_arrays["nll_array"]), 2000)
        self.assertEqual(len(result.np_arrays["valid_ids"]), 2000)

    def test_missing_covariances(self):
        traj_ref = helpers.perturbed_reference(self.traj_est, self.rng)
        self.traj_est.covariances = None
        with self.assertRaises(MetricsException):
            self.process(traj_ref)

    def test_no_valid_covariances(self):
        traj_ref = helpers.perturbed_reference(self.traj_est, self.rng)
        self.traj_est.covariances = np.zeros((2000, 6, 6))
        with self.assertRaises(MetricsException):
            self.process(traj_ref)

    def test_unsupported_pose_relation(self):
        with self.assertRaises(MetricsException):
            NEES(PoseRelation.rotation_angle_deg)

    def test_statistics_before_processing(self):
        with self.assertRaises(MetricsException):
            NEES().get_consistency_statistics()


class TestClopperPearson(unittest.TestCase):
    def test_matches_scipy_binomtest(self):
        for successes, trials in ((0, 10), (3, 10), (10, 10), (950, 1000)):
            interval = stats.binomtest(successes, trials).proportion_ci()
            self.assertTrue(
                np.allclose(
                    metrics.clopper_pearson_interval(successes, trials),
                    (interval.low, interval.high),
                )
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
