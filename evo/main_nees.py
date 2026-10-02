"""
Helper for calculating the normalized estimation error squared (NEES) of pose
covariances, as used by evo_nees.
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

import logging

from evo.core import lie_algebra, metrics
from evo.core.result import Result
from evo.core.trajectory import PosePath3D, PoseTrajectory3D

logger = logging.getLogger(__name__)

SEP = "-" * 80  # separator line


def nees(
    traj_ref: PosePath3D,
    traj_est: PosePath3D,
    pose_relation: metrics.PoseRelation = (
        metrics.PoseRelation.full_transformation
    ),
    *,
    align: bool = False,
    correct_scale: bool = False,
    n_to_align: int = -1,
    align_origin: bool = False,
) -> Result:
    """
    Calculates the NEES of the covariances of traj_est w.r.t. traj_ref.
    The trajectories have to be associated already (e.g. synchronized).
    Like in APE, traj_est is modified in place if it's aligned.
    The arrays of the result only contain the poses with a positive definite
    covariance, their indices are in the "valid_ids" array.
    """
    if align and align_origin:
        raise ValueError("align and align_origin can't be used simultaneously")

    ref_name = traj_ref.name or "reference"
    est_name = traj_est.name or "estimate"

    # Align the trajectories.
    only_scale = correct_scale and not align
    alignment_transformation = None
    if align or correct_scale:
        logger.debug(SEP)
        alignment_transformation = lie_algebra.sim3(
            *traj_est.align(traj_ref, correct_scale, only_scale, n=n_to_align)
        )
    if align_origin:
        logger.debug(SEP)
        alignment_transformation = traj_est.align_origin(traj_ref)

    # Calculate NEES.
    logger.debug(SEP)
    nees_metric = metrics.NEES(pose_relation)
    nees_metric.process_data((traj_ref, traj_est))

    title = str(nees_metric)
    if align and not correct_scale:
        title += "\n(with SE(3) Umeyama alignment)"
    elif align and correct_scale:
        title += "\n(with Sim(3) Umeyama alignment)"
    elif only_scale:
        title += "\n(scale corrected)"
    elif not align_origin:
        title += "\n(not aligned)"
    if (align or correct_scale) and n_to_align != -1:
        title += f" (aligned poses: {n_to_align})"
    if align_origin:
        title += "\n(with origin alignment)"

    nees_result = nees_metric.get_result(ref_name, est_name)
    nees_result.info["title"] = title
    nees_result.info["dof"] = nees_metric.dof

    logger.debug(SEP)
    logger.info(nees_result.pretty_str())

    nees_result.add_trajectory(ref_name, traj_ref)
    nees_result.add_trajectory(est_name, traj_est)
    valid_ids = nees_metric.valid_ids
    if isinstance(traj_est, PoseTrajectory3D):
        timestamps = traj_est.timestamps[valid_ids]
        nees_result.add_np_array(
            "seconds_from_start", timestamps - traj_est.timestamps[0]
        )
        nees_result.add_np_array("timestamps", timestamps)
        nees_result.add_np_array(
            "distances_from_start", traj_ref.distances[valid_ids]
        )
        nees_result.add_np_array("distances", traj_est.distances[valid_ids])

    if alignment_transformation is not None:
        nees_result.add_np_array(
            "alignment_transformation_sim3", alignment_transformation
        )

    return nees_result
