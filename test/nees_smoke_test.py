#!/usr/bin/env python

import os
import shutil
import subprocess as sp
import tempfile
from pathlib import Path

import numpy as np

from evo.core import lie_algebra as lie
from evo.core import trajectory
from evo.tools import file_interface

tmp_dir = Path("tmp")
cfg_dir = Path("cfg/nees")
here = Path(__file__).absolute().parent

# always run in script location
os.chdir(here)

ref_file = "data/fr2_desk_groundtruth.txt"


def write_estimate(path: Path, covariance_scale: float) -> None:
    """
    Writes every 50th pose of the reference with errors sampled from random
    covariances, which are written scaled by covariance_scale.
    The covariance of the 4th pose is indefinite and has to be skipped.
    """
    rng = np.random.default_rng(0)
    traj = file_interface.read_tum_trajectory_file(ref_file)
    traj.reduce_to_ids(np.arange(0, traj.num_poses, 50))
    n = traj.num_poses
    a = rng.normal(scale=0.01, size=(n, 6, 6))
    # Not too tight, so that the uncertainty of an alignment estimated from
    # the same poses doesn't bias the NEES.
    covariances = a @ a.transpose(0, 2, 1) + 1e-4 * np.eye(6)
    poses = [
        p.dot(lie.se3_exp(rng.multivariate_normal(np.zeros(6), c)))
        for p, c in zip(traj.poses_se3, covariances)
    ]
    covariances = covariance_scale * covariances
    covariances[3] = -np.eye(6)
    xyz, quat_wxyz = trajectory.se3_poses_to_xyz_quat_wxyz(poses)
    header = [
        "timestamp",
        "keyframe_id",
        *["twc_" + s for s in file_interface.POSE_COLUMN_SUFFIXES],
        *["cov_tcw_" + s for s in file_interface.COVARIANCE_COLUMN_SUFFIXES],
    ]
    rows = np.column_stack(
        (
            traj.timestamps,
            np.arange(n),
            xyz,
            np.roll(quat_wxyz, -1, axis=1),
            covariances.reshape(n, 36),
        )
    )
    np.savetxt(path, rows, delimiter=",", header=",".join(header), comments="")


def run(cmd: str) -> None:
    print(f"[smoke test] {cmd}")
    sp.check_output(cmd.split(" "), cwd=here)


try:
    with tempfile.TemporaryDirectory() as data_dir:
        estimates = {
            "consistent": Path(data_dir) / "consistent.csv",
            "overconfident": Path(data_dir) / "overconfident.csv",
        }
        write_estimate(estimates["consistent"], covariance_scale=1.0)
        write_estimate(estimates["overconfident"], covariance_scale=0.25)

        for cfg in sorted(cfg_dir.iterdir()):
            tmp_dir.mkdir(exist_ok=True)
            run(f"evo_nees tum {ref_file} {estimates['consistent']} -c {cfg}")
            shutil.rmtree(tmp_dir)

        # The results have to be usable by evo_res, and the ANEES has to
        # detect the overconfident covariances.
        tmp_dir.mkdir(exist_ok=True)
        for name, est_file in estimates.items():
            run(
                f"evo_nees tum {ref_file} {est_file} -a --no_warnings "
                f"--save_results tmp/{name}.zip"
            )
        run(
            "evo_res tmp/consistent.zip tmp/overconfident.zip --no_warnings "
            "--save_table tmp/table.csv"
        )
        assert (tmp_dir / "table.csv").exists()
        for name in estimates:
            stats = file_interface.load_res_file(tmp_dir / f"{name}.zip").stats
            assert stats["rejected"] == 1, stats["rejected"]
            in_bounds = (
                stats["anees_95_low"] < stats["anees"] < stats["anees_95_high"]
            )
            assert in_bounds == (name == "consistent"), (name, stats["anees"])
except sp.CalledProcessError as e:
    print(e.output.decode("utf-8"))
    raise
finally:
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
