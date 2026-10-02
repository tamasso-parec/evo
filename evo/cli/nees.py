import argparse
import copy
import logging

import evo.cli.common_ape_rpe as common
from evo.core.result import Result
from evo.core.sync import SyncMethod
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.main_nees import nees
from evo.tools import file_interface, log
from evo.tools.settings import SETTINGS

logger = logging.getLogger(__name__)

SEP = "-" * 80


def load_trajectories(
    args: argparse.Namespace,
) -> tuple[PoseTrajectory3D, PoseTrajectory3D]:
    """
    Loads the reference and the estimate with covariances of the CLI args.
    """
    if args.subcommand == "tum":
        traj_ref = file_interface.read_tum_trajectory_file(args.ref_file)
    elif args.subcommand == "euroc":
        traj_ref = file_interface.read_euroc_csv_trajectory(args.state_gt_csv)
    else:
        raise KeyError(f"unknown sub-command: {args.subcommand}")
    traj_est = file_interface.read_csv_trajectory_with_covariances(
        args.est_file
    )
    return traj_ref, traj_est


def plot_result(
    args: argparse.Namespace,
    result: Result,
    traj_ref: PosePath3D,
    traj_est: PosePath3D,
    traj_ref_full: PosePath3D | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from scipy import stats

    from evo.tools import plot

    dof = result.info["dof"]
    # Only the poses with a valid covariance have a NEES value.
    valid_ids = result.np_arrays["valid_ids"]
    if len(valid_ids) != traj_est.num_poses:
        if traj_ref_full is None:
            traj_ref_full = traj_ref
        traj_ref = copy.deepcopy(traj_ref)
        traj_ref.reduce_to_ids(valid_ids)
        traj_est = copy.deepcopy(traj_est)
        traj_est.reduce_to_ids(valid_ids)

    width, height = SETTINGS.plot_figsize
    fig_consistency = plt.figure(figsize=(2 * width, 2 / 3 * height))
    plot.nees_consistency(
        fig_consistency,
        result.np_arrays["error_array"],
        dof,
        title=result.info["title"],
    )
    common.plot_result(
        args,
        result,
        traj_ref,
        traj_est,
        traj_ref_full=traj_ref_full,
        statistics={
            "ANEES": result.stats["anees"],
            f"expected ({dof} dof)": dof,
            f"$\\chi^2_{{{dof}}}$ 95$\\%$": stats.chi2.ppf(0.95, dof),
        },
        extra_figures={"consistency": fig_consistency},
    )


def run(args: argparse.Namespace) -> None:
    log.configure_logging(
        args.verbose, args.silent, args.debug, local_logfile=args.logfile
    )
    if args.debug:
        from pprint import pformat

        parser_str = pformat({arg: getattr(args, arg) for arg in vars(args)})
        logger.debug(f"main_parser config:\n{parser_str}")
    logger.debug(SEP)

    traj_ref, traj_est = load_trajectories(args)
    pose_relation = common.get_pose_relation(args)

    traj_ref_full = None
    if args.plot_full_ref:
        traj_ref_full = copy.deepcopy(traj_ref)

    # Downsample or filtering has to be done before synchronization.
    # Otherwise filtering might mess up the sync.
    common.downsample_or_filter(args, traj_ref, traj_est)

    logger.debug(SEP)
    if args.t_start or args.t_end:
        if args.t_start:
            logger.info(f"Using time range start: {args.t_start}s")
        if args.t_end:
            logger.info(f"Using time range end: {args.t_end}s")
        traj_ref.reduce_to_time_range(args.t_start, args.t_end)
    sync_method = SyncMethod(args.sync_method)
    logger.debug(f"Synchronizing trajectories ({sync_method.value} method)...")
    traj_ref, traj_est = traj_ref.sync_with(
        traj_est,
        sync_method=sync_method,
        max_diff=args.t_max_diff,
        offset_2=args.t_offset,
    )

    result = nees(
        traj_ref=traj_ref,
        traj_est=traj_est,
        pose_relation=pose_relation,
        align=args.align,
        correct_scale=args.correct_scale,
        n_to_align=args.n_to_align,
        align_origin=args.align_origin,
    )

    if args.plot or args.save_plot:
        plot_result(
            args,
            result,
            traj_ref,
            traj_est,
            traj_ref_full=traj_ref_full,
        )

    if args.save_results:
        logger.debug(SEP)
        if not SETTINGS.save_traj_in_zip:
            result.trajectories.clear()
        file_interface.save_res_file(
            args.save_results, result, confirm_overwrite=not args.no_warnings
        )
