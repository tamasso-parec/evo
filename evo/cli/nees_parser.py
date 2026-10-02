import argparse

from evo.tools.settings import SETTINGS


def parser() -> argparse.ArgumentParser:
    basic_desc = (
        "Normalized estimation error squared (NEES) app for testing the "
        "consistency of pose covariances"
    )
    lic = "(c) evo authors"
    est_help = (
        "estimated trajectory .csv file with pose covariances, see "
        "evo.tools.file_interface.read_csv_trajectory_with_covariances"
    )

    shared_parser = argparse.ArgumentParser(add_help=False)
    algo_opts = shared_parser.add_argument_group("algorithm options")
    output_opts = shared_parser.add_argument_group("output options")
    usability_opts = shared_parser.add_argument_group("usability options")

    algo_opts.add_argument(
        "-r",
        "--pose_relation",
        default="full",
        help="pose relation on which the NEES is based: "
        "full (6 dof), or the marginal trans_part / rot_part (3 dof)",
        choices=["full", "trans_part", "rot_part"],
    )
    algo_opts.add_argument(
        "-s",
        "--correct_scale",
        action="store_true",
        help="correct scale with Umeyama's method",
    )
    algo_opts.add_argument(
        "--n_to_align",
        help="the number of poses to use for Umeyama alignment, "
        "counted from the start (default: all)",
        default=-1,
        type=int,
    )
    algo_opts.add_argument(
        "--downsample",
        type=int,
        help="Downsample trajectories to max N poses.",
    )
    algo_opts.add_argument(
        "--motion_filter",
        type=float,
        nargs=2,
        metavar=("DISTANCE", "ANGLE_DEGREES"),
        help="Filters out poses if the distance or angle to the previous one "
        " is below the threshold distance or angle. "
        "Angle is expected in degrees.",
    )

    align_opts = algo_opts.add_mutually_exclusive_group()
    align_opts.add_argument(
        "-a",
        "--align",
        help="alignment with Umeyama's method (no scale)",
        action="store_true",
    )
    align_opts.add_argument(
        "--align_origin",
        help="align the trajectory origin to the origin of the reference "
        "trajectory",
        action="store_true",
    )

    output_opts.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="show plot window",
    )
    output_opts.add_argument(
        "--plot_mode",
        default=SETTINGS.plot_mode_default,
        help="the axes for plot projection",
        choices=["xy", "xz", "yx", "yz", "zx", "zy", "xyz"],
    )
    output_opts.add_argument(
        "--plot_x_dimension",
        choices=["index", "seconds", "distances"],
        default="seconds",
        help="dimension that is used on the x-axis of the raw value plot"
        "(default: seconds)",
    )
    output_opts.add_argument(
        "--plot_colormap_max",
        type=float,
        help="the upper bound used for the color map plot "
        "(default: maximum NEES value)",
    )
    output_opts.add_argument(
        "--plot_colormap_min",
        type=float,
        help="the lower bound used for the color map plot "
        "(default: minimum NEES value)",
    )
    output_opts.add_argument(
        "--plot_colormap_max_percentile",
        type=float,
        help="percentile of the NEES distribution to be used "
        "as the upper bound of the color map plot "
        "(in %%, overrides --plot_colormap_max)",
    )
    output_opts.add_argument(
        "--plot_full_ref",
        action="store_true",
        help="plot the full, unsynchronized reference trajectory",
    )
    output_opts.add_argument(
        "--ros_map_yaml",
        help="yaml file of an ROS 2D map image (.pgm/.png)"
        " that will be drawn into the plot",
        default=None,
    )
    output_opts.add_argument(
        "--map_tile",
        help="CRS code of a map tile layer to add to the plot. "
        "Requires geo-referenced poses and the contextily package installed.",
    )
    output_opts.add_argument(
        "--save_plot", default=None, help="path to save plot"
    )
    output_opts.add_argument(
        "--save_results", help=".zip file path to store results"
    )
    output_opts.add_argument(
        "--logfile", help="Local logfile path.", default=None
    )
    usability_opts.add_argument(
        "--no_warnings",
        action="store_true",
        help="no warnings requiring user confirmation",
    )
    usability_opts.add_argument(
        "-v", "--verbose", action="store_true", help="verbose output"
    )
    usability_opts.add_argument(
        "--silent", action="store_true", help="don't print any output"
    )
    usability_opts.add_argument(
        "--debug",
        action="store_true",
        help="verbose output with additional debug info",
    )
    usability_opts.add_argument(
        "-c",
        "--config",
        help=".json file with parameters (priority over command line args)",
    )

    main_parser = argparse.ArgumentParser(description=f"{basic_desc} {lic}")
    sub_parsers = main_parser.add_subparsers(dest="subcommand")
    sub_parsers.required = True

    tum_parser = sub_parsers.add_parser(
        "tum",
        parents=[shared_parser],
        description=f"{basic_desc} for a TUM reference trajectory - {lic}",
    )
    tum_parser.add_argument("ref_file", help="reference trajectory file")
    tum_parser.add_argument("est_file", help=est_help)

    euroc_parser = sub_parsers.add_parser(
        "euroc",
        parents=[shared_parser],
        description=f"{basic_desc} for a EuRoC MAV reference - {lic}",
    )
    euroc_parser.add_argument(
        "state_gt_csv",
        help="ground truth: <seq>/mav0/state_groundtruth_estimate0/data.csv",
    )
    euroc_parser.add_argument("est_file", help=est_help)

    # Add time-sync options to parser of trajectory formats.
    for trajectory_parser in {euroc_parser, tum_parser}:
        trajectory_parser.add_argument(
            "--t_max_diff",
            type=float,
            default=0.01,
            help="maximum timestamp difference for data association",
        )
        trajectory_parser.add_argument(
            "--t_offset",
            type=float,
            default=0.0,
            help="constant timestamp offset for data association",
        )
        trajectory_parser.add_argument(
            "--sync_method",
            type=str,
            default="nearest_time",
            choices=["nearest_time", "interpolation"],
            help="method for the time synchronization of the trajectories: "
            "nearest_time associates poses with the closest matching "
            "timestamps (see --t_max_diff), "
            "interpolation resamples the reference at the timestamps "
            "of the estimate (requires a denser reference, covariances "
            "of the estimate can't be interpolated)",
        )
        trajectory_parser.add_argument(
            "--t_start",
            type=float,
            default=None,
            help="only use data with timestamps "
            "greater or equal this start time",
        )
        trajectory_parser.add_argument(
            "--t_end",
            type=float,
            default=None,
            help="only use data with timestamps less or equal this end time",
        )

    return main_parser
