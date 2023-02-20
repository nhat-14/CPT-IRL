import argparse

def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="A simulator for olfactory searches")
    """
    parser.add_argument(
        "--version",
        action="version",
        version="bombyxsim {ver}".format(ver=__version__))
    """
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        dest="input_dir",
        help='Path of the directory with odor plume data',
        required=True)
    parser.add_argument(
        "-c", "--conf",
        type=str,
        dest="conf",
        help='Path to config file (json)',
        required=False)
    parser.add_argument(
        "-n", "--nruns",
        type=int,
        dest="Nruns",
        help='Number of simulation runs',
        required=True)
    parser.add_argument(
        "-t", "--tlim",
        type=int,
        dest="tlim",
        help='Simulation time limit in seconds',
        default=120)
    parser.add_argument("--hit-prob",
                        type=float,
                        dest="hit_prob",
                        help='Hit probability',
                        default=1.0)
    parser.add_argument(
        "-a", "--animation",
        dest="animation",
        help='Draw animation',
        action="store_true",
        default=0)
    parser.add_argument(
        "--plot-traj",
        dest="plt_traj",
        help='Plot trajectories',
        action="store_true",
        default=0)
    parser.add_argument(
        "--save-csv",
        dest="save_log",
        help='Save log to csv',
        action="store_true",
        default=0)
    return parser.parse_args(args)