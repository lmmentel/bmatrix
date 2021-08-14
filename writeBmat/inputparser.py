from __future__ import print_function, division

import argparse
import os
import sys

import numpy as np

if sys.version_info.major == 3:
    import configparser as cp
else:
    import ConfigParser as cp


def get_input_args():
    "Get the command line options and parse the config file if present"

    return parse_args(create_parser())


def create_parser():
    "Create the parser and define all the arguments"

    parser = argparse.ArgumentParser()

    parser.add_argument("filename", help="Name of the file with coordinates")

    parser.add_argument(
        "--config", default="INPDAT", type=str, help="configuration/input file"
    )

    parser.add_argument(
        "-c",
        "--coordinates",
        choices=["cartesian", "fractional"],
        default="cartesian",
        help="Coordinates",
    )

    parser.add_argument(
        "--relax",
        action="store_true",
        help="False: realxation of atomic positions only (default), "
        "True: relaxation of atomic pos.  and lat. param",
    )

    parser.add_argument(
        "--cart",
        action="store_true",
        help="if specified optimization in cartesians, if not optimization in "
        "delocalized internals (default)",
    )

    parser.add_argument(
        "--torsions", action="store_false", help="generate torsions, default: No"
    )

    parser.add_argument(
        "--hessian",
        choices=["cartesian", "internal", "lindhs", "fischers"],
        default="internal",
        help="initialize hessian as a diag matrix of a given coord/model",
    )

    parser.add_argument(
        "--hupdate",
        choices=[
            "noupdate",
            "gdiis",
            "bfgs",
            "bfgsts",
            "sr1",
            "psb",
            "sr1/psb",
            "sr1/bfgs",
        ],
        default="gdiis",
        help="hessian update formula",
    )

    parser.add_argument(
        "--hfresh",
        default=0,
        type=int,
        help="specifies how offen should be hessian re-initialised (0 - never,1-every step...)",
    )

    parser.add_argument(
        "--hupdim",
        default=2,
        help="dimension of history for BFGS update, can differ from 2 only if HREFRESH=1",
    )

    parser.add_argument(
        "--optengine",
        choices=["diis", "rfo", "prfo(ts)", "qnr"],
        default="diis",
        help='"engine" for optimization,',
    )

    parser.add_argument(
        "--linesearch", action="store_true", help="perform line search, default: No"
    )

    parser.add_argument(
        "--linemax", default=5, type=int, help="max number of line minimizations"
    )

    parser.add_argument(
        "--constconj", action="store_true", help="constraints as defined, default: No"
    )

    parser.add_argument(
        "--hsing",
        default=1.0e-4,
        type=float,
        help="minimal allowed ratio of eigenvalue of Hessian to its maximal eigenvalue",
    )

    parser.add_argument(
        "--potim",
        default=1,
        help="the value of diagonal component of initial component if hessian=0 or "
        "if internal and cartesian components are mixed",
    )

    parser.add_argument(
        "--nfree",
        default=10,
        help="number of history-steps involved in DIIS, 1(steepest descent)-10",
    )

    parser.add_argument(
        "--nsw", default=200, type=int, help="maximal number of relaxation steps"
    )

    parser.add_argument(
        "--steplim",
        default=0.3,
        type=float,
        help="maximal allowed step in internal coodrs (bohr for bonds, rad. for "
        "angles and torsions)",
    )

    parser.add_argument(
        "--gcriter",
        default=0.05,
        type=float,
        help="convergence criteria for gradients (eV/A)",
    )

    parser.add_argument(
        "--ecriter",
        default=1.0e-2,
        type=float,
        help="convergence criteria for energy (eV)",
    )

    parser.add_argument(
        "--scriter",
        default=1.0e-1,
        type=float,
        help="convergence criteria for geometry step (A)",
    )

    parser.add_argument(
        "--ascale",
        default=1.0,
        type=float,
        help="scaling factor for covalent atomic radii. if set to zero, only "
        "cartesian coords will be detected",
    )

    parser.add_argument(
        "--bscale",
        default=2.0,
        type=float,
        help="as ASCALE multiplied bz this if more fragments is present in the cell",
    )

    parser.add_argument("--cscale", type=float, help="")

    parser.add_argument(
        "--anglecrit",
        default=6,
        type=int,
        help="criterion for internal-coordinates (bonding angles) detection scheme",
    )

    parser.add_argument(
        "--torsioncrit",
        default=4,
        type=int,
        help="criterion for internal-coordinates (torsions) detection scheme",
    )

    parser.add_argument(
        "--vdwrad", default=0.0, type=float, help="radius for empirical vdw force field"
    )

    parser.add_argument(
        "--fragcoord",
        default=1,
        type=int,
        help="if more fragments this determine what to do: "
        "  0 : add cartesians for all but the largest fragments, "
        "  1 : add longer distances, "
        "  2 : add inverse power distances (1/R), "
        "  3 : add 1/R^6). "
        "in cases of 1, 2 and 3 the distances are "
        "generated using BSCALE*ASCALE. new coordinates "
        "are not used for generation of angles, torsions etc...",
    )

    parser.add_argument(
        "--primset", action="store_true", help="reset primitive internals, default: No"
    )

    parser.add_argument("--ts", default=0)

    parser.add_argument(
        "--rigidstress", action="store_true", help="0/1 --->actual/given stress"
    )

    parser.add_argument("--scalebt", default=1.0)

    parser.add_argument("--subst", default=100)

    parser.add_argument(
        "--pulayguess",
        nargs="+",
        help="this will be subtracted from stress tensor, "
        "makes sense only for full relaxations (RELAX=1)",
    )

    parser.add_argument(
        "--rstress",
        nargs="+",
        help="if rigidstress=1: only for test purposes "
        "(replaces the actual stress by this line)",
    )

    return parser


def parse_args(parser):
    "Parse the argument from the command line and from the config file"

    args = parser.parse_args()

    if os.path.exists(args.config):

        config = cp.SafeConfigParser()
        config.read(args.config)
        defaults = dict(config.items("default"))

        arg_dict = args.__dict__
        for key, value in defaults.items():
            if isinstance(value, list):
                arg_dict[key].extend(value)
            else:
                arg_dict[key] = type(arg_dict[key])(value)

        if "atomic radii" in config.sections():
            atradii = dict(config.items("atomic radii"))
        else:
            atradii = {}
    else:
        print("Configuration file: {} not found, uning defaults.".format(args.config))
        atradii = {}

    args.atradii = atradii
    args.cscale = args.cscale or args.ascale

    args.hupdim = max(args.hupdim, 2)
    if args.ts == 1 and args.hupdate in [1, 6]:
        args.hupdate = 5
    if args.cart == 1:
        args.ascale = 0.0
    if args.pulayguess is None:
        args.pulayguess = np.zeros(6, dtype=float)
    else:
        args.pylayguess = np.array(args.pulayguess, dtype=float)
    if args.rstress is None:
        args.rstress = np.zeros(6, dtype=float)
    else:
        args.rstress = np.array(args.rstress, dtype=float)

    args.homeaddress = os.getcwd()
    args.compaddress = os.getcwd()

    return args
