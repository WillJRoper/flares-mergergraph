import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mega.core.talking_utils import pad_print_middle


def get_galaxy_info():

    # Define paths
    path = "/cosma/home/dp004/dc-rope1/cosma7/FLARES/flares-mergergraph/"
    halo_base = "data/halos/MEGAFLARES_halos_<reg>_<snap>.hdf5"
    graph_base = "data/dgraph/MEGAFLARES_graph_<reg>_<snap>.hdf5"

    # Get ID from command line
    grp = int(sys.argv[1])
    subgrp = int(sys.argv[2])
    region = sys.argv[3]
    tag = sys.argv[4]
    cont_type = int(sys.argv[5])

    # Replace place holders
    halo_base = halo_base.replace("<reg>", region)
    graph_base = graph_base.replace("<reg>", region)
    halo_base = halo_base.replace("<snap>", tag)
    graph_base = graph_base.replace("<snap>", tag)

    # Open halo file
    hdf = h5py.File(path + halo_base, "r")

    # Find this galaxy's MEGA ID
    grps = hdf["group_number"][...]
    subgrps = hdf["subgroup_number"][...]
    print(grps, subgrps)
    mega_ind = np.where(np.logical_and(grps == grp, subgrps == subgrp))[0]

    print("GroupNumber:", grp, "SubGroupNumber:", subgrp,
          "is MEGA halo:", mega_ind)

    hdf.close()

    # Open the graph file
    hdf = h5py.File(path + graph_base, "r")

    # Access this snapshot
    snap_root = hdf

    # Get the start index and stride
    prog_start = snap_root["prog_start_index"][mega_ind][0]
    prog_stride = snap_root["n_progs"][mega_ind][0]

    # How many halos are we dealing with?
    nhalo = snap_root["n_progs"].size

    # Lets get the data and clean it up
    true_nprog = prog_stride
    prog_mass_cont = snap_root["ProgMassContribution"][prog_start: prog_start +
                                                       prog_stride] * 10 ** 10
    prog_npart_cont = snap_root["ProgNPartContribution"][prog_start: prog_start +
                                                         prog_stride]
    okinds = prog_mass_cont[:, cont_type] > 10 ** 8
    nprog_major = prog_mass_cont[okinds, :].shape[0]
    prog_mass_cont = prog_mass_cont[okinds, :]
    prog_npart_cont = prog_npart_cont[okinds, :]

    hdf.close()

    # Print out this halos results from the graph
    header = "=" * 7 + \
        " LINKING DATA FOR GALAXY: (%d, %d) " % (grp, subgrp) + "=" * 7
    length = len(header)
    print(pad_print_middle("Nprog_all:", true_nprog, length=length))
    print(pad_print_middle("Nprog_major:", nprog_major, length=length))
    print(pad_print_middle("ProgMassContribution:", prog_mass_cont, length=length))
    print(pad_print_middle("ProgNPartContribution:", prog_npart_cont, length=length))
    print("=" * length)


if len(sys.argv) > 1:
    get_galaxy_info()
