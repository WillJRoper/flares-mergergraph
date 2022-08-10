import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt


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

    print(prog_start, prog_stride, prog_start + prog_stride)

    # How many halos are we dealing with?
    nhalo = snap_root.attrs["nhalo"]

    # Print out this halos results from the graph
    print("======== LINKING DATA FOR GALAXY: (%d, %d) ========"
          % (grp, subgrp))
    for key in snap_root.keys():
        if "Desc" in key:
            continue
        print(key)
        if snap_root[key].shape[0] == nhalo:
            print(key, "->", snap_root[key][mega_ind])
        else:
            print(key, ":")
            print(snap_root[key]
                  [prog_start: prog_start + prog_stride])
    print("=" * len("======== LINKING DATA FOR GALAXY: (%d, %d) "
                    "========" % (grp, subgrp)))


if len(sys.argv) > 1:
    get_galaxy_info()
