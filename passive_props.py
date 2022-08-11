import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mega.core.talking_utils import pad_print_middle


def print_info(grp, subgrp, mega_ind, true_nprog, nprog_major, prog_mass_cont,
               prog_npart_cont):

    pad = 15
    header = "=" * pad + \
        " LINKING DATA FOR GALAXY: (%d, %d = %d) " % (
            grp, subgrp, mega_ind) + "=" * pad
    length = len(header)
    print(pad_print_middle("Nprog_all:", true_nprog, length=length))
    print(pad_print_middle("Nprog_major:", nprog_major, length=length))
    print(pad_print_middle("ProgMassContribution:", prog_mass_cont, length=length))
    print(pad_print_middle("ProgNPartContribution:", prog_npart_cont, length=length))
    print("=" * length)


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
    print_info(grp, subgrp, mega_ind, true_nprog, nprog_major, prog_mass_cont,
               prog_npart_cont)


def plot_merger_ssfr():

    # Define paths
    path = "/cosma/home/dp004/dc-rope1/cosma7/FLARES/flares-mergergraph/"
    halo_base = "data/halos/MEGAFLARES_halos_<reg>_<snap>.hdf5"
    graph_base = "data/dgraph/MEGAFLARES_graph_<reg>_<snap>.hdf5"
    qui_path = "/cosma7/data/dp004/dc-love2/codes/flares_passive/" + \
        "analysis/data/select_quiescent.h5"
    flares_path = "/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline" + \
        "/data/flares.hdf5"

    # Open flares files
    hdf_q = h5py.File(qui_path, "r")
    hdf_f = h5py.File(flares_path, "r")

    # What snapshot are we doing?
    snap = "010_z005p000"

    # Set up lists to collect our results
    tot_nprogs = []
    tot_ssfrs = []
    central = []

    # Loop over regions
    for reg in hdf_f.keys():

        print("Region", reg)

        # Get FLARES galaxy IDs
        grps = hdf_f[reg][snap]["Galaxy"]["GroupNumber"]
        subgrps = hdf_f[reg][snap]["Galaxy"]["SubGroupNumber"]

        # Get sSFRs
        ssfrs = hdf_q[snap][reg]["sSFR"]["50Myr"][...]

        # Open MEGA file
        this_halo_base = halo_base.replace("<reg>", reg)
        this_halo_base = this_halo_base.replace("<snap>", snap)
        this_graph_base = graph_base.replace("<reg>", reg)
        this_graph_base = this_graph_base.replace("<snap>", snap)
        hdf_halo = h5py.File(this_halo_base, "r")
        hdf_graph = h5py.File(this_graph_base, "r")

        # Get MEGA galaxy IDs and masses
        mega_grps = hdf_halo["group_number"][...]
        mega_subgrps = hdf_halo["subgroup_number"][...]
        masses = hdf_halo["masses"][...]
        hdf_halo.close()

        # Get contribution information
        prog_mass_conts = hdf_graph["ProgMassContribution"][...]
        prog_npart_conts = hdf_graph["ProgNPartContribution"][...]
        start_index = hdf_graph["prog_start_index"][...]
        nprog = hdf_graph["n_prog"][...]
        hdf_graph.close()

        # Loop over MEGA galaxies getting the corresponding FLARES galaxy
        for ind in range(grps.size):

            # Extract this group, subgroup and ssfrs
            g, sg = grps[ind], subgrps[ind]
            ssfr = ssfrs[ind]

            # Which mega galaxy is this?
            mega_ind = np.where(np.logical_and(mega_grps == g,
                                               mega_subgrps == sg))[0]

            # Get this galaxy's data
            start = start_index[mega_ind]
            stride = nprog[mega_ind]
            prog_cont = prog_mass_conts[start: start + stride]
            prog_ncont = prog_npart_conts[start: start + stride]
            mass = masses[mega_ind]

            # Limit galaxy's contribution to those contributing at least 10%
            tot_prog_cont = np.sum(prog_cont, axis=1)
            frac_prog_cont = tot_prog_cont / np.sum(mass)
            okinds = frac_prog_cont > 0.1

            # Get only "true" contributions
            nprog = tot_prog_cont[okinds].size

            if ssfr < 10**-1:
                print_info(g, sg, mega_ind, stride, nprog,
                           prog_cont[okinds, :], prog_ncont[okinds, :])

            # Include this result
            tot_nprogs.append(nprog)
            tot_ssfrs.append(ssfr)
            if sg == 0:
                central.append(1)
            else:
                central.append(0)

    hdf_f.close()
    hdf_q.close()

    # Convert to arrays
    tot_nprogs = np.array(tot_nprogs, dtype=int)
    tot_ssfrs = np.array(tot_ssfrs, dtype=np.float64)
    central = np.array(central, dtype=bool)

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot the scatter of "normal" galaxies
    ax.scatter(np.log10(tot_ssfrs), tot_nprogs,
               marker=".", color="k", alpha=0.6)

    # Add lines
    ax.axvline(-0.1, linestyle="--", color="k", alpha=0.6)
    ax.axvline(-0.2, linestyle="dotted", color="k", alpha=0.6)

    # Plot the scatter for passive galaxies split by central status
    okinds = tot_ssfrs < 10 ** -1
    cent = central[okinds]
    ax.scatter(np.log10(tot_ssfrs[okinds][cent]), tot_nprogs[okinds][cent],
               marker="o", color="r", label="Central")
    ax.scatter(np.log10(tot_ssfrs[okinds][~cent]), tot_nprogs[okinds][~cent],
               marker="o", color="g", label="Satellite")

    # Label axes
    ax.set_xlabel("$\log_{10}(\mathrm{sSFR} / \mathrm={Gyr})$")
    ax.set_ylabel("$N_{prog}$")

    # Legend
    ax.legend()

    # Save figure
    fig.savefig("passive_nprog.png", bbox_inches="tight", dpi=100)


if len(sys.argv) > 1:
    get_galaxy_info()
else:
    plot_merger_ssfr()