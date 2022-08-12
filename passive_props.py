import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mega.core.talking_utils import pad_print_middle


def print_info(reg, grp, subgrp, mega_ind, true_nprog, nprog_major,
               prog_halo_ids, prog_mass_cont, prog_npart_cont, mass):

    # Convert units on mass
    prog_mass_cont = np.log10(prog_mass_cont)

    header = "{:=^90}".format(
        "LINKING DATA FOR GALAXY: (%d, %d = %d) in REGION %s " % (
            grp, subgrp, mega_ind, reg))
    length = len(header)
    print(header)
    print(pad_print_middle("| Nprog_all:", str(true_nprog) + " |", length=length))
    print("|" + "-" * (length - 2) + "|")
    print(pad_print_middle("| Nprog_major:", str(nprog_major) + " |",
                           length=length))
    print("|" + "-" * (length - 2) + "|")
    print(pad_print_middle("| log10(M_tot/M_sun):",
                           "%.2f |" % (np.log10(mass)),
                           length=length))
    print("|" + "-" * (length - 2) + "|")
    print(pad_print_middle("| ProgMassContribution:", "|", length=length))
    print(pad_print_middle("| ProgenitorID", "log10(M_cont/M_sun) |",
                           length=length))
    for i, prog in enumerate(prog_halo_ids):
        print(pad_print_middle(
            "| " + str(prog) + ":",
            "[%.2f %.2f %.2f %.2f %.2f %.2f] |" % (prog_mass_cont[i, 0],
                                                   prog_mass_cont[i, 1],
                                                   prog_mass_cont[i, 2],
                                                   prog_mass_cont[i, 3],
                                                   prog_mass_cont[i, 4],
                                                   prog_mass_cont[i, 5]),
            length=length))
    print("|" + "-" * (length - 2) + "|")
    print(pad_print_middle("| ProgNPartContribution:", "|", length=length))
    print(pad_print_middle("| ProgenitorID", "N_cont |",
                           length=length))
    for i, prog in enumerate(prog_halo_ids):
        print(pad_print_middle("| " + str(prog) + ":", prog_npart_cont[i, :],
                               length=length - 2), "|")
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

    # Get command line args
    pcent = float(sys.argv[1])

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

        # Get FLARES galaxy IDs
        grps = hdf_f[reg][snap]["Galaxy"]["GroupNumber"]
        subgrps = hdf_f[reg][snap]["Galaxy"]["SubGroupNumber"]
        star_masses = hdf_f[reg][snap]["Galaxy"]["Mstar_aperture"]["30"][...]

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
        split_masses = hdf_halo["masses"][...]
        hdf_halo.close()

        print(split_masses)

        # Get contribution information
        prog_mass_conts = hdf_graph["ProgMassContribution"][...]
        prog_npart_conts = hdf_graph["ProgNPartContribution"][...]
        prog_ids = hdf_graph["ProgHaloIDs"][...]
        start_index = hdf_graph["prog_start_index"][...]
        nprogs = hdf_graph["n_progs"][...]
        hdf_graph.close()

        # Loop over MEGA galaxies getting the corresponding FLARES galaxy
        for ind in range(grps.size):

            # Extract this group, subgroup and ssfrs
            g, sg = grps[ind], subgrps[ind]
            ssfr = ssfrs[ind]
            smass = star_masses[ind] * 10 ** 10

            # Which mega galaxy is this?
            mega_ind = np.where(np.logical_and(mega_grps == g,
                                               mega_subgrps == sg))[0]

            # Get this galaxy's data
            start = start_index[mega_ind][0]
            stride = nprogs[mega_ind][0]
            mass = masses[mega_ind] * 10 ** 10

            # Apply mass cut
            if smass < 10 ** 9:
                continue

            if stride == 0:
                nprog = 0
            else:
                prog_cont = prog_mass_conts[start: start + stride] * 10 ** 10
                prog_ncont = prog_npart_conts[start: start + stride]
                progs = prog_ids[start: start + stride]
                mass = masses[mega_ind] * 10 ** 10

                # Limit galaxy's contribution to those contributing at least 10%
                tot_prog_cont = np.sum(prog_cont, axis=1)
                frac_prog_cont = tot_prog_cont / mass
                okinds = frac_prog_cont > pcent

                # Get only "true" contributions
                nprog = tot_prog_cont[okinds].size

                if ssfr < -1:
                    print_info(reg, g, sg, mega_ind, stride, nprog,
                               progs[okinds], prog_cont[okinds, :],
                               prog_ncont[okinds, :], mass)

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
    ax.scatter(tot_ssfrs, tot_nprogs,
               marker=".", color="k", alpha=0.6)

    # Add lines
    ax.axvline(-1, linestyle="--", color="k", alpha=0.6)
    ax.axvline(-2, linestyle="dotted", color="k", alpha=0.6)

    # Plot the scatter for passive galaxies split by central status
    okinds = tot_ssfrs < -1
    cent = central[okinds]
    ax.scatter(tot_ssfrs[okinds][cent], tot_nprogs[okinds][cent],
               marker="o", color="r", label="Central")
    ax.scatter(tot_ssfrs[okinds][~cent], tot_nprogs[okinds][~cent],
               marker="o", color="g", label="Satellite")

    # Label axes
    ax.set_xlabel("$\log_{10}(\mathrm{sSFR} / \mathrm{Gyr})$")
    ax.set_ylabel("$N_{prog}$")

    # Legend
    ax.legend()

    # Save figure
    fig.savefig("passive_nprog_%d_pcent.png" % (pcent * 100),
                bbox_inches="tight", dpi=100)

    # Define and print out merger fractions
    tot_frac = tot_nprogs[tot_nprogs > 1].size / tot_nprogs.size
    pass_frac = tot_nprogs[okinds][tot_nprogs[okinds]
                                   > 1].size / tot_nprogs[okinds].size
    print("The passive merger fraction is %f compared to the total %f" %
          (pass_frac, tot_frac))


if len(sys.argv) > 2:
    get_galaxy_info()
else:
    plot_merger_ssfr()
