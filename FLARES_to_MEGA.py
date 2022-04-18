import sys

import core.param_utils as p_utils
import core.utilities as utils
import h5py
import mpi4py
import numpy as np
from core.collect_result import collect_halos
from core.halo import Halo
from core.serial_io import write_data
from core.talking_utils import message
from core.talking_utils import say_hello
from core.timing import TicToc
from core.timing import timer
from mpi4py import MPI

from eagle_IO import eagle_IO as E

mpi4py.rc.recv_mprobe = False

# Initializations and preliminaries
comm = MPI.COMM_WORLD  # get MPI communicator object
size = comm.size  # total number of processes
rank = comm.rank  # rank of this process
status = MPI.Status()  # get MPI status object


@timer("Reading")
def get_data(tictoc, reg, tag, meta, inputpath):
    # Define sim path
    sim_path = inputpath.replace("<reg>", reg)
    single_file = sim_path + "snapshot_" + tag + "/snap_" + tag + ".0.hdf5"

    # Open single file and get DM particle mass
    hdf = h5py.File(single_file, "r")
    part_dm_mass = hdf["Header"].attrs["MassTable"][1] * 10 ** 10 / meta.h
    hdf.close()

    # Define the NULL value in GADGET files
    null = 1073741824

    # Read in all the relevant data
    true_part_ids = E.read_array("SNAP", sim_path, tag,
                                 "PartType1/ParticleIDs", numThreads=8)
    part_ids = E.read_array("PARTDATA", sim_path, tag,
                            "PartType1/ParticleIDs", numThreads=8)
    part_grp_ids = E.read_array("PARTDATA", sim_path, tag,
                                "PartType1/GroupNumber", numThreads=8)
    part_subgrp_ids = E.read_array("PARTDATA", sim_path, tag,
                                   "PartType1/SubGroupNumber", numThreads=8)
    part_pos = E.read_array("PARTDATA", sim_path, tag,
                            "PartType1/Coordinates", numThreads=8, noH=True)
    part_vel = E.read_array("PARTDATA", sim_path, tag,
                            "PartType1/Velocity", numThreads=8, noH=True,
                            physicalUnits=True)

    # Get the number of particles we are dealing with
    npart = part_ids.size
    true_npart = true_part_ids.size

    # Let's bin the particles and split the work up
    rank_bins = np.linspace(0, npart, size + 1, dtype=int)

    # Initialise dictionaries to store sorted particles
    length_dict = {}
    dm_pid_dict = {}
    dm_ind_dict = {}
    dm_pos_dict = {}
    dm_vel_dict = {}
    dm_masses_dict = {}

    # Loop over the particles on this rank
    for ind in range(rank_bins[rank], rank_bins[rank + 1]):

        # Is this particle in a subgroup?
        if part_subgrp_ids[ind] == null:
            continue

        # Define this halo's key
        key = (part_grp_ids[ind], part_subgrp_ids[ind])

        # Add this particle to the halo
        length_dict.setdefault(key, 0)
        length_dict[key] += 1
        dm_pid_dict.setdefault(key, []).append(part_ids[ind])
        dm_ind_dict.setdefault(key, []).append(ind)
        dm_pos_dict.setdefault(key, []).append(part_pos[ind, :])
        dm_vel_dict.setdefault(key, []).append(part_vel[ind, :])
        dm_masses_dict.setdefault(key, []).append(part_dm_mass)

    # Now need collect on master
    all_length = comm.gather(length_dict, root=0)
    all_dm_pid = comm.gather(dm_pid_dict, root=0)
    all_dm_ind = comm.gather(dm_ind_dict, root=0)
    all_dm_pos = comm.gather(dm_pos_dict, root=0)
    all_dm_vel = comm.gather(dm_vel_dict, root=0)
    all_dm_masses = comm.gather(dm_masses_dict, root=0)
    if rank == 0:

        # Loop over halos from other ranks
        for r in range(len(all_length)):
            if r == 0:
                continue

            # Loop over halos
            for key in all_length[r]:
                # Add this particle to the halo
                length_dict.setdefault(key, 0)
                length_dict[key] += all_length[r][key]

                dm_pid_dict.setdefault(key, []).extend(all_dm_pid[r][key])
                dm_ind_dict.setdefault(key, []).extend(all_dm_ind[r][key])
                dm_pos_dict.setdefault(key, []).extend(all_dm_pos[r][key])
                dm_vel_dict.setdefault(key, []).extend(all_dm_vel[r][key])
                dm_masses_dict.setdefault(key,
                                          []).extend(all_dm_masses[r][key])

        # Loop over halos and clean any spurious (npart<10)
        ini_keys = list(length_dict.keys())
        for key in ini_keys:

            if length_dict[key] < 10:
                del length_dict[key]
                del dm_pid_dict[key]
                del dm_ind_dict[key]
                del dm_pos_dict[key]
                del dm_vel_dict[key]
                del dm_masses_dict[key]

        # Now we can sort our halos
        keys = length_dict.keys()
        vals = length_dict.values()
        keys = np.array(list(keys), dtype=object)
        vals = np.array(list(vals), dtype=int)
        sinds = np.argsort(vals)
        keys = keys[sinds, :]

        # Define arrays and lists to hold sorted halos
        nhalos = keys.shape[0]
        all_dm_begin = np.zeros(nhalos, dtype=int)
        all_dm_len = np.zeros(nhalos, dtype=int)
        all_grpid = np.zeros(nhalos, dtype=int)
        all_subgrpid = np.zeros(nhalos, dtype=int)
        all_halo_ids = np.zeros(nhalos, dtype=int)
        all_dm_pid = []
        all_dm_ind = []
        all_dm_pos = []
        all_dm_vel = []
        all_dm_masses = []

        # Loop over keys storing their results
        for ihalo, key in enumerate(keys):
            # Get group and subgroup ID
            key = tuple(key)
            grp, subgrp = key[0], key[1]

            # Store data
            all_halo_ids[ihalo] = ihalo
            all_dm_begin[ihalo] = len(all_dm_pid)
            all_dm_len[ihalo] = length_dict[key]
            all_grpid[ihalo] = grp
            all_subgrpid[ihalo] = subgrp
            all_dm_pid.extend(dm_pid_dict[key])
            all_dm_ind.extend(dm_ind_dict[key])
            all_dm_pos.extend(dm_pos_dict[key])
            all_dm_vel.extend(dm_vel_dict[key])
            all_dm_masses.extend(dm_masses_dict[key])

        # Convert all keys to arrays
        all_dm_pid = np.array(all_dm_pid, dtype=int)
        all_dm_ind = np.array(all_dm_ind, dtype=int)
        all_dm_pos = np.array(all_dm_pos, dtype=np.float64)
        all_dm_vel = np.array(all_dm_vel, dtype=np.float64)
        all_dm_masses = np.array(all_dm_masses, dtype=np.float64)

        # Define the number of particles sorted
        npart_sorted = len(all_dm_pid)

        # Lets bin the halos onto ranks
        shared_npart = npart_sorted / size

        # Loop over ranks populating them with roughly equal numbers
        # of particles
        halos_on_rank = {r: [] for r in range(size)}
        ihalo = 0
        for r in range(size):

            # Keep track of allocated particles
            nparts_to_send = 0

            # Allocate this halo
            while nparts_to_send <= shared_npart and ihalo < all_dm_len.size:
                halos_on_rank[r].append(ihalo)
                nparts_to_send += all_dm_len[ihalo]
                ihalo += 1

            # Allocate any leftovers
            # NOTE: leads to the last rank having more work but
            # these are small halos due to halo ordering
            while ihalo < all_dm_len.size:
                halos_on_rank[r].append(ihalo)
                ihalo += 1

        # Get how many particles each rank should expect
        nparts_on_rank = [0, ] * size
        for r, halos in enumerate(halos_on_rank):
            print(r, halos)
            for ihalo in halos:
                nparts_on_rank[r] += all_dm_len[ihalo]

        # Get how many halos each rank should expect, and their
        # particle offset
        nhalos_on_rank = [0, ] * size
        offsets = [0, ] * size
        for r, halos in enumerate(halos_on_rank):
            offsets[r] = all_dm_begin[halos[0]]
            nhalos_on_rank[r] = len(halos)

        halo_ids = None
        dm_len = None
        grpid = None
        subgrpid = None
        dm_pid = None
        dm_ind = None
        dm_pos = None
        dm_vel = None
        dm_masses = None

    else:
        halos_on_rank = None
        nhalos_on_rank = None
        nparts_on_rank = None
        all_halo_ids = None
        all_dm_begin = None
        all_dm_len = None
        all_grpid = None
        all_subgrpid = None
        all_dm_pid = None
        all_dm_ind = None
        all_dm_pos = None
        all_dm_vel = None
        all_dm_masses = None
        
    # Broadcast what each rank should be expecting
    nparts_on_rank = comm.bcast(nparts_on_rank, root=0)
    nhalos_on_rank = comm.bcast(nhalos_on_rank, root=0)
    
    # Now we can finally distribute the halos and particles
    if rank == 0:
        
        # Loop over ranks
        for r in range(size):

            # Get the data to send
            rank_begin = all_dm_begin[halos_on_rank[r][0]]
            rank_len = nparts_on_rank[r]
            halo_slice = (halos_on_rank[r][0], halos_on_rank[r][1] + 1)
            part_slice = (rank_begin, rank_begin + rank_len)

            if r == rank:

                # Get the master rank's halos
                halo_ids = all_halo_ids[halo_slice[0]: halo_slice[1]]
                dm_len = all_dm_len[halo_slice[0]: halo_slice[1]]
                grpid = all_grpid[halo_slice[0]: halo_slice[1]]
                subgrpid = all_subgrpid[halo_slice[0]: halo_slice[1]]
                dm_pid = all_dm_pid[part_slice[0]: part_slice[1]]
                dm_ind = all_dm_ind[part_slice[0]: part_slice[1]]
                dm_pos = all_dm_pos[part_slice[0]: part_slice[1]]
                dm_vel = all_dm_vel[part_slice[0]: part_slice[1]]
                dm_masses = all_dm_masses[part_slice[0]: part_slice[1]]

            else:

                # Post sends
                comm.Isend(all_halo_ids[halo_slice[0]: halo_slice[1]],
                           dest=r, tag=0)
                comm.Isend(all_dm_len[halo_slice[0]: halo_slice[1]],
                           dest=r, tag=1)
                comm.Isend(all_grpid[halo_slice[0]: halo_slice[1]],
                           dest=r, tag=2)
                comm.Isend(all_subgrpid[halo_slice[0]: halo_slice[1]],
                           dest=r, tag=3)
                comm.Isend(all_dm_pid[part_slice[0]: part_slice[1]],
                           dest=r, tag=4)
                comm.Isend(all_dm_ind[part_slice[0]: part_slice[1]],
                           dest=r, tag=5)
                comm.Isend(all_dm_pos[part_slice[0]: part_slice[1]],
                           dest=r, tag=6)
                comm.Isend(all_dm_vel[part_slice[0]: part_slice[1]],
                           dest=r, tag=7)
                comm.Isend(all_dm_masses[part_slice[0]: part_slice[1]],
                           dest=r, tag=8)
    
    else:

        # Create receive buffers
        halo_ids = np.empty(nhalos_on_rank[rank], dtype=int)
        dm_len = np.empty(nhalos_on_rank[rank], dtype=int)
        grpid = np.empty(nhalos_on_rank[rank], dtype=int)
        subgrpid = np.empty(nhalos_on_rank[rank], dtype=int)
        dm_pid = np.empty(nparts_on_rank[rank], dtype=int)
        dm_ind = np.empty(nparts_on_rank[rank], dtype=int)
        dm_pos = np.empty((nparts_on_rank[rank], 3), dtype=np.float64)
        dm_vel = np.empty((nparts_on_rank[rank], 3), dtype=np.float64)
        dm_masses = np.empty(nparts_on_rank[rank], dtype=np.float64)

        # Receive
        comm.Recv(dm_len, source=0, tag=1)
        comm.Recv(grpid, source=0, tag=2)
        comm.Recv(subgrpid, source=0, tag=3)
        comm.Recv(dm_pid, source=0, tag=4)
        comm.Recv(dm_ind, source=0, tag=5)
        comm.Recv(dm_pos, source=0, tag=6)
        comm.Recv(dm_vel, source=0, tag=7)
        comm.Recv(dm_masses, source=0, tag=8)

    return (halo_ids, dm_len, grpid, subgrpid, dm_pid, dm_ind, dm_pos, dm_vel,
            dm_masses, part_ids, true_npart)


def main():
    # Make the region list
    regions = []
    for reg in range(0, 40):
        if reg < 10:
            regions.append("0" + str(reg))
        else:
            regions.append(str(reg))

    # Read the parameter file
    paramfile = sys.argv[1]
    (inputs, flags, params, cosmology,
     simulation) = p_utils.read_param(paramfile)

    # Get job index
    job_ind = int(sys.argv[2])

    # Load the snapshot list
    snaplist = ["000_z015p000", "001_z014p000", "002_z013p000", "003_z012p000",
                "004_z011p000", "005_z010p000", "006_z009p000", "007_z008p000",
                "008_z007p000", "009_z006p000", "010_z005p000", "011_z004p770"]

    # Get the snapshot index
    snap_ind = job_ind % len(snaplist)

    reg_snaps = []
    for reg in reversed(regions):

        for snap in snaplist:
            reg_snaps.append((reg, snap))

    # Get snapshot and region
    reg, snap = reg_snaps[job_ind]

    # Get redshift
    z_str = snap.split('z')[1].split('p')
    z = float(z_str[0] + '.' + z_str[1])

    # Set up object containing housekeeping metadata
    meta = p_utils.Metadata(snaplist, snap_ind, cosmology,
                            params["llcoeff"], params["sub_llcoeff"], inputs,
                            None,
                            inputs["haloSavePath"], params["ini_alpha_v"],
                            params["min_alpha_v"], params["decrement"],
                            flags["verbose"], flags["subs"],
                            params["N_cells"], flags["profile"],
                            inputs["profilingPath"], cosmology["h"],
                            (simulation["comoving_DM_softening"],
                             simulation["max_physical_DM_softening"]),
                            dmo=True, periodic=0, boxsize=[3200, 3200, 3200],
                            npart=[0, 10 ** 7, 0, 0, 0, 0], z=z,
                            tot_mass=10 ** 13)

    meta.rank = rank
    meta.nranks = size

    if rank == 0:
        say_hello(meta)
        message(rank, "Running on Region %s and snap %s" % (reg, snap))

    # Instantiate timer
    tictoc = TicToc(meta)
    tictoc.start()
    meta.tictoc = tictoc

    # Get the particle data for all particle types in the current snapshot
    (halo_ids, dm_len, grpid, subgrpid, dm_pid, dm_ind, dm_pos, dm_vel,
     dm_masses, dm_snap_part_ids, true_npart) = get_data(tictoc, reg,
                                                         snap, meta,
                                                         inputs["data"])

    # Set npart
    meta.npart[1] = dm_snap_part_ids.size

    if rank == 0:
        message(rank, "Npart: %d ~ %d^3" % (true_npart,
                                            int(true_npart ** (1 / 3))))
        message(rank, "Nhalo: %d" % len(dm_len))

    # Define part type array
    dm_part_types = np.full_like(dm_pid, 1)

    # Initialise dictionary for mega halo objects
    results = {}

    # Loop over galaxies and create mega objects
    b = 0
    for ihalo, l in (halo_ids, dm_len):

        # Compute end
        e = b + l

        if len(dm_ind[b:e]) < 10:
            continue

        # Store this halo
        results[ihalo] = Halo(tictoc, dm_ind[b:e],
                              (grpid[ihalo], subgrpid[ihalo]),
                              dm_pid[b:e], dm_pos[b:e, :], dm_vel[b:e, :],
                              dm_part_types[b:e],
                              dm_masses[b:e], 10, meta)
        results[ihalo].clean_halo()
        results[ihalo].memory = utils.get_size(results[ihalo])

        # Set new begin
        b = e

    # Collect results from all processes bit by bit
    tictoc.start_func_time("Collecting")

    # We're collecting to master
    if rank == 0:

        # Define the number of ranks we must receive from
        send_ranks = size - 1
        closed_workers = 0

        # Initialise list of collected results
        collected_results = [results, ]

        # Loop until nothing left to receive
        while closed_workers < send_ranks:
            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,
                             status=status)
            tag = status.Get_tag()

            if tag == 0:

                # Store received results
                collected_results.append(data)

            elif tag == 1:

                closed_workers += 1

    else:

        collected_results = None

        # Loop until we've sent all our results
        while len(results) > 0:

            # Get some results to send
            # (limiting to 1GB of communication for safety)
            subset = {}
            subset_size = 0
            while subset_size < 1024 and len(results) > 0:

                # Get halo
                key, halo = results.popitem()

                # Add this halos memory
                subset_size += halo.memory * 10 ** -6

                # Include this halo
                subset[key] = halo

            # Send the halo subset
            comm.send(subset, dest=0, tag=0)

        # Send the finished signal
        comm.send(None, dest=0, tag=1)

    tictoc.stop_func_time()

    if rank == 0:

        # Lets combine all the halos we have collected from the other ranks
        res_tup = collect_halos(tictoc, meta, collected_results,
                                [{}, ] * size)
        (newPhaseID, newPhaseSubID, results_dict, sub_results_dict) = res_tup

        if meta.verbose:
            tictoc.report("Combining results")

        # Create extra data arrays
        extra_data = {"group_number": np.zeros(len(results), dtype=int),
                      "subgroup_number": np.zeros(len(results), dtype=int)}
        for ihalo in results:
            grp, subgrp = results[ihalo].shifted_inds
            extra_data["group_number"][ihalo] = grp
            extra_data["subgroup_number"][ihalo] = subgrp

        # Write out file
        write_data(tictoc, meta, newPhaseID, newPhaseSubID,
                   results_dict=results_dict, sub_results_dict={},
                   sim_pids=dm_snap_part_ids, basename_mod=reg,
                   extra_data=extra_data)

        if meta.verbose:
            tictoc.report("Writing")

    tictoc.end()
    tictoc.end_report(comm)


if __name__ == "__main__":
    main()
