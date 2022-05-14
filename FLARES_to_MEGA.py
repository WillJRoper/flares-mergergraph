import sys
import os

import mega.core.param_utils as p_utils
import mega.core.utilities as utils
import h5py
import mpi4py
import numpy as np
from mega.core.collect_result import collect_halos
from mega.halo_core.halo import Halo
from mega.core.serial_io import write_data
from mega.core.talking_utils import message, say_hello
from mega.core.timing import TicToc
from mega.core.timing import timer
from mpi4py import MPI

from eagle_IO import eagle_IO as E

mpi4py.rc.recv_mprobe = False

# Initializations and preliminaries
comm = MPI.COMM_WORLD  # get MPI communicator object
size = comm.size  # total number of processes
rank = comm.rank  # rank of this process
status = MPI.Status()  # get MPI status object


class HiddenPrints:
    """ A class to supress printing from outside functions
        (https://stackoverflow.com/questions/8391411/
        how-to-block-calls-to-print)

    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


@timer("Reading")
def get_data(tictoc, reg, tag, meta, inputpath):
    # Define sim path
    sim_path = inputpath.replace("<reg>", reg)
    single_file = sim_path + "snapshot_" + tag + "/snap_" + tag + ".0.hdf5"

    # Open single file and get DM particle mass
    hdf = h5py.File(single_file, "r")
    part_dm_mass = hdf["Header"].attrs["MassTable"][1] * 10 ** 10 / meta.h
    hdf.close()

    # Define the NULL value in SUBFIND files
    null = 1073741824

    # Read in all the relevant data
    with HiddenPrints():
        true_part_ids = E.read_array("SNAP", sim_path, tag,
                                     "PartType1/ParticleIDs", numThreads=8)
        part_ids = E.read_array("PARTDATA", sim_path, tag,
                                "PartType1/ParticleIDs", numThreads=8)
        part_grp_ids = E.read_array("PARTDATA", sim_path, tag,
                                    "PartType1/GroupNumber", numThreads=8)
        part_subgrp_ids = E.read_array("PARTDATA", sim_path, tag,
                                       "/PartType1/SubGroupNumber", numThreads=8)
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
        sinds = np.argsort(vals)[::-1]
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
            halos_on_rank[-1].append(ihalo)
            ihalo += 1

        # Get how many particles each rank should expect
        nparts_on_rank = [0, ] * size
        for r in halos_on_rank:
            for ihalo in halos_on_rank[r]:
                nparts_on_rank[r] += all_dm_len[ihalo]

        # Get how many halos each rank should expect, and their
        # particle offset
        nhalos_on_rank = [0, ] * size
        for r in halos_on_rank:
            nhalos_on_rank[r] = len(halos_on_rank[r])

        # Set up dictionaries for communicating data        
        halo_ids = [None for r in range(size)]
        dm_len = [None for r in range(size)]
        grpid = [None for r in range(size)]
        subgrpid = [None for r in range(size)]
        dm_pid = [None for r in range(size)]
        dm_ind = [None for r in range(size)]
        dm_pos = [None for r in range(size)]
        dm_vel = [None for r in range(size)]
        dm_masses = [None for r in range(size)]

        # Loop over ranks
        for r in range(size):

            # Get the data to send
            if len(halos_on_rank[r]) > 0:
                rank_begin = all_dm_begin[np.min(halos_on_rank[r])]
                rank_len = nparts_on_rank[r]
                halo_slice = (np.min(halos_on_rank[r]),
                              np.max(halos_on_rank[r]) + 1)
                part_slice = (rank_begin, rank_begin + rank_len)

                # Get store data
                halo_ids[r] = all_halo_ids[halo_slice[0]: halo_slice[1]]
                dm_len[r] = all_dm_len[halo_slice[0]: halo_slice[1]]
                grpid[r] = all_grpid[halo_slice[0]: halo_slice[1]]
                subgrpid[r] = all_subgrpid[halo_slice[0]: halo_slice[1]]
                dm_pid[r] = all_dm_pid[part_slice[0]: part_slice[1]]
                dm_ind[r] = all_dm_ind[part_slice[0]: part_slice[1]]
                dm_pos[r] = all_dm_pos[part_slice[0]: part_slice[1], :]
                dm_vel[r] = all_dm_vel[part_slice[0]: part_slice[1], :]
                dm_masses[r] = all_dm_masses[part_slice[0]: part_slice[1]]
            else:
                halo_ids[r] = np.array([], dtype=int)
                dm_len[r] = np.array([], dtype=int)
                grpid[r] = np.array([], dtype=int)
                subgrpid[r] = np.array([], dtype=int)
                dm_pid[r] = np.array([], dtype=int)
                dm_ind[r] = np.array([], dtype=int)
                dm_pos[r] = np.array([], dtype=np.float64)
                dm_vel[r] = np.array([], dtype=np.float64)
                dm_masses[r] = np.array([], dtype=np.float64)

    else:
        nhalos = None
        halo_ids = None
        dm_begin = None
        dm_len = None
        grpid = None
        subgrpid = None
        dm_pid = None
        dm_ind = None
        dm_pos = None
        dm_vel = None
        dm_masses = None

    # Broadcast the number of halos
    nhalos = comm.bcast(nhalos, root=0)

    # Scatter the results of the decompostion
    halo_ids = comm.scatter(halo_ids, root=0)
    dm_len = comm.scatter(dm_len, root=0)
    grpid = comm.scatter(grpid, root=0)
    subgrpid = comm.scatter(subgrpid, root=0)
    dm_pid = comm.scatter(dm_pid, root=0)
    dm_ind = comm.scatter(dm_ind, root=0)
    dm_pos = comm.scatter(dm_pos, root=0)
    dm_vel = comm.scatter(dm_vel, root=0)
    dm_masses = comm.scatter(dm_masses, root=0)

    return (halo_ids, dm_len, grpid, subgrpid, dm_pid, dm_ind, dm_pos, dm_vel,
            dm_masses, part_ids, true_npart, nhalos)


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
    meta = p_utils.Metadata(snaplist, snap_ind, cosmology, inputs,
                            flags, params, simulation,
                            boxsize=[3200, 3200, 3200],
                            npart=[0, 10 ** 7, 0, 0, 0, 0],
                            z=z, tot_mass=10 ** 13)

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
     dm_masses, dm_snap_part_ids, true_npart, nhalos) = get_data(tictoc, reg,
                                                                 snap, meta,
                                                                 inputs[
                                                                     "data"])

    # Set npart
    meta.npart[1] = dm_snap_part_ids.size

    if rank == 0:
        message(rank, "Npart: %d ~ %d^3" % (true_npart,
                                            int(true_npart ** (1 / 3))))
        message(rank, "Nhalo: %d" % nhalos)

    # Define part type array
    dm_part_types = np.full_like(dm_pid, 1)

    # Initialise dictionary for mega halo objects
    results = {}

    # Loop over galaxies and create mega objects
    b = 0
    halo_offset = halo_ids[0]
    for ihalo, l in zip(halo_ids, dm_len):

        # Compute end
        e = b + l

        if len(dm_ind[b:e]) < 10:
            continue

        # Store this halo
        results[ihalo] = Halo(tictoc, dm_ind[b:e],
                              (grpid[ihalo - halo_offset],
                               subgrpid[ihalo - halo_offset]),
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
