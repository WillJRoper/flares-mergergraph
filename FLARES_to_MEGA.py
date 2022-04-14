import sys

import core.param_utils as p_utils
import core.utilities as utils
from core.collect_result import collect_halos
from core.halo import Halo
from core.serial_io import write_data
from core.talking_utils import message
from core.talking_utils import say_hello
from core.timing import TicToc
from core.timing import timer

from mpi4py import MPI
import h5py
import mpi4py
import numpy as np

from eagle_IO import eagle_IO as E

mpi4py.rc.recv_mprobe = False

# Initializations and preliminaries
comm = MPI.COMM_WORLD  # get MPI communicator object
size = comm.size  # total number of processes
rank = comm.rank  # rank of this process
status = MPI.Status()  # get MPI status object


@timer("Reading")
def get_data(tictoc, reg, tag):
    # Define sim path
    sim_path = "/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/" \
               "G-EAGLE_" + reg + "/data/"

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
                            "PartType1/Coordinates", numThreads=8)
    part_vel = E.read_array("PARTDATA", sim_path, tag,
                            "PartType1/Velocity", numThreads=8)

    # Get the number of particles we are dealing with
    npart = part_ids.size
    true_npart = true_part_ids.size

    # Define master file path so we can simply get the DM part mass
    master_path = rF"/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/" \
                  rF"data/FLARES_{reg}_sp_info.hdf5"

    # Get master file data and compute the DM particle mass
    with h5py.File(master_path, "r") as hf:
        temp_dm_len = np.array(hf[tag + "/Galaxy"].get("DM_Length"),
                          dtype=np.int64)
        subgrp_dm_mass = np.array(hf[tag + "/Galaxy"].get("Mdm"),
                                  dtype=np.float64) * 10 ** 10
        part_dm_mass = subgrp_dm_mass / temp_dm_len

    # Let's bin the particles and split the work up
    rank_bins = np.linspace(0, npart, size + 1, dtype=int)

    # Initialise dictionary to store sorted particles
    halos = {"length": {}, "dm_pid": {},
             "dm_ind": {}, "dm_pos": {}, "dm_vel": {}, "dm_masses": {}}

    # Loop over the particles on this rank
    for ind in range(rank_bins[rank], rank_bins[rank + 1]):

        # Is this particle in a subgroup?
        if part_subgrp_ids[ind] == null:
            continue

        # Define this halo's key
        key = (part_grp_ids[ind], part_subgrp_ids[ind])

        # Add this particle to the halo
        halos["length"].setdefault(key, 0)
        halos["length"][key] += 1
        halos["dm_pid"].setdefault(key, []).append(part_ids)
        halos["dm_ind"].setdefault(key, []).append(ind)
        halos["dm_pos"].setdefault(key, []).append(part_pos[ind, :])
        halos["dm_vel"].setdefault(key, []).append(part_vel[ind, :])
        halos["dm_masses"].setdefault(key, []).append(part_dm_mass)

    # Need collect on master
    all_halos = comm.gather(halos, root=0)
    if rank == 0:

        # Loop over halos from other ranks
        for r, d in enumerate(all_halos):
            if r == 0:
                continue

            # Loop over halos
            for key in d["length"]:

                # Add this particle to the halo
                halos["length"].setdefault(key, 0)
                halos["length"][key] += d["length"][key]
                halos["dm_pid"].setdefault(key, []).extend(d["dm_pid"][key])
                halos["dm_ind"].setdefault(key, []).extend(d["dm_ind"][key])
                halos["dm_pos"].setdefault(key, []).extend(d["dm_pos"][key])
                halos["dm_vel"].setdefault(key, []).extend(d["dm_vel"][key])
                halos["dm_masses"].setdefault(key,
                                              []).extend(d["dm_masses"][key])

        # Now we can sort our halos
        keys = halos["length"].keys()
        vals = halos["length"].values()
        keys = np.array(list(keys), dtype=object)
        vals = np.array(list(vals), dtype=int)
        sinds = np.argsort(vals)
        keys = keys[sinds, :]
        
        # Define dictionary holding the sorted results
        sorted_halos = {"dm_begin": np.zeros(keys.shape[0], dtype=int),
                        "dm_len": np.zeros(keys.shape[0], dtype=int),
                        "grpid": np.zeros(keys.shape[0], dtype=int),
                        "subgrpid": np.zeros(keys.shape[0], dtype=int),
                        "dm_pid": [], "dm_ind": [], "dm_pos": [], 
                        "dm_vel": [], "dm_masses": []}
        
        # Loop over keys storing their results
        for ihalo, key in enumerate(keys):
            
            # Get group and subgroup ID
            key = tuple(key)
            grp, subgrp = key[0], key[1]
            
            # Store data
            sorted_halos["dm_begin"][ihalo] = len(sorted_halos["dm_pid"])
            sorted_halos["dm_len"][ihalo] = halos["length"][key]
            sorted_halos["grpid"][ihalo] = grp
            sorted_halos["subgrpid"][ihalo] = subgrp
            sorted_halos["dm_pid"].extend(halos["dm_pid"][key])
            sorted_halos["dm_ind"].extend(halos["dm_ind"][key])
            sorted_halos["dm_pos"].extend(halos["dm_pos"][key])
            sorted_halos["dm_vel"].extend(halos["dm_vel"][key])
            sorted_halos["dm_masses"].extend(halos["dm_masses"][key])

    else:
        sorted_halos = None
            
    # Lets broadcast what we've combined
    sorted_halos = comm.bcast(sorted_halos, root=0)

    return (sorted_halos["dm_len"], sorted_halos["grpid"],
            sorted_halos["subgrpid"], sorted_halos["dm_pid"],
            sorted_halos["dm_ind"], sorted_halos["dm_begin"],
            sorted_halos["dm_pos"], sorted_halos["dm_vel"],
            sorted_halos["dm_masses"], part_ids, true_npart)


def main(reg):
    # Read the parameter file
    paramfile = sys.argv[1]
    (inputs, flags, params, cosmology,
     simulation) = p_utils.read_param(paramfile)

    snap_ind = int(sys.argv[3])

    # Load the snapshot list
    snaplist = ["000_z015p000", "001_z014p000", "002_z013p000", "003_z012p000",
                "004_z011p000", "005_z010p000", "006_z009p000", "007_z008p000",
                "008_z007p000", "009_z006p000", "010_z005p000", "011_z004p770"]

    # Get snapshot
    snap = snaplist[snap_ind]

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
    (dm_len, grpid, subgrpid, dm_pid, dm_ind, dmbegin, dm_pos, dm_vel,
     dm_masses, dm_snap_part_ids, true_npart) = get_data(tictoc, reg, snap)

    # Set npart
    meta.npart[1] = dm_snap_part_ids.size

    if rank == 0:
        message(rank, "Npart: %d ~ %d^3" % (true_npart,
                                            int(true_npart ** (1 / 3))))
        message(rank, "Nhalo: %d" % len(dmbegin))

    # Define part type array
    dm_part_types = np.full_like(dm_pid, 1)

    # Initialise dictionary for mega halo objects
    results = {}

    # Split up galaxies across nodes
    if len(dmbegin) > meta.nranks:
        rank_halobins = np.linspace(0, len(dmbegin), size + 1, dtype=int)
    else:  # handle the case where there are less halos than ranks
        rank_halobins = []
        for halo in range(len(dmbegin) + 1):
            rank_halobins.append(halo)
        while len(rank_halobins) < size + 1:
            rank_halobins.append(0)

    # Loop over galaxies and create mega objects
    ihalo = rank_halobins[rank]
    for b, l in zip(dmbegin[rank_halobins[rank]: rank_halobins[rank + 1]],
                    dm_len[rank_halobins[rank]: rank_halobins[rank + 1]]):
        # Compute end
        e = b + l

        # Store this halo
        results[ihalo] = Halo(tictoc, dm_ind[b:e], None, dm_pid[b:e],
                              dm_pos[b:e, :], dm_vel[b:e, :],
                              dm_part_types[b:e],
                              dm_masses[b:e], 10, meta)
        results[ihalo].memory = utils.get_size(results[ihalo])
        ihalo += 1

    # Collect child process results
    tictoc.start_func_time("Collecting")
    collected_results = comm.gather(results, root=0)
    tictoc.stop_func_time()

    if rank == 0:

        # Lets combine all the halos we have collected from the other ranks
        res_tup = collect_halos(tictoc, meta, collected_results,
                                [{}, ] * size)
        (newPhaseID, newPhaseSubID, results_dict, haloID_dict,
         sub_results_dict, subhaloID_dict, phase_part_haloids) = res_tup

        if meta.verbose:
            tictoc.report("Combining results")

        # Write out file
        write_data(tictoc, meta, newPhaseID, newPhaseSubID,
                   results_dict=results_dict, sub_results_dict={},
                   sim_pids=dm_snap_part_ids)

        if meta.verbose:
            tictoc.report("Writing")

    tictoc.end()
    tictoc.end_report(comm)


regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append("0" + str(reg))
    else:
        regions.append(str(reg))

if __name__ == "__main__":
    main(regions[int(sys.argv[2])])
