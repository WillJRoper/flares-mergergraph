import sys

import h5py
import numpy as np
from eagle_IO import eagle_IO as E
import mpi4py
from mpi4py import MPI

from core.halo import Halo
from core.serial_io import write_data
import core.param_utils as p_utils
from core.timing import TicToc
from core.collect_result import collect_halos
from core.talking_utils import say_hello
from core.talking_utils import message
import core.utilities as utils


mpi4py.rc.recv_mprobe = False

# Initializations and preliminaries
comm = MPI.COMM_WORLD  # get MPI communicator object
size = comm.size  # total number of processes
rank = comm.rank  # rank of this process
status = MPI.Status()  # get MPI status object


def get_data(reg, tag):

    # Define sim path
    sim_path = "/cosma/home/dp004/dc-rope1/FLARES/FLARES-1/" \
               "G-EAGLE_" + reg + "/data/"

    # Read in all the relevant data
    part_ids = E.read_array("SNAP", sim_path, tag,
                            "PartType1/ParticleIDs", numThreads=8)
    part_grp_ids = E.read_array("SNAP", sim_path, tag,
                                "PartType1/GroupNumber", numThreads=8)
    part_subgrp_ids = E.read_array("SNAP", sim_path, tag,
                                "PartType1/SubGroupNumber", numThreads=8)
    part_pos = E.read_array("SNAP", sim_path, tag,
                                "PartType1/Coordinates", numThreads=8)
    part_vel = E.read_array("SNAP", sim_path, tag,
                                "PartType1/Velocity", numThreads=8)

    # Get the number of particles we are dealing with
    npart = part_ids.size

    # Define master file path so we can simply get the DM part mass
    master_path = rF"/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/" \
                  rF"data/FLARES_{reg}_sp_info.hdf5"

    # Get master file data and compute the DM particle mass
    with h5py.File(master_path, "r") as hf:
        dm_len = np.array(hf[tag + "/Galaxy"].get("DM_Length"),
                          dtype=np.int64)
        subgrp_dm_mass = np.array(hf[tag + "/Galaxy"].get("Mdm"),
                                  dtype=np.float64) * 10 ** 10
        part_dm_mass = subgrp_dm_mass / dm_len

    # Let's bin the particles and split the work up
    rank_bins = np.linspace(0, npart, size + 1, dtype=int)

    # Initialise dictionary to store sorted particles
    halos = {"length": {}, "grpid": {}, "subgrpid": {}, "dm_pid": {},
             "dm_ind": {}, "dm_pos": {}, "dm_vel": {}, "dm_masses": {}}

    # Define the NULL value in GADGET files
    null = 2**30

    # Loop over the particles on this rank
    for ind in range(rank_bins[rank], rank_bins[rank + 1]):
        print(part_subgrp_ids[ind])
        # Is this particle in a subgroup?
        if part_subgrp_ids[ind] == null:
            continue


    # Create pointer arrays
    dmbegin = np.zeros(len(dm_len), dtype=np.int64)
    dmbegin[1:] = np.cumsum(dm_len)[:-1]
    dmend = np.cumsum(dm_len)
    
    dm_snap_part_ids = E.read_array("SNAP", sim_path, tag,
                                    "PartType1/ParticleIDs", numThreads=8)

    return (dm_len, grpid, subgrpid, dm_pid, dm_ind, dmbegin, dmend,
            dm_pos, dm_vel, dm_masses, dm_snap_part_ids)


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
                            npart=[0, 10**7, 0, 0, 0, 0], z=z, tot_mass=10**13)

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
    (dm_len, grpid, subgrpid, dm_pid, dm_ind, dmbegin, dmend, dm_pos, dm_vel,
     dm_masses, dm_snap_part_ids) = get_data(reg, snap)

    # Set npart
    meta.npart[1] = dm_snap_part_ids.size

    if rank == 0:
        message(rank, "Npart: %d ~ %d^3" % (meta.npart[1],
                                        int(meta.npart[1] ** (1/3))))
        message(rank, "Nhalo: %d" % len(dmbegin))

    # Define part type array
    dm_part_types = np.full_like(dm_pid, 1)

    # Initialise dictionary for mega halo objects
    results = {}

    # Split up galaxies across nodes
    if len(dmbegin) > meta.nranks:
        rank_halobins = np.linspace(0, len(dmbegin), size + 1, dtype=int)
    else:
        rank_halobins = []
        for halo in range(len(dmbegin)):
            rank_halobins.append()

    # Loop over galaxies and create mega objects
    ihalo = rank_halobins[rank]
    for b, l in zip(dmbegin[rank_halobins[rank]: rank_halobins[rank + 1]],
                    dmend[rank_halobins[rank]: rank_halobins[rank + 1]]):
        
        # Compute end
        e = b + l

        # Store this halo
        results[ihalo] = Halo(tictoc, dm_ind[b:e], None, dm_pid[b:e], 
                              dm_pos[b:e, :], dm_vel[b:e, :], dm_part_types[b:e],
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
