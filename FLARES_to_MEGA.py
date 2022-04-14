import os
import sys

import h5py
import numpy as np
from eagle_IO import eagle_IO as E

from core.halo import Halo
from core.serial_io import write_data
import core.param_utils as p_utils
from core.timing import TicToc


def get_data(ii, tag, inp="FLARES"):
    num = str(ii)
    if inp == "FLARES":
        if len(num) == 1:
            num = "0" + num

        sim = rF"/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/" \
              rF"FLARES_{num}_sp_info.hdf5"

    else:
        sim = rF"/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/" \
              rF"EAGLE_{inp}_sp_info.hdf5"

    # Get dark matter datasets
    with h5py.File(sim, "r") as hf:
        dm_len = np.array(hf[tag + "/Galaxy"].get("DM_Length"),
                          dtype=np.int64)
        grpid = np.array(hf[tag + "/Galaxy"].get("GroupNumber"),
                         dtype=np.int64)
        subgrpid = np.array(hf[tag + "/Galaxy"].get("SubGroupNumber"),
                            dtype=np.int64)
        dm_pid = np.array(hf[tag + "/Particle"].get("DM_ID"),
                          dtype=np.int64)
        dm_ind = np.array(hf[tag + "/Particle"].get("DM_Index"),
                          dtype=np.int64)
        dm_pos = np.array(hf[tag + "/Particle"].get("DM_Coordinates"),
                          dtype=np.int64)
        dm_vel = np.array(hf[tag + "/Particle"].get("DM_Vel"),
                          dtype=np.int64)
        subgrp_dm_mass = np.array(hf[tag + "/Galaxy"].get("Mdm"),
                                  dtype=np.float64) * 10 ** 10
        part_dm_mass = subgrp_dm_mass / dm_len
        dm_masses = np.full(dm_pid.size, part_dm_mass[0], dtype=np.float64)

    # Create pointer arrays
    dmbegin = np.zeros(len(dm_len), dtype=np.int64)
    dmbegin[1:] = np.cumsum(dm_len)[:-1]
    dmend = np.cumsum(dm_len)
    
    dm_snap_part_ids = E.read_array("SNAP", 
                                    "/cosma/home/dp004/dc-rope1/FLARES/"
                                    "FLARES-1/G-EAGLE_" + str(reg) + "/data",
                                     tag, "PartType1/ParticleIDs",
                                     numThreads=8)

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

    print("Running on Region %s and snap %s" % (reg, snap))

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

    # Instantiate timer
    tictoc = TicToc(meta)
    tictoc.start()
    meta.tictoc = tictoc

    # Get the particle data for all particle types in the current snapshot
    (dm_len, grpid, subgrpid, dm_pid, dm_ind, dmbegin, dmend, dm_pos, dm_vel,
     dm_masses, dm_snap_part_ids) = get_data(reg, snap, inp="FLARES")
    print(dm_pos.shape, dm_vel.shape, dm_masses.shape)
    # Set npart
    meta.npart[1] = dm_snap_part_ids.size

    print("Npart: %d" % meta.npart[1])

    # Define part type array
    dm_part_types = np.full_like(dm_pid, 1)

    # Initialise dictionary for mega halo objects
    results = {}

    # Loop over galaxies and create mega objects
    newPhaseID = 0
    for (ihalo, b), l in zip(enumerate(dmbegin), dmend):

        print(ihalo, end="\r")
        
        # Compute end
        e = b + l

        # Store this halo
        results[ihalo] = Halo(tictoc, dm_ind[b:e], None, dm_pid[b:e], 
                              dm_pos[b:e, :], dm_vel[b:e, :], dm_part_types[b:e],
                              dm_masses[b:e], 10, meta)
        newPhaseID = ihalo

    # Write out file
    write_data(tictoc, meta, newPhaseID, newPhaseSubID=0,
               results_dict=results, sub_results_dict={},
               pre_sort_part_haloids=None, sim_pids=dm_snap_part_ids)
        

regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append("0" + str(reg))
    else:
        regions.append(str(reg))


if __name__ == "__main__":
    main(regions[int(sys.argv[2])])
