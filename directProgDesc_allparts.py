#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import os
import sys

import h5py
import matplotlib
import numpy as np

matplotlib.use('Agg')


def get_link_data(all_linked_halos, start_ind, nlinked_halos):
    """ A helper function for extracting a halo's linked halos
        (i.e. progenitors and descendants)

    :param all_linked_halos: Array containing all progenitors and descendants.
    :type all_linked_halos: float[N_linked halos]
    :param start_ind: The start index for this halos progenitors or descendents
                      elements in all_linked_halos
    :type start_ind: int
    :param nlinked_halos: The number of progenitors or descendents (linked halos)
                          the halo in question has
    :type nlinked_halos: int
    :return:
    """

    return all_linked_halos[start_ind: start_ind + nlinked_halos]


def dmgetLinks(current_halo_pids, prog_snap_grpIDs, prog_snap_subgrpIDs,
               desc_snap_grpIDs, desc_snap_subgrpIDs, prog_snap_part_masses,
               desc_snap_part_masses, prog_snap_part_types,
               desc_snap_part_types):

    # Set up dictionaries for the mass contributions
    prog_mass_contribution = {}
    desc_mass_contribution = {}

    # =============== Find Progenitor IDs ===============

    # If any progenitor halos exist (i.e. The current snapshot ID is not 000, enforced in the main function)
    if prog_snap_grpIDs.size != 0:

        # Find the halo IDs of the current halo's particles in the progenitor snapshot by indexing the
        # progenitor snapshot's particle halo IDs array with the halo's particle IDs, this can be done
        # since the particle halo IDs array is sorted by particle ID.
        pre_prog_grpids = prog_snap_grpIDs[current_halo_pids]
        pre_prog_subgrpids = prog_snap_subgrpIDs[current_halo_pids]
        prog_part_types = prog_snap_part_types[current_halo_pids]
        prog_part_masses = prog_snap_part_masses[current_halo_pids]

        # Combine IDs to get the unique entries from both groups and subgroups
        halo_ids = [str(grp) + "." + str(subgrp).zfill(6)
                    for grp, subgrp in zip(pre_prog_grpids,
                                           pre_prog_subgrpids)]

        # Find the unique halo IDs and the number of times each appears
        uniprog_ids, uniprog_counts = np.unique(halo_ids,
                                                       return_counts=True)

        # Remove single particle halos (ID=-2), since np.unique returns a sorted array this can be
        # done by removing the first value.
        if uniprog_ids[0] == str(-2) + "." + str(-2).zfill(6):
            uniprog_ids = uniprog_ids[1:]
            uniprog_counts = uniprog_counts[1:]

        okinds = uniprog_counts >= 10
        uniprog_ids = uniprog_ids[okinds]
        uniprog_counts = uniprog_counts[okinds]

        # Find the number of progenitor halos from the size of the unique array
        nprog = uniprog_ids.size

        # Sort the halo IDs and number of particles in each progenitor halo by their contribution to the
        # current halo (number of particles from the current halo in the progenitor or descendant)
        sorting_inds = uniprog_counts.argsort()[::-1]
        prog_ids = uniprog_ids[sorting_inds]
        prog_npart_contribution = uniprog_counts[sorting_inds]

        # Uncombine group and subgroup ids
        prog_grpids = np.full(prog_ids.size, -1)
        prog_subgrpids = np.full(prog_ids.size, -1)
        for ind, haloid in enumerate(prog_ids):
            prog_grpids[ind] = int(haloid.split(".")[0])
            prog_subgrpids[ind] = int(haloid.split(".")[-1])

        for part_type in set(prog_part_types):
            for grp, subgrp in zip(prog_grpids, prog_subgrpids):
                okinds = np.logical_and(pre_prog_grpids == grp, 
                                        pre_prog_subgrpids == subgrp)
                if prog_part_masses[okinds].size > 0:
                    prog_mass_contribution.setdefault(part_type, []).append(
                        np.sum(prog_part_masses[okinds]))
                else:
                    prog_mass_contribution.setdefault(part_type, []).append(0.)

    # If there is no progenitor store Null values
    else:
        nprog = -1
        prog_grpids = np.array([], copy=False, dtype=int)
        prog_subgrpids = np.array([], copy=False, dtype=int)
        prog_npart_contribution = np.array([], copy=False, dtype=int)

    # =============== Find Descendant IDs ===============

    # If any descendant halos exist (i.e. The current snapshot ID is not 000, enforced in the main function)
    if desc_snap_grpIDs.size != 0:

        # Find the halo IDs of the current halo's particles in the descendant snapshot by indexing the
        # descendant snapshot's particle halo IDs array with the halo's particle IDs, this can be done
        # since the particle halo IDs array is sorted by particle ID.
        pre_desc_grpids = desc_snap_grpIDs[current_halo_pids]
        pre_desc_subgrpids = desc_snap_subgrpIDs[current_halo_pids]
        desc_part_types = desc_snap_part_types[current_halo_pids]
        desc_part_masses = desc_snap_part_masses[current_halo_pids]

        # Combine IDs to get the unique entries from both groups and subgroups
        halo_ids = [str(grp) + "." + str(subgrp).zfill(6)
                    for grp, subgrp in zip(pre_desc_grpids,
                                           pre_desc_subgrpids)]

        # Find the unique halo IDs and the number of times each appears
        unidesc_ids, unidesc_counts = np.unique(halo_ids,
                                                       return_counts=True)

        # Remove single particle halos (ID=-2), since np.unique returns a sorted array this can be
        # done by removing the first value.
        if unidesc_ids[0] == str(-2) + "." + str(-2).zfill(6):
            unidesc_ids = unidesc_ids[1:]
            unidesc_counts = unidesc_counts[1:]

        okinds = unidesc_counts >= 10
        unidesc_ids = unidesc_ids[okinds]
        unidesc_counts = unidesc_counts[okinds]

        # Find the number of descendant halos from the size of the unique array
        ndesc = unidesc_ids.size

        # Sort the halo IDs and number of particles in each descendant halo by their contribution to the
        # current halo (number of particles from the current halo in the descendant or descendant)
        sorting_inds = unidesc_counts.argsort()[::-1]
        desc_ids = unidesc_ids[sorting_inds]
        desc_npart_contribution = unidesc_counts[sorting_inds]

        # Uncombine group and subgroup ids
        desc_grpids = np.full(desc_ids.size, -1)
        desc_subgrpids = np.full(desc_ids.size, -1)
        for ind, haloid in enumerate(desc_ids):
            desc_grpids[ind] = int(haloid.split(".")[0])
            desc_subgrpids[ind] = int(haloid.split(".")[-1])

        for part_type in set(desc_part_types):
            for grp, subgrp in zip(desc_grpids, desc_subgrpids):
                okinds = np.logical_and(pre_desc_grpids == grp, 
                                        pre_desc_subgrpids == subgrp)
                if desc_part_masses[okinds].size > 0:
                    desc_mass_contribution.setdefault(part_type, []).append(
                        np.sum(desc_part_masses[okinds]))
                else:
                    desc_mass_contribution.setdefault(part_type, []).append(0.)

    # If there is no descendant store Null values
    else:
        ndesc = -1
        desc_grpids = np.array([], copy=False, dtype=int)
        desc_subgrpids = np.array([], copy=False, dtype=int)
        desc_npart_contribution = np.array([], copy=False, dtype=int)

    return (nprog, prog_grpids, prog_subgrpids, prog_npart_contribution,
            prog_mass_contribution,
            ndesc, desc_grpids, desc_subgrpids, desc_npart_contribution,
            desc_mass_contribution,
            current_halo_pids)


def get_current_part_ind_dict(begins, lengths, part_ids, sinds, grp_ids,
                              sub_grpids):
    subhalo_id_part_inds = {}
    halo_id_part_inds = {}
    for b, l, grp, subgrp in zip(begins, lengths, grp_ids, sub_grpids):
        subhalo_id_part_inds.setdefault((grp, subgrp), set()).update(
            set(part_ids[b: b + l]))
        halo_id_part_inds.setdefault((grp, subgrp), set()).update(
            set(sinds[b: b + l]))
    return subhalo_id_part_inds, halo_id_part_inds


def get_progdesc_part_ind_dict(reg, snap):
    # Get the particle data for all particle types in the current snapshot
    (s_len, g_len, dm_len, grpid, subgrpid, s_pid, g_pid, dm_pid,
     S_mass, G_mass, DM_mass, sbegin, send,
     gbegin, gend, dmbegin, dmend, g_gal_mass_dict, dm_gal_mass_dict, 
     s_gal_mass_dict) = get_data(reg, snap, inp='FLARES')
    
    gal_masses = {}
    gal_masses[0] = g_gal_mass_dict
    gal_masses[1] = dm_gal_mass_dict
    gal_masses[4] = s_gal_mass_dict

    gas_part_types = np.full_like(g_pid, 0)
    dm_part_types = np.full_like(dm_pid, 1)
    star_part_types = np.full_like(s_pid, 4)
    # bh_part_types = np.full_like(bh_snap_part_ids, 5)

    # part_ids = np.concatenate([g_pid, dm_pid, s_pid, bh_pid])
    # part_types = np.concatenate([gas_part_types, dm_part_types,
    #                              star_part_types, bh_part_types])

    # Combine particle pids and types into a single array
    part_ids = np.concatenate([g_pid, dm_pid, s_pid])
    part_masses = np.concatenate([G_mass, DM_mass, S_mass])
    part_types = np.concatenate([gas_part_types, dm_part_types,
                                 star_part_types])

    # We need repeats of the group and subgroup arrays for the
    # single particle type arrays
    multi_part_grpids = np.concatenate([grpid, grpid, grpid])
    multi_part_subgrpids = np.concatenate([subgrpid, subgrpid, subgrpid])

    # Combine the pointer arrays to reference the single large array
    lengths = np.concatenate([g_len, dm_len, s_len])
    begins = []
    begins.extend(gbegin)
    offset = begins[-1] + g_len[-1]
    for b in dmbegin:
        begins.append(b + offset)
    offset = begins[-1] + dm_len[-1]
    for b in sbegin:
        begins.append(b + offset)

    # Get the indices which sort the particle IDs
    sinds = np.argsort(part_ids)

    tup = get_current_part_ind_dict(begins, lengths, part_ids, sinds,
                                    multi_part_grpids, multi_part_subgrpids)
    subhalo_id_part_inds, halo_id_part_inds = tup

    snap_grpIDs = np.full(len(part_ids), -2, dtype=float)
    snap_subgrpIDs = np.full(len(part_ids), -2, dtype=float)
    for key in halo_id_part_inds:
        pinds = list(halo_id_part_inds[key])
        if len(pinds) < 20:
            continue
        snap_grpIDs[pinds] = key[0]
        snap_subgrpIDs[pinds] = key[1]

    return snap_grpIDs, snap_subgrpIDs, part_types, part_masses, gal_masses


def get_data(ii, tag, inp='FLARES'):
    num = str(ii)
    if inp == 'FLARES':
        if len(num) == 1:
            num = '0' + num

        sim = rF"/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/" \
              rF"FLARES_{num}_sp_info.hdf5"

    else:
        sim = rF"/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/" \
              rF"EAGLE_{inp}_sp_info.hdf5"

    with h5py.File(sim, 'r') as hf:
        try:
            s_len = np.array(hf[tag + '/Galaxy'].get('S_Length'),
                             dtype=np.int64)
            s_gal_mass = np.array(hf[tag + '/Galaxy'].get('Mstar'),
                             dtype=np.int64)
        except ValueError:
            s_len = np.array([], dtype=np.int64)
            s_gal_mass = np.array([], dtype=np.float64)
        g_len = hf[tag + '/Galaxy'].get('G_Length')
        g_gal_mass = hf[tag + '/Galaxy'].get('Mgas')
        dm_len = hf[tag + '/Galaxy'].get('DM_Length')
        dm_gal_mass = hf[tag + '/Galaxy'].get('Mdm')

        grpid = np.array(hf[tag + '/Galaxy'].get('GroupNumber'),
                         dtype=np.int64)
        subgrpid = np.array(hf[tag + '/Galaxy'].get('SubGroupNumber'),
                            dtype=np.int64)

        s_pid = np.array(hf[tag + '/Particle'].get('S_ID'),
                         dtype=np.int64)
        g_pid = np.array(hf[tag + '/Particle'].get('G_ID'),
                         dtype=np.int64)
        dm_pid = np.array(hf[tag + '/Particle'].get('DM_ID'),
                          dtype=np.int64)

        subgrp_dm_mass = np.array(hf[tag + '/Galaxy'].get('Mdm'),
                                  dtype=np.float64) * 10 ** 10

        part_dm_mass = subgrp_dm_mass / dm_len

        S_mass = np.array(hf[tag + '/Particle'].get('S_Mass'),
                          dtype=np.float64) * 10 ** 10
        G_mass = np.array(hf[tag + '/Particle'].get('G_Mass'),
                          dtype=np.float64) * 10 ** 10
        DM_mass = np.full(dm_pid.size, part_dm_mass[0], dtype=np.float64)

        sbegin = np.zeros(len(s_len), dtype=np.int64)
        sbegin[1:] = np.cumsum(s_len)[:-1]
        send = np.cumsum(s_len)

        gbegin = np.zeros(len(s_len), dtype=np.int64)
        gbegin[1:] = np.cumsum(s_len)[:-1]
        gend = np.cumsum(s_len)

        dmbegin = np.zeros(len(s_len), dtype=np.int64)
        dmbegin[1:] = np.cumsum(s_len)[:-1]
        dmend = np.cumsum(s_len)

        g_gal_mass_dict = dict(zip(zip(grpid, subgrpid), g_gal_mass))
        dm_gal_mass_dict = dict(zip(zip(grpid, subgrpid), dm_gal_mass))
        s_gal_mass_dict = dict(zip(zip(grpid, subgrpid), s_gal_mass))

    return (s_len, g_len, dm_len, grpid, subgrpid, s_pid, g_pid, dm_pid,
            S_mass, G_mass, DM_mass,
            sbegin, send, gbegin, gend, dmbegin, dmend,
            g_gal_mass_dict, dm_gal_mass_dict, s_gal_mass_dict)


def partDirectProgDesc(reg, snap, prog_snap, desc_snap):
    """ A function which cycles through all halos in a snapshot finding and writing out the
    direct progenitor and descendant data.

    :param snapshot: The snapshot ID.
    :param halopath: The filepath to the halo finder HDF5 file.
    :param savepath: The filepath to the directory where the Merger Graph should be written out to.
    :param part_threshold: The mass (number of particles) threshold defining a halo.

    :return: None
    """

    # Get the particle data for all particle types in the current snapshot
    (s_len, g_len, dm_len, grpid, subgrpid, s_pid, g_pid, dm_pid,
     S_mass, G_mass, DM_mass, sbegin, send,
     gbegin, gend, dmbegin, dmend, g_gal_mass_dict, dm_gal_mass_dict, 
     s_gal_mass_dict) = get_data(reg, snap, inp='FLARES')

    gas_part_types = np.full_like(g_pid, 0)
    dm_part_types = np.full_like(dm_pid, 1)
    star_part_types = np.full_like(s_pid, 4)
    # bh_part_types = np.full_like(bh_snap_part_ids, 5)

    # part_ids = np.concatenate([g_pid, dm_pid, s_pid, bh_pid])
    # part_types = np.concatenate([gas_part_types, dm_part_types,
    #                              star_part_types, bh_part_types])

    # Combine particle pids and types into a single array
    part_ids = np.concatenate([g_pid, dm_pid, s_pid])
    part_types = np.concatenate([gas_part_types, dm_part_types,
                                 star_part_types])

    # We need repeats of the group and subgroup arrays for the
    # single particle type arrays
    multi_part_grpids = np.concatenate([grpid, grpid, grpid])
    multi_part_subgrpids = np.concatenate([subgrpid, subgrpid, subgrpid])

    # Combine the pointer arrays to reference the single large array
    print(g_len, dm_len, s_len)
    lengths = np.concatenate([g_len, dm_len, s_len])
    begins = []
    begins.extend(gbegin)
    offset = begins[-1] + g_len[-1]
    for b in dmbegin:
        begins.append(b + offset)
    offset = begins[-1] + dm_len[-1]
    for b in sbegin:
        begins.append(b + offset)

    # Get the indices which sort the particle IDs
    sinds = np.argsort(part_ids)

    # =============== Current Snapshot ===============

    tup = get_current_part_ind_dict(begins, lengths, part_ids, sinds,
                                    multi_part_grpids, multi_part_subgrpids)
    subhalo_id_part_inds, halo_id_part_inds = tup

    # =============== Progenitor Snapshot ===============

    # Only look for progenitor data if there is a descendant snapshot
    if prog_snap != None:

        res = get_progdesc_part_ind_dict(reg, prog_snap)
        (prog_snap_grpIDs, prog_snap_subgrpIDs, prog_part_types, 
         prog_part_masses, prog_gal_masses) = res

    else:  # Assign an empty array if the snapshot is less than the earliest (000)
        prog_snap_grpIDs = np.array([], copy=False)
        prog_snap_subgrpIDs = np.array([], copy=False)
        prog_part_types = np.array([], copy=False)
        prog_part_masses = np.array([], copy=False)
        prog_gal_masses = {pt: np.array([], copy=False) for pt in [0, 1, 4]}

    # =============== Descendant Snapshot ===============

    # Only look for descendant data if there is a descendant snapshot
    if desc_snap != None:

        res = get_progdesc_part_ind_dict(reg, desc_snap)
        (desc_snap_grpIDs, desc_snap_subgrpIDs, desc_part_types, 
         desc_part_masses, desc_gal_masses) = res

    else:  # Assign an empty array if the snapshot is less than the earliest (000)
        desc_snap_grpIDs = np.array([], copy=False)
        desc_snap_subgrpIDs = np.array([], copy=False)
        desc_part_types = np.array([], copy=False)
        desc_part_masses = np.array([], copy=False)
        desc_gal_masses = {pt: np.array([], copy=False) for pt in [0, 1, 4]}

    # =============== Find all Direct Progenitors And Descendant Of Halos In This Snapshot ===============

    results = {}

    # Loop through all the halos in this snapshot
    for num, haloID in enumerate(halo_id_part_inds.keys()):

        if len(halo_id_part_inds[haloID]) < 20:
            continue

        # =============== Current Halo ===============

        current_halo_pids = np.array(list(halo_id_part_inds[haloID]))

        # =============== Run The Direct Progenitor and Descendant Finder ===============

        # Run the progenitor/descendant finder
        results[haloID] = dmgetLinks(current_halo_pids,
                                     prog_snap_grpIDs,
                                     prog_snap_subgrpIDs,
                                     desc_snap_grpIDs,
                                     desc_snap_subgrpIDs, 
                                     prog_part_masses,
                                     desc_part_masses, 
                                     prog_part_types,
                                     desc_part_types)

    print('Processed', len(results.keys()), 'halos in snapshot', snap)

    return (results, part_ids, halo_id_part_inds, 
            prog_gal_masses, desc_gal_masses)


def get_int_sim_ids(simhaloID):
    grp = int(simhaloID)
    subgrp = str(simhaloID).split(".")[1]
    if len(subgrp) != 5:
        pad = 5 - len(subgrp)
        for i in range(pad):
            subgrp += "0"

    return int(grp), int(subgrp)


def write_hdf5(hdf, key, data):
    hdf.create_dataset(key, shape=data.shape,  dtype=data.dtype,
                       data=data, compression='gzip')
    print(key, "written data of shape", data.shape, "and type", data.dtype)


def mainDirectProgDesc(reg, snap, prog_snap, desc_snap,
                       savepath='MergerGraphs/', part_types=(0, 1, 4)):
    
    # Get the graph links based on the dark matter
    res = partDirectProgDesc(reg, snap, prog_snap, desc_snap)
    results, part_ids, part_inds, prog_gal_masses, desc_gal_masses = res

    # Set up arrays to store host results
    nhalo = len(results.keys())
    sim_grp_haloids = np.full(nhalo, -2, dtype=float)
    sim_subgrp_haloids = np.full(nhalo, -2, dtype=float)
    halo_nparts = np.full(nhalo, -2, dtype=int)
    nprogs = np.full(nhalo, -2, dtype=int)
    ndescs = np.full(nhalo, -2, dtype=int)
    prog_start_index = np.full(nhalo, -2, dtype=int)
    desc_start_index = np.full(nhalo, -2, dtype=int)

    progs_grp = []
    descs_grp = []
    progs_subgrp = []
    descs_subgrp = []
    prog_npart_conts = []
    desc_npart_conts = []
    prog_mass_conts = {pt: [] for pt in part_types}
    desc_mass_conts = {pt: [] for pt in part_types}
    prog_masses = {pt: [] for pt in part_types}
    desc_masses = {pt: [] for pt in part_types}

    for num, simhaloID in enumerate(results.keys()):

        grp, subgrp = simhaloID.split(".")
        sim_grp_haloids[num] = grp
        sim_subgrp_haloids[num] = subgrp

        haloID = num

        (nprog, prog_grpids, prog_subgrpids, prog_npart_contribution,
         prog_mass_contribution,
         ndesc, desc_grpids, desc_subgrpids, desc_npart_contribution,
         desc_mass_contribution,
         current_halo_pids) = results[simhaloID]

        if nprog > 0:
            prog_start_index[haloID] = len(progs_grp)
            progs_grp.extend(prog_grpids)
            progs_subgrp.extend(prog_subgrpids)
            prog_npart_conts.extend(prog_npart_contribution)
            for pt in part_types:
                prog_mass_conts[pt].extend(prog_mass_contribution[pt])
                for pgrp, psubgrp in zip(prog_grpids, prog_subgrpids):
                    prog_masses[pt].append(prog_gal_masses[(pgrp, psubgrp)])
        else:
            prog_start_index[haloID] = 2 ** 30

        if ndesc > 0:
            desc_start_index[haloID] = len(descs_grp)
            descs_grp.extend(desc_grpids)
            descs_subgrp.extend(desc_subgrpids)
            desc_npart_conts.extend(desc_npart_contribution)
            for pt in part_types:
                desc_mass_conts[pt].extend(desc_mass_contribution[pt])
                for dgrp, dsubgrp in zip(desc_grpids, desc_subgrpids):
                    desc_masses[pt].append(desc_gal_masses[(dgrp, dsubgrp)])
        else:
            desc_start_index[haloID] = 2 ** 30

        # Write out the data produced
        nprogs[haloID] = nprog  # number of progenitors
        ndescs[haloID] = ndesc  # number of descendants
        halo_nparts[int(haloID)] = current_halo_pids.size  # mass of the halo

    progs_grp = np.array(progs_grp)
    descs_grp = np.array(descs_grp)
    progs_subgrp = np.array(progs_subgrp)
    descs_subgrp = np.array(descs_subgrp)
    prog_mass_conts = np.array(prog_mass_conts)
    desc_mass_conts = np.array(desc_mass_conts)
    for pt in part_types:
        prog_mass_conts[pt] = np.array(prog_mass_conts[pt])
        desc_mass_conts[pt] = np.array(desc_mass_conts[pt])
        prog_masses[pt] = np.array(prog_masses[pt])
        desc_masses[pt] = np.array(desc_masses[pt])

    # Create file to store this snapshots graph results
    hdf = h5py.File(savepath + 'SubMgraph_' + snap + '.hdf5', 'w')

    write_hdf5(hdf, 'SUBFIND_Group_IDs', sim_grp_haloids)
    write_hdf5(hdf, 'SUBFIND_SubGroup_IDs', sim_subgrp_haloids)
    write_hdf5(hdf, 'nProgs', nprogs)
    write_hdf5(hdf, 'nDescs', ndescs)
    write_hdf5(hdf, 'nParts', halo_nparts)
    write_hdf5(hdf, 'Prog_Start_Index', prog_start_index)
    write_hdf5(hdf, 'Desc_Start_Index', desc_start_index)

    write_hdf5(hdf, 'prog_group_ids', progs_grp)
    write_hdf5(hdf, 'desc_group_ids', descs_grp)
    write_hdf5(hdf, 'prog_subgroup_ids', progs_subgrp)
    write_hdf5(hdf, 'desc_subgroup_ids', descs_subgrp)
    write_hdf5(hdf, 'prog_npart_contribution', prog_mass_conts)
    write_hdf5(hdf, 'desc_npart_contribution', desc_mass_conts)

    print("Progs", np.unique(nprogs[sim_grp_haloids >= 0], return_counts=True))
    print("Descs", np.unique(ndescs[sim_grp_haloids >= 0], return_counts=True))

    for pt in part_types:

        write_hdf5(hdf, 'prog_parttype' + str(pt) + '_mass_contribution',
                   prog_mass_conts[pt])
        write_hdf5(hdf, 'desc_parttype' + str(pt) + '_mass_contribution',
                   desc_mass_conts[pt])
        write_hdf5(hdf, 'prog_parttype' + str(pt) + '_masses',
                   prog_masses[pt])
        write_hdf5(hdf, 'desc_parttype' + str(pt) + '_masses',
                   desc_masses[pt])

    hdf.close()


regions = []
for reg in range(0, 40):
    if reg < 10:
        regions.append('0' + str(reg))
    else:
        regions.append(str(reg))

snaps = ['000_z015p000', '001_z014p000', '002_z013p000', '003_z012p000',
         '004_z011p000', '005_z010p000',
         '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000',
         '010_z005p000', '011_z004p770']
prog_snaps = [None, '000_z015p000', '001_z014p000', '002_z013p000',
              '003_z012p000', '004_z011p000', '005_z010p000',
              '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000',
              '010_z005p000']
desc_snaps = ['001_z014p000', '002_z013p000', '003_z012p000', '004_z011p000',
              '005_z010p000',
              '006_z009p000', '007_z008p000', '008_z007p000', '009_z006p000',
              '010_z005p000', '011_z004p770', None]

if __name__ == '__main__':

    reg_snaps = []
    for reg in reversed(regions):

        try:
            os.mkdir(
                '/cosma7/data/dp004/FLARES/FLARES-1/MergerGraphs/FLARES_' + reg)
        except OSError:
            pass

        for snap, prog_snap, desc_snap in zip(snaps, prog_snaps, desc_snaps):
            reg_snaps.append((reg, prog_snap, snap, desc_snap))

    ind = int(sys.argv[1])
    print(ind)
    print(reg_snaps[ind])

    mainDirectProgDesc(reg=reg_snaps[ind][0], snap=reg_snaps[ind][2],
                       prog_snap=reg_snaps[ind][1],
                       desc_snap=reg_snaps[ind][3],
                       savepath='/cosma7/data/dp004/FLARES/FLARES-1/'
                                'MergerGraphs/FLARES_'
                                + reg_snaps[ind][0] + '/')
