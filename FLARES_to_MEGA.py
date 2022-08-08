import sys
import os

import h5py
import mpi4py
import numpy as np
import mega.core.param_utils as p_utils
import mega.core.utilities as utils
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
    single_file = sim_path.replace("<snap>", tag)
    sim_path = "/".join([s for s in single_file.split("/") if "snap" not in s])

    # Open single file and get DM particle mass
    hdf = h5py.File(single_file, "r")
    part_mass = hdf["Header"].attrs["MassTable"][1] * 10 ** 10 / meta.h
    hdf.close()

    # Define the NULL value in SUBFIND files
    null = 1073741824

    # Set up dictionary for particle ids to be stitched later
    true_part_ids = {}

    # Read in all the relevant data
    with HiddenPrints():
        true_part_ids[1] = E.read_array("SNAP", sim_path, tag,
                                        "PartType1/ParticleIDs", numThreads=8)
        part_ids = E.read_array("PARTDATA", sim_path, tag,
                                "PartType1/ParticleIDs", numThreads=8)
        part_grp_ids = E.read_array("PARTDATA", sim_path, tag,
                                    "PartType1/GroupNumber", numThreads=8)
        part_subgrp_ids = E.read_array("PARTDATA", sim_path, tag,
                                       "/PartType1/SubGroupNumber",
                                       numThreads=8)
        part_pos = E.read_array("PARTDATA", sim_path, tag,
                                "PartType1/Coordinates", numThreads=8,
                                noH=True)
        part_vel = E.read_array("PARTDATA", sim_path, tag,
                                "PartType1/Velocity", numThreads=8, noH=True,
                                physicalUnits=True)

    # Get the number of particles we are dealing with
    npart = part_ids.size

    # Let's bin the particles and split the work up
    rank_bins = np.linspace(0, npart, size + 1, dtype=int)

    # Initialise dictionaries to store sorted particles
    length_dict = {}
    pid_dict = {}
    ind_dict = {}
    posx_dict = {}
    posy_dict = {}
    posz_dict = {}
    velx_dict = {}
    vely_dict = {}
    velz_dict = {}
    masses_dict = {}
    part_types_dict = {}

    # Loop over the particles on this rank
    for ind in range(rank_bins[rank], rank_bins[rank + 1]):

        # Is this particle in a subgroup?
        if part_subgrp_ids[ind] == null or part_grp_ids[ind] == null:
            continue

        # Define this halo's key
        key = (part_grp_ids[ind], part_subgrp_ids[ind])

        # Add this particle to the halo
        length_dict.setdefault(key, 0)
        length_dict[key] += 1
        pid_dict.setdefault(key, []).append(part_ids[ind])
        ind_dict.setdefault(key, []).append(ind)
        posx_dict.setdefault(key, []).append(part_pos[ind, 0])
        velx_dict.setdefault(key, []).append(part_vel[ind, 0])
        posy_dict.setdefault(key, []).append(part_pos[ind, 1])
        vely_dict.setdefault(key, []).append(part_vel[ind, 1])
        posz_dict.setdefault(key, []).append(part_pos[ind, 2])
        velz_dict.setdefault(key, []).append(part_vel[ind, 2])
        masses_dict.setdefault(key, []).append(part_mass)
        part_types_dict.setdefault(key, []).append(1)

    for part_type in [0, 4, 5]:

        # Read in all the relevant data
        with HiddenPrints():
            try:
                true_part_ids[part_type] = E.read_array("SNAP", sim_path, tag,
                                                        "PartType%d/ParticleIDs" % part_type,
                                                        numThreads=8)
            except (ValueError, KeyError):
                true_part_ids[part_type] = np.array([])
            try:
                part_ids = E.read_array("PARTDATA", sim_path, tag,
                                        "PartType%d/ParticleIDs" % part_type,
                                        numThreads=8)
                part_grp_ids = E.read_array("PARTDATA", sim_path, tag,
                                            "PartType%d/GroupNumber" % part_type,
                                            numThreads=8)
                part_subgrp_ids = E.read_array("PARTDATA", sim_path, tag,
                                               "/PartType%d/SubGroupNumber" % part_type,
                                               numThreads=8)
                part_pos = E.read_array("PARTDATA", sim_path, tag,
                                        "PartType%d/Coordinates" % part_type,
                                        numThreads=8, noH=True)
                part_vel = E.read_array("PARTDATA", sim_path, tag,
                                        "PartType%d/Velocity" % part_type,
                                        numThreads=8, noH=True,
                                        physicalUnits=True)
                part_mass = E.read_array("PARTDATA", sim_path, tag,
                                         "PartType%d/Mass" % part_type,
                                         numThreads=8, noH=True,
                                         physicalUnits=True)
            except (ValueError, KeyError):
                part_ids = np.array([])
                part_grp_ids = np.array([])
                part_subgrp_ids = np.array([])
                part_pos = np.array([])
                part_vel = np.array([])
                part_mass = np.array([])

        # Skip particle types that are not present
        if part_ids.size == 0:
            continue

        # Get the number of particles we are dealing with
        npart = part_ids.size

        # Let's bin the particles and split the work up
        rank_bins = np.linspace(0, npart, size + 1, dtype=int)

        # Loop over the particles on this rank
        for ind in range(rank_bins[rank], rank_bins[rank + 1]):

            # Is this particle in a subgroup?
            if part_subgrp_ids[ind] == null or part_grp_ids[ind] == null:
                continue

            # Define this halo's key
            key = (part_grp_ids[ind], part_subgrp_ids[ind])

            # Add this particle to the halo
            length_dict.setdefault(key, 0)
            length_dict[key] += 1
            pid_dict.setdefault(key, []).append(part_ids[ind])
            ind_dict.setdefault(key, []).append(ind)
            posx_dict.setdefault(key, []).append(part_pos[ind, 0])
            velx_dict.setdefault(key, []).append(part_vel[ind, 0])
            posy_dict.setdefault(key, []).append(part_pos[ind, 1])
            vely_dict.setdefault(key, []).append(part_vel[ind, 1])
            posz_dict.setdefault(key, []).append(part_pos[ind, 2])
            velz_dict.setdefault(key, []).append(part_vel[ind, 2])
            masses_dict.setdefault(key, []).append(part_mass[ind])
            part_types_dict.setdefault(key, []).append(part_type)

    if rank == 0:
        # Set up lists for master
        all_length = []
        all_pid = []
        all_ind = []
        all_posx = []
        all_velx = []
        all_posy = []
        all_vely = []
        all_posz = []
        all_velz = []
        all_masses = []
        all_part_types = []

    # Communicate 1000 galaxies at a time
    ngal = np.inf
    while ngal > 0:

        proxy_length_dict = {}
        proxy_pid_dict = {}
        proxy_ind_dict = {}
        proxy_posx_dict = {}
        proxy_velx_dict = {}
        proxy_posy_dict = {}
        proxy_vely_dict = {}
        proxy_posz_dict = {}
        proxy_velz_dict = {}
        proxy_masses_dict = {}
        proxy_part_types_dict = {}

        # Define wieght of communcation
        weight = 0

        # Loop until communication is large or there's nothing to send
        while weight < 1000 and len(length_dict) > 0:

            key, val = length_dict.popitem()
            proxy_length_dict[key] = val
            key, val = pid_dict.popitem()
            proxy_pid_dict[key] = val
            key, val = ind_dict.popitem()
            proxy_ind_dict[key] = val
            key, val = posx_dict.popitem()
            proxy_posx_dict[key] = val
            key, val = velx_dict.popitem()
            proxy_velx_dict[key] = val
            key, val = posy_dict.popitem()
            proxy_posy_dict[key] = val
            key, val = vely_dict.popitem()
            proxy_vely_dict[key] = val
            key, val = posz_dict.popitem()
            proxy_posz_dict[key] = val
            key, val = velz_dict.popitem()
            proxy_velz_dict[key] = val
            key, val = masses_dict.popitem()
            proxy_masses_dict[key] = val
            key, val = part_types_dict.popitem()
            proxy_part_types_dict[key] = val

            # Are we sending too much yet?
            weight = len(proxy_length_dict)

        # Now need collect on master
        proxy_all_length = comm.gather(proxy_length_dict, root=0)
        proxy_all_pid = comm.gather(proxy_pid_dict, root=0)
        proxy_all_ind = comm.gather(proxy_ind_dict, root=0)
        proxy_all_posx = comm.gather(proxy_posx_dict, root=0)
        proxy_all_velx = comm.gather(proxy_velx_dict, root=0)
        proxy_all_posy = comm.gather(proxy_posy_dict, root=0)
        proxy_all_vely = comm.gather(proxy_vely_dict, root=0)
        proxy_all_posz = comm.gather(proxy_posz_dict, root=0)
        proxy_all_velz = comm.gather(proxy_velz_dict, root=0)
        proxy_all_masses = comm.gather(proxy_masses_dict, root=0)
        proxy_all_part_types = comm.gather(proxy_part_types_dict, root=0)
        all_ngal = comm.gather(len(length_dict), root=0)
        if rank == 0:
            ngal = np.sum(all_ngal)
            all_length.extend(proxy_all_length)
            all_pid.extend(proxy_all_pid)
            all_ind.extend(proxy_all_ind)
            all_posx.extend(proxy_all_posx)
            all_velx.extend(proxy_all_velx)
            all_posy.extend(proxy_all_posy)
            all_vely.extend(proxy_all_vely)
            all_posz.extend(proxy_all_posz)
            all_velz.extend(proxy_all_velz)
            all_masses.extend(proxy_all_masses)
            all_part_types.extend(proxy_all_part_types)

        ngal = comm.bcast(ngal, root=0)

    if rank == 0:

        # Loop over halos from other ranks
        for r in range(len(all_length)):

            # Loop over halos (no guarantee popitem does them in
            # the exact same order for each dictionary so do individul loops)
            for key in all_length[r]:
                # Add this particle to the halo
                length_dict.setdefault(key, 0)
                length_dict[key] += all_length[r][key]
            for key in all_pid[r]:
                pid_dict.setdefault(key, []).extend(all_pid[r][key])
            for key in all_ind[r]:
                ind_dict.setdefault(key, []).extend(all_ind[r][key])
            for key in all_posx[r]:
                posx_dict.setdefault(key, []).extend(all_posx[r][key])
            for key in all_velx[r]:
                velx_dict.setdefault(key, []).extend(all_velx[r][key])
            for key in all_posy[r]:
                posy_dict.setdefault(key, []).extend(all_posy[r][key])
            for key in all_vely[r]:
                vely_dict.setdefault(key, []).extend(all_vely[r][key])
            for key in all_posz[r]:
                posz_dict.setdefault(key, []).extend(all_posz[r][key])
            for key in all_velz[r]:
                velz_dict.setdefault(key, []).extend(all_velz[r][key])
            for key in all_masses[r]:
                masses_dict.setdefault(key,
                                       []).extend(all_masses[r][key])
            for key in all_part_types[r]:
                part_types_dict.setdefault(key,
                                           []).extend(all_part_types[r][key])

        # Loop over halos and clean any spurious (npart<10)
        ini_keys = list(length_dict.keys())
        for key in ini_keys:

            if length_dict[key] < 10:
                del length_dict[key]
                del pid_dict[key]
                del ind_dict[key]
                del posx_dict[key]
                del velx_dict[key]
                del posy_dict[key]
                del vely_dict[key]
                del posz_dict[key]
                del velz_dict[key]
                del masses_dict[key]
                del part_types_dict[key]

        # Now we can sort our halos
        keys = length_dict.keys()
        vals = length_dict.values()
        keys = np.array(list(keys), dtype=object)
        vals = np.array(list(vals), dtype=int)
        sinds = np.argsort(vals)[::-1]
        keys = keys[sinds, :]

        # Define arrays and lists to hold sorted halos
        nhalos = keys.shape[0]
        all_begin = np.zeros(nhalos, dtype=int)
        all_len = np.zeros(nhalos, dtype=int)
        all_grpid = np.zeros(nhalos, dtype=int)
        all_subgrpid = np.zeros(nhalos, dtype=int)
        all_halo_ids = np.zeros(nhalos, dtype=int)
        all_pid = []
        all_ind = []
        all_posx = []
        all_velx = []
        all_posy = []
        all_vely = []
        all_posz = []
        all_velz = []
        all_masses = []
        all_part_types = []

        # Loop over keys storing their results
        for ihalo, key in enumerate(keys):
            # Get group and subgroup ID
            key = tuple(key)
            grp, subgrp = key[0], key[1]

            # Store data
            all_halo_ids[ihalo] = ihalo
            all_begin[ihalo] = len(all_pid)
            all_len[ihalo] = length_dict[key]
            all_grpid[ihalo] = grp
            all_subgrpid[ihalo] = subgrp
            all_pid.extend(pid_dict[key])
            all_ind.extend(ind_dict[key])
            all_posx.extend(posx_dict[key])
            all_velx.extend(velx_dict[key])
            all_posy.extend(posy_dict[key])
            all_vely.extend(vely_dict[key])
            all_posz.extend(posz_dict[key])
            all_velz.extend(velz_dict[key])
            all_masses.extend(masses_dict[key])
            all_part_types.extend(part_types_dict[key])

        # Convert all keys to arrays
        all_pid = np.array(all_pid, dtype=int)
        all_ind = np.array(all_ind, dtype=int)
        all_posx = np.array(all_posx, dtype=np.float64)
        all_velx = np.array(all_velx, dtype=np.float64)
        all_posy = np.array(all_posy, dtype=np.float64)
        all_vely = np.array(all_vely, dtype=np.float64)
        all_posz = np.array(all_posz, dtype=np.float64)
        all_velz = np.array(all_velz, dtype=np.float64)
        all_masses = np.array(all_masses, dtype=np.float64)
        all_part_types = np.array(all_part_types, dtype=int)

        # Define the number of particles sorted
        npart_sorted = len(all_pid)

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
            while nparts_to_send <= shared_npart and ihalo < all_len.size:
                halos_on_rank[r].append(ihalo)
                nparts_to_send += all_len[ihalo]
                ihalo += 1

        # Allocate any leftovers
        # NOTE: leads to the last rank having more work but
        # these are small halos due to halo ordering
        while ihalo < all_len.size:
            halos_on_rank[-1].append(ihalo)
            ihalo += 1

        # Get how many particles each rank should expect
        nparts_on_rank = [0, ] * size
        for r in halos_on_rank:
            for ihalo in halos_on_rank[r]:
                nparts_on_rank[r] += all_len[ihalo]

        # Get how many halos each rank should expect, and their
        # particle offset
        nhalos_on_rank = [0, ] * size
        for r in halos_on_rank:
            nhalos_on_rank[r] = len(halos_on_rank[r])

        # Set up dictionaries for communicating data
        halo_ids = [None for r in range(size)]
        length = [None for r in range(size)]
        grpid = [None for r in range(size)]
        subgrpid = [None for r in range(size)]
        pid = [None for r in range(size)]
        ind = [None for r in range(size)]
        posx = [None for r in range(size)]
        velx = [None for r in range(size)]
        posy = [None for r in range(size)]
        vely = [None for r in range(size)]
        posz = [None for r in range(size)]
        velz = [None for r in range(size)]
        masses = [None for r in range(size)]
        types = [None for r in range(size)]

        # Loop over ranks
        for r in range(size):

            # Get the data to send
            if len(halos_on_rank[r]) > 0:
                rank_begin = all_begin[np.min(halos_on_rank[r])]
                rank_len = nparts_on_rank[r]
                halo_slice = (np.min(halos_on_rank[r]),
                              np.max(halos_on_rank[r]) + 1)
                part_slice = (rank_begin, rank_begin + rank_len)

                # Get store data
                halo_ids[r] = all_halo_ids[halo_slice[0]: halo_slice[1]]
                length[r] = all_len[halo_slice[0]: halo_slice[1]]
                grpid[r] = all_grpid[halo_slice[0]: halo_slice[1]]
                subgrpid[r] = all_subgrpid[halo_slice[0]: halo_slice[1]]
                pid[r] = all_pid[part_slice[0]: part_slice[1]]
                ind[r] = all_ind[part_slice[0]: part_slice[1]]
                posx[r] = all_posx[part_slice[0]: part_slice[1]]
                velx[r] = all_velx[part_slice[0]: part_slice[1]]
                posy[r] = all_posy[part_slice[0]: part_slice[1]]
                vely[r] = all_vely[part_slice[0]: part_slice[1]]
                posz[r] = all_posz[part_slice[0]: part_slice[1]]
                velz[r] = all_velz[part_slice[0]: part_slice[1]]
                masses[r] = all_masses[part_slice[0]: part_slice[1]]
                types[r] = all_part_types[part_slice[0]: part_slice[1]]
            else:
                halo_ids[r] = np.array([], dtype=int)
                length[r] = np.array([], dtype=int)
                grpid[r] = np.array([], dtype=int)
                subgrpid[r] = np.array([], dtype=int)
                pid[r] = np.array([], dtype=int)
                ind[r] = np.array([], dtype=int)
                posx[r] = np.array([], dtype=np.float64)
                velx[r] = np.array([], dtype=np.float64)
                posy[r] = np.array([], dtype=np.float64)
                vely[r] = np.array([], dtype=np.float64)
                posy[r] = np.array([], dtype=np.float64)
                vely[r] = np.array([], dtype=np.float64)
                masses[r] = np.array([], dtype=np.float64)
                types[r] = np.array([], dtype=np.float64)

    else:
        nhalos = None
        halo_ids = None
        begin = None
        length = None
        grpid = None
        subgrpid = None
        pid = None
        ind = None
        posx = None
        velx = None
        posy = None
        vely = None
        posz = None
        velz = None
        masses = None
        types = None

    # Broadcast the number of halos
    nhalos = comm.bcast(nhalos, root=0)

    # Scatter the results of the decompostion
    halo_ids = comm.scatter(halo_ids, root=0)
    length = comm.scatter(length, root=0)
    grpid = comm.scatter(grpid, root=0)
    subgrpid = comm.scatter(subgrpid, root=0)
    pid = comm.scatter(pid, root=0)
    ind = comm.scatter(ind, root=0)
    posx = comm.scatter(posx, root=0)
    velx = comm.scatter(velx, root=0)
    posy = comm.scatter(posy, root=0)
    vely = comm.scatter(vely, root=0)
    posz = comm.scatter(posz, root=0)
    velz = comm.scatter(velz, root=0)
    masses = comm.scatter(masses, root=0)
    types = comm.scatter(types, root=0)

    # Combine position and velocity coordinates
    if posx.size > 0:
        pos = np.column_stack((posx, posy, posz))
        vel = np.column_stack((velx, vely, velz))
    else:
        pos = np.array([[], [], []])
        vel = np.array([[], [], []])

    return (halo_ids, length, grpid, subgrpid, pid, ind, pos, vel,
            masses, types, true_part_ids, nhalos)


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
                "008_z007p000", "009_z006p000", "010_z005p000"]

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

    # Define path to a single snapshot file for metadata
    sim_path = inputs["data"].replace("<reg>", reg)
    single_file = sim_path.replace("<snap>", snap)
    inputs["halo_basename"] = inputs["halo_basename"].replace("<reg>", reg)

    # Exit if the file exists
    if os.path.isfile(inputs["haloSavePath"] + inputs["halo_basename"]
                      + snap + ".hdf5"):
        if rank == 0:
            print(inputs["haloSavePath"] + inputs["halo_basename"]
                  + snap + ".hdf5", "exists")
        return

    # Open single file and get DM particle mass
    try:
        hdf = h5py.File(single_file, "r")
        boxsize = hdf["Header"].attrs["BoxSize"] / 0.6777
        nparts = hdf["Header"].attrs["NumPart_Total"]
        hdf.close()
    except OSError:
        if rank == 0:
            print("File is not on cosma7")
        return

    # Set up object containing housekeeping metadata
    meta = p_utils.Metadata(snaplist, snap_ind, cosmology, inputs,
                            flags, params, simulation,
                            boxsize=[boxsize, boxsize, boxsize],
                            npart=nparts,
                            z=z)

    meta.rank = rank
    meta.nranks = size

    if rank == 0:
        say_hello(meta)
        message(rank, "Running on Region %s and snap %s (id=%d)"
                % (reg, snap, job_ind))

    # Instantiate timer
    tictoc = TicToc(meta)
    tictoc.start()
    meta.tictoc = tictoc

    # Get the particle data for all particle types in the current snapshot
    (halo_ids, length, grpid, subgrpid, pid, ind, pos, vel,
     masses, part_types, pre_snap_part_ids,  nhalos) = get_data(tictoc,
                                                                reg,
                                                                snap,
                                                                meta,
                                                                inputs[
                                                                    "data"])

    # Initialise array to store all particle IDs
    snap_part_ids = np.zeros(np.sum(meta.npart), dtype=int)

    # Read particle IDs to store combined particle ids array
    for part_type in meta.part_types:
        offset = meta.part_ind_offset[part_type]
        snap_part_ids[offset: offset + meta.npart[part_type]
                      ] = pre_snap_part_ids[part_type]

    if rank == 0:
        message(rank, "Nhalo: %d" % nhalos)
        message(rank, "Npart_dm: %d" % meta.npart[1])
        message(rank, "Npart_gas: %d" % meta.npart[0])
        message(rank, "Npart_star: %d" % meta.npart[4])
        message(rank, "Npart_bh: %d" % meta.npart[5])
        message(rank, "Npart_tot: %d" % snap_part_ids.size)

    # Initialise dictionary for mega halo objects
    results = {}

    # Loop over galaxies and create mega objects
    b = 0
    if len(halo_ids) > 0:
        halo_offset = halo_ids[0]
        for ihalo, l in zip(halo_ids, length):

            # Compute end
            e = b + l

            if len(ind[b:e]) < 10:
                b = e
                continue

            # Dummy internal energy
            int_nrg = np.zeros_like(masses[b:e])

            # Store this halo
            results[ihalo] = Halo(tictoc, ind[b:e],
                                  (grpid[ihalo - halo_offset],
                                   subgrpid[ihalo - halo_offset]),
                                  pid[b:e], pos[b:e, :], vel[b:e, :],
                                  part_types[b:e],
                                  masses[b:e], int_nrg, 10, meta,
                                  calc_energy=False)
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
                   sim_pids=snap_part_ids,
                   extra_data=extra_data)

        if meta.verbose:
            tictoc.report("Writing")

    tictoc.end()
    tictoc.end_report(comm)


if __name__ == "__main__":
    main()
