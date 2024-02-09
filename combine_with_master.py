"""Combine the FLARES master file and mergergraph file."""
import h5py
import numpy as np


def extract_group_subgroup_ids(float_id):
    """
    Extracts the group ID and subgroup ID from a combined float representation.

    Parameters:
    float_id (float): The combined float of the form grpID.%05d' %subgrpID.

    Returns:
    tuple: A tuple containing the group ID (int) and subgroup ID (int).
    """
    # Separate the integer part (grpID) and the fractional part
    grpID = int(float_id)
    fractional_part = float_id - grpID

    # Multiply the fractional part by 100000 and round it to get subgrpID
    subgrpID = int(fractional_part * 100000)

    return grpID, subgrpID


def exclude_and_copy_group(src_group, dest_group, exclude_group_names):
    """
    Recursively copies contents from src_group to dest_group, excluding the specified group.
    """
    for item_name, item in src_group.items():
        # Construct the path for the current item
        path = f"{src_group.name}/{item_name}".lstrip("/")

        # Skip the excluded group and its contents
        skip = False
        for exclude_group_name in exclude_group_names:
            if exclude_group_name in path:
                skip = True
                break
        if skip:
            continue

        print(f"Copying {path}")

        # If it's a group, create it in the destination and recurse
        if isinstance(item, h5py.Group):
            dest_sub_group = dest_group.require_group(item_name)
            exclude_and_copy_group(item, dest_sub_group, exclude_group_names)
        # If it's a dataset, copy it directly
        elif isinstance(item, h5py.Dataset):
            src_group.copy(item_name, dest_group)


def copy_hdf5_excluding_group(
    original_file_path, new_file_path, exclude_group_names
):
    with h5py.File(original_file_path, "r") as original_file, h5py.File(
        new_file_path, "w"
    ) as new_file:
        # Start the recursive copying from the root, excluding the specified group
        exclude_and_copy_group(original_file, new_file, exclude_group_names)


# Define the input files
master_file = (
    "/cosma7/data/dp004/dc-payy1/my_files/flares_pipeline/data/flares.hdf5"
)
new_file = "flares_with_mergers.hdf5"

# Define the mergergraph directory
mega_path = "/cosma7/data/dp004/FLARES/FLARES-1/MergerGraphs/"

# Make a new copy
copy_hdf5_excluding_group(
    master_file,
    new_file,
    [
        "Particle",
        "BPASS",
        "011_z004p770",
    ],
)

# Define the regions
regs = []
for reg in range(40):
    regs.append(str(reg).zfill(2))

# Define  the snapshots
flares_snaps = [
    "001_z014p000",
    "002_z013p000",
    "003_z012p000",
    "004_z011p000",
    "005_z010p000",
    "006_z009p000",
    "007_z008p000",
    "008_z007p000",
    "009_z006p000",
    "010_z005p000",
]


with h5py.File(new_file, "r+") as hdf_master:
    # Loop over the regions
    for reg in regs:
        # Loop over the snapshots
        for snap in flares_snaps:
            print(reg, snap)
            # Get the galaxy group
            gal_grp = hdf_master[f"{reg}/{snap}/Galaxy"]

            # Define the mergergraph file
            mergergraph_file = (
                mega_path + f"GEAGLE_{reg}/SubMgraph_{snap}.hdf5"
            )
            # Open the mergergraph file
            with h5py.File(mergergraph_file, "r") as hdf_mergergraph:
                # Get the master group and subgroup IDs
                master_grpIDs = gal_grp["GroupNumber"][:]
                master_subgrpIDs = gal_grp["SubGroupNumber"][:]

                # Get the mergergraph group and subgroup IDs
                mergergraph_subfindIDs = hdf_mergergraph["SUBFIND_halo_IDs"][:]
                mergergraph_grpIDs = []
                mergergraph_subgrpIDs = []
                for subfindID in mergergraph_subfindIDs:
                    grpID, subgrpID = extract_group_subgroup_ids(subfindID)
                    mergergraph_grpIDs.append(grpID)
                    mergergraph_subgrpIDs.append(subgrpID)

                # Get the progenitor and descendant pointers and numbers
                prog_ptrs = hdf_mergergraph["Prog_Start_Index"][:]
                n_progs = hdf_mergergraph["nProgs"][:]

                # Make a nice look up table for the merger graph pointer and
                # length
                mergergraph_lookup = {}
                for i, (grpID, subgrpID) in enumerate(
                    zip(mergergraph_grpIDs, mergergraph_subgrpIDs)
                ):
                    mergergraph_lookup[(grpID, subgrpID)] = (
                        prog_ptrs[i],
                        n_progs[i],
                    )

                # Loop over galaxies in the master file getting the data
                pointers = np.zeros(len(master_grpIDs), dtype=np.int32)
                nprogs = np.zeros(len(master_grpIDs), dtype=np.int32)
                prog_star_ms = []
                for ind, (grp, subgrp) in enumerate(
                    zip(master_grpIDs, master_subgrpIDs)
                ):
                    # Skip galaxies without an entry
                    if (grp, subgrp) not in mergergraph_lookup:
                        nprogs[ind] = 0
                        pointers[ind] = -1
                        continue

                    # Get the mergergraph pointer and length
                    prog_start_index, nprog = mergergraph_lookup[(grp, subgrp)]

                    # Get the progenitor masses
                    prog_masses = hdf_mergergraph["prog_stellar_masses"][
                        prog_start_index : prog_start_index + nprog
                    ]

                    # Perform a stellar mass cut
                    prog_masses = prog_masses[prog_masses > 1e8]

                    # Sort by stellar mass (they're sorted by DM mass in the
                    # MEGA file)
                    prog_masses = np.sort(prog_masses)[::-1]

                    # Store this data
                    nprogs[ind] = prog_masses.size
                    pointers[ind] = len(prog_star_ms)
                    prog_star_ms.extend(prog_masses)

                # Add the data to the master file under a new group
                gal_grp.create_group("MergerGraph")
                gal_grp["MergerGraph"].create_dataset(
                    "Prog_Start_Index", data=pointers
                )
                gal_grp["MergerGraph"].create_dataset("nProgs", data=nprogs)
                gal_grp["MergerGraph"].create_dataset(
                    "prog_stellar_masses", data=np.array(prog_star_ms)
                )
