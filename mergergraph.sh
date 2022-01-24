#!/bin/bash -l
#SBATCH --ntasks 1 # The number of cores you need...
#SBATCH --array=1-480%50
#SBATCH --cpus-per-task=8
#SBATCH -J MEGA-FLARES #Give it something meaningful.
#SBATCH -o logs/output_Halo.%J.out
#SBATCH -e logs/error_Halo.%J.err
#SBATCH -p cosma6 #or some other partition, e.g. cosma, cosma6, etc.
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 72:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=wjr21@sussex.ac.uk #PLEASE PUT YOUR EMAIL ADDRESS HERE (without the <>)

# Run the job from the following directory - change this to point to your own personal space on /lustre
cd /cosma7/data/dp004/dc-rope1/FLARES/flares

module purge
#load the modules used to build your program.
module load pythonconda3/4.5.4

source activate flares-env

i=$(($SLURM_ARRAY_TASK_ID - 1))

# Run the program
./directProgDesc_allparts.py $i

source deactivate

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit

