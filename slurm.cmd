#!/bin/bash

#SBATCH --job-name=smoke-test
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=u3013241@connect.hku.hk
#SBATCH --partition=intel
#SBATCH --qos=normal
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=%x.out.%j
#SBATCH --error=%x.err.%j

set -e

cd "${SLURM_SUBMIT_DIR}"

echo "Job start: $(date)"
echo "Host: $(hostname)"
echo "SLURM_NTASKS: ${SLURM_NTASKS}"

module load mpich/gcc/4.1.2-gcc12.3
module load singularity

unset CC CXX FC F77 F90 MPICC MPICXX MPIFC

SANDBOX_DIR="${HOME}/Desktop/singularity-images/fenicsx-0.11.0-sdbx"
PROJECT_DIR="${HOME}/Desktop/code-projects/adaptive-maxwell-mpi"

mpirun -n "${SLURM_NTASKS}" \
    singularity exec -w "${SANDBOX_DIR}" \
    bash -c "
        cd '${PROJECT_DIR}'
        python main.py run configs/smoke-test.yaml
    "

echo "Job finish: $(date)"