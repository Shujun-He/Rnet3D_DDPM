# from glob import glob
# import os

# pdbs=glob("pdbs/*_native_1.pdb")+glob("pdbs/*_predicted*.pdb")

# os.system("mkdir recon_pdbs")

# for pdb in pdbs:
#     target_path=pdb.replace("pdbs/","recon_pdbs/")
#     os.system(f"../Arena {pdb} {target_path}")


from glob import glob
import os
from multiprocessing import Pool, cpu_count

# Create output directory if it doesn't exist
os.makedirs("recon_pdbs", exist_ok=True)

# Collect all pdb files
pdbs = glob("pdbs/*_native_1.pdb") + glob("pdbs/*_predicted*.pdb")

# Define worker function
def run_arena(pdb):
    target_path = pdb.replace("pdbs/", "recon_pdbs/")
    return os.system(f"../Arena {pdb} {target_path}")

if __name__ == "__main__":
    # Use a pool of workers
    with Pool(processes=cpu_count()) as pool:
        pool.map(run_arena, pdbs)
