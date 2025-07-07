import pickle
from tqdm import tqdm

with open("../dataprocessing_allatom/deduped_pdb_xyz_data.pkl", "rb") as f:
    data = pickle.load(f)


for i in tqdm(range(len(data['sequence']))):
    xyz= data['xyz'][i]

    #change key 'phosphate' to 'all' if it contains 'phosphate'
    for j in range(len(data['xyz'][i])):
        if 'phosphate' in data['xyz'][i][j].keys():
            data['xyz'][i][j]['all'] = data['xyz'][i][j]['phosphate']
            del data['xyz'][i][j]['phosphate']



with open("../fixed_deduped_pdb_xyz_data.pkl", "wb+") as f:
    pickle.dump(data, f)