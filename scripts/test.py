import numpy as np
import scipy.io as sio


with np.load("/home/rab/AI-Aided-ENF-Timestamping/data/datasets/ENF_WHU/02_preprocessed/queries/001/001_0_24000.npz") as data:
    print(data['signal'])

orig_mat_data = sio.loadmat("/home/rab/AI-Aided-ENF-Timestamping/data/datasets/ENF_BGU/01_raw/ENF-BGU/electric_room_monday_23_10_exp1_sec_30_90.mat")
print(len(orig_mat_data['x'].flatten()))

mat_data = sio.loadmat("/home/rab/AI-Aided-ENF-Timestamping/data/datasets/ENF_BGU/01_raw/ENF-BGU/1min/exp1/exp1_sec_30_90.mat")
# print(mat_data)

new_data = {
    'signal': mat_data['x_filtered'].flatten(),
    'sample_rate': mat_data['fs'].item()
}

print(len(new_data['signal']))