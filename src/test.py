import h5py
def check_hdf5_content(hdf5_path):
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        keys = list(hdf5_file.keys())
        print("Keys in HDF5 file:", keys)

# Kiểm tra nội dung tệp HDF5
check_hdf5_content(r"C:\Users\dangk\B4\shisen\data\MPIIFaceGaze\preprocessed\Image\face_lmk.hdf5")