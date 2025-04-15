import h5py

# Open the HDF5 file
with h5py.File("imagenet.hdf5", "r") as f:
    def print_structure(name, obj):
        print(name)

    f.visititems(print_structure)