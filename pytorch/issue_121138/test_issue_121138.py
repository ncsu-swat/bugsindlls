import torch
import numpy as np
import h5py
import psutil

issue_no = '121138'
print('Pytorch issue no.', issue_no)

process = psutil.Process()
print(process.memory_info().rss/1024/1024)  # used memory in mb
outfile="testingmemleak.h5"

fill_value = "value".encode('ascii')
large_array_bad = np.full((400,2000,8), fill_value)
# write a nice file
with h5py.File(outfile, 'w') as file:
    file.create_dataset('bad', data=large_array_bad)
print(process.memory_info().rss/1024/1024)

def test_f():
    flag = False
    for i in range(100):
        # read a nice file
        data_dict={}  # cleared each iteration so memory should stay constant
        with h5py.File(outfile,'r') as file:
            for label, data in file.items():
                data_dict[label] = data[()]
        print(process.memory_info().rss/1024/1024)
        tensor_data = []  # cleared each iteration so memory should stay constant
        for label, data in data_dict.items():  # try to convert to torch tensors
            try:
                tensor = torch.from_numpy(data)
                tensor_data.append(tensor)
            except TypeError:
                print("couldn't convert from numpy")
        print(process.memory_info().rss/1024/1024)
        if process.memory_info().rss/1024/1024 > 1000:
            print("Too much memory used, memory leak detected")
            flag = True
            break
    
    assert flag