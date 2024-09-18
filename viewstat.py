import glob
import pickle
import numpy as np
import itertools

def getfilelist(dir, ext):
    return glob.glob(f"{dir}/*{ext}")


dir_list = ['./12qubit_cyclic_depth3/512batch_100epoch', './12qubit_cyclic_depth5/512batch_100epoch', \
            './12qubit_multi_depth3/512batch_100epoch', './12qubit_multi_depth5/512batch_100epoch', \
            './12qubit_single_depth5/512batch_100epoch',]

scheme_list = ['./3plus3_', './3processor_multi_', './6single_']
depth_list = ['depth5/512batch_100epoch','depth7/512batch_100epoch','depth9/512batch_100epoch']

ext = '.pickle'

for directory in dir_list:

    files = getfilelist(directory, ext)

    a = None
    nocount=0
    nclist = []
    paramlen = 0

    accuracy_list = []
    for i, filename in enumerate(files):
        with open(filename, 'rb') as f:
            a = pickle.load(f)
            accuracy_list.append(a['test_accuracy'])
            f.close()

        try: a['parameters']
        except:
            print('Error!')
            nocount += 1
            nclist.append(filename)
            continue
        if not i:
            #if 'multi' in filename:
            #    print(a['parameters'])
            paramlen = len(a['parameters'])


    accuracy_list = np.array(accuracy_list)

    print(directory)
    print(f'# of logs = {len(files)}')
    print(f'# of Parameters = {paramlen}')
    print(f'Average Accuracy = {np.average(accuracy_list)}')
    print(f'Sigma = {np.std(accuracy_list)}\n')