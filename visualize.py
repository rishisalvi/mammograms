#%% 
import numpy as np 
import pandas as pd
from glob import glob
import pydicom

#%%
df = pd.read_csv('mass_case_description_train_set.csv')
#folder = 
paths = glob('CBIS\manifest-ZkhPvrLo5216730872708713142\CBIS-DDSM\Calc-Test_P_00038_LEFT_CC' + '1-1.dcm')
paths
# %%
"""import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files
filename = get_testdata_files("1-1.dcm")
ds = pydicom.dcmread(filename)
plt.imshow(ds.pixel_array, cmap=plt.cm.gray) """
# %%
from pydicom import dcmread
ds = dcmread(r"C:\Users\rishi\Documents\MammogramData\CBIS\manifest-ZkhPvrLo5216730872708713142\CBIS-DDSM\Calc-Test_P_00038_LEFT_CC\08-29-2017-DDSM-NA-96009\1.000000-full mammogram images-63992\1-1.dcm")
print(ds)
plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
# %%
from pydicom import dcmread
#ds = dcmread(r"C:\Users\rishi\Documents\MammogramData\CBIS\manifest-ZkhPvrLo5216730872708713142\CBIS-DDSM\Calc-Test_P_00041_LEFT_CC\08-29-2017-DDSM-NA-96009\1.000000-full mammogram images-63992\1-1.dcm")
file = r"C:\Users\rishi\Documents\MammogramData\CBIS\manifest-ZkhPvrLo5216730872708713142\CBIS-DDSM\Calc-Test_P_00041_LEFT_CC/**/**/1-1.dcm"
paths = glob(file)
paths
ds1 = dcmread(paths[0])
print(ds1)
plt.imshow(ds1.pixel_array, cmap=plt.cm.gray) 
# %%
import matplotlib.pyplot as plt
import numpy as np
rows = 1
cols = 2
images = []
images.insert(0, ds)
images.insert(1, ds1)

plt.figure(figsize=(20, 40))

for i in range(rows * cols):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(images[i].pixel_array, cmap=plt.cm.gray)
    plt.title(f'Image {i+1}', fontsize=25)
    plt.axis('off') 
plt.show()

# %%
df = pd.read_csv('calc_case_description_test_set.csv')
patients = df.iloc[:, 0]
patients = set(patients)
patients = list(patients)
patients.sort()
# %%
from pydicom import dcmread
images = []
valid_patients = []
for i in range(len(patients)):
    if ("P_00906" >= patients[i]):
        file = r"C:\Users\rishi\Documents\MammogramData\CBIS\**\CBIS-DDSM\Calc-Test_" + patients[i] + "_LEFT_CC/**/**/1-1.dcm"
        paths = glob(file)
        if (len(paths) > 0):
            data = dcmread(paths[0])
            images.append(data)
            valid_patients.append(patients[i])
print(len(images))
# %%
rows = 5
cols = 7

plt.figure(figsize=(20, 40))

for i in range(rows * cols):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(images[i].pixel_array, cmap=plt.cm.gray) 
    plt.title(valid_patients[i], fontsize=25)
    plt.axis('off') 
plt.show()