import zipfile
import os
import shutil

file_path = '/Users/bw/Dataset/MIPI_demosaic_hybridevs/val/'

os.chdir(file_path)

cwd = os.getcwd()
print(cwd)
shutil.make_archive('submit0208','zip','submit0208')

