import os
from glob import glob
import numpy as np
import vtk


def readnrrd(filename):
    """Read image in nrrd format."""
    reader = vtk.vtkNrrdReader()
    reader.SetFileName(filename)
    reader.Update()
    info = reader.GetInformation()
    return reader.GetOutput(), info


def writenifti(image, filename, info):
    """Write nifti file."""
    writer = vtk.vtkNIFTIImageWriter()
    writer.SetInputData(image)
    writer.SetFileName(filename)
    writer.SetInformation(info)
    writer.Write()


if __name__ == '__main__':
    # Define the paths for data folders
    data2_folder = r'e:\SZBL-test1\data2'


    data2_CT0_paths = [
        os.path.join(data2_folder, "CT-0", x)
        for x in os.listdir(os.path.join(data2_folder, "CT-0"))
    ]

    data2_CT1_paths = [
        os.path.join(data2_folder, "CT-1", x)
        for x in os.listdir(os.path.join(data2_folder, "CT-1"))
    ]

    # Combine all paths
    all_paths = data2_CT0_paths + data2_CT1_paths

    # Process each file
    for file in all_paths:
        m, info = readnrrd(file)

        # Get the relative path from the original data folder
        relative_path = os.path.relpath(file, data2_folder)

        # Generate the save path based on the relative path
        save_folder = os.path.join(r'e:\SZBL-test1\1', os.path.dirname(relative_path))
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, os.path.basename(file).replace('.nrrd', '.nii.gz'))

        writenifti(m, save_path, info)
