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
 
 
def writenifti(image,filename, info):
    """Write nifti file."""
    writer = vtk.vtkNIFTIImageWriter()
    writer.SetInputData(image)
    writer.SetFileName(filename)
    writer.SetInformation(info)
    writer.Write()
 
 
if __name__ == '__main__':
    baseDir = os.path.normpath(r'C:\Users\Admin\Desktop\EGFR total\EGFRzhong\EGFRzhong\data\0')
    files = glob(baseDir+'/*.nrrd')
    for file in files:
        m, info = readnrrd(file)
        writenifti(m,  file.replace( '.nrrd','.nii.gz'), info)
