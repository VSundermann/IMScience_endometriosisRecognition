import os
import pandas as pd
import numpy as np
from scipy.ndimage import binary_closing
import SimpleITK as sitk

def ImgResample(image, out_spacing=(0.5, 0.5, 0.5), out_size=None, is_label=False, pad_value=0):
    """Resamples an image to given element spacing and output size."""

    original_spacing = np.array(image.GetSpacing())
    original_size = np.array(image.GetSize())

    if out_size is None:
        out_size = np.round(np.array(original_size * original_spacing / np.array(out_spacing))).astype(int)
    else:
        out_size = np.array(out_size)

    original_direction = np.array(image.GetDirection()).reshape(len(original_spacing),-1)
    original_center = (np.array(original_size, dtype=float) - 1.0) / 2.0 * original_spacing
    out_center = (np.array(out_size, dtype=float) - 1.0) / 2.0 * np.array(out_spacing)

    original_center = np.matmul(original_direction, original_center)
    out_center = np.matmul(original_direction, out_center)
    out_origin = np.array(image.GetOrigin()) + (original_center - out_center)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size.tolist())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(out_origin.tolist())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        #resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
        #resample.SetInterpolator(sitk.sitkLinear)

    return resample.Execute(sitk.Cast(image, sitk.sitkFloat32))

#################################################################
## DATA PATHS AND DIRECTORIES
#################################################################

# Change these if you want to use a different dataset
original_dataset_dir = '../dataset_raw/UT-EndoMRI/D2_TCPW'
dataset_info = pd.read_csv("../dataset_raw/dataset_info_D2.csv", header=0)

output_dir = '../dataset_resample'

# Main directories
MRI_full_dataset_dir = os.path.join(output_dir, 'mri_full')
label_full_dataset_dir = os.path.join(output_dir, 'label_full')
MRI_filtered_dir = os.path.join(output_dir, 'mri_filtered')
label_filtered_dir = os.path.join(output_dir, 'label_filtered')
MRI_full_slices_dir = os.path.join(output_dir, 'mri_full_slices')
label_full_slices_dir = os.path.join(output_dir, 'label_full_slices')
MRI_filtered_slices_dir = os.path.join(output_dir, 'mri_filtered_slices')
label_filtered_slices_dir = os.path.join(output_dir, 'label_filtered_slices')

main_directories = [MRI_full_dataset_dir, label_full_dataset_dir, MRI_filtered_dir, label_filtered_dir, MRI_full_slices_dir, label_full_slices_dir, MRI_filtered_slices_dir, label_filtered_slices_dir]

for directory in main_directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Subdirectories by annotation type
# Not necessary for now, only working with em
#annotation_directories = ['ov', 'ut', 'em', 'cy']
#
#for directory in main_directories:
#    for annotation in annotation_directories:
#        if not os.path.exists(os.path.join(directory, annotation)):
#            os.makedirs(os.path.join(directory, annotation))

# Subdirectories by MRI type
mri_types_directories = ['T1', 'T1FS', 'T2', 'T2FS']

for directory in main_directories:
    for mri_type in mri_types_directories:
        if not os.path.exists(os.path.join(directory, mri_type)):
            os.makedirs(os.path.join(directory, mri_type))

#################################################################
## DATA PROCESSING
#################################################################   

# Return the slices of the image that contain annotations
def check_annotation(image):
    annotations = []

    image_array = sitk.GetArrayFromImage(image)

    for i in range(image_array.shape[0]):
        if np.any(image_array[i] > 0):
            annotations.append(i)
    
    return annotations

def ImgResample_slices(image, label=False):
    resampled_slices = []
    for i in range(image.GetDepth()):
        # Resample the 2D slice
        resampled_slice = ImgResample(image[:, :, i], out_size=(512, 512), out_spacing=(0.5, 0.5), is_label=label)

        # Execute closing to clean GT image from random, erroneous, markers
        resampled_slice_closed = binary_closing(sitk.GetArrayFromImage(resampled_slice), iterations=2)

        resampled_slice_closed = sitk.GetImageFromArray(resampled_slice_closed)
        

        # Reset origin to avoid JoinSeries error about physical space mismatch
        # We will restore the full 3D origin later
        resampled_slice_closed.SetOrigin((0.0, 0.0))
        
        resampled_slices.append(resampled_slice_closed)

    # Stack the 2D slices back into a 3D volume
    new_image = sitk.JoinSeries(resampled_slices)

    # Restore the original 3D geometry
    # Note: JoinSeries creates a volume where the 3rd dimension is the stack axis.
    # We need to ensure spacing and origin are correct for the 3D volume.
    new_image.SetSpacing((0.5, 0.5, image.GetSpacing()[2]))
    new_image.SetOrigin(image.GetOrigin())
    new_image.SetDirection(image.GetDirection())

    return new_image
    
def saver_function(full_image, filtered_image, directory_full, directory_filtered, directory_full_slices, directory_filtered_slices, patient_id, mri_type, individual_slices=True):
    sitk.WriteImage(full_image, os.path.join(directory_full, patient_id + '_' + mri_type + '.nii.gz'))
    sitk.WriteImage(full_image, os.path.join(directory_full, mri_type, patient_id + '_' + mri_type + '.nii.gz'))
    sitk.WriteImage(filtered_image, os.path.join(directory_filtered, patient_id + '_' + mri_type + '.nii.gz'))
    sitk.WriteImage(filtered_image, os.path.join(directory_filtered, mri_type, patient_id + '_' + mri_type + '.nii.gz'))

    if individual_slices:       
        for i in range(full_image.GetDepth()):   
            sitk.WriteImage(full_image[:, :, i], os.path.join(directory_full_slices, patient_id + '_' + mri_type + '_' + str(i) + '.nii.gz'))
            sitk.WriteImage(full_image[:, :, i], os.path.join(directory_full_slices, mri_type, patient_id + '_' + mri_type + '_' + str(i) + '.nii.gz'))

        for i in range(filtered_image.GetDepth()):
            sitk.WriteImage(filtered_image[:, :, i], os.path.join(directory_filtered_slices, patient_id + '_' + mri_type + '_' + str(i) + '.nii.gz'))
            sitk.WriteImage(filtered_image[:, :, i], os.path.join(directory_filtered_slices, mri_type, patient_id + '_' + mri_type + '_' + str(i) + '.nii.gz'))


for index, row in dataset_info.iterrows():
    patient_id = row['Patient ID']
    
    if row['Endometrioma'] >= 1:
        gt_image = sitk.ReadImage(os.path.join(original_dataset_dir, patient_id, patient_id + '_em.nii.gz'), sitk.sitkFloat32)
        slices = check_annotation(gt_image)
        
        gt_image = ImgResample_slices(gt_image, label=True)
        
        gt_filtered_array = sitk.GetArrayFromImage(gt_image)[slices, :, :]
        gt_slices = sitk.GetImageFromArray(gt_filtered_array)
        
        """ if row['T1'] >= 1:
            T1_image = sitk.ReadImage(os.path.join(original_dataset_dir, patient_id, patient_id + '_T1.nii.gz'))
            T1_image = ImgResample_slices(T1_image, label=False)

            T1_filtered_array = sitk.GetArrayFromImage(T1_image)[slices, :, :]
            T1_slices = sitk.GetImageFromArray(T1_filtered_array)
            
            saver_function(T1_image, T1_slices, MRI_full_dataset_dir, MRI_filtered_dir, MRI_full_slices_dir, MRI_filtered_slices_dir, patient_id, 'T1')
            saver_function(gt_image, gt_slices, label_full_dataset_dir, label_filtered_dir, label_full_slices_dir, label_filtered_slices_dir, patient_id, 'T1')
        
        if row['T1FS'] >= 1:
            T1FS_image = sitk.ReadImage(os.path.join(original_dataset_dir, patient_id, patient_id + '_T1FS.nii.gz'))
            T1FS_image = ImgResample_slices(T1FS_image, label=False)
            
            T1FS_filtered_array = sitk.GetArrayFromImage(T1FS_image)[slices, :, :]
            T1FS_slices = sitk.GetImageFromArray(T1FS_filtered_array)

            saver_function(T1FS_image, T1FS_slices, MRI_full_dataset_dir, MRI_filtered_dir, MRI_full_slices_dir, MRI_filtered_slices_dir, patient_id, 'T1FS')
            saver_function(gt_image, gt_slices, label_full_dataset_dir, label_filtered_dir, label_full_slices_dir, label_filtered_slices_dir, patient_id, 'T1FS')
        """
        if row['T2'] >= 1:
            T2_image = sitk.ReadImage(os.path.join(original_dataset_dir, patient_id, patient_id + '_T2.nii.gz'))
            T2_image = ImgResample_slices(T2_image, label=False)
            
            T2_filtered_array = sitk.GetArrayFromImage(T2_image)[slices, :, :]
            T2_slices = sitk.GetImageFromArray(T2_filtered_array)

            saver_function(T2_image, T2_slices, MRI_full_dataset_dir, MRI_filtered_dir, MRI_full_slices_dir, MRI_filtered_slices_dir, patient_id, 'T2')
            saver_function(gt_image, gt_slices, label_full_dataset_dir, label_filtered_dir, label_full_slices_dir, label_filtered_slices_dir, patient_id, 'T2')
            
        if row['T2FS'] >= 1:
            T2FS_image = sitk.ReadImage(os.path.join(original_dataset_dir, patient_id, patient_id + '_T2FS.nii.gz'))
            T2FS_image = ImgResample_slices(T2FS_image, label=False)
            
            T2FS_filtered_array = sitk.GetArrayFromImage(T2FS_image)[slices, :, :]
            T2FS_slices = sitk.GetImageFromArray(T2FS_filtered_array)

            saver_function(T2FS_image, T2FS_slices, MRI_full_dataset_dir, MRI_filtered_dir, MRI_full_slices_dir, MRI_filtered_slices_dir, patient_id, 'T2FS')
            saver_function(gt_image, gt_slices, label_full_dataset_dir, label_filtered_dir, label_full_slices_dir, label_filtered_slices_dir, patient_id, 'T2FS')
