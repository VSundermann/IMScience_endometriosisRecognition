import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import RAovSeg_tools as tools
import os


T1_image_original = sitk.ReadImage('../dataset_raw/UT-EndoMRI/D2_TCPW/D2-060/D2-060_T1.nii.gz', sitk.sitkFloat64)
T1FS_image_original = sitk.ReadImage('../dataset_raw/UT-EndoMRI/D2_TCPW/D2-060/D2-060_T1FS.nii.gz', sitk.sitkFloat64)
T2_image_original = sitk.ReadImage('../dataset_raw/UT-EndoMRI/D2_TCPW/D2-060/D2-060_T2.nii.gz', sitk.sitkFloat64)
T2FS_image_original = sitk.ReadImage('../dataset_raw/UT-EndoMRI/D2_TCPW/D2-060/D2-060_T2FS.nii.gz', sitk.sitkFloat64)

T1_array_original = sitk.GetArrayFromImage(T1_image_original)
T1FS_array_original = sitk.GetArrayFromImage(T1FS_image_original)
T2_array_original = sitk.GetArrayFromImage(T2_image_original)
T2FS_array_original = sitk.GetArrayFromImage(T2FS_image_original)

T1_image_processed = sitk.ReadImage('../dataset_processed/MRI_images/ut/D2-060_T1.nii.gz', sitk.sitkFloat64)
T1FS_image_processed = sitk.ReadImage('../dataset_processed/MRI_images/ut/D2-060_T1FS.nii.gz', sitk.sitkFloat64)
T2_image_processed = sitk.ReadImage('../dataset_processed/MRI_images/ut/D2-060_T2.nii.gz', sitk.sitkFloat64)
T2FS_image_processed = sitk.ReadImage('../dataset_processed/MRI_images/ut/D2-060_T2FS.nii.gz', sitk.sitkFloat64)

T1_array_processed = sitk.GetArrayFromImage(T1_image_processed)
T1FS_array_processed = sitk.GetArrayFromImage(T1FS_image_processed)
T2_array_processed = sitk.GetArrayFromImage(T2_image_processed)
T2FS_array_processed = sitk.GetArrayFromImage(T2FS_image_processed)

plt.subplot(2,4,1)
plt.imshow(T1_array_original[18], cmap='gray')
plt.subplot(2,4,2)
plt.imshow(T1FS_array_original[18], cmap='gray')
plt.subplot(2,4,3)
plt.imshow(T2_array_original[18], cmap='gray')
plt.subplot(2,4,4)
plt.imshow(T2FS_array_original[18], cmap='gray')
plt.subplot(2,4,5)
plt.imshow(T1_array_processed[18], cmap='gray')
plt.subplot(2,4,6)
plt.imshow(T1FS_array_processed[18], cmap='gray')
plt.subplot(2,4,7)
plt.imshow(T2_array_processed[18], cmap='gray')
plt.subplot(2,4,8)
plt.imshow(T2FS_array_processed[18], cmap='gray')
plt.show()

"""
    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(Lb[18], cmap='gray')
    plt.title("Ground Truth")
    plt.subplot(1, 3, 2)
    plt.imshow(Pred[18], cmap='gray')
    plt.title("Prediction")
    plt.subplot(1, 3, 3)
    plt.imshow(Pred_postprocessed[18], cmap='gray')
    plt.title("Postprocessed")
    plt.show()

    # Dice calculation
    dsc1 = tools.dsc_cal_np(Pred,Lb)
    dsc2 = tools.dsc_cal_np(Pred_postprocessed,Lb)
    print(f"The DSC between groundtruth and prediction is {dsc1}")
    print(f"The DSC between groundtruth and postprocessed prediction is {dsc2}")
"""