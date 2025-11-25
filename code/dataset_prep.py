import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import RAovSeg_tools as tools

def load_image(image_path):
    """Load a medical image using SimpleITK and return as a NumPy array."""
    image = sitk.ReadImage(image_path, sitk.sitkFloat32)
    #image_array = sitk.GetArrayFromImage(image)
    return image

def preprocess_image(image, out_spacing=(0.35, 0.35, 6.0), out_size=(512, 512, 38), norm_type="percentile_clip", percentile_low=1, percentile_high=99, o1=0.24, o2=0.3):
    """Preprocess the image: resample, normalize, and apply custom preprocessing."""
    # Resample the image
    resampled_image = tools.ImgResample(image, out_spacing=out_spacing, out_size=out_size, is_label=False, pad_value=0)
    
    resampled_array = sitk.GetArrayFromImage(resampled_image)
    # Normalize the image
    normalized_image = tools.ImgNorm(resampled_array, norm_type=norm_type, percentile_low=percentile_low, percentile_high=percentile_high)
    
    # Custom preprocessing
    preprocessed_image = tools.preprocess_(normalized_image, o1=o1, o2=o2)
    
    return preprocessed_image

dataset_info = pd.DataFrame(columns=['Patient ID', 'T1', 'T1FS', 'T2', 'T2FS', 'Ovary', 'Uterus', 'Endometrioma', 'Cyst', 'Cul de sac', 'Other File'])
patient_info = ['', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

dataset_dir='../dataset_raw'
image_dir=os.path.join(dataset_dir,'UT-EndoMRI/D1_MHS')

processed_dataset_dir = '../dataset_processed'
processed_labels_dir = '../dataset_processed/labels'
proc_ov_dir = '../dataset_processed/labels/ov'
proc_ut_dir = '../dataset_processed/labels/ut'
proc_em_dir = '../dataset_processed/labels/em'
proc_cy_dir = '../dataset_processed/labels/cy'
proc_cds_dir = '../dataset_processed/labels/cds'

if not os.path.exists(processed_dataset_dir):
    os.makedirs(processed_dataset_dir)
if not os.path.exists(processed_labels_dir):
    os.makedirs(processed_labels_dir)
if not os.path.exists(proc_ov_dir):
    os.makedirs(proc_ov_dir)
if not os.path.exists(proc_ut_dir):
    os.makedirs(proc_ut_dir)
if not os.path.exists(proc_em_dir):
    os.makedirs(proc_em_dir)
if not os.path.exists(proc_cy_dir):
    os.makedirs(proc_cy_dir)
if not os.path.exists(proc_cds_dir):
    os.makedirs(proc_cds_dir)

cy_files = ('cy.nii.gz','cy_r1.nii.gz','cy_r2.nii.gz','cy_r3.nii.gz')
em_files = ('em.nii.gz','em_r1.nii.gz','em_r2.nii.gz','em_r3.nii.gz')
ut_files = ('ut.nii.gz','ut_r1.nii.gz','ut_r2.nii.gz','ut_r3.nii.gz')
ov_files = ('ov.nii.gz','ov_r1.nii.gz','ov_r2.nii.gz','ov_r3.nii.gz')
cds_files = ('cds.nii.gz','cds_r1.nii.gz','cds_r2.nii.gz','cds_r3.nii.gz')

for folder in os.listdir(image_dir):
    folder_path=os.path.join(image_dir,folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(cy_files):
                image_path=os.path.join(folder_path,file)
                image_array=load_image(image_path)
                preprocessed_image=tools.ImgResample(image_array, out_spacing=(0.35, 0.35, 6.0), out_size=(512, 512, 38), is_label=True, pad_value=0)
                patient_info[8] += 1
                sitk.WriteImage(preprocessed_image, os.path.join(proc_cy_dir, file))
            elif file.endswith(em_files):
                image_path=os.path.join(folder_path,file)
                image_array=load_image(image_path)
                preprocessed_image=tools.ImgResample(image_array, out_spacing=(0.35, 0.35, 6.0), out_size=(512, 512, 38), is_label=True, pad_value=0)
                patient_info[7] += 1
                sitk.WriteImage(preprocessed_image, os.path.join(proc_em_dir, file))
            elif file.endswith(ut_files):
                image_path=os.path.join(folder_path,file)
                image_array=load_image(image_path)
                preprocessed_image=tools.ImgResample(image_array, out_spacing=(0.35, 0.35, 6.0), out_size=(512, 512, 38), is_label=True, pad_value=0)
                patient_info[6] += 1
                sitk.WriteImage(preprocessed_image, os.path.join(proc_ut_dir, file))
            elif file.endswith(ov_files):
                image_path=os.path.join(folder_path,file)
                image_array=load_image(image_path)
                preprocessed_image=tools.ImgResample(image_array, out_spacing=(0.35, 0.35, 6.0), out_size=(512, 512, 38), is_label=True, pad_value=0)
                patient_info[5] += 1
                sitk.WriteImage(preprocessed_image, os.path.join(proc_ov_dir, file))
            elif file.endswith(cds_files):
                image_path=os.path.join(folder_path,file)
                image_array=load_image(image_path)
                preprocessed_image=tools.ImgResample(image_array, out_spacing=(0.35, 0.35, 6.0), out_size=(512, 512, 38), is_label=True, pad_value=0)
                patient_info[9] += 1
                sitk.WriteImage(preprocessed_image, os.path.join(proc_cds_dir, file))
            elif file.endswith('T1.nii.gz'):
                image_path=os.path.join(folder_path,file)
                image_array=load_image(image_path)
                preprocessed_image=preprocess_image(image_array)
                patient_info[1] += 1
                sitk.WriteImage(sitk.GetImageFromArray(preprocessed_image), os.path.join(processed_dataset_dir, file))
            elif file.endswith('T1FS.nii.gz'):
                image_path=os.path.join(folder_path,file)
                image_array=load_image(image_path)
                preprocessed_image=preprocess_image(image_array)
                patient_info[2] += 1
                sitk.WriteImage(sitk.GetImageFromArray(preprocessed_image), os.path.join(processed_dataset_dir, file))
            elif file.endswith('T2.nii.gz'):
                image_path=os.path.join(folder_path,file)
                image_array=load_image(image_path)
                preprocessed_image=preprocess_image(image_array)
                patient_info[3] += 1
                sitk.WriteImage(sitk.GetImageFromArray(preprocessed_image), os.path.join(processed_dataset_dir, file))
            elif file.endswith('T2FS.nii.gz'):
                image_path=os.path.join(folder_path,file)
                image_array=load_image(image_path)
                preprocessed_image=preprocess_image(image_array)
                patient_info[4] += 1
                sitk.WriteImage(sitk.GetImageFromArray(preprocessed_image), os.path.join(processed_dataset_dir, file))
            else:
                image_path=os.path.join(folder_path,file)
                image_array=load_image(image_path)
                preprocessed_image=preprocess_image(image_array)
                patient_info[10] += 1
                sitk.WriteImage(sitk.GetImageFromArray(preprocessed_image), os.path.join(processed_dataset_dir, file))
                

    patient_info[0] = folder
    dataset_info.loc[len(dataset_info)] = patient_info
    print(f"Patient {folder} processed")
    patient_info = ['', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    dataset_info.to_csv(os.path.join(processed_dataset_dir, 'dataset_info_D1.csv'), index=False)    
        

