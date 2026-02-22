import pandas as pd
import os
import shutil

dataset_dir = '../dataset_processed'
labels_dir = os.path.join(dataset_dir, 'labels/em')

dataset_info = pd.read_csv('../dataset_processed/dataset_info_D2.csv')

OV_labels_dir = os.path.join(labels_dir, 'ov')
UT_labels_dir = os.path.join(labels_dir, 'ut')
EM_labels_dir = os.path.join(labels_dir, 'em')
CY_labels_dir = os.path.join(labels_dir, 'cy')
CDS_labels_dir = os.path.join(labels_dir, 'cds')

if not os.path.exists(labels_dir):
    os.makedirs(labels_dir)
if not os.path.exists(OV_labels_dir):
    os.makedirs(OV_labels_dir)
if not os.path.exists(UT_labels_dir):
    os.makedirs(UT_labels_dir)
if not os.path.exists(EM_labels_dir):
    os.makedirs(EM_labels_dir)
if not os.path.exists(CY_labels_dir):
    os.makedirs(CY_labels_dir)
if not os.path.exists(CDS_labels_dir):
    os.makedirs(CDS_labels_dir)

for index, row in dataset_info.iterrows():
    patient_id = row['Patient ID']
    T1_MRI_image_path = os.path.join(labels_dir, f'{patient_id}_em.nii.gz')
    T1FS_MRI_image_path = os.path.join(labels_dir, f'{patient_id}_em.nii.gz')
    T2_MRI_image_path = os.path.join(labels_dir, f'{patient_id}_em.nii.gz')
    T2FS_MRI_image_path = os.path.join(labels_dir, f'{patient_id}_em.nii.gz')
    PAT_MRI_image_path = os.path.join(labels_dir, f'{patient_id}_em.nii.gz')

    """ if row['Ovary'] == 1:
        if row['T1'] == 1:
            shutil.copy(T1_MRI_image_path, os.path.join(OV_labels_dir, f'{patient_id}_T1.nii.gz'))
        if row['T1FS'] == 1:
            shutil.copy(T1FS_MRI_image_path, os.path.join(OV_labels_dir, f'{patient_id}_T1FS.nii.gz'))
        if row['T2'] == 1:
            shutil.copy(T2_MRI_image_path, os.path.join(OV_labels_dir, f'{patient_id}_T2.nii.gz'))
        if row['T2FS'] == 1:
            shutil.copy(T2FS_MRI_image_path, os.path.join(OV_labels_dir, f'{patient_id}_T2FS.nii.gz'))
        if row['Other File'] == 1:
            shutil.copy(PAT_MRI_image_path, os.path.join(OV_labels_dir, f'{patient_id}_pat.nii.gz'))
    if row['Uterus'] == 1:
        if row['T1'] == 1:
            shutil.copy(T1_MRI_image_path, os.path.join(UT_labels_dir, f'{patient_id}_T1.nii.gz'))
        if row['T1FS'] == 1:
            shutil.copy(T1FS_MRI_image_path, os.path.join(UT_labels_dir, f'{patient_id}_T1FS.nii.gz'))
        if row['T2'] == 1:
            shutil.copy(T2_MRI_image_path, os.path.join(UT_labels_dir, f'{patient_id}_T2.nii.gz'))  
        if row['T2FS'] == 1:
            shutil.copy(T2FS_MRI_image_path, os.path.join(UT_labels_dir, f'{patient_id}_T2FS.nii.gz'))
        if row['Other File'] == 1:
            shutil.copy(PAT_MRI_image_path, os.path.join(UT_labels_dir, f'{patient_id}_pat.nii.gz')) """
    if row['Endometrioma'] == 1:
        if row['T1'] == 1:
            shutil.copy(T1_MRI_image_path, os.path.join(EM_labels_dir, f'{patient_id}_em_T1.nii.gz'))
        if row['T1FS'] == 1:
            shutil.copy(T1FS_MRI_image_path, os.path.join(EM_labels_dir, f'{patient_id}_em_T1FS.nii.gz'))
        if row['T2'] == 1:
            shutil.copy(T2_MRI_image_path, os.path.join(EM_labels_dir, f'{patient_id}_em_T2.nii.gz'))
        if row['T2FS'] == 1:
            shutil.copy(T2FS_MRI_image_path, os.path.join(EM_labels_dir, f'{patient_id}_em_T2FS.nii.gz'))
        if row['Other File'] == 1:
            shutil.copy(PAT_MRI_image_path, os.path.join(EM_labels_dir, f'{patient_id}_em_pat.nii.gz'))
    """ if row['Cyst'] == 1:
        if row['T1'] == 1:
            shutil.copy(T1_MRI_image_path, os.path.join(CY_labels_dir, f'{patient_id}_T1.nii.gz'))
        if row['T1FS'] == 1:
            shutil.copy(T1FS_MRI_image_path, os.path.join(CY_labels_dir, f'{patient_id}_T1FS.nii.gz'))
        if row['T2'] == 1:
            shutil.copy(T2_MRI_image_path, os.path.join(CY_labels_dir, f'{patient_id}_T2.nii.gz'))
        if row['T2FS'] == 1:
            shutil.copy(T2FS_MRI_image_path, os.path.join(CY_labels_dir, f'{patient_id}_T2FS.nii.gz'))
        if row['Other File'] == 1:
            shutil.copy(PAT_MRI_image_path, os.path.join(CY_labels_dir, f'{patient_id}_pat.nii.gz'))
    if row['Cul de sac'] == 1:
        if row['T1'] == 1:
            shutil.copy(T1_MRI_image_path, os.path.join(CDS_labels_dir, f'{patient_id}_T1.nii.gz'))
        if row['T1FS'] == 1:
            shutil.copy(T1FS_MRI_image_path, os.path.join(CDS_labels_dir, f'{patient_id}_T1FS.nii.gz'))
        if row['T2'] == 1:
            shutil.copy(T2_MRI_image_path, os.path.join(CDS_labels_dir, f'{patient_id}_T2.nii.gz')) 
        if row['T2FS'] == 1:
            shutil.copy(T2FS_MRI_image_path, os.path.join(CDS_labels_dir, f'{patient_id}_T2FS.nii.gz'))
        if row['Other File'] == 1:
            shutil.copy(PAT_MRI_image_path, os.path.join(CDS_labels_dir, f'{patient_id}_pat.nii.gz')) """

    
    # Delete the original files
    """ os.remove(T1_MRI_image_path)
    os.remove(T1FS_MRI_image_path)
    os.remove(T2_MRI_image_path)
    os.remove(T2FS_MRI_image_path)
    os.remove(PAT_MRI_image_path) """