import os
from glob import glob
import torch
import SimpleITK as sitk

from monai import config
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import Activations, AsDiscrete, Compose, LoadImage, SaveImage, ScaleIntensity

main_directory = '../dataset_processed'
MRI_images_dir = os.path.join(main_directory, 'MRI_images/em')
labels_dir = os.path.join(main_directory, 'labels/em')

images = glob(os.path.join(MRI_images_dir, 'D2*.nii.gz'))
labels = glob(os.path.join(labels_dir, 'D2*.nii.gz'))

# define transforms for image and segmentation
imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])
segtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])
val_ds = ArrayDataset(images, imtrans, labels, segtrans)

# sliding window inference for one image at every iteration
val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
saver = SaveImage(output_dir="./output", output_ext=".png", output_postfix="seg", scale=255)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

model.load_state_dict(torch.load("best_metric_model_segmentation2d_array.pth", weights_only=True))
model.eval()

with torch.no_grad():
    for val_data in val_loader:
        val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
        # define sliding window size and batch size for windows inference
        roi_size = (96, 96)
        sw_batch_size = 4
        val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
        val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
        val_labels = decollate_batch(val_labels)
        # compute metric for current iteration
        dice_metric(y_pred=val_outputs, y=val_labels)
        for val_output in val_outputs:
            saver(val_output)
    # aggregate the final mean dice result
    print("evaluation metric:", dice_metric.aggregate().item())
    # reset the status
    dice_metric.reset()