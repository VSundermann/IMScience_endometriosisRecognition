import SimpleITK as sitk
import RAovSeg_tools as tools
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import torch
from torch.utils.tensorboard import SummaryWriter
import monai
import PIL
import os

from monai.engines import SupervisedTrainer
from monai.data import decollate_batch, DataLoader, ArrayDataset
from monai.visualize import plot_2d_or_3d_image
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks import eval_mode
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    ResampleToMatchd,
)

##########################################
### DATASET SETUP
##########################################

main_directory = '../dataset_processed'
MRI_images_dir = os.path.join(main_directory, 'MRI_images/ov')
labels_dir = os.path.join(main_directory, 'labels/ov')

images = glob(os.path.join(MRI_images_dir, 'D2*.nii.gz'))
labels = glob(os.path.join(labels_dir, 'D2*.nii.gz'))

"""
plt.subplots(3, 3, figsize=(8, 8))
for i, k in enumerate(np.random.randint(num_total, size=9)):
    im = sitk.ReadImage(image_files_list[k])
    arr = sitk.GetArrayFromImage(im)
    plt.subplot(3, 3, i + 1)
    plt.xlabel(class_names[image_class[k]])
    plt.imshow(arr[18], cmap="gray")
plt.tight_layout()
plt.show()
"""
##########################################
### DIVISION OF TRAINING, TEST AND VALIDATION SETS
##########################################

train_imtrans = Compose(
        [
            LoadImage(image_only=True, ensure_channel_first=True),
            ScaleIntensity(),
            #RandSpatialCrop((96, 96), random_size=False),
            #RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        ]
    )

train_segtrans = Compose(
        [
            LoadImage(image_only=True, ensure_channel_first=True),
            ScaleIntensity(),
            #RandSpatialCrop((96, 96), random_size=False),
            #RandRotate90(prob=0.5, spatial_axes=(0, 1)),
        ]
    )

val_imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])
val_segtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])

 # define array dataset, data loader
check_ds = ArrayDataset(images, train_imtrans, labels, train_segtrans)
check_loader = DataLoader(check_ds, batch_size=10, num_workers=2, pin_memory=torch.cuda.is_available())
im, seg = monai.utils.misc.first(check_loader)
print(im.shape, seg.shape)

# create a training data loader
train_ds = ArrayDataset(images[:20], train_imtrans, labels[:20], train_segtrans)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())

# create a validation data loader
val_ds = ArrayDataset(images[-20:], val_imtrans, labels[-20:], val_segtrans)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

##########################################
### MODEL PIPELINE
##########################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
    # ResClass
    # It was trained on 2D MRI slices from all training subjects, utilizing 3,252 slices for training and 2,168 slices for validation.
    # The model architecture is a two-layer 2D ResNet18 with 8 and 16 features in the respective layers.
    # Binary Cross Entropy with Logits Loss (BCEWithLogitsLoss) was used to train the classifier


    classification_model = monai.networks.nets.ResNetFeatures(model_name="resnet18").to(device)
    resclass_loss = torch.nn.BCEWithLogitsLoss()
    optimizer_classification = torch.optim.Adam(classification_model.parameters(), lr=1e-4)


    trainer = SupervisedTrainer(
        device=device,
        max_epochs=2,
        train_data_loader=train_loader,
        network=classification_model,
        optimizer=optimizer_classification,
        loss_function=resclass_loss,
    )

    trainer.run()


    max_items_to_print = 10
    with eval_mode(classification_model):
        for item in preprocessed_images:
            prob = np.array(classification_model(item["image"].to(device)).detach().to("cpu"))[0]
            pred = class_names[prob.argmax()]
            gt = item["class_name"][0]
            print(f"Class prediction is {pred}. Ground-truth: {gt}")
            max_items_to_print -= 1
            if max_items_to_print == 0:
                break 
"""

# AttUSeg
# 594 MRI slices for training and 136 MRI slices for validation.
# This model was developed using a four-layer Attention U-Net architecture, with 16, 32, 64, and 128 features in each layer.
# The Focal Tversky Loss function, with parameters α = 0.8, β = 0.2, and γ = 1.33, was employed for training
# To mitigate overfitting, we increased the size of the validation set, 
# incorporated a dropout layer with a probability of 0.2, and applied L2 regularization

segmentation_model = monai.networks.nets.AttentionUnet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2, 2),
    dropout=0.2,
).to(device)
#attuseg_loss = tools.focal_tversky()
optimizer_segmentation = torch.optim.Adam(segmentation_model.parameters(), lr=1e-4, weight_decay=1e-5)

val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
metric_values = list()
writer = SummaryWriter()

for epoch in range(4):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{10}")
    segmentation_model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer_segmentation.zero_grad()
        outputs = segmentation_model(inputs)
        loss = tools.focal_tversky(labels, outputs)
        loss.backward()
        optimizer_segmentation.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        segmentation_model.eval()
        with torch.no_grad():
            val_images = None
            val_labels = None
            val_outputs = None
            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                roi_size = (96, 96)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, segmentation_model)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)
            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(segmentation_model.state_dict(), "best_metric_model_segmentation2d_array.pth")
                print("saved new best metric model")
            print(
                "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                    epoch + 1, metric, best_metric, best_metric_epoch
                )
            )
            writer.add_scalar("val_mean_dice", metric, epoch + 1)
            # plot the last model output as GIF image in TensorBoard with the corresponding image and label
            plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
            plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
            plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer.close()