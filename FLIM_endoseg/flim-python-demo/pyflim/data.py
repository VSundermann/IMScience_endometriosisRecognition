import os
from torch.utils.data import Dataset
from torchvision import transforms, utils
from skimage import io, transform
from skimage import color
from skimage import measure
import torchvision.transforms.functional as F
import torch
import numpy as np
from pyflim import util
from PIL import Image
import torch
import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class FLIMData(Dataset):
    def __init__(self, orig_folder, marker_folder=None, images_list=None, lab_norm=True, label_folder=None, orig_ext=".png", marker_ext="-seeds.txt", label_ext=".png",transform=None, bits=8, convert_gray_to_lab=False):
        """
        Arguments:
            orig_folder (string): Path to the folder with the original images.
            marker_folder (string): Path to the folder with the marker files for each training image.
            training_images_list (string or list): List of training images or path to the txt file with training images (if blank, all files within the orig folder will be considered).
            label_folder (string): Path to the folder with the ground-truth annotation for each image.
            orig_ext (string): File extension to be considered when looking at the orig_folder.
            marker_ext (string): Extension to append to the filename of the original final when looking at the marker_folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            bits (int): Number of bits usage in color information, default is 8 (for 256 colors)
        """
        self.marker_folder = marker_folder
        self.orig_folder = orig_folder
        self.label_folder = label_folder
        self.lab_norm = lab_norm

        if(marker_folder is not None):
            self.mode = "train"
        else:
            self.mode = "test"
        if(images_list is not None):
            self.file_list = FLIMData.get_file_list(images_list)
        else:
            self.file_list = util.readAllFilesFromFolder(orig_folder, orig_ext)
        
        if(self.mode == "train"):
            self.file_list = FLIMData.filter_from_folder(self.file_list, marker_folder, marker_ext)
        
        assert len(self.file_list) >= 1, "No training images were loaded"
        self.file_list = sorted(self.file_list)
        self.orig_ext = orig_ext
        self.marker_ext = marker_ext
        self.label_ext = label_ext
        self.transform = transform
        if(self.transform == None):
            self.batchable = False
        self.results = []
        self.ctb = self.get_blue_to_red_color_table(bits=bits)
        self.bits=bits
        self.convert_gray_to_lab=convert_gray_to_lab

    def test(self):
        self.mode = "test"

    def train(self):
        self.mode = "train"

    def filter_from_folder(read_list, marker_folder, marker_ext):
        if(marker_folder is None):
            return read_list
        index_to_remove = []
        for i,filename in enumerate(read_list):
            marker_file = marker_folder+filename.split(".")[0]+marker_ext
            if not os.path.isfile(marker_file):
                index_to_remove.append(i)
        filtered_list = [i for j, i in enumerate(read_list) if j not in index_to_remove]
        return filtered_list
        

    def get_file_list(file_list):
        if isinstance(file_list, str):
            return util.readFileList(file_list)
        elif isinstance(file_list, list):
            return  file_list
        else:
            assert False, "Image list parameter is neither a list nor a string"
            
    #This function is different than the one implemented in the LibIFT and provide worst results
    def rgb2labnorm(rbg_image):
        lab = color.rgb2lab(rbg_image)
        lab[:,:,0]/=100
        lab[:,:,1]=(lab[:,:,1]+87)/(87+99)
        lab[:,:,2]=(lab[:,:,2]+108)/(108+95)

        return lab

    def _labf(x):
        if x >= 8.85645167903563082e-3:
            return x ** (0.33333333333)
        else:
            return (841.0 / 108.0) * (x) + (4.0 / 29.0)
    
    def image_to_lab(image):
        return FLIMData._image_to_lab(image)

    def image_to_lab_norm(image):
        return FLIMData._image_to_lab_norm(image)      
    
    def get_blue_to_red_color_table(self, bits):
        n_colors=2**bits
        if bits == 8:
            img_dtype = np.uint8
        elif bits == 16:
            img_dtype = np.uint16
        colors = np.arange(0, n_colors).reshape(-1)
        colors = 4 * (colors/n_colors) + 1
        ctb = np.zeros((*colors.shape, 3)).astype(img_dtype)

        ctb[:, 0] = (n_colors - 1) * \
                  np.maximum(0, (3 - abs(colors - 4) - abs(colors - 5)) / 2)
        ctb[:, 1] = (n_colors - 1) * \
                  np.maximum(0, (4 - abs(colors - 2) - abs(colors - 4)) / 2)
        ctb[:, 2] = (n_colors - 1) * \
                  np.maximum(0, (3 - abs(colors - 1) - abs(colors - 2)) / 2)
      
        return ctb
    
    def gray_to_colored(self, gray_image):
        rgb_image = np.zeros((*gray_image.shape, 3))
        rgb_image[:, :, :] = self.ctb[gray_image]
      
        return rgb_image
      
    def _image_to_lab(image):
        image = image / image.max()
    
        labf_v = np.vectorize(FLIMData._labf)
    
        new_image = np.zeros_like(image)
        R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    
        X = (
            0.4123955889674142161 * R
            + 0.3575834307637148171 * G
            + 0.1804926473817015735 * B
        )
        Y = (
            0.2125862307855955516 * R
            + 0.7151703037034108499 * G
            + 0.07220049864333622685 * B
        )
        Z = (
            0.01929721549174694484 * R
            + 0.1191838645808485318 * G
            + 0.9504971251315797660 * B
        )
    
        X = labf_v(X / 0.950456)
        Y = labf_v(Y / 1.0)
        Z = labf_v(Z / 1.088754)
    
        new_image[:, :, 0] = 116 * Y - 16
        new_image[:, :, 1] = 500 * (X - Y)
        new_image[:, :, 2] = 200 * (Y - Z)
    
        return new_image

    def _image_to_lab_norm(image):
        new_image = FLIMData._image_to_lab(image)

        new_image[:, :, 0] = new_image[:, :, 0] / 99.998337
        new_image[:, :, 1] = (new_image[:, :, 1] + 86.182236) / (86.182236 + 98.258614)
        new_image[:, :, 2] = (new_image[:, :, 2] + 107.867744) / (107.867744 + 94.481682)

        return new_image
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        image_path = os.path.join(self.orig_folder,self.file_list[idx])
        if(len(image_path.split(".")) == 1):
            image_path+=self.orig_ext
        
        image_data = io.imread(image_path)
        if (len(image_data.shape) == 2 and self.convert_gray_to_lab):
            image_data = self.gray_to_colored(image_data)
            image = FLIMData.image_to_lab_norm(image_data) if self.lab_norm else FLIMData.image_to_lab(image_data)
        elif(len(image_data.shape) == 3):
            image = FLIMData.image_to_lab_norm(image_data) if self.lab_norm else FLIMData.image_to_lab(image_data)
        else:
            image = np.expand_dims(image_data, 2).astype(np.float32)

        original_size = np.array(image.shape)

        marker_label_image = None
        if(self.marker_folder is not None):
            marker_file = os.path.join(self.marker_folder,self.file_list[idx].split(".")[0]+self.marker_ext)
            marker_image = FLIMData.label_markers_by_component(FLIMData.get_marker_image(marker_file))
            marker_label_image = FLIMData.get_marker_image(marker_file)
        else:
            marker_image = None
            
        if(self.label_folder is not None):
            label_file = os.path.join(self.label_folder,self.file_list[idx].split(".")[0]+self.label_ext) # UPDATE
            label = io.imread(label_file)
        else:
            label = None

        saliency = None
        if(len(self.results) >= idx+1):
            saliency=self.results[idx]

        sample = {'image': image, "image_path":image_path, 'original_size': original_size }

        if(label is not None):
            sample['label'] = label
        if(saliency is not None):
            sample['saliency'] = saliency
        if(marker_image is not None):
            sample['markers'] = marker_image
        if(marker_label_image is not None):
            sample['markers_label'] = marker_label_image
        
        if self.transform:
            sample = self.transform(sample)

        return sample


    def read_image_markers(marker_file):
        marker_list = []
        with open(marker_file) as reader:
            full_file = reader.readlines()
        for line in full_file:
            line_split = line.split(" ")
            if(len(line_split) == 3):
                n_markers = line_split[0]
                image_size = (int(line_split[2].split("\n")[0]), int(line_split[1]))
            else:
                marker_class = int(line_split[3].split("\\")[0])+1
                marker_list.append((int(line_split[1]),int(line_split[0]), marker_class))

        return marker_list, image_size

    def get_marker_image(marker_file):
        if(marker_file.endswith(".txt")):
            markers, image_size = FLIMData.read_image_markers(marker_file)
            marker_image = np.zeros((image_size[0], image_size[1]))
            for m in markers:
                marker_image[m[0], m[1]] = m[2]#+1
            return marker_image
        elif(marker_file.endswith((".png",".jpg"))):
            marker_image = io.imread(marker_file)
            ones_array = np.ones_like(marker_image)
            marker_array = np.where(marker_image > 0, ones_array+1, ones_array) # Normalize image color values to represent binary labels, image must be GT
            return marker_array
        else:
            raise Exception('Unsuported Image type, use .txt, .png or .jpg')

    def label_markers_by_component(marker_image):
        bin_image = np.copy(marker_image)
        bin_image[bin_image>0] = 1
        component_image = measure.label(bin_image, background=0)

        return component_image

    def save_images(self, output_folder):
        if not os.path.exists(output_folder): 
            os.makedirs(output_folder) 
        for sample in self:
            saliency = sample["saliency"]
            image_file = sample["image_path"].split("/")[-1]
            
            io.imsave(output_folder+"/"+image_file, saliency)
        
        
#-----------------------------------Transforms-----------------------------------------------
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, sample):
        image, image_path = sample['image'], sample['image_path']
        label = None
        saliency = None
        markers = None
        if 'label' in sample:
            label = sample['label']
        if 'saliency' in sample:
            saliency = sample['saliency']
        if 'markers' in sample:
            markers = sample['markers']
        if 'markers_label' in sample:
            markers_label = sample['markers_label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        image_new = transform.resize(image, (new_h, new_w))
        if(markers is not None):
            scale = np.array([new_h, new_w]) / np.array([h,w])

            index_list = [(x,y) for x,y in np.transpose(np.where(markers>0))]
            new_index_list = [(x,y) for x,y in  np.array([scale*x for x in index_list]).astype(int)]
            markers_new = np.zeros((int(round(markers.shape[0]*scale[0])), int(round(markers.shape[1]*scale[1]))))
            markers_label_new = np.zeros((int(round(markers_label.shape[0]*scale[0])), int(round(markers_label.shape[1]*scale[1]))))
            for (x,y),(x_,y_) in zip(new_index_list, index_list):
                markers_new[x,y] = markers[x_,y_]
                markers_label_new[x,y] = markers_label[x_,y_]
        else:
            markers_new = None
            
        
        if(label is not None):
            label_new = transform.resize(label, (new_h, new_w))
        else:
            label_new = None

        out_sample = {'image': image_new, "image_path":image_path, 'original_size': sample['original_size']}
        
        if label is not None:
            out_sample['label'] = label
        if saliency is not None:
            out_sample['saliency'] = saliency
        if markers_new is not None:
            out_sample['markers'] = markers_new
            out_sample['markers_label'] = markers_label_new
            
        return out_sample
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        
        image, image_path = sample['image'], sample['image_path']
        label = None
        saliency = None
        markers = None
        if 'label' in sample:
            label = sample['label']
        if 'saliency' in sample:
            saliency = sample['saliency']
        if 'markers' in sample:
            markers = sample['markers']
        if 'markers_label' in sample:
            markers_label = sample['markers_label']
        
        image_new = torch.from_numpy(image).permute((2, 0, 1)).float()
        if(label is not None):
            label_new = torch.unsqueeze(torch.from_numpy(label), 0)
        else:
            label_new = None
            
        
        out_sample = {'image': image_new, "image_path":image_path, 'original_size': sample['original_size']}
        
        if label is not None:
            out_sample['label'] = label
        if saliency is not None:
            out_sample['saliency'] = saliency
        if markers is not None:
            out_sample['markers'] = markers
            out_sample['markers_label'] = markers_label
            
        return out_sample
