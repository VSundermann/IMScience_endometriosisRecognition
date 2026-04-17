from skimage.filters import threshold_otsu
import numpy as np
import os
from skimage import io, measure
from skimage import morphology
import xml.etree.ElementTree as ET
from skimage.color import rgb2gray
import gc
import torch
from torch.nn import Parameter

# def filter_component_by_area(saliency, area_range=[2000,12000]):
#     sal = saliency[0][0]
#     bin_sal = np.copy(sal)
#     #threshold = threshold_otsu(saliency)**2
#     #print(threshold)
#     bin_sal[sal>0] = 1
#     #bin_sal=morphology.area_opening(bin_sal, area_threshold=64, connectivity=2).astype(int)
#     sal_components_image = measure.label(bin_sal, background=0, connectivity=2)
#     sal_nb_components = sal_components_image.max()
    
#     bbs = []
#     for c_ in range(1,sal_nb_components+1):
#         area = len(sal_components_image[sal_components_image == c_])
        
#         if (area < area_range[0] or area > area_range[1]):# or sal[sal_components_image == c_].mean() < threshold_otsu(saliency)):
#             saliency[0][0][sal_components_image == c_] = 0

def filter_component_by_area(saliency, area_range=[1800,10000]):
    filtered_sal = []
    for sal in saliency:
        sal = sal[0]
        bin_sal = np.copy(sal)
        #threshold = threshold_otsu(saliency)**2
        bin_sal[sal>threshold_otsu(sal)] = 1
        bin_sal[sal<=threshold_otsu(sal)] = 0
        sal[bin_sal == 0] = 0
        #bin_sal=morphology.area_opening(bin_sal, area_threshold=64, connectivity=2).astype(int)
        sal_components_image = measure.label(bin_sal, background=0, connectivity=2)
        sal_nb_components = sal_components_image.max()
        
        bbs = []
        for c_ in range(1,sal_nb_components+1):
            area = len(sal_components_image[sal_components_image == c_])
            
            if (area < area_range[0] or area > area_range[1]):# or sal[sal_components_image == c_].mean() < threshold_otsu(saliency)):
                sal[sal_components_image == c_] = 0

def draw_bbs(image, bbs, label_bbs, border_radius, color=[0,121,255], label_color=[0,255,255]):
    drawn_image = np.copy(image)
    combined_color = [0,255,0]
    for bounding_box in bbs:
        min_x = bounding_box[0]
        min_y = bounding_box[1]
        max_x = bounding_box[2]
        max_y = bounding_box[3]
        drawn_image[min_x-border_radius:min_x+border_radius, min_y:max_y,:] = color
        drawn_image[max_x-border_radius:max_x+border_radius, min_y:max_y,:] = color
        drawn_image[min_x:max_x, min_y-border_radius:min_y+border_radius,:] = color
        drawn_image[min_x:max_x, max_y-border_radius:max_y+border_radius,:] = color
        
    for bounding_box in label_bbs:
        min_x = bounding_box[0]
        min_y = bounding_box[1]
        max_x = bounding_box[2]
        max_y = bounding_box[3]
        for x in range(min_x-border_radius, min_x+border_radius):
            for y in range(min_y, max_y):
                if (drawn_image[x,y,:] == color).all() or (drawn_image[x,y,:] == combined_color).all():
                    drawn_image[x,y,:] = combined_color
                else:
                    drawn_image[x,y,:] = label_color
                    
        for x in range(max_x-border_radius, max_x+border_radius):
            for y in range(min_y, max_y):
                if (drawn_image[x,y,:] == color).all() or (drawn_image[x,y,:] == combined_color).all():
                    drawn_image[x,y,:] = combined_color
                else:
                    drawn_image[x,y,:] = label_color
                    
        for y in range(min_y-border_radius, min_y+border_radius):
            for x in range(min_x, max_x):
                if (drawn_image[x,y,:] == color).all() or (drawn_image[x,y,:] == combined_color).all():
                    drawn_image[x,y,:] = combined_color
                else:
                    drawn_image[x,y,:] = label_color
                    
        for y in range(max_y-border_radius, max_y+border_radius):
            for x in range(min_x, max_x):
                if (drawn_image[x,y,:] == color).all() or (drawn_image[x,y,:] == combined_color).all():
                    drawn_image[x,y,:] = combined_color
                else:
                    drawn_image[x,y,:] = label_color
                    
                #drawn_image[max_x-border_radius:max_x+border_radius, min_y:max_y,:] = label_color
                #drawn_image[min_x:max_x, min_y-border_radius:min_y+border_radius,:] = label_color
                #drawn_image[min_x:max_x, max_y-border_radius:max_y+border_radius,:] = label_color
    return drawn_image
    

def readFileList(file_path):
    filelist = open(file_path).read().splitlines()
    return filelist

def readAllFilesFromFolder(file_path, extension):
    filelist = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f)) and f.split(".")[1] == extension]
    return filelist

def binarize_saliency(saliency):
    threshold = threshold_otsu(saliency)
    bin_sal = np.copy(saliency)
    bin_sal[saliency>threshold] = 1
    bin_sal[saliency<=threshold] = 0

    #bin_sal=morphology.area_opening(bin_sal, area_threshold=64, connectivity=2).astype(int)
    #bin_sal=morphology.area_closing(bin_sal, area_threshold=32, connectivity=2).astype(int)
    #bin_sal=morphology.isotropic_erosion(bin_sal, radius=3).astype(int)
    
    return bin_sal

def scale_bb(bb, scale, image_size):
    x_size = bb[2]-bb[0]
    y_size = bb[3]-bb[1]
    x_change = abs(int((x_size*scale) - x_size) // 2)
    y_change = abs(int((y_size*scale) - y_size) // 2)
    bb[0]=max(0, bb[0] - x_change)
    bb[1]=max(0, bb[1] - y_change)
    bb[2]=min(image_size[0], bb[2] + x_change)
    bb[3]=min(image_size[1], bb[3] + y_change)

def bb_is_in_range(bb, size_range):
    if(size_range == None):
        return True
    bb_area = (bb[3] - bb[1]) * (bb[2] - bb[0])
    if(bb_area < size_range[0] or bb_area > size_range[1]):
        return False
    return True
    
def find_minimum_bounding_box(components, component_label, xsize, ysize):
    indices = np.argwhere(components==component_label)
    min_x = max(np.min(indices[:,0])-xsize, 0)
    max_x = min(np.max(indices[:,0])+xsize, components.shape[0])
    min_y = max(np.min(indices[:,1])-ysize, 0)
    max_y = min(np.max(indices[:,1])+ysize, components.shape[1])
       
    bounding_box = [min_x,min_y,max_x,max_y]
    
    return bounding_box

def get_files_extension(folder):
    ext = None
    for f in os.listdir(folder):
        if os.path.isfile(folder+"/"+f):
            ext = f.split(".")[-1]
    assert ext != None, "No files found in "+folder
    return ext
        
def get_bbs_from_saliency_folder(folder, file_list, scale=1.0, bbs_size_range=None):
    files_sal = []
    if(file_list == None):
        files_list =  os.listdir(folder)
    else:
        files_list = file_list
        
    for sal_file in files_list:
        if os.path.isfile(folder+sal_file):
            saliency = io.imread(folder+sal_file)
            if(len(saliency.shape) == 3):
                saliency = saliency[:,:,0]
            bin_sal = binarize_saliency(saliency)
            sal_components_image = measure.label(bin_sal, background=0, connectivity=2)
            sal_nb_components = sal_components_image.max()
            
            bbs = []
            for c_ in range(1,sal_nb_components+1):
                bb = find_minimum_bounding_box(sal_components_image, c_, 0, 0)
                if(scale != 1.0):
                    scale_bb(bb, scale, bin_sal.shape)
                if(bb_is_in_range(bb, bbs_size_range)):
                    bbs.append(bb)
            files_sal.append(bbs)
    return files_sal

def get_bbs_from_saliency(filename, ext, scale=1.0, bbs_size_range=None):
    if(not os.path.isfile(filename+"."+ext)):
        return [], None
    #assert os.path.isfile(filename+"."+ext), "Could not load saliency file: "+filename+"."+ext
    
    saliency = io.imread(filename+"."+ext)
    if(len(saliency.shape) == 3):
        saliency = saliency[:,:,0]

    image_size = saliency.shape
    bin_sal = binarize_saliency(saliency)
    sal_components_image = measure.label(bin_sal, background=0, connectivity=2)
    sal_nb_components = sal_components_image.max()
    
    bbs = []
    for c_ in range(1,sal_nb_components+1):
        bb = find_minimum_bounding_box(sal_components_image, c_, 0, 0)
        if(scale != 1.0):
            scale_bb(bb, scale, bin_sal.shape)
        if(bbs_size_range != None):
            if(bb_is_in_range(bb, bbs_size_range)):
                bbs.append(bb)
        else:
            bbs.append(bb)
    return bbs, image_size

def get_annotation_from_obj(obj):
    label = obj.findtext('name')
    bndbox = obj.find('bndbox')
    xmin = int(float(bndbox.findtext('xmin'))) - 1
    ymin = int(float(bndbox.findtext('ymin'))) - 1
    xmax = int(float(bndbox.findtext('xmax')))
    ymax = int(float(bndbox.findtext('ymax')))
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    return [ymin, xmin, ymax, xmax]

def get_bbs_from_xml(filename, ext, scale=1.0, bbs_size_range=None):
    ann_tree = ET.parse(filename+"."+ext)
    ann_root = ann_tree.getroot()
    
    bbs=[]
    for size in ann_root.findall('size'):
        width = size.findtext('width')
        height = size.findtext('height')
        image_size = [int(width), int(height)]
        
    for obj in ann_root.findall('object'):
        bb = get_annotation_from_obj(obj)
        bbs.append(bb)

    return bbs, image_size
    
def load_bb(file_with_path, ext, scale=1.0, bbs_size_range=None):
    assert ext == "png" or ext == "xml" or ext == "jpg", "Extension "+ext+" not allowed"
    if(ext == "xml"):
        bbs, image_size = get_bbs_from_xml(file_with_path, ext, bbs_size_range)
    elif(ext == "png" or ext == "jpg"):
        bbs, image_size = get_bbs_from_saliency(file_with_path, ext, scale=scale, bbs_size_range=bbs_size_range)
    return bbs, image_size


def get_bbs_from_json(folder):
    return False

def get_bbs_from_folder(folder, file_list, scale=1.0, bbs_size_range=None):
    ext = get_files_extension(folder)
    if(ext == "png" or ext == "jpg"):
        return get_bbs_from_saliency_folder(folder, file_list=file_list, scale=scale, bbs_size_range=bbs_size_range)
    elif(ext == "xml"):
        return get_bbs_from_xml_folder(folder)
    elif(ext == "json"):
        return get_bbs_from_json_folder(folder)
        
        
        
        
#------------------- GPU TRACKING UTILITIES -------------------


dtype_memory_size_dict = {
    torch.float64: 64/8,
    torch.double: 64/8,
    torch.float32: 32/8,
    torch.float: 32/8,
    torch.float16: 16/8,
    torch.half: 16/8,
    torch.int64: 64/8,
    torch.long: 64/8,
    torch.int32: 32/8,
    torch.int: 32/8,
    torch.int16: 16/8,
    torch.short: 16/6,
    torch.uint8: 8/8,
    torch.int8: 8/8,
}
# compatibility of torch1.0
if getattr(torch, "bfloat16", None) is not None:
    dtype_memory_size_dict[torch.bfloat16] = 16/8
if getattr(torch, "bool", None) is not None:
    dtype_memory_size_dict[torch.bool] = 8/8 # pytorch use 1 byte for a bool, see https://github.com/pytorch/pytorch/issues/41571

def get_mem_space(x):
    try:
        ret = dtype_memory_size_dict[x]
    except KeyError:
        print(f"dtype {x} is not supported!")
    return ret

class MemTracker(object):
    """
    Class used to track pytorch memory usage
    Arguments:
        detail(bool, default True): whether the function shows the detail gpu memory usage
        path(str): where to save log file
        verbose(bool, default False): whether show the trivial exception
        device(int): GPU number, default is 0
    """
    def __init__(self, detail=True, path='', verbose=False, device=0):
        self.print_detail = detail
        self.last_tensor_sizes = set()
        self.verbose = verbose
        self.begin = True
        self.device = device

    def get_tensors(self):
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    tensor = obj
                else:
                    continue
                if tensor.is_cuda:
                    yield tensor
            except Exception as e:
                if self.verbose:
                    print('A trivial exception occured: {}'.format(e))

    def get_tensor_usage(self):
        sizes = [np.prod(np.array(tensor.size())) * get_mem_space(tensor.dtype) for tensor in self.get_tensors()]
        return np.sum(sizes) / 1024**2

    def get_allocate_usage(self):
        return torch.cuda.memory_allocated() / 1024**2

    def clear_cache(self):
        gc.collect()
        torch.cuda.empty_cache()

    def print_all_gpu_tensor(self, file=None):
        for x in self.get_tensors():
            print(x.size(), x.dtype, np.prod(np.array(x.size()))*get_mem_space(x.dtype)/1024**2, file=file)

    def track(self):
        """
        Track the GPU memory usage
        """

        return  self.get_allocate_usage()
def get_model_n_params(model, type_size=1):
    if(model.network_type == "regular"):
        para = sum([np.prod(list(p.size())) for p in model.parameters()])
        return para * type_size / 1000
    if(model.network_type == "dregular_sw"):
        para = 0
        for layer in model.layers:
            para += sum([np.prod(list(p.size())) for p in Parameter(layer.conv.weight)])
        return para * type_size / 1000
    elif(model.network_type == "separable"):
        para = 0
        for layer in model.layers:
            para += sum([np.prod(list(p.size())) for p in Parameter(layer.conv.conv[0].weight)]) + sum([np.prod(list(p.size())) for p in Parameter(layer.conv.conv[1].weight)])
        return para * type_size / 1000
    elif(model.network_type == "dseparable_sw"):
        para = 0
        for layer in model.layers:
            para += sum([np.prod(list(p.size())) for p in Parameter(layer.conv.depthwise_convs["weight"])]) + sum([np.prod(list(p.size())) for p in Parameter(layer.conv.pointwise_convs["weight"])])
        return para * type_size / 1000
    elif(model.network_type == "dseparable_mw"):
        para = 0
        for layer in model.layers:
            for d in layer.conv.depthwise_convs["weights"].keys():
                para += sum([np.prod(list(p.size())) for p in Parameter(layer.conv.depthwise_convs["weights"][d])]) + sum([np.prod(list(p.size())) for p in Parameter(layer.conv.pointwise_convs["weights"][d])])
        return para * type_size / 1000
    elif(model.network_type == "dregular_mw"):
        para = 0
        for layer in model.layers:
            for d in layer.conv.weights.keys():
                para += sum([np.prod(list(p.size())) for p in Parameter(layer.conv.weights[d])])
        return para * type_size / 1000
    else:
        return 0
