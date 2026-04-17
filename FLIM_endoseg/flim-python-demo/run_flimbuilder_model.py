import random
import time
from model import flim, arch, data, metrics, util
import numpy as np
from torch.utils.data import DataLoader
import argparse
import torch
import os

def get_metrics(dataset_folder, output_folder, file_list, scale_factor=1.0, bb_size_range=None):

    print("Computing metrics...")
    metricas = dict()
        
    results_folder = output_folder
    label_folder = dataset_folder+"/label/"
    
    metricas = metrics.FLIMMetrics()
    metricas.evaluate_detection_results(results_folder, label_folder,file_list=util.readFileList(file_list), bb_scale=scale_factor, bbs_size_range=bb_size_range)
        
    metricas.print_results()
    metricas.save_pr_curve(output_folder+"/"+"pr_curve.png")
    
    
if __name__ == "__main__":
    try:
        ap = argparse.ArgumentParser()
    except:
        ap.print_help()
        sys.exit(0)
    ap.add_argument("-i", "--input_dataset", required=True,	help="path to the folder with <orig> <label> <markers> folders and split files")
    ap.add_argument("-l", "--file_list", required=True,	help="path to the file-list file")
    ap.add_argument("-a", "--arch_file", required=True,	help="path to the architecture file <arch.json>")
    ap.add_argument("-m", "--trained_model_folder", required=True,	help="path to the FLIMBuilder trained model folder")
    ap.add_argument("-d", "--device", required=False, default="cpu", help="device where the model will run (e.g., 'cpu' or 'cuda:0')")
    ap.add_argument("-s", "--scale_factor", required=False, default="1.0", help="Factor to scale the bounding boxes")
    ap.add_argument("-r", "--bb_size_range", required=False, default=None, help="Size range for the bounding_box (E.g. [1200, 30000])")
    ap.add_argument("-f", "--filter_saliency_component", action="store_true", help="Apply size filter in the saliency map directly")
    ap.add_argument("-o", "--output_folder", required=True, help="path to the folder to save the results")
    args = vars(ap.parse_args())
    
    print("Starting and validating parameters...")
    dataset_folder = args["input_dataset"]
    output_folder = args["output_folder"]
    arch_file = args["arch_file"]
    device = args["device"]
    scale_factor = float(args["scale_factor"])
    bb_size_range_s = args["bb_size_range"]
    pre_trained_weights = args["trained_model_folder"]
    file_list = args["file_list"]
    if(bb_size_range_s != None):
        min_size = int(bb_size_range_s.split(",")[0][1:])
        max_size = int(bb_size_range_s.split(",")[1][:-1])
        bb_size_range = [min_size, max_size]
    else:
        bb_size_range = None
    filter_by_size = args["filter_saliency_component"]
    
    orig_folder = dataset_folder+"/orig/"
    marker_folder = dataset_folder+"/markers/"
    label_folder = dataset_folder+"/label/"
    orig_ext = ".png"
    label_ext = ".png"
    
    if not os.path.exists(output_folder): 
            os.makedirs(output_folder) 
    
    print("Loading architecture...")
    architecture = arch.FLIMArchitecture(arch_file)
    assert architecture is not None, "Could not load architecture from "+arch_file
    model = flim.FLIMModel(architecture, adaptation_function="robust_weights", device=device, filter_by_size=filter_by_size)
    assert model is not None, "Failed to create model from architecture"
    
    print("Loading weights...")
    model.load_ift_flim(pre_trained_weights)

    #Run network
    print("Running validation...")
    dataset = data.FLIMData(orig_folder, images_list=file_list, orig_ext=orig_ext,
                                             transform=data.transforms.Compose([data.ToTensor()]))
    start = time.time()
    model.forward(dataset, output_folder)
    stop = time.time()
    print('Forward pass in:', stop - start, 'seconds')
    get_metrics(dataset_folder, output_folder, file_list=file_list, scale_factor=scale_factor, bb_size_range=bb_size_range)
