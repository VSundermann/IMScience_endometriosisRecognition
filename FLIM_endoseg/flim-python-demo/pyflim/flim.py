from pyflim import layers, data, arch, util
import numpy as np
import torch
from torch.nn import Parameter, Conv2d
import torch.nn.functional as F
from skimage import io
import os

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from torch.utils.data import DataLoader
from skimage.util.shape import view_as_windows
from torch.nn import Conv2d,Sequential
import faiss
import random
import torch.nn as nn
import math

class FLIMModel(nn.Module):
    def __init__(self, architecture, learning_rate=1.0, decoder_type="vanilla_adaptive_decoder", adaptation_function="robust_weights", device="cpu", filter_by_size=False, use_bias=False, track_gpu_stats=False, network_type="regular", dilation_values=None, **kwargs):
        torch.set_grad_enabled(False)
        super(FLIMModel,self).__init__()
        self.layers = nn.ModuleList()
        self.architecture = architecture
        self.network_type = network_type
        self.dilation_values = dilation_values
        for l in range(self.architecture.nlayers):
            layer = layers.FLIMLayer(architecture.layers[l], network_type=network_type, dilation_values=dilation_values)
            self.layers.append(layer)

        #sklearn kmeans is not giving consistent results, using faiss instead
        #self.kmeans = FLIMModel.cluster_patches
        self.kmeans = FLIMModel.cluster_patches_faiss
        self.use_bias = use_bias
        if device!="cpu":
            self.kmeans = FLIMModel.cluster_patches_faiss
        self.normalization = FLIMModel.marker_based_normalization
        decoder_input_size = architecture.layers[self.architecture.nlayers-1]["noutput_channels"]
        self.decoder = layers.FLIMAdaptiveDecoderLayer(decoder_input_size, decoder_type=decoder_type, adaptation_function=adaptation_function, filter_by_size=filter_by_size, device=device, **kwargs)
        self.device = device
        self.to(device)
        self.track_gpu_stats = track_gpu_stats
        self.max_gpu_usage = 0.0

    def shift_weights(in_weights, total_kernel_size, ninput_channels):
        kn = total_kernel_size
        mid_index = int(kn/2)+1
        
        in_channels = ninput_channels
        
        weights = np.concatenate((in_weights[in_channels:mid_index*in_channels],in_weights[0:in_channels],in_weights[mid_index*in_channels:]), axis=0)
        return weights


    def load_ift_flim(self, folder):
        for l in range(self.architecture.nlayers):
            layer_weight_file = folder+"/conv"+str(l+1)+"-kernels.npy"
            total_kernel_size = self.architecture.layers[l]["kernel_size"][0] * self.architecture.layers[l]["kernel_size"][1] #* self.architecture.layers[l]["ninput_channels"]
            kernels = FLIMModel.shift_weights(np.load(layer_weight_file), total_kernel_size, self.architecture.layers[l]["ninput_channels"])
            kernels = kernels.transpose()
            kernels = kernels.reshape(self.architecture.layers[l]["noutput_channels"], 
                                      self.architecture.layers[l]["kernel_size"][0], self.architecture.layers[l]["kernel_size"][1],self.architecture.layers[l]["ninput_channels"])
            kernels = kernels.transpose(0,3,1,2)
            weights = torch.from_numpy(kernels).float().to(self.device)

            #weights = torch.from_numpy(kernels).float()
            with torch.no_grad():
                self.layers[l].conv.weight = Parameter(weights)
            means_file = folder+"/conv"+str(l+1)+"-mean.txt"
            self.layers[l].normalization_parameters["mean"] = np.zeros(self.architecture.layers[l]["ninput_channels"])
            with open(means_file) as file:
                for line in file:
                    for i,m in enumerate(line.split(" ")):
                        if(m != ""):
                            self.layers[l].normalization_parameters["mean"][i] = float(m)
            std_file = folder+"/conv"+str(l+1)+"-stdev.txt"
            self.layers[l].normalization_parameters["std"] = np.zeros(self.architecture.layers[l]["ninput_channels"])
            with open(std_file) as file:
                for line in file:
                    for i,s in enumerate(line.split(" ")):
                        if(s != ""):
                            self.layers[l].normalization_parameters["std"][i] = float(s)
            self.layers[l].conv.to(self.device)
            if(self.use_bias):
                bias_file = folder+"/conv"+str(l+1)+"-bias.txt"
                with open(bias_file) as file:
                    N = file.readline()
                    N = int(N[:-1]) #number of elements
                    self.layers[l].normalization_parameters["bias"] = np.zeros(N)
                    for line in file:
                        for i,s in enumerate(line.split(" ")):
                            if(s != ""):
                                self.layers[l].normalization_parameters["bias"][i] = float(s)

                with torch.no_grad():
                    bias_ = torch.from_numpy(self.layers[l].normalization_parameters["bias"]).float()
                    self.layers[l].conv.bias = Parameter(bias_)

    def learn_layer_weights_multi_dilation(self, X, M, l):
        layer_parameters = self.architecture.layers[l]
        patchsize=layer_parameters["kernel_size"]

        for d in self.dilation_values:
            combined_kernel_candidates = []
            kernel_labels = []
            for x,marker_image_labels in zip(X, M):
                features = x.detach()
                features = features.cpu().permute(1,2,0).numpy()
                
                patches = FLIMModel.patchify(features, patchsize, d)
                marker_image = data.FLIMData.label_markers_by_component(marker_image_labels)
                label_patches = FLIMModel.patchify(marker_image_labels, patchsize, layer_parameters["dilation_rate"])
                kernel_candidates = []
                
                for m_label in range(1,marker_image.max()+1):
                    component_marker = np.zeros(marker_image.shape)
                    component_marker[marker_image==m_label] = 1
    
                    representatives, indices = FLIMModel.select_patches(patches, component_marker)
    
                    if(representatives.shape[0] > 0):
                        kernel_candidates_, selected_points = self.kmeans(representatives, layer_parameters["nkernels_per_marker"])

                        selected_points_real = indices[selected_points]

                        if(selected_points is None):
                            for f in indices:
                                kernel_labels.append(label_patches[f, patchsize[0]//2, patchsize[1]//2])
                        else:
                            for f in selected_points_real:
                                kernel_labels.append(label_patches[f, patchsize[0]//2, patchsize[1]//2])
                                
                        for k in kernel_candidates_:
                            kernel_candidates.append(k)
                
                kernel_candidates = np.array(kernel_candidates)
                if(kernel_candidates.shape[0] != 0):
                    for k in kernel_candidates:
                        combined_kernel_candidates.append(k)
                    
            combined_kernel_candidates = np.array(combined_kernel_candidates)
            if(combined_kernel_candidates.shape[0] < layer_parameters["noutput_channels"]):
                layer_parameters["noutput_channels"] = combined_kernel_candidates.shape[0]
                if(l+1 != self.architecture.nlayers):
                    self.architecture.layers[l+1]["ninput_channels"] = combined_kernel_candidates.shape[0]
                if(self.network_type == "dregular_mw"):
                    self.layers[l].conv = layers.MultiDilatedConvolutionMultiWeight(in_channels=layer_parameters["ninput_channels"], out_channels=layer_parameters["noutput_channels"], 
                               kernel_size=layer_parameters["kernel_size"], padding_mode='reflect', dilation_values=self.dilation_values)
                elif(self.network_type == "dseparable_mw"):
                    del self.layers[l].conv
                    self.layers[l].conv = layers.DilatedSeparableConvolutionMultiWeight(in_channels=layer_parameters["ninput_channels"], out_channels=layer_parameters["noutput_channels"], 
                           kernel_size=layer_parameters["kernel_size"], dilation_values=self.dilation_values)
                    if(l+1 != self.architecture.nlayers):
                        layer_parameters_ = self.architecture.layers[l+1]
                        del self.layers[l+1].conv
                        self.layers[l+1].conv = layers.DilatedSeparableConvolutionMultiWeight(in_channels=layer_parameters["ninput_channels"], out_channels=layer_parameters["noutput_channels"], 
                               kernel_size=layer_parameters["kernel_size"], dilation_values=self.dilation_values)

            
            kernels, selected_points = self.kmeans(combined_kernel_candidates, layer_parameters["noutput_channels"])
            selected_kernels_labels=[]

            if(selected_points is None):
                for f in kernel_labels:
                    selected_kernels_labels.append(f)
            else:
                for f in selected_points:
                    selected_kernels_labels.append(int(kernel_labels[f]))
                
            if(not self.network_type == "dseparable_mw"):
                FLIMModel.unit_norm_kernels(kernels)
    
            if(self.use_bias):
                for i,std in enumerate(self.layers[l].normalization_parameters["std"]):
                    kernels[:,:,:,i]/=std
                mean_shifted = np.copy(kernels)
                for i,mean in enumerate(self.layers[l].normalization_parameters["mean"]):
                    mean_shifted[:,:,:,i]*=mean
                bias_ = (-1)*mean_shifted.sum(axis=(1,2,3))
            weights = torch.from_numpy(kernels).permute(0,3,2,1).float().to(self.device)
            
            self.layers[l].marker_labels = torch.Tensor((np.array(selected_kernels_labels)-1).reshape(-1))
            if(self.network_type == "dseparable_mw"):
                shape_vector = weights.shape
                depth_wise_kernel = weights.mean(axis=0).unsqueeze(axis=1).to(self.device)
                FLIMModel.unit_norm_kernels(depth_wise_kernel)
                mean_sum = weights.mean(axis=(0,2,3)).sum()
                mean_percentage = (weights.mean(axis=(0,2,3)) * weights.std(axis=(0,2,3))) / mean_sum
                point_wise_kernel = (weights.mean(axis=(2,3))*mean_percentage).unsqueeze(2).unsqueeze(3).to(self.device)
                FLIMModel.unit_norm_kernels(point_wise_kernel)
                
                values, vectors = torch.linalg.eig(weights)
                self.layers[l].conv.depthwise_convs["weights"][str(d)] = depth_wise_kernel
                self.layers[l].conv.pointwise_convs["weights"][str(d)] = point_wise_kernel
            elif(self.network_type == "dregular_mw"):
                weights = torch.from_numpy(kernels).permute(0,3,2,1).float().to(self.device)
                if(self.use_bias):
                    bias = torch.from_numpy(bias_).float().to(self.device)
                with torch.no_grad():
                    self.layers[l].conv.weights[str(d)] = weights
            
            del kernels


    def learn_layer_weights(self, X, M, l):
        layer_parameters = self.architecture.layers[l]
        patchsize=layer_parameters["kernel_size"]

        combined_kernel_candidates = []
        kernel_labels = []
        for x,marker_image_labels in zip(X, M):
            features = x.detach()
            features = features.cpu().permute(1,2,0).numpy()
            
            patches = FLIMModel.patchify(features, patchsize, layer_parameters["dilation_rate"])
            marker_image = data.FLIMData.label_markers_by_component(marker_image_labels)
            label_patches = FLIMModel.patchify(marker_image_labels, patchsize, layer_parameters["dilation_rate"])
            kernel_candidates = []
            
            for m_label in range(1,marker_image.max()+1):
                component_marker = np.zeros(marker_image.shape)
                component_marker[marker_image==m_label] = 1
                representatives, indices = FLIMModel.select_patches(patches, component_marker)
                if(representatives.shape[0] > 0):
                    kernel_candidates_, selected_points = self.kmeans(representatives, layer_parameters["nkernels_per_marker"])
                    
                    selected_points_real = indices[selected_points]

                    if(selected_points is None):
                        for f in indices:
                            kernel_labels.append(label_patches[f, patchsize[0]//2, patchsize[1]//2])
                    else:
                        for f in selected_points_real:
                            kernel_labels.append(label_patches[f, patchsize[0]//2, patchsize[1]//2])
                            
                    for k in kernel_candidates_:
                        kernel_candidates.append(k)
            
            kernel_candidates = np.array(kernel_candidates)
            if(kernel_candidates.shape[0] != 0):
                for k in kernel_candidates:
                    combined_kernel_candidates.append(k)
                
        combined_kernel_candidates = np.array(combined_kernel_candidates)
        if(combined_kernel_candidates.shape[0] < layer_parameters["noutput_channels"]):
            layer_parameters["noutput_channels"] = combined_kernel_candidates.shape[0]
            if(l+1 != self.architecture.nlayers):
                self.architecture.layers[l+1]["ninput_channels"] = combined_kernel_candidates.shape[0]
                
            if(self.network_type == "regular" or self.network_type == "dregular_sw"):
                self.layers[l].conv = Conv2d(in_channels=layer_parameters["ninput_channels"], out_channels=layer_parameters["noutput_channels"], 
                           kernel_size=layer_parameters["kernel_size"], 
                           dilation=layer_parameters["dilation_rate"],padding=layer_parameters["dilation_rate"]*(layer_parameters["kernel_size"][0]-1)//2, padding_mode='reflect')
            elif(self.network_type == "dseparable_sw"):
                del self.layers[l].conv
                self.layers[l].conv = layers.DilatedSeparableConvolution(in_channels=layer_parameters["ninput_channels"], out_channels=layer_parameters["noutput_channels"], 
                           kernel_size=layer_parameters["kernel_size"], dilation_values=self.dilation_values)
                if(l+1 != self.architecture.nlayers):
                    layer_parameters_ = self.architecture.layers[l+1]
                    del self.layers[l+1].conv
                    self.layers[l+1].conv = layers.DilatedSeparableConvolution(in_channels=layer_parameters_["ninput_channels"], out_channels=layer_parameters_["noutput_channels"], 
                               kernel_size=layer_parameters_["kernel_size"], dilation_values=self.dilation_values)
        
        kernels, selected_points = self.kmeans(combined_kernel_candidates, layer_parameters["noutput_channels"])
        selected_kernels_labels=[]

        if(selected_points is None):
            for f in kernel_labels:
                selected_kernels_labels.append(f)
        else:
            for f in selected_points:
                selected_kernels_labels.append(int(kernel_labels[f]))
                
        if(not self.network_type == "dseparable_sw" or self.network_type == "separable"):
            FLIMModel.unit_norm_kernels(kernels)

        if(self.use_bias):
            for i,std in enumerate(self.layers[l].normalization_parameters["std"]):
                kernels[:,:,:,i]/=std
            #kernels/=self.layers[l].normalization_parameters["std"]
            mean_shifted = np.copy(kernels)
            for i,mean in enumerate(self.layers[l].normalization_parameters["mean"]):
                mean_shifted[:,:,:,i]*=mean
            bias_ = (-1)*mean_shifted.sum(axis=(1,2,3))
        weights = torch.from_numpy(kernels).permute(0,3,2,1).float()
        
        self.layers[l].marker_labels = torch.Tensor((np.array(selected_kernels_labels)-1).reshape(-1))
        if(self.network_type == "dseparable_sw" or self.network_type == "separable"):
            shape_vector = weights.shape
            depth_wise_kernel = weights.mean(axis=0).unsqueeze(axis=1).to(self.device)
            print("Depth wise", depth_wise_kernel.shape)
            FLIMModel.unit_norm_kernels(depth_wise_kernel)
            mu_c = weights.mean(axis=(0,2,3))
            sigma_c = weights.std(axis=(0,2,3))
            beta = mu_c.sum()
            mean_percentage = (mu_c * sigma_c) / beta
            point_wise_kernel = (weights.mean(axis=(2,3))*mean_percentage).unsqueeze(2).unsqueeze(3).to(self.device)
            print(weights.mean(axis=(2,3)).shape, weights.shape, mu_c.shape)
            print("Point wise", point_wise_kernel.shape)
            FLIMModel.unit_norm_kernels(point_wise_kernel)
            
            if(self.network_type == "dseparable_sw"):
                self.layers[l].conv.depthwise_convs["weight"] = depth_wise_kernel
                self.layers[l].conv.pointwise_convs["weight"] = point_wise_kernel
            if(self.network_type == "separable"):
                self.layers[l].conv.conv[0].weight = Parameter(depth_wise_kernel.to(self.device))
                self.layers[l].conv.conv[0].bias = Parameter(torch.zeros(depth_wise_kernel.shape[0]).to(self.device))
                self.layers[l].conv.conv[1].weight = Parameter(point_wise_kernel.to(self.device))
                self.layers[l].conv.conv[1].bias = Parameter(torch.zeros(point_wise_kernel.shape[0]).to(self.device))
        elif(self.network_type == "regular"):
            weights = torch.from_numpy(kernels).permute(0,3,2,1).float().to(self.device)
            if(self.use_bias):
                bias = torch.from_numpy(bias_).float().to(self.device)
            else:
                bias = torch.zeros(kernels.shape[0]).to(self.device)
            with torch.no_grad():
                self.layers[l].conv.weight = Parameter(weights)
                #if(self.use_bias):
                self.layers[l].conv.bias = Parameter(bias)
                
        elif(self.network_type == "dregular_sw"):
            weights = torch.from_numpy(kernels).permute(0,3,2,1).float().to(self.device)
            self.layers[l].conv.weight = weights
        
    def update_markers(self, M,scales):
        M_new = []
        if(isinstance(scales[0], np.ndarray)):
            for m,scale in zip(M, scales):
                index_list = [(x,y) for x,y in np.swapaxes(np.where(m>0), 1,0)]
                new_index_list = [(x,y) for x,y in np.array([scale*a for a in index_list]).astype(int)]
                m_new = np.zeros((int(round(m.shape[0]*scale[0])), int(round(m.shape[1]*scale[1]))))
                for (x,y),(x_,y_) in zip(new_index_list, index_list):
                    m_new[x,y] = m[x_,y_]
                M_new.append(m_new.astype(np.uint8))
        else:
            for m in M:
                index_list = [(x,y) for x,y in np.transpose(np.where(m>0))]
                new_index_list = [(int(round(x*scales[0])),int(round(y*scales[1]))) for x,y in  np.array([x for x in index_list]).astype(np.uint8)]
                m_new = np.zeros((int(round(m.shape[0]*scales[0])), int(round(m.shape[1]*scales[1]))))
                for (x,y),(x_,y_) in zip(new_index_list, index_list):
                    m_new[x,y] = m[x_,y_]
                M_new.append(m_new.astype(np.uint8))
                
        return M_new

    def batch_fit(self, dataset, learning_rate=1.0):
        scales = [1.0,1.0]
        M = []
        for sample_batch in dataset:
            X = sample_batch["image"].float()
            previous_size = sample_batch["image"].shape[2:4]
            s = sample_batch["image"].numpy().shape[2:4]
            scales = [s[0]/previous_size[0],s[1]/previous_size[1]]

            for m in sample_batch["markers_label"]:                            
                marker_image = m
                M.append(marker_image.numpy().astype(np.uint8))
        for l in range(self.architecture.nlayers):
            self.layers[l].normalization_parameters = FLIMModel.find_marker_norm_parameters(X, M, self.architecture.layers[l]["kernel_size"], self.architecture.stdev_factor)
            if(not self.use_bias):
                X = self.normalization(X, self.layers[l].normalization_parameters)

            self.learn_layer_weights(X, M, l)

            #Update data
            previous_size = X.shape[2:4]
            X = self.layers[l].conv(X.to(self.device))
            X = self.layers[l].activation(X)
            X = self.layers[l].pool(X)
            s = X.shape[2:4]
            scales = [s[0]/previous_size[0],s[1]/previous_size[1]]
            M = self.update_markers(M, scales)
        if(self.device != "cpu"):
            for l in range(self.architecture.nlayers):
                self.layers[l].conv.to(self.device)
        
    def fit(self, dataset, learning_rate=1.0):
        if(isinstance(dataset, DataLoader)):
            self.batch_fit(dataset, learning_rate=learning_rate)
        else:
            X = []
            M = []
            previous_size = []
            scales = []
            for sample in dataset:
                X.append(sample["image"])
                previous_size.append(sample["image"].shape[1:3])
                scales.append(np.array(sample["image"].shape)[1:3] / np.array(sample["image"].shape)[1:3])

                marker_image = sample["markers_label"]
                M.append(marker_image.astype(np.uint8))
            for l in range(self.architecture.nlayers):
                self.layers[l].normalization_parameters = FLIMModel.find_marker_norm_parameters(X, M, self.architecture.layers[l]["kernel_size"], self.architecture.stdev_factor)
                X_new = []
                if(not self.use_bias):
                    for x in X:
                        y = self.normalization(x.unsqueeze(0), self.layers[l].normalization_parameters)
                        X_new.append(y.squeeze(0))
                del X
                X = X_new
                if(self.network_type == "dregular_mw" or self.network_type == "dseparable_mw"):
                    self.learn_layer_weights_multi_dilation(X, M, l)
                else:
                    self.learn_layer_weights(X, M, l)

                #Update data
                for i in range(len(X)):
                    X[i] = self.layers[l].conv(torch.unsqueeze(X[i], 0).to(self.device))
                    X[i] = self.layers[l].activation(X[i])
                    X[i] = self.layers[l].pool(X[i]).squeeze()
                    scales[i] = np.array(X[i].shape)[1:3] / np.array(previous_size[i])
                    previous_size[i] = np.array(X[i].shape)[1:3]
                    torch.cuda.empty_cache()
                M = self.update_markers(M, scales)
        if(self.network_type == "regular"):
            if(self.device != "cpu"):
                for l in range(self.architecture.nlayers):
                    self.layers[l].conv.weight.to(self.device)
                    self.layers[l].conv.to(self.device)
    def run(self, dataset, output_folder=None, decoder_layer=None, activations_folder=None): # decoder_layer should be None, since it checks wether the value is None, rather than -1
        if not os.path.exists(output_folder): 
            os.makedirs(output_folder) 
        if activations_folder is not None:
            if not os.path.exists(activations_folder): 
                os.makedirs(activations_folder) 
        if(isinstance(dataset, DataLoader)):
            original_sizes = None
            image_files = None
            for sample_batch in dataset:
                X = sample_batch["image"].float().to(self.device)
                Y = self.forward(X, decoder_layer)#self.layers[decoder_layer].marker_labels.clone(), decoder_layer) # Forward only receives 2 arguments, the marker_labels can be accessed inside forward anyway
                
                original_sizes = sample_batch['original_size']
                image_paths = sample_batch["image_path"]

                for image_path,y,original_size in zip(image_paths, Y, original_sizes):
                    out_size = y.shape[-2:]
                    if(out_size[0] != original_size[0] or out_size[1] != original_size[1]):
                        y = F.interpolate(y.unsqueeze(0), [original_size[0], original_size[1]], mode='bilinear', align_corners=True).squeeze(0)
                    saliency = y[0].detach().numpy().astype(np.uint8)
                    image_file = image_path.split("/")[-1]
                    io.imsave(output_folder+"/"+image_file, saliency, check_contrast=False)
                    for b,c in enumerate(X):
                        channel = ((c/c.max())*255).detach().numpy().astype(np.uint8)
                        io.imsave(activations_folder+"/"+str(b)+"-"+image_file, channel)
                    del saliency
                del X
                del Y
        else:
            for sample in dataset:
                X = sample["image"].to(self.device)
                original_size = sample["original_size"]
                y = self.forward(X.unsqueeze(0), decoder_layer)#self.layers[decoder_layer].marker_labels.clone(), decoder_layer) # Check comment on line 428
                out_size = y.shape[-2:]
                if(out_size[0] != original_size[0] or out_size[1] != original_size[1]):
                    y = F.interpolate(y, [original_size[0], original_size[1]], mode='bilinear', align_corners=True)
                saliency = y[0][0].detach().numpy().astype(np.uint8)
                image_file = sample["image_path"].split("/")[-1]
                io.imsave(output_folder+"/"+image_file, saliency, check_contrast=False)
                    #----- Uncomment the lines below to look at the images activations ----
                for b,c in enumerate(X):
                    channel = ((c/c.max())*255).detach().numpy().astype(np.uint8)
                    io.imsave(activations_folder+"/"+str(b)+"-"+image_file, channel)
                del y
                del X
                del saliency
    
    def forward(self, X, decoder_layer=None):
        original_size = (X.shape[-2:])
        gpu_tracker = util.MemTracker()

        decoder_layer = self.architecture.nlayers - 1 if decoder_layer == None else decoder_layer

        y = None
        for l in range(self.architecture.nlayers):
            if(not self.use_bias):
                X = self.normalization(X, self.layers[l].normalization_parameters)
            X = self.layers[l].conv(X)
            X = self.layers[l].activation(X)
            X = self.layers[l].pool(X)
            if(self.track_gpu_stats):
                if self.max_gpu_usage < gpu_tracker.track():
                    self.max_gpu_usage = gpu_tracker.track()
            if(l == decoder_layer):
                y = self.decoder(X.detach().clone(), original_size, self.layers[l].marker_labels.detach().clone())
                break
        return y
        
    def find_marker_norm_parameters(X, M, kernel_size=[3,3], std_factor=0.01, dilation_rate=1):
        all_patches = []
        for x,marker_image in zip(X, M):
            features = x.detach()
            features = features.permute(1,2,0).cpu().numpy().astype(float)
            selected_features = features[marker_image>0]
            #patches = FLIMModel.patchify(features, kernel_size, dilation_rate)
            #Learning Normalization Parameters
            #all_marker_patches = FLIMModel.select_patches(patches, marker_image)
            #all_patches.append(all_marker_patches)
            all_patches.append(selected_features)
        patch_dataset = np.array([p for patches in all_patches for p in patches])
        
        return FLIMModel.find_norm_parameters(patch_dataset, std_factor)

    def find_norm_parameters(X, std_factor=0.01):
        norm_parameters = dict()
        norm_parameters["std"] = X.std(axis=0)+std_factor
        norm_parameters["mean"] = X.mean(axis=0)
        return norm_parameters
        
    def marker_based_normalization(X, norm_parameters):
        X = FLIMModel.z_score_normalization(X, norm_parameters)
        return X
            
    def z_score_normalization(X, norm_parameters=None):
        if(norm_parameters == None):
            norm_parameters = FLIMModel.find_norm_parameters(X)
        X_norm = torch.clone(X)
        for b in range(X_norm.shape[1]):
            band_mean = norm_parameters["mean"][b]
            band_std = norm_parameters["std"][b]
            X_norm[:,b,:,:] = (X_norm[:,b,:,:] - band_mean) / band_std
        return X_norm
    
    def unit_norm_kernels(K):
        for k in K:
            if(torch.is_tensor(k)):
                sum = torch.sqrt(torch.sum(torch.square(k)))
                if (sum > 0.001):
                    k /= sum
            else:
                sum = np.sqrt(np.sum(np.square(k)))
                if (sum > 0.001):
                    k /= sum

    def patchify(img, patch_shape, dilation_ratio=1):
        if(len(img.shape) > 2):
            pad = dilation_ratio*(patch_shape[0]-1)//2
            padded = np.pad(img, ((pad, pad), (pad, pad), (0,0)), 'constant', constant_values=0)
            X, Y, F = padded.shape
            x, y = patch_shape

            k = (dilation_ratio-1)*(x-1) + x

            patches_ = view_as_windows(padded,(k, k, F)).reshape((-1,k,k,F))
            shape = patches_.shape
            mask = np.zeros(patches_.shape[1:])
            mask[0::dilation_ratio, 0::dilation_ratio, :] = 1
            patches = patches_[:,mask==1]
            n_patches = patches.shape[0]
            patches = patches.reshape((n_patches, math.ceil(shape[1]/dilation_ratio), math.ceil(shape[2]/dilation_ratio), shape[3]))
            contiguous_patches = np.ascontiguousarray(patches)
        else:
            pad = dilation_ratio*(patch_shape[0]-1)//2
            padded = np.pad(img, ((pad, pad), (pad, pad)), 'constant', constant_values=0)
            X, Y = padded.shape
            x, y = patch_shape

            k = (dilation_ratio-1)*(x-1) + x

            patches_ = view_as_windows(padded,(k, k)).reshape((-1,k,k))
            shape = patches_.shape
            mask = np.zeros(patches_.shape[1:])
            mask[0::dilation_ratio, 0::dilation_ratio] = 1
            patches = patches_[:,mask==1]
            n_patches = patches.shape[0]
            patches = patches.reshape((n_patches, math.ceil(shape[1]/dilation_ratio), math.ceil(shape[2]/dilation_ratio)))
            contiguous_patches = np.ascontiguousarray(patches)
        return contiguous_patches


    def randomInit(low, high, nelems):
        assert low < high, "Low is greater than High ("+str(low)+","+str(high)+")"
        total_of_elems = high-low+1
        assert nelems < total_of_elems, "Nelems ("+str(nelems)+") is greater than the total of integer number in the range: ["+str(low)+","+str(high)+"]"

        selected = np.zeros(nelems)
        values = np.zeros(total_of_elems)
        count = np.zeros(total_of_elems)

        t = 0
        for i in range(low, high):
            values[t] = i
            count[t] = 100
            t+=1

        if(nelems == total_of_elems):
            return values

        #Randomly select samples
        t = 0
        roof = total_of_elems - 1
        while(t < nelems):
            i = random.randint(0, roof)
            #i = t
            v = values[i]

            if(count[i] == 0):
                selected[t] = v
                swap = values[i]
                values[i] = values[roof]
                values[roof] = swap
                swap = count[i]
                count[i] = count[roof]
                count[roof] = swap
                t+=1
                roof-=1
            else:
                count[i]-=1
        return selected.astype(np.uint8)

    def cluster_patches(patches, n_clusters=8):
        n, x, y, f = patches.shape
        X = np.reshape(patches, (n,x*y*f))
        kmeans = KMeans(n_clusters = n_clusters, n_init=1, max_iter=100, tol=0.00000000001, copy_x=True, random_state = 3425, verbose=0)
        kmeans.fit(X)
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
        return kmeans.cluster_centers_.reshape((n_clusters,x,y,f))

    def cluster_patches_faiss(patches, n_clusters):
        niter = 100
        verbose = False
        n, x, y, f = patches.shape
        if(n > n_clusters):
            X = np.reshape(patches, (n,x*y*f))
            d = X.shape[1]
            kmeans = faiss.Kmeans(d, n_clusters, niter=niter, verbose=verbose,
                                  min_points_per_centroid = 1, max_points_per_centroid = 10000000, seed=3425, nredo=1, update_index=True)
            kmeans.train(X)
            closest, _ = pairwise_distances_argmin_min(kmeans.centroids, X)
            return kmeans.centroids.reshape((n_clusters,x,y,f)), closest
            #return np.take(patches, closest, 0)
        else:
            return patches.reshape((n,x,y,f)), None
    
    def select_patches(patches, marker):
        _, x, y, f = patches.shape
        pad = (x-1)//2
        marker_p = np.pad(marker, pad, 'constant', constant_values=0)
        X, Y = marker_p.shape
        rx = x//2
        ry = y//2
        mask=marker_p[rx:X-rx,ry:Y-ry]
        mask=mask.flatten()
        
        return patches[mask>0], np.argwhere(mask>0)
