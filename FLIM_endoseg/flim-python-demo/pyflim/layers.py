import torch
from torch.nn import Conv2d, MaxPool2d, AvgPool2d, ReLU, Sequential, Parameter
from sklearn.metrics import jaccard_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import fbeta_score
import numpy as np
import torch.nn.functional as F
import math
from skimage.filters import threshold_otsu
from skimage import io, transform
from PIL import Image
from pyflim import util
from skimage.util import view_as_windows

class SeparableConvolution(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding_mode='reflect', stride=1, d=1):
        super(SeparableConvolution, self).__init__()
        self.conv = torch.nn.Sequential(
                Conv2d(in_channels=in_channels, out_channels=in_channels, 
                            kernel_size=kernel_size, dilation=d, stride=stride,
                            padding=d*(kernel_size-1)//2, padding_mode=padding_mode, 
                            groups=in_channels),
                #ReLU(inplace=True),
                Conv2d(in_channels=in_channels, 
                           out_channels=out_channels, kernel_size=1, stride=1),
                #ReLU(inplace=True))
        )
    def forward(self, x):
        return self.conv(x)
    

class DilatedSeparableConvolution(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_values=None):
        super(DilatedSeparableConvolution, self).__init__()
        self.depthwise_convs = dict()
        self.pointwise_convs = dict()
        self.layer_dict = dict()
        
        if(dilation_values is None):
            self.dilation_values = [1,2,3]
        else:
            self.dilation_values = dilation_values
        self.activation = ReLU()
        
        self.layer_dict["in_channels"] = in_channels
        self.layer_dict["out_channels"] = out_channels
        self.layer_dict["kernel_size"] = kernel_size
        # self.conv = torch.nn.Sequential(
        #         Conv2d(in_channels=in_channels, out_channels=out_channels, 
        #                     kernel_size=kernel_size, dilation=d, stride=stride,
        #                     padding=d*(kernel_size-1)//2, padding_mode=padding_mode, 
        #                     groups=in_channels),
        #         #ReLU(inplace=True),
        #         Conv2d(in_channels=in_channels, 
        #                    out_channels=out_channels, kernel_size=1, stride=1),
        #         #ReLU(inplace=True))
        # )

    def set_and_run(self, device, d, x):
        conv = Conv2d(in_channels=self.layer_dict["in_channels"], out_channels=self.layer_dict["in_channels"], 
                            kernel_size=self.layer_dict["kernel_size"][0], dilation=d, stride=1,
                            padding=d*(self.layer_dict["kernel_size"][0]-1)//2, padding_mode='reflect', 
                            groups=self.layer_dict["in_channels"])
        conv.weight = Parameter(self.depthwise_convs["weight"]).to(device)
        conv.bias = Parameter(torch.zeros(self.depthwise_convs["weight"].shape[0]).to(device))
        x_ = conv(x)
        del x
        conv = Conv2d(in_channels=self.layer_dict["in_channels"], 
                           out_channels=self.layer_dict["out_channels"], kernel_size=1, stride=1)
        conv.weight = Parameter(self.pointwise_convs["weight"].to(device))
        conv.bias = Parameter(torch.zeros(self.pointwise_convs["weight"].shape[0]).to(device))
        y = conv(x_)
        del x_
        return y

    def forward(self, x):
        y_sum = None
        dev = x.get_device()
        if(dev == -1):
            device = "cpu"
        else:
            device = "cuda:"+str(dev)
        for i in range(len(self.dilation_values)):
            d = self.dilation_values[i]
            y = self.set_and_run(dev, d, x)
            if(y_sum == None):
                y_sum = torch.zeros(y.shape).to(device)
            y_sum+=y
            del y
        del x
        return y_sum
        
class DilatedSeparableConvolutionMultiWeight(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding_mode, dilation_values=None):
        super(DilatedSeparableConvolutionMultiWeight, self).__init__()
        self.depthwise_convs = dict()
        self.pointwise_convs = dict()
        
        if(dilation_values is None):
            self.dilation_values = [1,2,3]
        else:
            self.dilation_values = dilation_values
        self.activation = ReLU()
        
        self.depthwise_convs["in_channels"] = in_channels
        self.depthwise_convs["out_channels"] = in_channels*out_channels
        self.depthwise_convs["kernel_size"] = kernel_size
        self.depthwise_convs["groups"] = in_channels
        self.depthwise_convs["weights"] = dict()
        
        self.pointwise_convs["in_channels"] = in_channels*out_channels
        self.pointwise_convs["out_channels"] = out_channels
        self.pointwise_convs["kernel_size"] = 1
        self.pointwise_convs["groups"] = out_channels
        self.pointwise_convs["weights"] = dict()
            

    def forward(self, x):
        y_sum = None
        dev = x.get_device()
        if(dev == -1):
            device = "cpu"
        else:
            device = "cuda:"+str(dev)
        for i in range(len(self.dilation_values)):
            d = self.dilation_values[i]
            
            #Depth-wise
            layer_dict = self.depthwise_convs
            depthwise_conv = Conv2d(in_channels=layer_dict["in_channels"], out_channels=layer_dict["out_channels"], 
                            kernel_size=layer_dict["kernel_size"], dilation=d,
                            padding=d*(layer_dict["kernel_size"][0]-1)//2, padding_mode='reflect', 
                            groups=layer_dict["groups"])
            depthwise_conv.weight = Parameter(layer_dict["weights"][str(d)])
            bias = torch.zeros(layer_dict["weights"][str(d)].shape[0]).to(device)
            depthwise_conv.bias = Parameter(bias)
            y = depthwise_conv(x)
            del depthwise_conv
            
            #Point-wise                                
            layer_dict = self.pointwise_convs
            pointwise_conv = Conv2d(in_channels=layer_dict["in_channels"], 
                           out_channels=layer_dict["out_channels"], kernel_size=1, stride=1, groups=layer_dict["groups"])
            pointwise_conv.weight = Parameter(layer_dict["weights"][str(d)])
            bias = torch.zeros(layer_dict["weights"][str(d)].shape[0]).to(device)
            pointwise_conv.bias = Parameter(bias)
            
            y = pointwise_conv(y)
            del pointwise_conv
            y = self.activation(y)
            if(y_sum == None):
                y_sum = torch.zeros(y.shape).to(device)
            y_sum+=y
            del y
        del x
        return y_sum
        
class MultiDilatedConvolutionSingleWeight(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding_mode="reflect", dilation_values=None):
        super(MultiDilatedConvolutionSingleWeight, self).__init__()
        self.weight = None
        
        if(dilation_values is None):
            self.dilation_values = [1,2,3]
        else:
            self.dilation_values = dilation_values
        self.activation = ReLU()
            
        self.arch = dict()
        self.arch["in_channels"] = in_channels
        self.arch["out_channels"] = out_channels
        self.arch["kernel_size"] = kernel_size
        self.arch["padding_mode"] = padding_mode

    def forward(self, x):
        y_sum = None
        dev = x.get_device()
        if(dev == -1):
            device = "cpu"
        else:
            device = "cuda:"+str(dev)
        for i in range(len(self.dilation_values)):
            d = self.dilation_values[i]
            
            #Depth-wise
            conv = Conv2d(in_channels=self.arch["in_channels"], out_channels=self.arch["out_channels"], 
                            kernel_size=self.arch["kernel_size"], dilation=d,
                            padding=d*(self.arch["kernel_size"][0]-1)//2, padding_mode=self.arch["padding_mode"])
            bias = torch.zeros(self.weight.shape[0]).to(device)
            conv.weight = Parameter(self.weight)
            conv.bias = Parameter(bias)
            y = conv(x)
            del conv
            
            y = self.activation(y)
            if(y_sum == None):
                y_sum = torch.zeros(y.shape).to(device)
            y_sum+=y
            del y
        del x
        return y_sum

    def update_weight(self, depthwise_weight, pointwise_weight):
        for i in range(len(self.dilation_values)):
            d = depthwise_weight.get_device()
            if(d == -1):
                device = "cpu"
            else:
                device = "cuda:"+str(d)
                
            bias = torch.zeros(depthwise_weight.shape[0]).to(device)
            self.depthwise_convs[i].to(device)
            self.depthwise_convs[i].weight = Parameter(depthwise_weight)
            self.depthwise_convs[i].bias = Parameter(bias)
            
            self.pointwise_convs[i].to(device)
            bias = torch.zeros(pointwise_weight.shape[0]).to(device)
            self.pointwise_convs[i].weight = Parameter(pointwise_weight)
            self.pointwise_convs[i].bias = Parameter(bias)

class MultiDilatedConvolutionMultiWeight(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding_mode="reflect", dilation_values=None):
        super(MultiDilatedConvolutionMultiWeight, self).__init__()
        self.weights = dict()
        
        if(dilation_values is None):
            self.dilation_values = [1,2,3]
        else:
            self.dilation_values = dilation_values
        self.activation = ReLU()
            
        self.arch = dict()
        self.arch["in_channels"] = in_channels
        self.arch["out_channels"] = out_channels
        self.arch["kernel_size"] = kernel_size
        self.arch["padding_mode"] = padding_mode

    def forward(self, x):
        y_sum = None
        dev = x.get_device()
        if(dev == -1):
            device = "cpu"
        else:
            device = "cuda:"+str(dev)
        for i in range(len(self.dilation_values)):
            d = self.dilation_values[i]
            
            #Depth-wise
            conv = Conv2d(in_channels=self.arch["in_channels"], out_channels=self.arch["out_channels"], 
                            kernel_size=self.arch["kernel_size"], dilation=d,
                            padding=d*(self.arch["kernel_size"][0]-1)//2, padding_mode=self.arch["padding_mode"])
            bias = torch.zeros(self.weights[str(d)].shape[0]).to(device)
            conv.weight = Parameter(self.weights[str(d)])
            conv.bias = Parameter(bias)
            y = conv(x)
            del conv
            
            y = self.activation(y)
            if(y_sum == None):
                y_sum = torch.zeros(y.shape).to(device)
            y_sum+=y
            del y
        del x
        return y_sum
        
class DilatedSeparableConvolutionMultiWeight(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_values=None):
        super(DilatedSeparableConvolutionMultiWeight, self).__init__()
        self.depthwise_convs = dict()
        self.depthwise_convs["weights"] = dict()
        self.pointwise_convs = dict()
        self.pointwise_convs["weights"] = dict()
        self.layer_dict = dict()
        
        if(dilation_values is None):
            self.dilation_values = [1,2,3]
        else:
            self.dilation_values = dilation_values
        self.activation = ReLU()
        
        self.layer_dict["in_channels"] = in_channels
        self.layer_dict["out_channels"] = out_channels
        self.layer_dict["kernel_size"] = kernel_size

    def set_and_run(self, device, d, x):
        conv = Conv2d(in_channels=self.layer_dict["in_channels"], out_channels=self.layer_dict["in_channels"], 
                            kernel_size=self.layer_dict["kernel_size"][0], dilation=d, stride=1,
                            padding=d*(self.layer_dict["kernel_size"][0]-1)//2, padding_mode='reflect', 
                            groups=self.layer_dict["in_channels"])
        conv.weight = Parameter(self.depthwise_convs["weights"][str(d)].to(device))
        conv.bias = Parameter(torch.zeros(self.depthwise_convs["weights"][str(d)].shape[0]).to(device))
        x_ = conv(x)
        del x
        conv = Conv2d(in_channels=self.layer_dict["in_channels"], 
                           out_channels=self.layer_dict["out_channels"], kernel_size=1, stride=1)
        conv.weight = Parameter(self.pointwise_convs["weights"][str(d)].to(device))
        conv.bias = Parameter(torch.zeros(self.pointwise_convs["weights"][str(d)].shape[0]).to(device))
        y = conv(x_)
        del x_
        return y
            

    def forward(self, x):
        y_sum = None
        dev = x.get_device()
        if(dev == -1):
            device = "cpu"
        else:
            device = "cuda:"+str(dev)
        for i in range(len(self.dilation_values)):
            d = self.dilation_values[i]
            y = self.set_and_run(dev, d, x)
            if(y_sum == None):
                y_sum = torch.zeros(y.shape).to(device)
            y_sum+=y
            del y
        del x
        return y_sum
        
class MultiDilatedConvolutionSingleWeight(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding_mode="reflect", dilation_values=None):
        super(MultiDilatedConvolutionSingleWeight, self).__init__()
        self.weight = None
        
        if(dilation_values is None):
            self.dilation_values = [1,2,3]
        else:
            self.dilation_values = dilation_values
        self.activation = ReLU()
            
        self.arch = dict()
        self.arch["in_channels"] = in_channels
        self.arch["out_channels"] = out_channels
        self.arch["kernel_size"] = kernel_size
        self.arch["padding_mode"] = padding_mode

    def forward(self, x):
        y_sum = None
        dev = x.get_device()
        if(dev == -1):
            device = "cpu"
        else:
            device = "cuda:"+str(dev)
        for i in range(len(self.dilation_values)):
            d = self.dilation_values[i]
            
            #Depth-wise
            conv = Conv2d(in_channels=self.arch["in_channels"], out_channels=self.arch["out_channels"], 
                            kernel_size=self.arch["kernel_size"], dilation=d,
                            padding=d*(self.arch["kernel_size"][0]-1)//2, padding_mode=self.arch["padding_mode"])
            bias = torch.zeros(self.weight.shape[0]).to(device)
            conv.weight = Parameter(self.weight)
            conv.bias = Parameter(bias)
            y = conv(x)
            del conv
            
            y = self.activation(y)
            if(y_sum == None):
                y_sum = torch.zeros(y.shape).to(device)
            y_sum+=y
            del y
        del x
        return y_sum


class MultiDilatedConvolutionMultiWeight(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding_mode="reflect", dilation_values=None):
        super(MultiDilatedConvolutionMultiWeight, self).__init__()
        self.weights = dict()
        
        if(dilation_values is None):
            self.dilation_values = [1,2,3]
        else:
            self.dilation_values = dilation_values
        self.activation = ReLU()
            
        self.arch = dict()
        self.arch["in_channels"] = in_channels
        self.arch["out_channels"] = out_channels
        self.arch["kernel_size"] = kernel_size
        self.arch["padding_mode"] = padding_mode

    def forward(self, x):
        y_sum = None
        dev = x.get_device()
        if(dev == -1):
            device = "cpu"
        else:
            device = "cuda:"+str(dev)
        for i in range(len(self.dilation_values)):
            d = self.dilation_values[i]
            
            #Depth-wise
            conv = Conv2d(in_channels=self.arch["in_channels"], out_channels=self.arch["out_channels"], 
                            kernel_size=self.arch["kernel_size"], dilation=d,
                            padding=d*(self.arch["kernel_size"][0]-1)//2, padding_mode=self.arch["padding_mode"])
            bias = torch.zeros(self.weights[str(d)].shape[0]).to(device)
            conv.weight = Parameter(self.weights[str(d)])
            conv.bias = Parameter(bias)
            y = conv(x)
            del conv
            
            y = self.activation(y)
            if(y_sum == None):
                y_sum = torch.zeros(y.shape).to(device)
            y_sum+=y
            del y
        del x
        return y_sum

    def update_weight(self, depthwise_weight, pointwise_weight):
        for i in range(len(self.dilation_values)):
            d = depthwise_weight.get_device()
            if(d == -1):
                device = "cpu"
            else:
                device = "cuda:"+str(d)
                
            bias = torch.zeros(depthwise_weight.shape[0]).to(device)
            self.depthwise_convs[i].to(device)
            self.depthwise_convs[i].weight = Parameter(depthwise_weight)
            self.depthwise_convs[i].bias = Parameter(bias)
            
            self.pointwise_convs[i].to(device)
            bias = torch.zeros(pointwise_weight.shape[0]).to(device)
            self.pointwise_convs[i].weight = Parameter(pointwise_weight)
            self.pointwise_convs[i].bias = Parameter(bias)
            
class FLIMLayer(torch.nn.Module):
    def __init__(self, layer_parameters, device="cpu", network_type="regular", dilation_values=None):
        super(FLIMLayer, self).__init__()
        self.trained = False
        seed = 3
        torch.manual_seed(seed)
        self.device = device
        self.normalization_parameters = dict()
        self.marker_labels = torch.Tensor()
        if(network_type == "regular"):
            self.conv = Conv2d(in_channels=layer_parameters["ninput_channels"], out_channels=layer_parameters["noutput_channels"], 
                           kernel_size=layer_parameters["kernel_size"], stride=1, 
                           dilation=layer_parameters["dilation_rate"],padding=layer_parameters["dilation_rate"]*(layer_parameters["kernel_size"][0]-1)//2, padding_mode='reflect')
            
        if(network_type == "separable"):
            self.conv = SeparableConvolution(in_channels=layer_parameters["ninput_channels"], out_channels=layer_parameters["noutput_channels"], 
                            kernel_size=layer_parameters["kernel_size"][0], d=layer_parameters["dilation_rate"])
        elif(network_type == "dseparable_sw"):
            self.conv = DilatedSeparableConvolution(in_channels=layer_parameters["ninput_channels"], out_channels=layer_parameters["noutput_channels"], 
                            kernel_size=layer_parameters["kernel_size"], dilation_values=dilation_values)
        elif(network_type == "dseparable_mw"):
            self.conv = DilatedSeparableConvolutionMultiWeight(in_channels=layer_parameters["ninput_channels"], out_channels=layer_parameters["noutput_channels"], 
                            kernel_size=layer_parameters["kernel_size"], dilation_values=dilation_values)
        elif(network_type == "dregular_sw"):
            self.conv = MultiDilatedConvolutionSingleWeight(in_channels=layer_parameters["ninput_channels"], out_channels=layer_parameters["noutput_channels"], 
                            kernel_size=layer_parameters["kernel_size"], padding_mode='reflect', dilation_values=dilation_values)
        elif(network_type == "dregular_mw"):
            self.conv = MultiDilatedConvolutionMultiWeight(in_channels=layer_parameters["ninput_channels"], out_channels=layer_parameters["noutput_channels"], 
                            kernel_size=layer_parameters["kernel_size"], padding_mode='reflect', dilation_values=dilation_values)
        #self.conv.to(device)
        #padding_mode='reflect'
        #self.conv.bias.data.fill_(0.0)
        #self.conv.weight.data.fill_(0.0)
        if(layer_parameters["activation_function"] == "relu"):
            self.activation = ReLU()
        else:
            self.activation = ReLU()
            
        if(layer_parameters["pooling_type"] == "max_pool"):
            self.pool = MaxPool2d(kernel_size=layer_parameters["pooling_size"], stride=(layer_parameters["pooling_stride"], 
                                            layer_parameters["pooling_stride"]), padding=(layer_parameters["pooling_size"][0]-1)//2)
        elif(layer_parameters["pooling_type"] == "avg_pool"):
            self.pool = AvgPool2d(kernel_size=layer_parameters["pooling_size"], stride=(layer_parameters["pooling_stride"], 
                                            layer_parameters["pooling_stride"]), padding=(layer_parameters["pooling_size"][0]-1)//2)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
            
class FLIMAdaptiveDecoderLayer(torch.nn.Module):
    def __init__(self, input_size, error_function=None, grad_function=None, adaptation_function="robust_weights", decoder_type="vanilla_adaptive_decoder", filter_by_size=False, device="cpu", **kwargs):
        super(FLIMAdaptiveDecoderLayer, self).__init__()
        self.thresholds = None
        if(adaptation_function=="robust_weights"):
            self.adaptation_function = self.robust_adaptation_weights    
        self.weights = np.ones((1,input_size,1,1))
        self.error_function=error_function
        self.grad_function=grad_function
        self.normalization_parameters = dict()
        self.filter_by_size = filter_by_size
        self.device = device
        self.decoder_type = decoder_type
        self.kwargs = kwargs

    #Miscelaneous
    def normalize_by_band(feature):
        if(len(feature.shape) == 4):
            for im in range(feature.shape[0]):
                for b in range(feature.shape[1]):
                    max_ = feature[im,b,:,:].max()
                    min_ = feature[im,b,:,:].min()
                    if(max_ - min_ != 0.0):
                        feature[im,b,:,:] = 1 * ((feature[im,b,:,:] - min_) / (max_ - min_))
                    elif(max_ != 0.0):
                        feature[im,b,:,:] = 1 * ((feature[im,b,:,:] - min_) / (max_))
        if(len(feature.shape) == 3):
            for b in range(feature.shape[0]):
                max_ = feature[b,:,:].max()
                min_ = feature[b,:,:].min()
                if(max_ - min_ != 0.0):
                    feature[b,:,:] = 1 * ((feature[b,:,:] - min_) / (max_ - min_))
                elif(max_ != 0.0):
                    feature[b,:,:] = 1 * ((feature[b,:,:] - min_) / (max_))
    
    def circular_mask(self, radius):
        radius = int(radius)
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = x**2 + y**2 <= radius**2
        return mask.astype(float)

    def normalize_by_band_max(feature):
        if(len(feature.shape) == 4):
            for im in range(feature.shape[0]):
                for b in range(feature.shape[1]):
                    max_ = feature[im,b,:,:].max()
                    if(max_ != 0.0):
                        feature[im,b,:,:] = feature[im,b,:,:] / max_
        if(len(feature.shape) == 3):
            for b in range(feature.shape[0]):
                max_ = feature[b,:,:].max()
                if(max_ > 0.0):
                    feature[b,:,:] = feature[b,:,:] / max_

    def find_adaptive_threshold(feature):
        otsu_activations = []
        means_stds = []
        for f in feature:
            means = []
            for b in range(f.shape[0]):
                if(f[b,:,:].max() != 0.0):
                    band_image = f[b,:,:]#/feature[:,b,:,:].max()
                    means.append(band_image.mean())
            means = np.array(means)
            otsu_activation = threshold_otsu(means)
            otsu_activations.append(otsu_activation)
            means_stds.append(means.std()**2)

        return otsu_activations, means_stds

    def robust_adaptation_weights(self, feature):
        adaptation_weights = np.zeros((feature.shape[0], feature.shape[1], 1, 1))
        self.weights = np.ones((feature.shape[0],feature.shape[1],1,1))
        #img_size = [feature.shape[2], feature.shape[3]]
        thresholds, stdevs = FLIMAdaptiveDecoderLayer.find_adaptive_threshold(feature)
        for img in range(feature.shape[0]):
            threshold = thresholds[img]
            stdev = stdevs[img]
            if(threshold > 0.1):
                low_end = threshold - stdev
            else:
                low_end = threshold
            high_end = threshold + stdev
            for b in range(feature.shape[1]):
                if(feature[:,b,:,:].max() != 0.0):
                    band_image = feature[img,b,:,:]
                    otsu_band = threshold_otsu(band_image.squeeze())
                    bin_band = band_image[band_image > otsu_band]
                    proportion_fg = bin_band.shape[0]/(band_image.shape[0]*band_image.shape[1])
                    if(band_image.mean() >= high_end and proportion_fg > 0.2):
                        adaptation_weights[img,b,0,0] = -1
                    elif(band_image.mean() <= low_end and proportion_fg < 0.1):
                        adaptation_weights[img,b,0,0] = 1
                    else:
                        adaptation_weights[img,b,0,0] = 0
                
        adapted_weights = self.weights*adaptation_weights

        return adapted_weights

    def forward(self, feature, original_size = None, weights = None):
        if(self.decoder_type == "vanilla_adaptive_decoder"):
            return self.vanilla_adaptive_decoder(feature, original_size)
        elif(self.decoder_type == "decoder_2"):
            return self.adaptive_decoder2(feature, original_size=original_size, marker_labels=weights, **self.kwargs)
        elif(self.decoder_type == "decoder_3"):
            return self.adaptive_decoder3(feature, original_size=original_size, marker_labels=weights, **self.kwargs)    

    def vanilla_adaptive_decoder(self, feature, original_size = None, weights = None):
        if(original_size != None):
            interp_feature = F.interpolate(feature, [original_size[0], original_size[1]], mode='bilinear', align_corners=True)
        else:
            interp_feature = feature

        interp_feature_array = interp_feature.cpu().detach().numpy()
        FLIMAdaptiveDecoderLayer.normalize_by_band(interp_feature_array)
        
        adapted_weights = self.adaptation_function(interp_feature_array)
        if(weights is not None):
            adapted_weights = weights*adapted_weights
        del interp_feature_array

        kernel = torch.from_numpy(adapted_weights).float().to(self.device)
        y = []
        for img,k in zip(interp_feature, kernel):
            decoded = F.conv2d(img, kernel, padding=0, stride=1).cpu().detach().numpy()
            y.append(decoded*255)
        del interp_feature
        y = torch.from_numpy(np.array(y))
        y = FLIMAdaptiveDecoderLayer.relu(y)
        FLIMAdaptiveDecoderLayer.normalize_by_band(y)
        if(self.filter_by_size):
            util.filter_component_by_area(y)

        return torch.from_numpy(np.array(y*255))

    def view_as_windows_pytorch(self, image, shape, stride=None):
        windows = image.unfold(1, shape[0], stride[0])
        return windows.unfold(2, shape[1], stride[1])

    def adaptive_decoder2(self, feature, original_size = None, marker_labels = None, **kwargs):
        marker_labels=marker_labels
        interp_feature_array = feature.detach()
        FLIMAdaptiveDecoderLayer.normalize_by_band_max(interp_feature_array)
        adj_radius = kwargs.get('adj_radius')
        
        if(adj_radius == None):
            adj_radius = 1.5
        
        r = int(adj_radius)

        weights = torch.zeros(feature.shape[1:]).to(interp_feature_array.device)
        interp_feature_array = torch.nn.functional.pad(interp_feature_array, (r,r,r,r))
        mask = torch.tensor(self.circular_mask(adj_radius)).to(interp_feature_array.device)
        mask_size = mask.sum()
        mask_shape = mask.shape[0]
        mask = mask.reshape((1,1,1,mask.shape[0],mask.shape[0]))
        background_weights = int((marker_labels==0).sum())
        foreground_weights = int((marker_labels==1).sum())

        if(foreground_weights != 0):   
            window_0 = self.view_as_windows_pytorch(interp_feature_array[0,(marker_labels==1),:,:], (mask_shape,mask_shape), stride=[1,1]).permute((1,2,0,3,4))
            
            #circular adjacency
            window_0 = window_0 * mask
            sum = torch.sum(window_0, axis=(2,3,4), keepdim=True)
            mean_0 = sum/(foreground_weights*mask_size)
            
            var_0_ = torch.sum(torch.pow((window_0 - mean_0) * mask,2.0), axis=(2,3,4))
            var_0 = var_0_/(foreground_weights*mask_size)
            var_0[var_0 < 1e-7] = 1.0

            P0 = torch.exp(-torch.pow(interp_feature_array[:,:,r:-r,r:-r][0,:,:,:] - mean_0.squeeze().unsqueeze(0), 2) / (2.0 * var_0))
            
        if(background_weights != 0):
            window_1 = self.view_as_windows_pytorch(interp_feature_array[0,(marker_labels==0),:,:], (mask_shape,mask_shape), stride=[1,1]).permute((1,2,0,3,4))
            
            #circular adjacency
            window_1 = window_1 * mask
            sum = torch.sum(window_1, axis=(2,3,4), keepdim=True)
            mean_1 = sum/(background_weights*mask_size)

            var_1_ = torch.sum(torch.pow((window_1 - mean_1) * mask,2.0), axis=(2,3,4))
            var_1 = var_1_/(background_weights*mask_size)
            var_1[var_1 < 1e-7] = 1.0
            
            P1 = torch.exp(-torch.pow(interp_feature_array[:,:,r:-r,r:-r][0,:,:,:] - mean_1.squeeze().unsqueeze(0), 2) / (2.0 * var_1))
            
        weights_ = weights[(marker_labels==1),:,:]
        weights_[(P0[(marker_labels==1),:,:] > P1[(marker_labels==1),:,:])] = 1
        weights_[(P0[(marker_labels==1),:,:] <= P1[(marker_labels==1),:,:])] = 0
        weights[(marker_labels==1),:,:] = weights_

        weights_ = weights[(marker_labels==0),:,:]
        weights_[(P0[(marker_labels==0),:,:] < P1[(marker_labels==0),:,:])] = -1
        weights_[(P0[(marker_labels==0),:,:] >= P1[(marker_labels==0),:,:])] = 0
        weights[(marker_labels==0),:,:] = weights_
        
        if(r  > 0):
            interp_feature_array = interp_feature_array[:,:,r:-r,r:-r]
        
        salie = (interp_feature_array[0,:,:,:] * weights).sum(axis=0)
        
        salie = FLIMAdaptiveDecoderLayer.relu(salie)

        if(salie.max()-salie.min() != 0.0):
            salie = (salie - salie.min()) / (salie.max() - salie.min())
        elif(salie.max() != 0):
            salie = (salie - salie.min()) / (salie.max())
        
        del interp_feature_array
        
        if(original_size!=None):
            salie = F.interpolate(salie.unsqueeze(0).unsqueeze(0), [original_size[0], original_size[1]],mode='bilinear',align_corners=True)
            
        if(self.filter_by_size):
            util.filter_component_by_area(salie, self.filter_by_size)
        
        return salie * 255.0
    
    def adaptive_decoder3(self, feature, original_size = None, marker_labels = None, **kwargs):
        marker_labels=marker_labels
        interp_feature_array = feature.detach()
        FLIMAdaptiveDecoderLayer.normalize_by_band_max(interp_feature_array)
        adj_radius = kwargs.get('adj_radius')
        
        if(adj_radius == None):
            adj_radius = 1.5
        
        r = int(adj_radius)
        
        weights = torch.zeros(feature.shape[1:]).to(interp_feature_array.device)
        interp_feature_array = torch.nn.functional.pad(interp_feature_array, (r,r,r,r))
        mask = torch.tensor(self.circular_mask(adj_radius)).to(interp_feature_array.device)
        mask_size = mask.sum()
        mask_shape = mask.shape[0]
        mask = mask.reshape((1,1,1,mask.shape[0],mask.shape[0]))
        background_weights = int((marker_labels==0).sum())
        foreground_weights = int((marker_labels==1).sum())

        if(foreground_weights != 0):   
            window_0 = self.view_as_windows_pytorch(interp_feature_array[0,(marker_labels==1),:,:], (mask_shape,mask_shape), stride=[1,1]).permute((1,2,0,3,4))
            
            #circular adjacency
            window_0 = window_0 * mask
            sum = torch.sum(window_0, dim=(2,3,4))
            mean_0 = sum/(foreground_weights*mask_size)
                    
        if(background_weights != 0):
            window_1 = self.view_as_windows_pytorch(interp_feature_array[0,(marker_labels==0),:,:], (mask_shape,mask_shape), stride=[1,1]).permute((1,2,0,3,4))
            #circular adjacency
            window_1 = window_1 * mask
            sum = torch.sum(window_1, dim=(2,3,4))
            mean_1 = sum/(background_weights*mask_size)

        weights_ = weights[(marker_labels==1),:,:]
        weights_[:,((mean_0) > mean_1)] = 1
        weights[(marker_labels==1),:,:] = weights_

        weights_ = weights[(marker_labels==0),:,:]
        weights_[:,(mean_0 < (mean_1))] = -1
        weights[(marker_labels==0),:,:] = weights_

        if(r  > 0):
            interp_feature_array = interp_feature_array[:,:,r:-r,r:-r]

        salie = (interp_feature_array[0,:,:,:] * weights).sum(dim=0)
        
        salie = FLIMAdaptiveDecoderLayer.relu(salie)

        if(salie.max()-salie.min() != 0.0):
            salie = (salie - salie.min()) / (salie.max() - salie.min())
        elif(salie.max() != 0):
            salie = (salie - salie.min()) / (salie.max())
        
        del interp_feature_array
        
        if(original_size!=None):
            salie = F.interpolate(salie.unsqueeze(0).unsqueeze(0), [original_size[0], original_size[1]],mode='bilinear',align_corners=True)
        
        if(self.filter_by_size):
            util.filter_component_by_area(salie, self.filter_by_size)
        
        return salie * 255.0
       
    def relu(image):
        image[image < 0] = 0
        return image
