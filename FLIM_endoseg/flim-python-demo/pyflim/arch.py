import json

class FLIMArchitecture():
    def save_architecture():
        #TO BE IMPLEMENTED
        return 0

    def is_valid_pool(pooling_type):
        if(pooling_type != "max_pool" and pooling_type != "avg_pool"):
            return False
        return True
    
    def load_architecture(self, architecture_file, nchannels=3):
        f = open(architecture_file)

        data = json.load(f)
        f.close()
        self.nlayers = data["nlayers"]
        self.stdev_factor = data["stdev_factor"]
        self.layers = []

        for i in range(self.nlayers):
            self.layers.append(dict())
            self.layers[i]["kernel_size"] = data["layer"+str(i+1)]["conv"]["kernel_size"][:-1]
            self.layers[i]["dilation_rate"] = data["layer"+str(i+1)]["conv"]["dilation_rate"][0]
            self.layers[i]["nkernels_per_marker"] = data["layer"+str(i+1)]["conv"]["nkernels_per_marker"]
            self.layers[i]["noutput_channels"] = data["layer"+str(i+1)]["conv"]["noutput_channels"]
            if(i == 0):
                self.layers[i]["ninput_channels"] = nchannels
            else:
                self.layers[i]["ninput_channels"] = self.layers[i-1]["noutput_channels"]

            self.layers[i]["pooling_size"] = data["layer"+str(i+1)]["pooling"]["size"][:-1]
            self.layers[i]["pooling_stride"] = data["layer"+str(i+1)]["pooling"]["stride"]
            assert FLIMArchitecture.is_valid_pool(data["layer"+str(i+1)]["pooling"]["type"])==True, "Pooling type unavailable. Try either 'max_pool', or 'avg_pool'"
            self.layers[i]["pooling_type"] = data["layer"+str(i+1)]["pooling"]["type"]
            self.layers[i]["activation_function"] = "relu"
            
    def change_layer(self, layer, layer_number):
        self.layers[layer_number] = layer
        
    @staticmethod
    def create_layer(kernel_size=[3,3], dilation_rate=1, nkernels_per_marker=10, noutput_channels=32, pooling_size=[3,3], pooling_stride=2, pooling_type="max_pool", activation_function="relu"):
        layer = dict()
        layer["kernel_size"] = kernel_size
        layer["dilation_rate"] = dilation_rate
        layer["nkernels_per_marker"] = nkernels_per_marker
        layer["noutput_channels"] = noutput_channels

        layer["pooling_size"] = pooling_size
        layer["pooling_stride"] = pooling_stride
        layer["pooling_type"] = pooling_type
        layer["activation_function"] = activation_function
        
        return layer
            
    def add_layer(self, layer):
        if(self.nlayers==0):
            layer["ninput_channels"] = 3
        else:
            layer["ninput_channels"] = self.layers[self.nlayers-1]["noutput_channels"]
        self.layers.append(layer)
        self.nlayers+=1

    def remove_layer(self, layer_number=-1):
        if(len(self.layers) == 0):
            print("No layers to remove")
            return
        if(layer_number==-1):
            last_layer = len(self.layers)-1
            self.layers.pop(last_layer)
        else:
            self.layers = self.layers.pop(layer_number)
        self.nlayers-=1

    def __init__(self, architecture_file=None):
        
        if(architecture_file != None):
            self.load_architecture(architecture_file)
        else:
            self.nlayers = 0
            self.stdev_factor = 0.01
            self.layers = []
