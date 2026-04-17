import unittest
from model import arch
import numpy as np

class TestArchMethods(unittest.TestCase):

    def test_load_architecture(self):
        architecture = arch.FLIMArchitecture()
        architecture_file = "tests/example_arch.json"
        architecture.load_architecture(architecture_file)

        expected_n_layers = 2
        
        #------ Layer 1
        expected_kernel_size = [3,3]
        expected_dilation_rate = 1
        expected_nkernels_per_marker = 1
        expected_noutput_channels = 32
        expected_ninput_channels = 3
        expected_pooling_size = [3,3]
        expected_pooling_stride = 1
        expected_pooling_type = "max_pool"
        expected_activation_function = "relu"

        message = "Expected number of layers (2) different than the number read (" + str(architecture.nlayers)+")"
        self.assertEqual(architecture.nlayers, expected_n_layers, message)
        
        message = "(Layer 1 ) Expected kernel size ([3,3]) different than the number read (" + str(architecture.layers[0]["kernel_size"])+")"
        self.assertEqual(architecture.layers[0]["kernel_size"], expected_kernel_size, message)
        
        message = "(Layer 1 ) Expected dilation rate (1) different than the number read (" + str(architecture.layers[0]["dilation_rate"])+")"
        self.assertEqual(architecture.layers[0]["dilation_rate"], expected_dilation_rate, message)
        
        message = "(Layer 1 ) Expected n kernels per marker (1) different than the number read (" + str(architecture.layers[0]["nkernels_per_marker"])+")"
        self.assertEqual(architecture.layers[0]["nkernels_per_marker"], expected_nkernels_per_marker, message)
        
        message = "(Layer 1 ) Expected n output channels (32) different than the number read (" + str(architecture.layers[0]["noutput_channels"])+")"
        self.assertEqual(architecture.layers[0]["noutput_channels"], expected_noutput_channels, message)
        
        message = "(Layer 1 ) Expected n input channels (3) different than the number read (" + str(architecture.layers[0]["ninput_channels"])+")"
        self.assertEqual(architecture.layers[0]["ninput_channels"], expected_ninput_channels, message)
        
        message = "(Layer 1 ) Expected pooling size ([3,3]) different than the number read (" + str(architecture.layers[0]["pooling_size"])+")"
        self.assertEqual(architecture.layers[0]["pooling_size"], expected_pooling_size, message)
        
        message = "(Layer 1 ) Expected pooling stride (1) different than the number read (" + str(architecture.layers[0]["pooling_stride"])+")"
        self.assertEqual(architecture.layers[0]["pooling_stride"], expected_pooling_stride, message)
        
        message = "(Layer 1 ) Expected pooling type ('max_pool') different than the option read (" + str(architecture.layers[0]["pooling_type"])+")"
        self.assertEqual(architecture.layers[0]["pooling_type"], expected_pooling_type, message)

        message = "(Layer 1 ) Expected activation function ('relu') different than the option read (" + str(architecture.layers[0]["activation_function"])+")"
        self.assertEqual(architecture.layers[0]["activation_function"], expected_activation_function, message)
        
        #------ Layer 2
        expected_kernel_size = [5,5]
        expected_dilation_rate = 3
        expected_nkernels_per_marker = 5
        expected_noutput_channels = 64
        expected_ninput_channels = 32
        expected_pooling_size = [5,5]
        expected_pooling_stride = 1
        expected_pooling_type = "avg_pool"
        expected_activation_function = "relu"

        message = "Expected number of layers (2) different than the number read (" + str(architecture.nlayers)+")"
        self.assertEqual(architecture.nlayers, expected_n_layers, message)
        
        message = "(Layer 2 ) Expected kernel size ([5,5]) different than the number read (" + str(architecture.layers[1]["kernel_size"])+")"
        self.assertEqual(architecture.layers[1]["kernel_size"], expected_kernel_size, message)
        
        message = "(Layer 2 ) Expected dilation rate (1) different than the number read (" + str(architecture.layers[1]["dilation_rate"])+")"
        self.assertEqual(architecture.layers[1]["dilation_rate"], expected_dilation_rate, message)
        
        message = "(Layer 2 ) Expected n kernels per marker (1) different than the number read (" + str(architecture.layers[1]["nkernels_per_marker"])+")"
        self.assertEqual(architecture.layers[1]["nkernels_per_marker"], expected_nkernels_per_marker, message)
        
        message = "(Layer 2 ) Expected n output channels (32) different than the number read (" + str(architecture.layers[1]["noutput_channels"])+")"
        self.assertEqual(architecture.layers[1]["noutput_channels"], expected_noutput_channels, message)
        
        message = "(Layer 2 ) Expected n input channels (3) different than the number read (" + str(architecture.layers[1]["ninput_channels"])+")"
        self.assertEqual(architecture.layers[1]["ninput_channels"], expected_ninput_channels, message)
        
        message = "(Layer 2 ) Expected pooling size ([3,3]) different than the number read (" + str(architecture.layers[1]["pooling_size"])+")"
        self.assertEqual(architecture.layers[1]["pooling_size"], expected_pooling_size, message)
        
        message = "(Layer 2 ) Expected pooling stride (1) different than the number read (" + str(architecture.layers[1]["pooling_stride"])+")"
        self.assertEqual(architecture.layers[1]["pooling_stride"], expected_pooling_stride, message)
        
        message = "(Layer 2 ) Expected pooling type ('max_pool') different than the option read (" + str(architecture.layers[1]["pooling_type"])+")"
        self.assertEqual(architecture.layers[1]["pooling_type"], expected_pooling_type, message)

        message = "(Layer 2 ) Expected activation function ('relu') different than the option read (" + str(architecture.layers[1]["activation_function"])+")"
        self.assertEqual(architecture.layers[1]["activation_function"], expected_activation_function, message)

    def test_create_layer(self):
        layer = arch.FLIMArchitecture.create_layer(kernel_size=[7,7], dilation_rate=5, nkernels_per_marker=3, noutput_channels=128, pooling_size=[3,3], pooling_stride=3, pooling_type="max_pool", activation_function="relu")

    
        expected_kernel_size = [7,7]
        expected_dilation_rate = 5
        expected_nkernels_per_marker = 3
        expected_noutput_channels = 128
        expected_ninput_channels = 3
        expected_pooling_size = [3,3]
        expected_pooling_stride = 3
        expected_pooling_type = "max_pool"
        expected_activation_function = "relu"

        message = "Expected kernel size ([3,3]) different than the number read (" + str(layer["kernel_size"])+")"
        self.assertEqual(layer["kernel_size"], expected_kernel_size, message)
        
        message = "Expected dilation rate (1) different than the number read (" + str(layer["dilation_rate"])+")"
        self.assertEqual(layer["dilation_rate"], expected_dilation_rate, message)
        
        message = "Expected n kernels per marker (1) different than the number read (" + str(layer["nkernels_per_marker"])+")"
        self.assertEqual(layer["nkernels_per_marker"], expected_nkernels_per_marker, message)
        
        message = "Expected n output channels (32) different than the number read (" + str(layer["noutput_channels"])+")"
        self.assertEqual(layer["noutput_channels"], expected_noutput_channels, message)
        
        message = "Expected pooling size ([3,3]) different than the number read (" + str(layer["pooling_size"])+")"
        self.assertEqual(layer["pooling_size"], expected_pooling_size, message)
        
        message = "Expected pooling stride (1) different than the number read (" + str(layer["pooling_stride"])+")"
        self.assertEqual(layer["pooling_stride"], expected_pooling_stride, message)
        
        message = "Expected pooling type ('max_pool') different than the option read (" + str(layer["pooling_type"])+")"
        self.assertEqual(layer["pooling_type"], expected_pooling_type, message)

        message = "Expected activation function ('relu') different than the option read (" + str(layer["activation_function"])+")"
        self.assertEqual(layer["activation_function"], expected_activation_function, message)

    def test_add_layer(self):
        architecture = arch.FLIMArchitecture()
        layer = arch.FLIMArchitecture.create_layer()
        architecture.add_layer(layer)
        
        message = "(Layer 1) Expected n input channels to be 3, got (" + str(architecture.layers[0]["ninput_channels"])+") instead."
        self.assertEqual(architecture.layers[0]["ninput_channels"], 3, message)
        
        layer = arch.FLIMArchitecture.create_layer(noutput_channels=64)
        architecture.add_layer(layer)
        message = "(Layer 1) Expected n input channels to be 32, got (" + str(architecture.layers[1]["ninput_channels"])+") instead."
        self.assertEqual(architecture.layers[1]["ninput_channels"], 32, message)
        
        architecture.add_layer(layer)
        message = "(Layer 1) Expected n input channels to be 32, got (" + str(architecture.layers[1]["ninput_channels"])+") instead."
        self.assertEqual(architecture.layers[1]["ninput_channels"], 64, message)

    def test_remove_layer(self):
        architecture = arch.FLIMArchitecture()
        layer = arch.FLIMArchitecture.create_layer()
        architecture.add_layer(layer)
        layer = arch.FLIMArchitecture.create_layer(noutput_channels=64)
        architecture.add_layer(layer)
        layer = arch.FLIMArchitecture.create_layer(noutput_channels=128)
        architecture.add_layer(layer)

        architecture.remove_layer()
        message = "Expected n layers to be 2 but got (" + str(architecture.nlayers)+")"
        self.assertEqual(architecture.nlayers, 2, message)
        
        message = "Expected last layer's n output channel to be 64 but got (" + str(architecture.layers[architecture.nlayers-1]["noutput_channels"])+")"
        self.assertEqual(architecture.layers[architecture.nlayers-1]["noutput_channels"], 64, message)


if __name__ == '__main__':
    unittest.main()