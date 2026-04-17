import unittest
from model import data
from model import flim
import numpy as np
import torch

class TestFLIMMethods(unittest.TestCase):
    def setUp(self):
        self.rgb = np.random.uniform(low=0, high=255, size=(500,500,3)).astype(np.uint8)
        self.lab = data.FLIMData.rgb2labnorm(self.rgb)
        self.lab[99,99] = [0.001,0.002,0.003]
        self.lab[99,100] = [0.01,0.02,0.03]
        self.lab[99,101] = [0.001,0.002,0.003]
        self.lab[100,99] = [0.01,0.02,0.03]
        self.lab[100,100] = [0.1,0.2,0.3]
        self.lab[100,101] = [0.01,0.02,0.03]
        self.lab[101,99] = [0.001,0.002,0.003]
        self.lab[101,100] = [0.01,0.02,0.03]
        self.lab[101,101] = [0.001,0.002,0.003]

        self.lab[199,199] = [0.004,0.005,0.006]
        self.lab[199,200] = [0.04,0.05,0.06]
        self.lab[199,201] = [0.004,0.005,0.006]
        self.lab[200,199] = [0.04,0.05,0.06]
        self.lab[200,200] = [0.4,0.5,0.6]
        self.lab[200,201] = [0.04,0.05,0.06]
        self.lab[201,199] = [0.004,0.005,0.006]
        self.lab[201,200] = [0.04,0.05,0.06]
        self.lab[201,201] = [0.004,0.005,0.006]

        

        self.lab[299,299] = [0.007,0.008,0.009]
        self.lab[299,300] = [0.07,0.08,0.09]
        self.lab[299,301] = [0.007,0.008,0.009]
        self.lab[300,299] = [0.07,0.08,0.09]
        self.lab[300,300] = [0.7,0.8,0.9]
        self.lab[300,301] = [0.07,0.08,0.09]
        self.lab[301,299] = [0.007,0.008,0.009]
        self.lab[301,300] = [0.07,0.08,0.09]
        self.lab[301,301] = [0.007,0.008,0.009]

        self.marker_image = np.zeros((500,500))
        self.marker_image[100,100] = 1
        self.marker_image[200,200] = 1
        #self.marker_image[300,300] = 1

        self.patchsize = [3,3]
        self.patches = flim.FLIMModel.patchify(self.lab, self.patchsize)
        self.representatives = np.array(flim.FLIMModel.select_patches(self.patches, self.marker_image))
    
    # def test_patch_select(self):
    #     message = "Number of representatives should be 2 and got value" + str(self.representatives.shape[0])+" instead."
    #     self.assertEqual(self.representatives.shape[0], 2, message)

    #     expected_representatives = np.zeros((2,3,3,3))
    #     expected_representatives[0] = np.array([[[0.001,0.002,0.003],[0.01,0.02,0.03],[0.001,0.002,0.003]],[[0.01,0.02,0.03],[0.1,0.2,0.3],[0.01,0.02,0.03]],[[0.001,0.002,0.003],[0.01,0.02,0.03],[0.001,0.002,0.003]]])
    #     expected_representatives[1] = np.array([[[0.004,0.005,0.006],[0.04,0.05,0.06],[0.004,0.005,0.006]],[[0.04,0.05,0.06],[0.4,0.5,0.6],[0.04,0.05,0.06]],[[0.004,0.005,0.006],[0.04,0.05,0.06],[0.004,0.005,0.006]]])
        
    #     message = "Representatives should look like \n"+str(expected_representatives)+" \nbut instead is\n" + str(self.representatives)+"."
    #     self.assertEqual(np.array_equal(expected_representatives, self.representatives), True, message)

    def test_patch_clustering(self):
        n_clusters = 1
        #self.marker_image[300,300] = 1
        #self.representatives = np.array(flim.FLIMModel.select_patches(self.patches, self.marker_image))
        kernels = flim.FLIMModel.cluster_patches(self.representatives, n_clusters)

        
        expected_kernel = np.array([[[[0.0025,0.0035,0.0045],[0.025,0.035,0.045],[0.0025,0.0035,0.0045]],[[0.025,0.035,0.045],[0.25,0.35,0.45],[0.025,0.035,0.045]],[[0.0025,0.0035,0.0045],[0.025,0.035,0.045],[0.0025,0.0035,0.0045]]]])
        message = "Expected selected kernel to be \n"+str(expected_kernel)+" \nbut instead got\n" + str(kernels-expected_kernel)+"."
        self.assertLess((expected_kernel-kernels).sum(), 0.001, message)
        self.assertGreaterEqual((expected_kernel-kernels).sum(), 0.0, message)

    def test_unit_norm_kernels(self):
        n_clusters = 1
        self.marker_image[300,300] = 1
        self.representatives = np.array(flim.FLIMModel.select_patches(self.patches, self.marker_image))
        kernels = flim.FLIMModel.cluster_patches(self.representatives, n_clusters)
        flim.FLIMModel.unit_norm_kernels(kernels)
        message = "Expected weight vector to have norm 1, got norm ="+str(np.linalg.norm(kernels))+" instead"
        self.assertEqual(np.linalg.norm(kernels)>=0.999 and np.linalg.norm(kernels)<=1.00001, 1.0, message)

    def test_kernel_reshape(self):
        n_clusters = 1
        self.marker_image[300,300] = 1
        self.representatives = np.array(flim.FLIMModel.select_patches(self.patches, self.marker_image))
        kernels = []
        kernels_1 = flim.FLIMModel.cluster_patches(self.representatives, n_clusters)
        kernels.append(kernels_1)
        
        self.marker_image[100,100] = 0
        self.marker_image[200,200] = 0
        self.representatives = np.array(flim.FLIMModel.select_patches(self.patches, self.marker_image))
        kernels_2 = flim.FLIMModel.cluster_patches(self.representatives, n_clusters)
        
        kernels.append(kernels_2)
        kernels = np.array(kernels)
        
        kernel_candidates = np.reshape(kernels, (kernels.shape[0]*kernels.shape[1], kernels.shape[2], kernels.shape[3], kernels.shape[4]))
        message = "Expected kernel candidates shape to be (2,3,3,3) but got\n" + str(kernel_candidates.shape)+" instead"
        self.assertEqual(kernel_candidates.shape, (2,3,3,3), message)

        weights = torch.from_numpy(kernels_1).permute(0,3,2,1).float()
        expected_weights = torch.from_numpy(np.array([[[[0.0040, 0.0400, 0.0040],
          [0.0400, 0.4000, 0.0400],
          [0.0040, 0.0400, 0.0040]],

         [[0.0050, 0.0500, 0.0050],
          [0.0500, 0.5000, 0.0500],
          [0.0050, 0.0500, 0.0050]],

         [[0.0060, 0.0600, 0.0060],
          [0.0600, 0.6000, 0.0600],
          [0.0060, 0.0600, 0.0060]]]])).float()
        message = "Expected weight vector to be \n"+str(expected_weights)+"but got\n" + str(weights)+" instead"
        self.assertEqual(torch.equal(weights,expected_weights), True,message)

    def get_random_marker_image(size=(500,500), n_markers = 150):
        rng = np.random.default_rng()
        random_positions = rng.choice(size[0]*size[1], size=150, replace=False)
        marker_pixels = np.unravel_index(random_positions, size)

        markers = np.zeros(size)
        markers[marker_pixels] = 1

        return markers
        

    def test_kmeans_determinism(self):
        dataset = []
        for i in range(1,10):
            rgb = np.random.uniform(low=0, high=255, size=(500,500,3)).astype(np.uint8)
            lab = data.FLIMData.rgb2labnorm(self.rgb)
            marker_image = TestFLIMMethods.get_random_marker_image((500,500), 150)

            n_clusters = 16
            patches = flim.FLIMModel.patchify(lab, self.patchsize)
            
            representatives = np.array(flim.FLIMModel.select_patches(patches, marker_image)).astype(np.float32)
            #assert False, representatives.shape
            kernels = flim.FLIMModel.cluster_patches_faiss(representatives, n_clusters)
            kernels_2 = flim.FLIMModel.cluster_patches_faiss(representatives, n_clusters)
            message = "Expected selected kernel to be \n"+str(kernels_2)+" \nbut instead got\n" + str(kernels-kernels_2)+"."
            self.assertLess((kernels_2-kernels).sum(), 0.001, message)
            self.assertGreaterEqual((kernels_2-kernels).sum(), 0.0, message)

    def test_flim_determinism(self):
        

if __name__ == '__main__':
    unittest.main()