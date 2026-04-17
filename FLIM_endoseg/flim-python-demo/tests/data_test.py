import unittest
from model import data
import numpy as np

class TestDataMethods(unittest.TestCase):

    def test_rgb2labnorm(self):
        rgb = np.random.uniform(low=0, high=255, size=(500,500,3)).astype(np.uint8)
        rgb[0,0,0] = 255
        rgb[0,0,1] = 255
        rgb[0,0,2] = 255
        rgb[0,1,0] = 0
        rgb[0,1,1] = 0
        rgb[0,1,2] = 0
        lab = data.FLIMData.rgb2labnorm(rgb)
        message = "L should be between 0-1 and got value" + str(lab[:,:,0].max())+")"
        self.assertLessEqual(lab[:,:,0].max(), 1.0,message)
        
        message = "a should be between 0-1 and got value" + str(lab[:,:,1].max())+")"
        self.assertLessEqual(lab[:,:,1].max(), 1.0, message)
        
        message = "b should be between 0-1 and got value" + str(lab[:,:,2].max())+")"
        self.assertLessEqual(lab[:,:,1].max(), 1.0, message)

    def test_read_image_markers_file(self):
        n_markers = 50
        filename = "tests/example_marker_file.txt"
        marker_list, image_size = data.FLIMData.read_image_markers(filename)
        
        message = "Expected marker number (50) different than number read ("+str(len(marker_list))+")"
        self.assertEqual(len(marker_list), 50, message)
        message = "Expected image size ((500,500)) different than number read ("+str(image_size)+")"
        self.assertEqual(image_size, (500,500), message)

    def test_get_marker_image(self):
        n_markers = 50
        filename = "tests/example_marker_file.txt"
        marker_image = data.FLIMData.get_marker_image(filename)

        bin_marker = np.copy(marker_image)
        bin_marker[bin_marker>0] = 1
        message = "Expected number of marked pixels (50) different than pixels marked ("+str(bin_marker.sum())+")"
        self.assertEqual(bin_marker.sum(), 50, message)
        message = "Expected marker image sum (60) different than sum computed (possibly not reading the correct labels)("+str(marker_image.sum())+")"
        self.assertEqual(marker_image.sum(), 58, message)

    def test_label_markers_by_component(self):
        filename = "tests/example_marker_file.txt"
        marker_image = data.FLIMData.label_markers_by_component(data.FLIMData.get_marker_image(filename))
        
        message = "Expected number of marker labels (3) different than number of labels found ("+str(marker_image.max())+")"
        self.assertEqual(marker_image.max(), 3, message)
        
if __name__ == '__main__':
    unittest.main()