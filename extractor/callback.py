"""
MIT License

Copyright (c) 2019 Mingqi Yuan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
Encoding = UTF-8
By Mingqi, Yuan, 2019/3/31
Usage: Call the extractor
"""

from extractor.network import *
import matplotlib.pyplot as plt
import cv2

class Call():
    def get_feature(self, extractor_type, path_to_image):
        data = {
        'path_to_feature': '',
        'path_to_graph': '',
        'source_image': '',
        'histogram_to_feature': [],
    }
        if str(extractor_type) == 'LeNet5':
            data['path_to_feature'] = LeNet5.LeNet5(path_to_image= path_to_image)

        return data['path_to_feature'], data['path_to_graph']

    def get_histogram(self, path_to_feature):
        im = cv2.imread(path_to_feature, 0)
        return (plt.hist(im.ravel(), 256, [0, 256])[0])

