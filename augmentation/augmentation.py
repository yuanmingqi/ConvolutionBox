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
By Mingqi, Yuan, 2019/3/26
Usage: For images augmentation
"""

import os
import cv2
import shutil
import numpy as np
from utils.generate_unique_id import generate_unique_id

class Augmentation():
    def __init__(self, path_to_image_folder, path_to_save_folder, methods, factor):
        self.image_folder = path_to_image_folder
        self.save_folder = path_to_save_folder
        self.methods = methods
        self.factor = factor
        self.filename = os.listdir(self.image_folder)
        if len(self.filename) < 9 :
            for i in range(9 - len(self.filename)):
                self.filename.append(self.filename[0])

    def main(self):
        statistics = {'value': [0, 0, 0, 0, 0, 0, 0, 0, 0]}
        for i in range(self.factor):
            for item in self.filename:
                img_dir = os.path.join(self.image_folder, item)
                judge = np.random.randint(1, 10, 1)
                if judge == 1 and 1 in self.methods:
                    self.rotate(img_dir)
                    statistics['value'][0] = statistics['value'][0] + 1

                elif judge == 2 and 2 in self.methods:
                    self.zoom_in(img_dir)
                    statistics['value'][1] = statistics['value'][1] + 1

                elif judge == 3 and 3 in self.methods:
                    self.zoom_out(img_dir)
                    statistics['value'][2] = statistics['value'][2] + 1

                elif judge == 4 and 4 in self.methods:
                    self.shear(img_dir)
                    statistics['value'][3] = statistics['value'][3] + 1

                elif judge == 5 and 5 in self.methods:
                    self.mirroring(img_dir)
                    statistics['value'][4] = statistics['value'][4] + 1

                elif judge == 6 and 6 in self.methods:
                    self.saturation(img_dir)
                    statistics['value'][5] = statistics['value'][5] + 1

                elif judge == 7 and 7 in self.methods:
                    self.noise(img_dir)
                    statistics['value'][6] = statistics['value'][6] + 1

                elif judge == 8 and 8 in self.methods:
                    self.contrast_ratio(img_dir)
                    statistics['value'][7] = statistics['value'][7] + 1

                elif judge == 9 and 9 in self.methods:
                    self.brightness(img_dir)
                    statistics['value'][8] = statistics['value'][8] + 1

                else:
                    self.saturation(img_dir)
                    statistics['value'][5] = statistics['value'][5] + 1

        filename = os.listdir(self.save_folder)
        img_show = []
        for i in range(9):
            img_dir = os.path.join(self.save_folder, filename[i])
            shutil.copyfile(src=img_dir,
                            dst='static/images/augmentation/' + filename[i])
            img_show.append('static/images/augmentation/' + filename[i])

        return statistics, img_show

    def rotate(self, path_to_image):
        img = cv2.imread(path_to_image, 1)
        center = (img.shape[0] / 2, img.shape[0] / 2)
        angle = np.random.randint(1, 90, 1)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (img.shape[0], img.shape[1]))
        save_path = os.path.join(self.save_folder, 'rotate_' + str(angle[0]) + '_' + generate_unique_id())
        cv2.imwrite(save_path,rotated)

    def zoom_in(self, path_to_image):
        img = cv2.imread(path_to_image, 1)
        (h, w) = img.shape[:2]
        change = np.random.randint(1, 30, 1)
        dst_size = (h + change, w + change)
        method = cv2.INTER_NEAREST
        resized = cv2.resize(img, dst_size, interpolation=method)
        save_path = os.path.join(self.save_folder, 'zoom_in_' + str(change[0]) + '_' + generate_unique_id())
        cv2.imwrite(save_path, resized)

    def zoom_out(self, path_to_image):
        img = cv2.imread(path_to_image, 1)
        (h, w) = img.shape[:2]
        change = np.random.randint(1, 30, 1)
        dst_size = (h - change, w - change)
        method = cv2.INTER_NEAREST
        resized = cv2.resize(img, dst_size, interpolation=method)
        save_path = os.path.join(self.save_folder, 'zoom_out_' + str(change[0])+ '_' + generate_unique_id())
        cv2.imwrite(save_path, resized)

    def shear(self, path_to_image):
        img = cv2.imread(path_to_image, 1)
        (h, w) = img.shape[:2]
        cropped = img[int(h/4):int(h/2), int(w/4):int(w/2)]
        save_path = os.path.join(self.save_folder, 'shear_'+ generate_unique_id())
        cv2.imwrite(save_path, cropped)

    def mirroring(self, path_to_image):
        img = cv2.imread(path_to_image, 1)
        img_flip = cv2.flip(img, 1)
        save_path = os.path.join(self.save_folder, 'mirroring_' + generate_unique_id())
        cv2.imwrite(save_path, img_flip)

    def brightness(self, path_to_image):
        img = cv2.imread(path_to_image, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if np.random.randint(1, 10, 1) < 5:
            temp = img[:, :, 2] / 1.25
            img[:, :, 2] = temp
        else:
            temp = img[:, :, 2] * 1.25
            img[:, :, 2] = temp
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        save_path = os.path.join(self.save_folder, 'brightness_' + generate_unique_id())
        cv2.imwrite(save_path, img)

    def saturation(self, path_to_image):
        img = cv2.imread(path_to_image, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if np.random.randint(1, 10, 1) < 5:
            temp = img[:, :,1] / 1.55
            img[:, :, 1] = temp
        else:
            temp = img[:, :, 1] * 1.55
            img[:, :, 1] = temp
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        save_path = os.path.join(self.save_folder, 'saturation_' + generate_unique_id())
        cv2.imwrite(save_path, img)

    def noise(self, path_to_image):
        # 定义添加椒盐噪声的函数
        def SaltAndPepper(src, percetage):
            SP_NoiseImg = src
            SP_NoiseNum = int(percetage * src.shape[0] * src.shape[1])
            for i in range(SP_NoiseNum):
                randX = np.random.random_integers(0, src.shape[0] - 1)
                randY = np.random.random_integers(0, src.shape[1] - 1)
                if np.random.random_integers(0, 1) == 0:
                    SP_NoiseImg[randX, randY] = 0
                else:
                    SP_NoiseImg[randX, randY] = 255
            return SP_NoiseImg

        img = cv2.imread(path_to_image, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gauss_noiseImage = SaltAndPepper(img, 0.01)

        save_path = os.path.join(self.save_folder, 'noise_' + generate_unique_id())
        cv2.imwrite(save_path, gauss_noiseImage)

    def contrast_ratio(self, path_to_image):
        img = cv2.imread(path_to_image, 1)

        def gamma_trans(img, gamma):
            gamma_list = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
            gamma_table = np.round(np.array(gamma_list)).astype(np.uint8)
            return cv2.LUT(img, gamma_table)

        if np.random.randint(1, 10, 1) < 5:
            img_gamma = gamma_trans(img, np.random.rand(1))
        else:
            img_gamma = gamma_trans(img, 2 + np.random.rand(1))

        save_path = os.path.join(self.save_folder, 'contrast_ratio_' + generate_unique_id())
        cv2.imwrite(save_path, img_gamma)
