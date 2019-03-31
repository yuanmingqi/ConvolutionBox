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

from flask import Flask, request, render_template
from utils.encoder import encoder
from utils.get_image import get_image
from utils.get_histogram import get_histogram
from conv.standard_conv import standard_conv
from pooling.pooling import avg_pooling, max_pooling
from extractor.callback import Call
import json

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')

@app.route('/conv_show', methods=['post', 'get'])
def conv_show():
    data = {
        'path_conv1': '../static/timg1.gif',
        'path_conv2': '../static/timg1.gif',
        'path_conv3': '../static/timg1.gif',
        'path_conv4': '../static/timg1.gif',
        'source_image': '../static/timg1.gif',
        'histogram_1C': [],
        'histogram_2C': [],
        'histogram_3C': []
    }
    return render_template('convolution.html', data = data)

@app.route('/conv_do', methods=['post', 'get'])
def conv_do():
    path_to_image = get_image()
    filter_size = int(request.values.get('filter_size'))
    strides = int(request.values.get('strides'))
    padding = str(request.values.get('padding'))

    feature_path = standard_conv(strides=strides,
                                padding=padding,
                                filter_size=filter_size,
                                path_to_image=path_to_image)
    data = get_histogram(feature_path)
    data['source_image'] = path_to_image

    data = json.dumps(data, cls= encoder)
    data = json.loads(data)

    return render_template('convolution.html', data = data)

@app.route('/pooling_show', methods=['post', 'get'])
def pooling_show():
    data = {
        'path_avg_pooling': '../static/timg1.gif',
        'path_max_pooling': '../static/timg1.gif',
        'source_image': '../static/timg1.gif',
        'histogram_avg': [],
        'histogram_max': [],
    }
    return render_template('pooling.html', data = data)

@app.route('/pooling_do', methods=['post', 'get'])
def pooling_do():
    path_to_image = get_image()
    pool_size = int(request.values.get('pool_size'))
    strides = int(request.values.get('strides'))
    padding = str(request.values.get('padding'))

    data = {}
    data['path_avg_pooling'], data['histogram_avg'] = \
        avg_pooling(pool_size=pool_size,
                        strides=strides,
                        padding=padding,
                        path_to_image=path_to_image)

    data['path_max_pooling'], data['histogram_max'] = \
        max_pooling(pool_size=pool_size,
                       strides=strides,
                       padding=padding,
                       path_to_image=path_to_image)

    data['source_image'] = path_to_image

    data = json.dumps(data, cls=encoder)
    data = json.loads(data)

    return render_template('pooling.html', data = data)

@app.route('/extractor_show', methods=['post', 'get'])
def extractor_show():
    data = {
        'path_to_feature': '../static/timg1.gif',
        'path_to_graph': '../static/timg1.gif',
        'source_image': '../static/timg1.gif',
        'histogram_to_feature': [],
    }
    return render_template('extractor.html', data = data)

@app.route('/extractor_do', methods=['post', 'get'])
def extractor_do():
    path_to_image = get_image()
    extractor_type = request.values.get('extractor')

    data = {}
    data['source_image'] = path_to_image
    data['path_to_graph'] = 'static/graph/' + extractor_type + '.png'
    data['path_to_feature'] = Call().get_feature(path_to_image=path_to_image, extractor_type = extractor_type)
    data['path_to_feature'] = data['path_to_feature'][0]
    data['histogram_to_feature'] = Call().get_histogram(path_to_feature=data['path_to_feature'])


    data = json.dumps(data, cls=encoder)
    data = json.loads(data)

    return render_template('extractor.html', data = data)

@app.errorhandler(404)
def page_not_found():
    return render_template('404.html')

@app.errorhandler(500)
def internal_server_error():
    return render_template('500.html')

if __name__ == '__main__':
    app.run()




