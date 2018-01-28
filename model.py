import caffe
import numpy as np
from config import config


# API for model.
class Model:
    def __init__(self, caffemodel, sketch_model_prototxt, image_model_prototxt):
        self.sketch_net = caffe.Net(sketch_model_prototxt, caffemodel, caffe.TEST)
        self.img_net = caffe.Net(image_model_prototxt, caffemodel, caffe.TEST)
        self.output_layer_sketch = config.get('Network', 'output_layer_sketch')
        self.output_layer_image = config.get('Network', 'output_layer_image')
        self._init_transformer()

    def _init_transformer(self):
        self.transformer = caffe.io.Transformer({'data': np.shape(self.sketch_net.blobs['data'].data)})
        self.transformer.set_mean('data', np.array([104, 117, 123]))
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_channel_swap('data', (2, 1, 0))
        self.transformer.set_raw_scale('data', 255.0)

    # Gets features from each image in given image path list.
    # Result is MxN numpy array where M is number of given images and N
    # is number of extracted features.
    def get_img_features(self, image_path_list, sketch=False):
        feats = []
        for full_path in image_path_list:
            print 'Image:', full_path
            feats.append(self.get_feature(full_path, sketch=sketch))
        return np.asarray(feats)

    # Extract features from given image path.
    # Retrieves features as 1xN numpy array.
    def get_feature(self, full_path, sketch=False):
        layer = self.output_layer_sketch if sketch else self.output_layer_image
        net = self.sketch_net if sketch else self.img_net
        inp = (self.transformer.preprocess('data', caffe.io.load_image(full_path)))
        sketch_in = np.reshape([inp], np.shape(self.sketch_net.blobs['data'].data))
        query = net.forward(data=sketch_in)
        query = np.copy(query[layer])
        query = np.reshape(query, [np.shape(query)[1]])
        return query
