from model import Model
from os import walk
import pickle
from config import *


def save_obj(pkl_path, obj):
    with open(pkl_path, 'wb') as f:
        pickle.dump(obj, f)


def load_obj(pkl_path):
    with open(pkl_path, 'rb') as f:
        obj = pickle.load(f)
    return obj


# Extract features from all images in root_dir using given model
# Return dictionary containing lists of image_paths and features
def create_features_idx(model, root_dir, sketch=False):
    image_paths = []
    features = []
    print 'Calculating features'
    counter = 0
    for dirpath, dirnames, filenames in walk(root_dir):
        for img_path in filenames:
            counter += 1
            print 'Image', counter, '->', img_path
            full_path = dirpath + '/' + img_path
            feature = model.get_feature(full_path, sketch=sketch)
            image_paths.append(full_path)
            features.append(feature)
    return {'features': features, 'image_paths': image_paths}


# Gets n most similar images for given sketch
def get_n_most_similar(n, model, index, sketch_path, nbrs):
    query = model.get_feature(sketch_path, sketch=True)
    distances, indices = nbrs.kneighbors([query])
    return [(index['paths'][indices[0][i]], distances[0][i]) for i in range(n)]


# Gets n most distant images for given sketch
def get_n_most_distant(n, model, index, sketch_path, nbrs):
    query = model.get_feature(sketch_path, sketch=True)
    distances, indices = nbrs.kneighbors([query])
    start_idx = len(indices[0])-1
    end_idx = start_idx-n
    return [(index['paths'][indices[0][i]], distances[0][i]) for i in range(start_idx, end_idx, -1)]


# Creates model from model configuration files.
def create_model():
    caffemodel = config.get('Network', 'caffemodel')
    sketch_prototxt = config.get('Network', 'sketch_prototxt')
    image_prototxt = config.get('Network', 'image_prototxt')
    model = Model(caffemodel, sketch_prototxt, image_prototxt)
    return model


# Creates index dictionary using given model and image root directory.
def create_index(model, sketch=False):
    root_dir = config.get('Index', 'image_root_dir') if not sketch else config.get('Index', 'sketch_root_dir')
    return create_features_idx(model, root_dir, sketch=sketch)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Index')
    parser.add_argument('-config', type=str)
    args = parser.parse_args()
    parse_conf_file(args.config)
    model = create_model()
    index_dict = create_index(model)
    save_obj(config.get('Index', 'pickle_img_idx'), index_dict)
