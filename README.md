# sketchy
Simple tool for sketch based image retrieval.

Desktop application written in Python which allows user to draw sektches and then retrieve the most similiar images from image database.

Application is based on a [Sketchy](http://sketchy.eye.gatech.edu) paper. Trained neural network, image and sketch database can be downloaded from there and are used in this implementation.

[Caffe](http://caffe.berkeleyvision.org) is used for data processing and calculating neural network features.

### How to use

Important files:
- model.py: API of model for processing images.
- index.py: API for fetching nearest images in respect of users sketch query.
- main.py: Main script which show main window where user can draw sketches and retrieve images.
- config.py: Configuration object.
- config.cfg: Configuration file.

How to run:
1. Download all necessary files from [Sketchy](http://sketchy.eye.gatech.edu).
2. Create .cfg configuration file.
3. Run index.py to create mapping from image database paths to image feature vectors. Save that mapping locally as a serialized object (pickle).
```
python index.py -config config.cfg
```
4. Run main.py to draw sketches and query database.
```
python main.py -config config.cfg
```


Main window for drawing sketches

![](/app_images/main.png)

Displaying the most similar and most different images based on sketch

![](/app_images/query.png)


