from Tkinter import *
from PIL import Image, ImageDraw, ImageTk
import index as idx
from sklearn.neighbors import NearestNeighbors
import numpy as np
from config import *


class DrawBox:
    def __init__(self, title='Sketchy', n=3):
        # creating model (neural network)
        self.model = idx.create_model()
        # loading indexed images from pickle file (dictionary)
        self.index = idx.load_obj('img_idx.pkl')
        # index all feature vectors
        self.nbrs = NearestNeighbors(n_neighbors=np.size(self.index['features'], 0), algorithm='brute', metric='cosine'). \
            fit(self.index['features'])
        # number of retrieved images
        self.n = n

        # main window config
        self.fname = config.get('Canvas', 'query_sketch')
        self.root = Tk()
        self.root.title(title)
        # self.root.resizable(0, 0)

        # canvas and pil config
        self.draw_clr = '#000000'
        self.width = 256
        self.height = 256
        self.white = (255, 255, 255)

        # defining canvas
        self.canvas = Canvas(self.root, bg='white', width=self.width, height=self.height)
        # self.canvas.pack()
        self.canvas.grid(row=0, column=0)
        self.canvas.bind('<B1-Motion>', self.motion)

        # defining background PIL image that is the indentical as the one drawn in canvas
        self.sketch_img = Image.new("RGB", (self.width, self.height), self.white)
        self.draw_img = ImageDraw.Draw(self.sketch_img)

        # button frame
        self.button_frame = Frame(self.root)
        # defining button that runs the search
        self.run_button = Button(self.button_frame, text="Run", command=self.run).pack(side='left')
        self.clear_button = Button(self.button_frame, text="Clear", command=self.clear_canvas).pack(side='right')
        self.button_frame.grid(row=1, column=0)

    # Clear canvas button
    def clear_canvas(self):
        self.canvas.delete('all')
        self.sketch_img = Image.new("RGB", (self.width, self.height), self.white)
        self.draw_img = ImageDraw.Draw(self.sketch_img)

    # Button action, saves the PIL image as a file and makes query from that file.
    # Show n closest and furthest images from given sketch
    def run(self):
        self.sketch_img.save(self.fname)
        closest = idx.get_n_most_similar(self.n, self.model, self.index, self.fname, self.nbrs)
        furthest = idx.get_n_most_distant(self.n, self.model, self.index, self.fname, self.nbrs)
        i = 0

        for img_path, distance in closest:
            img = ImageTk.PhotoImage(Image.open(img_path))
            attr_name = 'result_closest' + str(i)
            setattr(self.root, attr_name, img)
            Label(self.root, image=getattr(self.root, attr_name)).grid(row=0, column=i+1)
            txt = 'Distance: ' + str(distance)
            Label(self.root, text=txt).grid(row=1, column=i+1)
            i += 1
        i = 0
        for img_path, distance in furthest:
            img = ImageTk.PhotoImage(Image.open(img_path))
            attr_name = 'result_furthest' + str(i)
            setattr(self.root, attr_name, img)
            Label(self.root, image=getattr(self.root, attr_name)).grid(row=2, column=i+1)
            txt = 'Distance: ' + str(distance)
            Label(self.root, text=txt).grid(row=3, column=i+1)
            i += 1

    # Mouse listener, draws simultaneously on canvas and background PIL image.
    def motion(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.draw_clr)
        self.draw_img.ellipse([x1, y1, x2, y2], fill=self.draw_clr)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser('Index')
    parser.add_argument('-config', type=str)
    args = parser.parse_args()
    parse_conf_file(args.config)
    draw_box = DrawBox()
    draw_box.root.mainloop()
