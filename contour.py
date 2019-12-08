import pylab
from PIL import Image

# This piece of code is responsible for making a contour of an animal shown on the page. It resizes the given
# image as well.

# read image to array
im = pylab.array(Image.open("testing_prediction_cat.jpg").convert("L"))

# create a new figure
pylab.figure(figsize=(0.5, 0.5))

# show contours with origin upper left corner
pylab.contour(im, levels=[245], colors="black", origin="image")
pylab.axis("equal")

# pylab.show()
pylab.savefig("testing_prediction_cat_cont.jpg")
