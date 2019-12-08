from PIL import Image

# This piece of code makes a list of 0 and 1 (black and white dots of the contour image with a size 50x50)
# It also saves the data into txt file.

img = Image.open("testing_prediction_cat_cont.jpg", mode="r")
img = img.convert('1') # convert image to black and white
img.save("testing_prediction_cat_cont_50_bw.jpg")

a = [1 if x == 255 else x for x in list(img.getdata())]

with open("testing_img_data.txt", "w") as f:
    f.write(str(a))

print(len(a))
