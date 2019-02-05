from PIL import Image

img = Image.open("testing_prediction_cat_cont.jpg", mode="r")
img = img.convert('1') # convert image to black and white
img.save("testing_prediction_cat_cont_50_bw.jpg")

a = [1 if x == 255 else x for x in list(img.getdata())]

with open("testing_img_data.txt", "w") as f:
    f.write(str(a))

print(len(a))
