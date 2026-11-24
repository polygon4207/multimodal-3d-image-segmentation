from PIL import Image

img = Image.open("data/lucchi/training.tif")
print(img.n_frames)  # nombre de pages (nombre de coupes) = 165
print(img.size)  # (largeur, hauteur) d'une page = (1024, 768)
