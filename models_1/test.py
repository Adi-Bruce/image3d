import sys

sys.path.insert(1,'/home/brucewayne/image3d/models_1')

from superpoint_superglue import SuperGlueMatcher

matcher = SuperGlueMatcher(device="cpu")

mk0,mk1, scores = matcher.match_pair('/home/brucewayne/Downloads/wallpaperflare.com_wallpaper.jpg','/home/brucewayne/Downloads/images.jpeg')

print(mk0.shape,mk1.shape,scores.shape)