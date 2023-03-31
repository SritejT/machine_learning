import urllib.request
import pandas as pd
from PIL import Image

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

train_set = pd.read_csv("datasets/facial_expressions/faceexp-comparison-data-train-public.csv",  error_bad_lines=False)

response = urllib.request.urlopen(train_set["http://farm5.staticflickr.com/4108/5185055338_1dec873bf3_b.jpg"][1])

image = Image.open(response)
image.show()