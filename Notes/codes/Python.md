# Python Notes

## Read images

Need to import: 
``` python
import cv2
from PIL import Image
import numpy as np
```

Basic methods:
``` python
image = cv2.imread(img_file)
image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
```

After above, we can do some extra processing: 

* Padding images
``` python 
def PaddingImage(image):
    image_size = image.size
    max_edge = max(image_size)
    padding_img = Image.new('RGB', (max_edge, max_edge))
    padding_img.paste(image, (0, 0))
    return padding_img
```

* Resize images:
``` python
def ResizeImage(image):
    return image.resize([608, 608], Image.ANTIALIAS)
```

For more details:
* `new(mode, size, color)`: create a new Image based on `mode`, `size` and `color`.
* `padding_image.paste(image, position)`: paste `image` on `padding_image` at (0, 0). 
