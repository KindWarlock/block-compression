import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from scipy.misc import face


def blocks(arr, yb, xb):
    new_shape = yb, xb
    new_arr = np.zeros(new_shape)
    block_size_y = arr.shape[0] // yb
    block_size_x = arr.shape[1] // xb

    for y in range(yb):
        # the last one may be a bit larger
        if y == yb - 1:
            end_y = arr.shape[0]
        else:
            end_y = block_size_y * (y + 1)
        for x in range(xb):
            if x == xb - 1:
                end_x = arr.shape[1]
            else:
                end_x = block_size_x * (x + 1)
            new_arr[y, x] = np.mean(arr[y * block_size_y:end_y, 
                                        x * block_size_x:end_x])

    return new_arr

# ЕНОТ
# img = face(gray=True)
# plt.subplot(121)
# plt.imshow(img)
# plt.subplot(122)
# plt.imshow(blocks(img, img.shape[0] // 10, img.shape[1] // 10))

# ШАКАЛЫ
img = mpimg.imread('images.jpeg')
for i in range(1, 11): 
    plt.subplot(1, 10, i) 
    plt.imshow(blocks(img, img.shape[0] // i, img.shape[1] // i))
    plt.title(f'{i} / 10')
    plt.xticks([])
    plt.yticks([])

plt.show()
