import matplotlib.pyplot as plt
import numpy as np
image = np.array([(1, 1, 1, 1, 1, 1, 1, 1, 1, 0),
    (0, 1, 0, 1, 1, 1, 0, 0, 0, 0),
    (0, 1, 1, 0, 1, 0, 1, 0, 1, 0),
    (0, 1, 1, 0, 1, 0, 1, 0, 1, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 1)]);
plt.imshow(image, cmap='gray')
plt.show()
noisy = image + 0.2 * np.random.rand(5, 10)
print(noisy)
noisy = noisy/noisy.max()
plt.imshow(noisy, cmap='gray')
plt.show()
rand_index = np.random.randint(0, 32)
rand_image = np.array(image)
rand_image[int(rand_index % 4),
int(rand_index / 4)] = np.where(
rand_image[int(rand_index % 4), int(rand_index / 4)] == 1,
0.0, 1.0)
plt.imshow(rand_image, cmap='gray')
print(rand_image)
plt.show()