# A script to fix corrupted RGB images

import sys
import numpy as np
import interpol
import matplotlib.pyplot as plt

def _main(filename):

    # Load corrupted image from file    
    I = plt.imread(filename)

    # Show the original, corrupted image on the screen
    plt.imshow(I)
    plt.show(block=True)

    # Figure out how to detect the defective red pixels

    bad_px = np.argwhere((I[:, :, 0] == 1) & (I[:, :, 1] == 0) & (I[:, :, 2] == 0))
                    
    # Send the corrupted image and list of bad pixels to be fixed
    I = interpol.pixelfix(I, bad_px)

    # Save the fixed image and show it on screen
    plt.imshow(I)
    plt.savefig(f"fixed_{filename}")
    plt.show(block=True)


if __name__ == '__main__':
    _main(sys.argv[1])
    