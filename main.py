import matplotlib.pyplot as plt
from teval import Image
from pathlib import Path
import numpy as np

sample_idx = 4

# path_ = Path("/home/ftpuser/ftp/Data/HHI_Aachen/sample3/img1")

path_ = Path(f"E:\measurementdata\HHI_Aachen\sample{sample_idx}\img1")


def main():
    if sample_idx == 3:
        options = {"cbar_min": 1, "cbar_max": 3.0}
    else:
        options = {"cbar_min": 1, "cbar_max": 3.0}
    img = Image(path_, options=options, sample_idx=sample_idx)
    img.plot_image()

    plt.show()


if __name__ == '__main__':
    main()

