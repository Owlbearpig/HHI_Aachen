from teval.Measurements.image import Image
from pathlib import Path

path_ = Path("/home/ftpuser/ftp/Data/HHI_Aachen/sample3/img1")


def main():
    img = Image(path_)
    img.plot_image()


if __name__ == '__main__':
    main()

