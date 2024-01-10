import numpy as np
from pathlib import Path
from PIL import Image
import struct

MAIN_FOLDER = Path(__file__).parent


def write_tensor_to_file(tensor, filename):
    int_list = tensor.reshape(-1).tolist()
    my_struct = struct.Struct("<" + "B" * len(int_list))

    with open(MAIN_FOLDER / filename, "wb") as f:
        f.write(my_struct.pack(*int_list))


def preprocess_image():
    filepath = MAIN_FOLDER / "ch3_img_in.jpg"
    img = Image.open(filepath)
    data = np.array(img.getdata())
    print(f"Image shape: {img.size}")
    write_tensor_to_file(data, "ch3_img_in.dat")
    return img.size


def postprocess_image(filepath, img_shape, bytes_per_pixel):
    length = img_shape[0] * img_shape[1] * bytes_per_pixel
    my_struct = struct.Struct("<" + "B" * length)

    with open(filepath, mode="rb") as f:
        data = f.read()

    int_list = my_struct.unpack(data)

    if bytes_per_pixel == 1:
        int_array = np.array(int_list).reshape(img_shape[::-1])
    else:
        int_array = np.array(int_list).reshape((img_shape[1], img_shape[0], bytes_per_pixel))

    img = Image.fromarray(int_array.astype(np.uint8))

    # this is required by Pillow to save image
    img = img.convert("RGB")

    img_out_filepath = MAIN_FOLDER / filepath.name.replace(".dat", ".png")
    img.save(img_out_filepath)


if __name__ == "__main__":
    img_shape = preprocess_image()

    for filename, bytes_per_pixel in [
        ("ch3_img_convert_out.dat", 1),
        ("ch3_img_blur_out.dat", 3),
    ]:
        filepath = MAIN_FOLDER / filename

        if filepath.exists():
            print(f"Processing {filename}")
            postprocess_image(filepath, img_shape, bytes_per_pixel)
