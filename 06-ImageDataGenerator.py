from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as pyplot



def augmentation(mode=0):
    if mode == 0:
        # shift in the range of [-200, 200]
        generator = ImageDataGenerator(width_shift_range=[-200, 200])

    elif mode == 1:
        # 90 degrees rotation
        generator = ImageDataGenerator(rotation_range=90)

    elif mode == 2:
        # change brightness ranging from 0.2 to 1.0
        generator = ImageDataGenerator(brightness_range=[0.2, 1.0])

    return generator


def visualizer(raw_img, generator):
    # load raw image as an array
    data = img_to_array(raw_img)
    samples = expand_dims(data, 0)
    it = generator.flow(samples, batch_size=1)

    # visualise augmentation
    for i in range(9):
        pyplot.subplot(330 + 1 + i)
        batch = it.next()
        image = batch[0].astype('uint8')
        pyplot.imshow(image)

    pyplot.show()


def main():
    img = load_img('./ferrari_mountain.jpg')
    generator = augmentation(mode=0)

    visualizer(img, generator)


if __name__ == "__main__":
    main()
