from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as pyplot



def augmentation(mode=0):
    if mode == 0:
        # ToDo : 너비를 기준으로 [-200, 200]범위에서 shift하는 augmentation을 설정합니다.
        generator = ImageDataGenerator(width_shift_range=[-200, 200])

    elif mode == 1:
        # ToDo: 90도 회전하는 augmentation을 설정합니다.
        generator = ImageDataGenerator(rotation_range=90)

    elif mode == 2:
        # ToDo: 0.2~1.0으로 밝기를 변화시키는 augmentation을 설정합니다.
        generator = ImageDataGenerator(brightness_range=[0.2, 1.0])

    return generator


def visualizer(raw_img, generator):
    # 이미지를 불러옵니다.
    data = img_to_array(raw_img)
    samples = expand_dims(data, 0)
    it = generator.flow(samples, batch_size=1)

    # 이미지를 augmentation 결과에 따라 시각화합니다.
    for i in range(9):
        pyplot.subplot(330 + 1 + i)
        batch = it.next()
        image = batch[0].astype('uint8')
        pyplot.imshow(image)

    pyplot.show()


def main():
    img = load_img('./ferrari_mountain.jpg')

    # mode 0, 1, 2를 바꾸어 augmentation의 동작을 달리 해보세요.
    datagen = augmentation(mode=2)

    # 코드가 작동한다고 판단되면 아래 주석을 해제해 결과를 확인해 보세요.
    visualizer(img, datagen)


if __name__ == "__main__":
    main()
