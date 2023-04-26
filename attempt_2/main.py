#КОД РАБОТАЕТ НА ВЕРСИИ PYTHON 3.10!!!!!!!!!!!!!!!!!!
# файл, где хранятся пользователи и инфа о них
import json
# штука для открытия фото
from PIL import Image
# удаление заднего фона картинки
import rembg
# для удаления фото, после моего работы с ними (чтоб память не загружать)
import os
# загрузка модели
from tensorflow.keras.models import load_model
# для перевода скачанной картинки в grayscale
import keras.utils as imag
import numpy as np
import cv2


class json_file:
    def write(data, file_name):
        data = json.dumps(data)
        data = json.loads(str(data))

        with open(file_name, 'w', encoding="utf-8") as file:
            json.dump(data, file, indent=4)

    def read(file_name):
        with open(file_name, 'r', encoding="utf-8") as file:
            return json.load(file)


class predict:
    def fashion(photo_file, photo):
        classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка',
                   'ботинки']

        model = load_model('fashion_mnist_dense.h5')

        image = Image.open(f"{photo.file_id}.jpg")

        # удаление фона с помощью rembg
        image = rembg.remove(image)

        # создание нового изображения с белым фоном аналогичного размера
        new_image = Image.new("RGB", image.size, (255, 255, 255))

        # наложение полученной png фотографии на белый фон
        new_image.paste(image, (0, 0), image)

        new_image.save(f"1_{photo.file_id}.jpg")

        img = imag.load_img(f"1_{photo.file_id}.jpg", target_size=(28, 28), color_mode='grayscale')

        x = imag.img_to_array(img)
        x = x.reshape(1, 784)
        x = 255 - x
        x /= 255

        os.remove(f"1_{photo.file_id}.jpg")

        prediction = model.predict(x)
        prediction = np.argmax(prediction)

        name = classes[prediction]
        return name


class color:
    def find(photo_file, photo):
        # # Объявление и нормализация изобажения
        I = cv2.imread(f"{photo.file_id}.jpg")
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

        x = I.shape[0] // 2
        y = I.shape[1] // 2

        # Получение цвета в массив с форматом RGB
        img = Image.open(f"{photo.file_id}.jpg")
        color = img.getpixel((x, y))
        os.remove(f"{photo.file_id}.jpg")

        # Интерпритация массива в название цвета
        if (-1, 150, 170) <= color <= (25, 175, 200):
            color = 'светло-голубой'
            # print(color)
            return color
        elif (255, 15, 0) <= color < (240, 45, 30):
            color = 'красный'
            # print(color)
            return color
        elif (240, 45, 30) <= color <= (185, 15, 0):
            color = 'темно - красный'
            # print(color)
            return color
        elif (255, 185, 130) <= color < (255, 165, 100):
            color = 'светло - оранжевый'
            # print(color)
            return color
        elif (255, 165, 100) <= color < (255, 115, 0):
            color = 'оранжевый'
            # print(color)
            return color
        elif (255, 115, 0) <= color <= (205, 90, 0):
            color = 'темно - оранжевый'
            # print(color)
            return color
        elif (255, 260, 185) <= color < (255, 250, 100):
            color = 'светло - желтый'
            # print(color)
            return color
        elif (255, 250, 100) <= color < (255, 250, 0):
            color = 'желтый'
            # print(color)
            return color
        elif (255, 250, 0) <= color <= (220, 215, 0):
            color = 'темно - желтый'
            # print(color)
            return color
        elif (205, 255, 200) <= color < (130, 255, 115):
            color = 'светло - зеленый'
            # print(color)
            return color
        elif (130, 255, 115) <= color < (25, 255, 0):
            color = 'зеленый'
            # print(color)
            return color
        elif (25, 255, 0) <= color <= (10, 100, 0):
            color = 'темно - зеленый'
            # print(color)
            return color
        elif (255, 195, 245) <= color < (255, 130, 230):
            color = 'светло - розовый'
            # print(color)
            return color
        elif (255, 130, 230) <= color < (255, 0, 210):
            color = 'розовый'
            # print(color)
            return color
        elif (255, 0, 210) <= color <= (190, 0, 155):
            color = 'темно - розовый'
            # print(color)
            return color
        elif (39, 188, 216) <= color <= (1, 147, 254):
            color = 'голубой'
            return color
        elif (5, 127, 250) <= color <= (56, 27, 228):
            color = 'синий'
            return color
        elif (98, 70, 185) <= color <= (169, 0, 255):
            color = 'фиолетовый'
            return color
        elif (170, 0, 255) <= color <= (236, 18, 237):
            color = 'сиреневый'
            return color
        elif (242, 13, 229) <= color <= (255, 0, 236):
            color = 'розовый'
            return color

# print(predict.fashion("2.jpg"))
# print(color.f("2.jpg"))
