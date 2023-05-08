#КОД РАБОТАЕТ НА ВЕРСИИ PYTHON 3.10!!!!!!!!!!!!!!!!!!
# файл, где хранятся пользователи и инфа о них
import json
# штука для открытия фото
from PIL import Image
# удаление заднего фона картинки
import rembg
# для удаления фото, после моего работы с ними (чтоб память не загружать)
import os
# 123
from sklearn.cluster import KMeans
# загрузка модели
from tensorflow.keras.models import load_model
# для перевода скачанной картинки в grayscale
from sklearn.cluster import KMeans
import keras.utils as imag
import numpy as np
import cv2


class JsonFile:
    def write(data, file_name):
        with open(file_name, 'w', encoding="utf-8") as file:
            json.dump(data, file, indent=4)

    def read(file_name):
        with open(file_name, 'r', encoding="utf-8") as file:
            return json.load(file)


class Predict:
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


class Color:
    def find(photo_file, photo):
        # # Объявление и нормализация изобажения
        I = cv2.imread(f"{photo.file_id}.jpg")
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

        x = I.shape[0] // 2
        y = I.shape[1] // 2

        # Получение цвета в массив с форматом RGB
        img = Image.open(f"{photo.file_id}.jpg")
        color = img.getpixel((x, y))
        color = [color]
        os.remove(f"{photo.file_id}.jpg")


        # Интерпритация массива в название цвета
        colors = np.array([
            [0, 0, 0],
            [255, 0, 0],
            [0, 128, 0],
            [0, 0, 255],
            [0, 255, 255],
            [255, 255, 255],
            [255, 165, 0],
            [255, 255, 0],
            [128, 0, 128],
            [255, 192, 203]
        ])

        # color_names = {
        #     (0, 0, 0): "чёрный",
        #     (128, 128, 128): "серый",
        #     (255, 0, 0): "красный",
        #     (0, 128, 0): "зелёный",
        #     (0, 0, 255): "синий",
        #     (0, 255, 255): "голубой",
        #     (255, 255, 255): "белый",
        #     (255, 165, 0): "оранжевый",
        #     (255, 255, 0): "жёлтый",
        #     (128, 0, 128): "фиолетовый",
        #     (165, 42, 42): "коричневый",
        #     (255, 192, 203): "розовый"
        # }

        # color = np.array(color)

        # closest_color_index = np.argmin(np.linalg.norm(colors - color, axis=1))

        # closest_color_name = color_names[tuple(colors[closest_color_index])]

        # print("Ваш цвет похож на: ", closest_color_name)
        kmeans = KMeans(n_clusters=10, random_state=0, n_init=10).fit(colors)

        predicted_label = kmeans.predict(color)

        # Нахождение ближайшего цвета по метке кластера
        closest_color = kmeans.cluster_centers_[predicted_label][0]

        def get_color_name(rgb):
            if np.array_equal(rgb, [0, 0, 0]):
                return "Черный"
            elif np.array_equal(rgb, [255, 0, 0]):
                return "Красный"
            elif np.array_equal(rgb, [0, 128, 0]):
                return "Зеленый"
            elif np.array_equal(rgb, [0, 0, 255]):
                return "Синий"
            elif np.array_equal(rgb, [0, 255, 255]):
                return "Голубой"
            elif np.array_equal(rgb, [255, 255, 255]):
                return "Белый"
            elif np.array_equal(rgb, [255, 165, 0]):
                return "Оранжевый"
            elif np.array_equal(rgb, [255, 255, 0]):
                return "Желтый"
            elif np.array_equal(rgb, [128, 0, 128]):
                return "Фиолетовый"
            elif np.array_equal(rgb, [255, 192, 203]):
                return "Розовый"
            else:
                return "Неизвестный"
            
        return(get_color_name(closest_color))


        # return closest_color_name
# print(predict.fashion("2.jpg"))
# print(color.f("2.jpg"))
