#Импорт необходимых библиотек
import cv2
import numpy as np
import matplotlib.pyplot as plt
#для выбора нужных файлов с диска
import tkinter as tk
from tkinter import filedialog, messagebox 

#Блок загрузки изображений с диска
def select_and_analyze():
    """Простой выбор файла и базовый анализ"""
    root = tk.Tk()
    root.withdraw()
file_path = filedialog.askopenfilename(
        title="Выберите спутниковый снимок",
        filetypes=[("Изображения", "*.jpg *.jpeg *.png *.tif *.bmp"), 
                  ("Все файлы", "*.*")]
    )

image_path = file_path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Преобразование в монохром
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Пороговая обработка для выделения светлых областей (ледник) Значение порога (для данного набора данных 200, но можно менять для других наборов данных)
threshold_value = 200
_, binary_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

#Морфологические операции
kernel = np.ones((5,5), np.uint8)
binary_mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel) # эрозия
binary_mask_cleaned = cv2.morphologyEx(binary_mask_cleaned, cv2.MORPH_OPEN, kernel) # делатация

#Поиск контуров
contours, _ = cv2.findContours(binary_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Фильтрация контуров по минимальной площади (для данного набора данных оптимально 3000)
min_contour_area = 3000
large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

#Расчёт площади ледника
#Сумма площадей всех найденных контуров
glacier_area_from_contours = sum(cv2.contourArea(cnt) for cnt in large_contours)
print(f"Площадь ледника (по контурам): {int(glacier_area_from_contours)} кв. пикс.")
gpl2 = glacier_area_from_contours*576/1000000
print(f"Площадь ледника (по контурам): {int(gpl2)} кв. км")

#Визуализация результата
#Вынос контуров на изображение
result_image = image_rgb.copy()
cv2.drawContours(result_image, large_contours, -1, (255, 0, 0), 1) 

#Отображение всех этапов
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].imshow(image_rgb)
axes[0, 0].set_title('Исходное изображение')
axes[0, 0].axis('off')

axes[0, 1].imshow(gray, cmap='gray')
axes[0, 1].set_title('Монохром(контрастный)')
axes[0, 1].axis('off')

axes[1, 0].imshow(binary_mask_cleaned, cmap='gray')
axes[1, 0].set_title('Бинарная маска (ледник = белый)')
axes[1, 0].axis('off')

axes[1, 1].imshow(result_image)
axes[1, 1].set_title('Обработанное изображение с контуром')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()
