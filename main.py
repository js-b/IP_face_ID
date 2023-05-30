import numpy as np
import face_recognition
import cv2
import os
from datetime import datetime, timedelta
import requests
import urllib

# Папка с известными лицами
directory = 'KnownFaces'

# Список для хранения кодировок лиц и имен
face_encodings_known = []
face_names = []

# Получаем список файлов из папки
file_list = os.listdir(directory)
print(file_list)

# Загружаем изображения и кодируем лица
for file_name in file_list:
    image = face_recognition.load_image_file(f'{directory}/{file_name}')
    face_encoding = face_recognition.face_encodings(image)[0]
    face_encodings_known.append(face_encoding)
    face_names.append(os.path.splitext(file_name)[0])

print(face_names)

# Папка для сохранения неизвестных лиц
unknown_directory = 'UnknownFaces'

# Создаем папку, если она не существует
if not os.path.exists(unknown_directory):
    os.makedirs(unknown_directory)

# Папка для сохранения присутствия
attendance_directory = 'Attendance'

# Создаем папку для сохранения присутствия, если она не существует
if not os.path.exists(attendance_directory):
    os.makedirs(attendance_directory)

# Словарь для хранения времени последней отметки для каждого человека
last_attendance_times = {}

# Временной интервал для повторной отметки в часах
repeat_interval = timedelta(hours=1)

# Функция для отметки присутствия
def mark_attendance(name):
    today = datetime.today().strftime('%Y-%m-%d')
    file_path = f"{attendance_directory}/{today}.csv"
    with open(file_path, "a") as f:
        now = datetime.now()
        time_string = now.strftime("%H:%M:%S")
        f.write(f'{name}, {time_string}\n')
    last_attendance_times[name] = now

# Функция для проверки времени последней отметки
def check_last_attendance(name):
    if name in last_attendance_times:
        last_time = last_attendance_times[name]
        if datetime.now() - last_time < repeat_interval:
            return False
    return True

# Функция для вывода процента совпадения лица
def show_face_match_percentage(face_distances):
    min_distance = min(face_distances)
    match_percentage = round((1 - min_distance) * 100)
    return match_percentage

# Получение видеопотока с IP-камеры
video_url = 'https://192.168.1.109:8080/video'
username = ' '  # Замените на свои реальные данные, если требуется аутентификация
password = ' '  # Замените на свои реальные данные, если требуется аутентификация
stream = requests.get(video_url, auth=(username, password), verify=False, stream=True)

# Открываем видеопоток с IP-камеры
capture = cv2.VideoCapture()
capture.open(video_url)

while True:
    # Считываем кадр из видеопотока
    success, frame = capture.read()

    # Уменьшаем размер кадра для ускорения обработки
    small_frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Находим расположение лиц и кодируем их в текущем кадре
    face_locations_current_frame = face_recognition.face_locations(small_frame)
    face_encodings_current_frame = face_recognition.face_encodings(small_frame, face_locations_current_frame)

    # Проверяем каждое лицо в текущем кадре
    for face_encoding, face_location in zip(face_encodings_current_frame, face_locations_current_frame):
        matches = face_recognition.compare_faces(face_encodings_known, face_encoding)
        face_distances = face_recognition.face_distance(face_encodings_known, face_encoding)
        match_index = np.argmin(face_distances)

        # Выводим процент совпадения лица
        match_percentage = show_face_match_percentage(face_distances)

        # Если найдено совпадение и процент совпадения больше 60
        if matches[match_index] and match_percentage > 60:
            name = face_names[match_index]
            top, right, bottom, left = [coordinate * 4 for coordinate in face_location]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Проверяем время последней отметки
            if check_last_attendance(name):
                top, right, bottom, left = [coordinate * 4 for coordinate in face_location]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                mark_attendance(name)
        else:
            # Рисуем прямоугольник вокруг неизвестного лица и добавляем надпись "Не опознан"
            top, right, bottom, left = [coordinate * 4 for coordinate in face_location]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, "Не опознан", (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        # Выводим процент совпадения лица
        cv2.putText(frame, f"Совпадение: {match_percentage}%", (left + 6, top - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    # Отображаем кадр в окне
    cv2.imshow("WebCam", frame)

    # Если нажата клавиша 'q', выходим из цикла
    if cv2.waitKey(1) == ord('q'):
        break

# Освобождаем ресурсы и закрываем окна
capture.release()
cv2.destroyAllWindows()
