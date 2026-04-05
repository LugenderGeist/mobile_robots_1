import cv2
import cv2.aruco as aruco
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import json


class FieldRectifier:
    """
    Класс для выравнивания поля и обнаружения ОДНОГО робота по ArUco метке
    С оптимизированными параметрами детектора для работы с искажёнными метками
    """

    def __init__(self, video_path: str, output_path: str = "output_video.mp4"):
        """
        Инициализация
        """
        self.video_path = video_path
        self.output_path = output_path
        self.field_width = None
        self.field_height = None
        self.corners = None
        self.H = None
        self.output_size = (720, 720)

        # Настройки ArUco для 7x7 меток
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_7X7_50)

        # Оптимизированные параметры детектора для работы с искажёнными метками
        self.aruco_params = aruco.DetectorParameters_create()
        self._configure_detector_params()

        # Хранение траектории
        self.robot_trajectory = []

    def _configure_detector_params(self):
        """
        Настройка параметров детектора ArUco для устойчивого распознавания
        меток, расположенных под углом и с искажениями
        """
        # === Адаптивная пороговая обработка ===
        # Увеличиваем диапазон размеров окна для адаптивной пороговой обработки
        # Это помогает при неравномерном освещении
        self.aruco_params.adaptiveThreshWinSizeMin = 3
        self.aruco_params.adaptiveThreshWinSizeMax = 31
        self.aruco_params.adaptiveThreshWinSizeStep = 4

        # Константа, вычитаемая из среднего взвешенного (стандартное значение)
        self.aruco_params.adaptiveThreshConstant = 7

        # === Фильтрация кандидатов по размеру ===
        # Минимальная длина периметра маркера (в пикселях)
        # Уменьшаем, чтобы находить даже маленькие метки
        self.aruco_params.minMarkerPerimeterRate = 0.01

        # Максимальная длина периметра маркера
        self.aruco_params.maxMarkerPerimeterRate = 0.8

        # === Фильтрация по форме ===
        # Минимальное расстояние между углами (отсеивает шумовые кандидаты)
        self.aruco_params.minCornerDistanceRate = 0.05

        # Минимальное расстояние между маркерами
        self.aruco_params.minMarkerDistanceRate = 0.05

        # === Параметры декодирования ===
        # Количество битов в границе маркера (стандартно 1)
        self.aruco_params.markerBorderBits = 1

        # Коррекция ошибок при декодировании
        # Увеличиваем для более толерантного распознавания искажённых меток
        self.aruco_params.errorCorrectionRate = 0.8

        # === Перспективные искажения ===
        # Количество пикселей на ячейку при удалении перспективы
        self.aruco_params.perspectiveRemovePixelPerCell = 4

        # Допустимый зазор при удалении перспективы
        self.aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.13

        # === Уточнение углов ===
        # Использовать уточнение углов (повышает точность)
        self.aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
        self.aruco_params.cornerRefinementWinSize = 5
        self.aruco_params.cornerRefinementMaxIterations = 30
        self.aruco_params.cornerRefinementMinAccuracy = 0.05

    def set_field_dimensions(self, width: float, height: float):
        """Задать реальные размеры поля"""
        self.field_width = width
        self.field_height = height
        print(f"✓ Размеры поля заданы: {width} x {height}")

    def set_corners_manually(self, frame: np.ndarray) -> np.ndarray:
        """Интерактивный выбор углов поля"""
        corners = []

        # Масштабируем для удобного отображения
        max_display_width = 1000
        max_display_height = 700

        h, w = frame.shape[:2]
        scale = min(max_display_width / w, max_display_height / h, 1.0)

        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            display_frame = cv2.resize(frame, (new_w, new_h))
            print(f"🖼️ Изображение уменьшено: {w}x{h} -> {new_w}x{new_h}")
        else:
            display_frame = frame.copy()
            scale = 1.0

        working_frame = display_frame.copy()

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                orig_x, orig_y = int(x / scale), int(y / scale)
                corners.append((orig_x, orig_y))

                cv2.circle(working_frame, (x, y), 6, (0, 255, 0), -1)
                cv2.putText(working_frame, str(len(corners)), (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.imshow("Select corners", working_frame)
                print(f"  Точка {len(corners)}: ({orig_x}, {orig_y})")

                if len(corners) == 4:
                    print("\n✅ Выбраны все 4 угла!")
                    print("📢 Нажмите 'q' для подтверждения")

        cv2.namedWindow("Select corners", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select corners", 1000, 700)
        cv2.imshow("Select corners", working_frame)
        cv2.setMouseCallback("Select corners", mouse_callback)

        print("\n📌 Выберите 4 угла поля в порядке: ЛВ -> ПВ -> ПН -> ЛН")
        print("👉 После выбора 4 точек нажмите 'q'")
        print("👉 Для отмены нажмите 'ESC'\n")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') and len(corners) == 4:
                break
            elif key == ord('q') and len(corners) != 4:
                print(f"⚠️ Выбрано {len(corners)} точек из 4")
            elif key == 27:
                cv2.destroyAllWindows()
                raise ValueError("Выбор углов отменён")

        cv2.destroyAllWindows()
        return np.array(corners, dtype=np.float32)

    def compute_homography(self) -> np.ndarray:
        """Вычислить матрицу гомографии для преобразования в 720x720"""
        if self.corners is None:
            raise ValueError("Не заданы углы поля")

        if self.field_width is None or self.field_height is None:
            raise ValueError("Не заданы реальные размеры поля")

        # Целевые координаты - квадрат 720x720
        dst_corners = np.array([
            [0, 0],
            [self.output_size[0], 0],
            [self.output_size[0], self.output_size[1]],
            [0, self.output_size[1]]
        ], dtype=np.float32)

        # Вычисляем гомографию
        self.H, _ = cv2.findHomography(self.corners, dst_corners)

        print(f"\n✓ Матрица гомографии вычислена")
        print(f"  Выходной размер: {self.output_size[0]} x {self.output_size[1]}")

        return self.H

    def detect_robot(self, frame: np.ndarray) -> Tuple[bool, int, Tuple[float, float], Tuple[float, float], np.ndarray]:
        """
        Обнаружить робота по ArUco метке на кадре

        Returns:
            found: найден ли робот
            marker_id: ID метки (если найден)
            center_pixel: координаты центра в пикселях выровненного изображения (x, y)
            center_real: координаты центра в реальных единицах поля (x, y)
            marker_corners: углы метки (для отрисовки)
        """
        # Преобразуем в оттенки серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Ищем метки
        corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        # Если метка не найдена
        if ids is None or len(ids) == 0:
            return False, -1, (0, 0), (0, 0), None

        # Берём первую найденную метку (у нас только одна)
        marker_id = ids[0][0]
        marker_corners = corners[0][0]

        # Вычисляем центр метки в исходных координатах
        center_x_orig = np.mean(marker_corners[:, 0])
        center_y_orig = np.mean(marker_corners[:, 1])

        # Преобразуем центр в выровненные координаты (вид сверху)
        point_in_original = np.array([[[center_x_orig, center_y_orig]]], dtype=np.float32)
        point_in_rectified = cv2.perspectiveTransform(point_in_original, self.H)

        center_x_rect = point_in_rectified[0][0][0]
        center_y_rect = point_in_rectified[0][0][1]

        # Преобразуем в реальные координаты поля
        scale_x = self.field_width / self.output_size[0]
        scale_y = self.field_height / self.output_size[1]

        real_x = center_x_rect * scale_x
        real_y = center_y_rect * scale_y

        return True, marker_id, (center_x_rect, center_y_rect), (real_x, real_y), marker_corners

    def draw_robot(self, frame: np.ndarray, marker_id: int, center_pixel: Tuple[float, float],
                   marker_corners: np.ndarray = None) -> np.ndarray:
        """
        Нарисовать робота на выровненном кадре

        Args:
            frame: выровненный кадр
            marker_id: ID метки
            center_pixel: координаты центра в пикселях
            marker_corners: углы метки для рисования контура

        Returns:
            кадр с отрисованным роботом
        """
        x, y = int(center_pixel[0]), int(center_pixel[1])

        # Если есть углы метки, рисуем зелёный контур
        if marker_corners is not None:
            corners_int = marker_corners.astype(np.int32)
            cv2.polylines(frame, [corners_int], True, (0, 255, 0), 2)

        # Рисуем красный круг в месте нахождения робота
        cv2.circle(frame, (x, y), 15, (0, 0, 255), -1)
        cv2.circle(frame, (x, y), 15, (255, 255, 255), 2)

        # Добавляем текст с ID
        cv2.putText(frame, f"Robot ID: {marker_id}", (x - 40, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Рисуем крестик в центре
        cv2.line(frame, (x - 8, y), (x + 8, y), (255, 255, 255), 2)
        cv2.line(frame, (x, y - 8), (x, y + 8), (255, 255, 255), 2)

        return frame

    def process_video(self) -> Dict:
        """
        Обработка видео: выравнивание поля и отслеживание робота
        """
        # Открываем видео
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {self.video_path}")

        # Получаем информацию о видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n📹 Информация о видео:")
        print(f"  Файл: {self.video_path}")
        print(f"  Всего кадров: {total_frames}")
        print(f"  FPS: {fps:.2f}")

        # Выбираем углы на первом кадре
        if self.corners is None:
            ret, first_frame = cap.read()
            if not ret:
                raise ValueError("Не удалось прочитать первый кадр")
            self.corners = self.set_corners_manually(first_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Вычисляем гомографию
        self.compute_homography()

        # Создаём папку для выходного видео
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        # Создаём видео-запись
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, self.output_size)

        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(self.output_path, fourcc, fps, self.output_size)

        print(f"\n🔄 Обработка видео и отслеживание робота...")
        print("=" * 60)

        frame_count = 0
        processed_count = 0
        self.robot_trajectory = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 1. Выравниваем поле
            rectified = cv2.warpPerspective(frame, self.H, self.output_size)

            # 2. Ищем робота на исходном кадре
            found, robot_id, center_pixel, center_real, marker_corners = self.detect_robot(frame)

            # 3. Если робот найден, рисуем его на выровненном кадре
            if found:
                rectified = self.draw_robot(rectified, robot_id, center_pixel, marker_corners)

                # Сохраняем данные о позиции
                self.robot_trajectory.append({
                    'frame': frame_count,
                    'x_pixel': center_pixel[0],
                    'y_pixel': center_pixel[1],
                    'x_real': center_real[0],
                    'y_real': center_real[1]
                })

                # Выводим информацию в консоль (каждый 30-й кадр)
                if frame_count % 30 == 0 or frame_count == 1:
                    print(f"  Кадр {frame_count}: Робот ID={robot_id}, "
                          f"позиция: ({center_real[0]:.2f}, {center_real[1]:.2f})")
            else:
                if frame_count % 100 == 0:
                    print(f"  Кадр {frame_count}: Робот не обнаружен")

            # 4. Записываем кадр в видео
            out.write(rectified)
            processed_count += 1

            # Прогресс
            if processed_count % 100 == 0:
                percent = (processed_count / total_frames) * 100
                print(f"  Прогресс: {processed_count}/{total_frames} кадров ({percent:.1f}%)")

        # Закрываем всё
        cap.release()
        out.release()

        print("=" * 60)
        print(f"\n✅ Обработка завершена!")
        print(f"  Обработано кадров: {processed_count}")
        print(f"  Робот обнаружен в {len(self.robot_trajectory)} кадрах")
        print(f"  Результат сохранён: {self.output_path}")

        detection_rate = (len(self.robot_trajectory) / processed_count) * 100 if processed_count > 0 else 0
        print(f"  Процент обнаружения: {detection_rate:.1f}%")

        return {
            'processed_frames': processed_count,
            'total_frames': total_frames,
            'fps': fps,
            'robot_detections': len(self.robot_trajectory),
            'detection_rate': detection_rate,
            'robot_trajectory': self.robot_trajectory
        }

    def save_trajectory(self, output_file: str = "robot_trajectory.json"):
        """Сохранить траекторию движения робота в JSON файл"""
        if not self.robot_trajectory:
            print("✗ Нет данных о траектории для сохранения")
            return

        data = {
            'field_size': (self.field_width, self.field_height),
            'video_file': self.video_path,
            'total_detections': len(self.robot_trajectory),
            'trajectory': self.robot_trajectory
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Траектория сохранена в {output_file}")

    def save_corners(self, corners_file: str = "field_corners.json"):
        """Сохранить координаты углов"""
        if self.corners is None:
            print("✗ Нет координат для сохранения")
            return

        data = {
            'corners': self.corners.tolist(),
            'field_width': self.field_width,
            'field_height': self.field_height,
            'video_file': self.video_path
        }

        with open(corners_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Координаты сохранены в {corners_file}")

    def load_corners(self, corners_file: str) -> bool:
        """Загрузить координаты углов"""
        try:
            with open(corners_file, 'r') as f:
                data = json.load(f)
                self.corners = np.array(data['corners'], dtype=np.float32)
                print(f"✓ Загружены координаты углов из {corners_file}")
                return True
        except Exception as e:
            print(f"✗ Не удалось загрузить координаты: {e}")
            return False