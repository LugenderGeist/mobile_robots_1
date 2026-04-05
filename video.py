import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import json


class FieldRectifier:

    def __init__(self, video_path: str, output_path: str = "output_video.mp4"):
        self.video_path = video_path
        self.output_path = output_path
        self.field_width = None
        self.field_height = None
        self.corners = None
        self.H = None
        self.output_size = (720, 720)  # ФИКСИРОВАННЫЙ РАЗМЕР 720x720

    def set_field_dimensions(self, width: float, height: float):
        self.field_width = width
        self.field_height = height
        print(f"✓ Размеры поля заданы: {width} x {height}")

    def set_corners_manually(self, frame: np.ndarray) -> np.ndarray:
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
        if self.corners is None:
            raise ValueError("Не заданы углы поля")

        if self.field_width is None or self.field_height is None:
            raise ValueError("Не заданы реальные размеры поля")

        dst_corners = np.array([
            [0, 0],  # левый верхний
            [self.output_size[0], 0],  # правый верхний
            [self.output_size[0], self.output_size[1]],  # правый нижний
            [0, self.output_size[1]]  # левый нижний
        ], dtype=np.float32)

        # Вычисляем гомографию
        self.H, _ = cv2.findHomography(self.corners, dst_corners)

        print(f"\n✓ Матрица гомографии вычислена")
        print(f"  Выходной размер: {self.output_size[0]} x {self.output_size[1]}")

        return self.H

    def process_video_to_video(self, corners_file: Optional[str] = None) -> dict:
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

        # Создаём видео-запись
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, self.output_size)

        if not out.isOpened():
            # Пробуем другой кодек
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(self.output_path, fourcc, fps, self.output_size)

        print(f"\n Обработка видео...")

        frame_count = 0
        processed_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Выравниваем кадр
            rectified = cv2.warpPerspective(frame, self.H, self.output_size)

            # Записываем в выходное видео
            out.write(rectified)

            processed_count += 1

            # Показываем прогресс
            if processed_count % 100 == 0:
                percent = (processed_count / total_frames) * 100
                print(f"  Обработано: {processed_count}/{total_frames} кадров ({percent:.1f}%)")

        # Закрываем всё
        cap.release()
        out.release()

        print(f"\n✅ Обработка завершена!")
        print(f"  Обработано кадров: {processed_count}")
        print(f"  Результат сохранён: {self.output_path}")

        # Проверяем размер файла
        output_file = Path(self.output_path)
        if output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"  Размер файла: {size_mb:.1f} MB")

        return {
            'processed_frames': processed_count,
            'total_frames': total_frames,
            'fps': fps,
            'output_size': self.output_size,
            'output_path': self.output_path
        }

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