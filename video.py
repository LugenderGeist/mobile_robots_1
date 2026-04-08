import cv2
import cv2.aruco as aruco
import numpy as np
from pathlib import Path
import json


class FieldRectifier:
    """
    Класс для выравнивания поля и обнаружения робота по ArUco метке
    """

    def __init__(self, video_path: str, output_path: str = "output_video.mp4"):
        self.video_path = video_path
        self.output_path = output_path
        self.field_width = None
        self.field_height = None
        self.corners = None
        self.H = None
        self.output_size = (720, 720)

        # ArUco для метки 6x6 (исправлено!)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.aruco_params = aruco.DetectorParameters_create()

        self.robot_trajectory = []

    def set_field_dimensions(self, width: float, height: float):
        self.field_width = width
        self.field_height = height
        print(f"✓ Размеры поля: {width} x {height}")

    def set_corners_manually(self, frame: np.ndarray) -> np.ndarray:
        corners = []

        h, w = frame.shape[:2]
        scale = min(1000 / w, 700 / h, 1.0)

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
                    print("\n✅ Выбраны все 4 угла! Нажмите 'q'")

        cv2.namedWindow("Select corners", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select corners", 1000, 700)
        cv2.imshow("Select corners", working_frame)
        cv2.setMouseCallback("Select corners", mouse_callback)

        print("\n📌 Выберите 4 угла поля: ЛВ -> ПВ -> ПН -> ЛН")
        print("👉 После выбора нажмите 'q'\n")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') and len(corners) == 4:
                break
            elif key == 27:
                cv2.destroyAllWindows()
                raise ValueError("Выбор отменён")

        cv2.destroyAllWindows()
        return np.array(corners, dtype=np.float32)

    def compute_homography(self) -> np.ndarray:
        if self.corners is None:
            raise ValueError("Нет углов поля")

        dst_corners = np.array([
            [0, 0], [self.output_size[0], 0],
            [self.output_size[0], self.output_size[1]], [0, self.output_size[1]]
        ], dtype=np.float32)

        self.H, _ = cv2.findHomography(self.corners, dst_corners)
        print(f"✓ Гомография вычислена, размер: {self.output_size[0]}x{self.output_size[1]}")
        return self.H

    def detect_robot(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict,
                                                     parameters=self.aruco_params)

        if ids is not None and len(ids) > 0:
            marker_id = ids[0][0]
            marker_corners = corners[0][0]

            center_x = np.mean(marker_corners[:, 0])
            center_y = np.mean(marker_corners[:, 1])

            point = np.array([[[center_x, center_y]]], dtype=np.float32)
            point_rect = cv2.perspectiveTransform(point, self.H)

            scale_x = self.field_width / self.output_size[0]
            scale_y = self.field_height / self.output_size[1]

            real_x = point_rect[0][0][0] * scale_x
            real_y = point_rect[0][0][1] * scale_y

            return True, marker_id, (point_rect[0][0][0], point_rect[0][0][1]), (real_x, real_y), marker_corners

        return False, -1, (0, 0), (0, 0), None

    def draw_axes_2d(self, frame: np.ndarray, marker_corners: np.ndarray, marker_id: int, axis_length: float = 60):
        """
        Нарисовать оси X и Y на ArUco метке (упрощённая версия)
        """
        # Убеждаемся, что marker_corners имеет правильную форму
        if len(marker_corners.shape) == 3:
            marker_corners = marker_corners[0]

        # Преобразуем углы в выровненные координаты
        pts = marker_corners.astype(np.float32).reshape(-1, 1, 2)
        rectified_corners = cv2.perspectiveTransform(pts, self.H)
        rectified_corners = rectified_corners.reshape(-1, 2)

        # Центр
        center = np.mean(rectified_corners, axis=0)
        cx, cy = int(center[0]), int(center[1])

        # Направления по углам метки
        # Вектор X: от угла 0 к углу 1 (вправо)
        dx = rectified_corners[1][0] - rectified_corners[0][0]
        dy = rectified_corners[1][1] - rectified_corners[0][1]
        length = np.sqrt(dx * dx + dy * dy)
        if length > 0:
            dx = dx / length * axis_length
            dy = dy / length * axis_length

        # Вектор Y: от угла 0 к углу 3 (вниз), но рисуем вверх
        ux = rectified_corners[3][0] - rectified_corners[0][0]
        uy = rectified_corners[3][1] - rectified_corners[0][1]
        ulength = np.sqrt(ux * ux + uy * uy)
        if ulength > 0:
            ux = ux / ulength * axis_length
            uy = uy / ulength * axis_length

        # Рисуем ось X (красная)
        x_end = (cx + int(dx), cy + int(dy))
        cv2.arrowedLine(frame, (cx, cy), x_end, (0, 0, 255), 2, tipLength=0.3)
        cv2.putText(frame, "X", (x_end[0] + 5, x_end[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Рисуем ось Y (зелёная) - вверх (противоположно направлению)
        y_end = (cx - int(ux), cy - int(uy))
        cv2.arrowedLine(frame, (cx, cy), y_end, (0, 255, 0), 2, tipLength=0.3)
        cv2.putText(frame, "Y", (y_end[0] + 5, y_end[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # ID метки
        cv2.putText(frame, f"ID:{marker_id}", (cx - 20, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Красный крестик
        cv2.line(frame, (cx - 8, cy), (cx + 8, cy), (0, 0, 255), 2)
        cv2.line(frame, (cx, cy - 8), (cx, cy + 8), (0, 0, 255), 2)

        return frame

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n📹 Видео: {self.video_path}")
        print(f"   Кадров: {total_frames}, FPS: {fps:.2f}")

        if self.corners is None:
            ret, first_frame = cap.read()
            if not ret:
                raise ValueError("Не удалось прочитать первый кадр")
            self.corners = self.set_corners_manually(first_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.compute_homography()

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                              fps, self.output_size)

        print(f"\n🔄 Обработка...\n")

        frame_count = 0
        processed = 0
        self.robot_trajectory = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            rectified = cv2.warpPerspective(frame, self.H, self.output_size)
            found, robot_id, center_pixel, center_real, marker_corners = self.detect_robot(frame)

            if found:
                rectified = self.draw_axes_2d(rectified, marker_corners, robot_id, axis_length=50)
                self.robot_trajectory.append({
                    'frame': frame_count,
                    'x_pixel': center_pixel[0],
                    'y_pixel': center_pixel[1],
                    'x_real': center_real[0],
                    'y_real': center_real[1]
                })

                if frame_count % 50 == 0:
                    print(f"  Кадр {frame_count}: Робот ID={robot_id}, "
                          f"позиция: ({center_real[0]:.2f}, {center_real[1]:.2f})")
            else:
                if frame_count % 100 == 0:
                    print(f"  Кадр {frame_count}: Робот не найден")

            out.write(rectified)
            processed += 1

            if processed % 100 == 0:
                print(f"  Прогресс: {processed}/{total_frames} ({processed * 100 / total_frames:.1f}%)")

        cap.release()
        out.release()

        print(f"\n✅ Обработка завершена!")
        print(f"   Обработано: {processed} кадров")
        print(f"   Обнаружений: {len(self.robot_trajectory)}")
        print(f"   Результат: {self.output_path}")

        return {
            'processed_frames': processed,
            'total_frames': total_frames,
            'fps': fps,
            'robot_detections': len(self.robot_trajectory)
        }

    def save_trajectory(self, output_file: str = "robot_trajectory.json"):
        """Сохранить траекторию движения робота в JSON файл"""
        if not self.robot_trajectory:
            print("✗ Нет данных о траектории для сохранения")
            return

        trajectory_json = []
        for point in self.robot_trajectory:
            trajectory_json.append({
                'frame': int(point['frame']),
                'x_pixel': float(point['x_pixel']),
                'y_pixel': float(point['y_pixel']),
                'x_real': float(point['x_real']),
                'y_real': float(point['y_real'])
            })

        data = {
            'field_size': [float(self.field_width), float(self.field_height)],
            'video_file': self.video_path,
            'total_detections': len(trajectory_json),
            'trajectory': trajectory_json
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Траектория сохранена в {output_file}")

    def save_corners(self, corners_file: str = "field_corners.json"):
        if self.corners is None:
            return
        data = {
            'corners': self.corners.tolist(),
            'field_width': self.field_width,
            'field_height': self.field_height
        }
        with open(corners_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Углы сохранены в {corners_file}")

    def load_corners(self, corners_file: str) -> bool:
        try:
            with open(corners_file, 'r') as f:
                data = json.load(f)
                self.corners = np.array(data['corners'], dtype=np.float32)
                self.field_width = data.get('field_width', self.field_width)
                self.field_height = data.get('field_height', self.field_height)
                print(f"✓ Углы загружены из {corners_file}")
                return True
        except Exception as e:
            print(f"✗ Не удалось загрузить углы: {e}")
            return False

