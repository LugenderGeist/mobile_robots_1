import cv2
import cv2.aruco as aruco
import numpy as np
from pathlib import Path
import json


class FieldRectifier:
    def __init__(self, video_path: str, output_path: str = "output_video.mp4"):
        self.video_path = video_path
        self.output_path = output_path
        self.field_width = None
        self.field_height = None
        self.corners = None
        self.H = None
        self.output_size = (720, 720)

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
        # Вычисляем обратную матрицу для преобразования из выровненных в исходные координаты
        self.H_inv, _ = cv2.findHomography(dst_corners, self.corners)

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

    def set_obstacle_params(self, edge_margin: int = 20,
                            robot_exclusion_radius: int = 60,
                            min_area: int = 500,
                            threshold_v: int = 100):
        """
        Установить параметры обнаружения препятствий
        """
        self.edge_margin = edge_margin
        self.robot_exclusion_radius = robot_exclusion_radius
        self.obstacle_min_area = min_area
        self.obstacle_threshold_v = threshold_v
        print(f"✓ Параметры обнаружения препятствий:")
        print(f"   Отступ от края (закрашивание): {edge_margin} px")
        print(f"   Зона исключения робота: {robot_exclusion_radius} px")
        print(f"   Мин. площадь: {min_area} px")
        print(f"   Порог яркости: {threshold_v}")

    def detect_obstacles(self, frame: np.ndarray, robot_center: tuple = None) -> list:
        """
        Обнаружить препятствия на поле (тёмные объекты, независимо от цвета)

        Args:
            frame: исходный кадр (BGR)
            robot_center: координаты центра робота в ВЫРОВНЕННЫХ пикселях (x, y)

        Returns:
            список препятствий с координатами в реальных единицах
        """
        # Используем параметры из self
        margin_pixels = getattr(self, 'margin_pixels', 5)
        robot_exclusion_radius = getattr(self, 'robot_exclusion_radius', 60)
        min_area = getattr(self, 'obstacle_min_area', 500)
        threshold_v = getattr(self, 'obstacle_threshold_v', 100)

        # 1. Конвертируем в HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 2. Извлекаем канал яркости (Value)
        v_channel = hsv[:, :, 2]

        # 3. Пороговая обработка по яркости
        _, mask = cv2.threshold(v_channel, threshold_v, 255, cv2.THRESH_BINARY_INV)

        # 4. Морфологические операции для очистки маски
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 5. Если задан центр робота, "вырезаем" зону вокруг него из маски
        if robot_center is not None:
            # Преобразуем центр робота из выровненных координат в исходные
            point_rect = np.array([[[robot_center[0], robot_center[1]]]], dtype=np.float32)
            point_orig = cv2.perspectiveTransform(point_rect, self.H_inv)

            if point_orig is not None and len(point_orig) > 0:
                cx_orig = int(point_orig[0][0][0])
                cy_orig = int(point_orig[0][0][1])

                # Создаём маску для зоны робота в исходных координатах
                robot_mask = np.zeros_like(mask)
                cv2.circle(robot_mask, (cx_orig, cy_orig), robot_exclusion_radius, 255, -1)

                # Убираем зону робота из маски препятствий
                mask = cv2.bitwise_and(mask, cv2.bitwise_not(robot_mask))

        # 6. Поиск контуров
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        obstacles = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            M = cv2.moments(contour)
            if M["m00"] != 0:
                center_x_orig = M["m10"] / M["m00"]
                center_y_orig = M["m01"] / M["m00"]
            else:
                continue

            # Преобразуем центр в выровненные координаты
            point = np.array([[[center_x_orig, center_y_orig]]], dtype=np.float32)
            point_rect = cv2.perspectiveTransform(point, self.H)

            center_x_rect = point_rect[0][0][0]
            center_y_rect = point_rect[0][0][1]

            # Проверка отступа от края
            if (center_x_rect < margin_pixels or
                    center_x_rect > self.output_size[0] - margin_pixels or
                    center_y_rect < margin_pixels or
                    center_y_rect > self.output_size[1] - margin_pixels):
                continue

            # Реальные координаты поля
            scale_x = self.field_width / self.output_size[0]
            scale_y = self.field_height / self.output_size[1]

            real_x = center_x_rect * scale_x
            real_y = center_y_rect * scale_y

            obstacles.append({
                'center_pixel': (center_x_rect, center_y_rect),
                'center_real': (real_x, real_y),
                'area': area,
                'contour': contour
            })

        return obstacles

    def draw_obstacles(self, frame: np.ndarray, obstacles: list, robot_center: tuple = None,
                       robot_radius: int = 50, edge_margin: int = 20) -> np.ndarray:
        """
        Нарисовать препятствия и жёлтую рамку (внутри которой ищем)
        """
        h, w = frame.shape[:2]

        # 1. Рисуем жёлтую рамку (граница области поиска препятствий)
        cv2.rectangle(frame, (edge_margin, edge_margin),
                      (w - edge_margin, h - edge_margin),
                      (0, 255, 255), 2)  # жёлтая рамка

        # 2. Рисуем зону вокруг робота (полупрозрачный серый круг)
        if robot_center is not None:
            cx, cy = int(robot_center[0]), int(robot_center[1])
            overlay = frame.copy()
            cv2.circle(overlay, (cx, cy), robot_radius, (100, 100, 100), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.circle(frame, (cx, cy), robot_radius, (200, 200, 200), 2)

        # 3. Рисуем препятствия
        for obs in obstacles:
            cx = int(obs['center_pixel'][0])
            cy = int(obs['center_pixel'][1])

            radius = int(np.sqrt(obs['area'] / np.pi))
            radius = max(radius, 10)

            cv2.circle(frame, (cx, cy), radius, (255, 0, 0), 2)
            cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (0, 255, 255), 2)
            cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (0, 255, 255), 2)

            cv2.putText(frame, f"({obs['center_real'][0]:.1f}, {obs['center_real'][1]:.1f})",
                        (cx - 40, cy - radius - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

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
        all_obstacles = []  # для сохранения траекторий препятствий

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            rectified = cv2.warpPerspective(frame, self.H, self.output_size)

            # Детекция робота
            found, robot_id, center_pixel, center_real, marker_corners = self.detect_robot(frame)

            # Детекция препятствий
            if found:
                obstacles = self.detect_obstacles(frame, robot_center=center_pixel)
            else:
                obstacles = self.detect_obstacles(frame, robot_center=None)

            # Отрисовка с рамкой
            rectified = self.draw_obstacles(rectified, obstacles,
                                            robot_center=center_pixel if found else None,
                                            robot_radius=self.robot_exclusion_radius,
                                            edge_margin=self.edge_margin)

            # Сохраняем данные о препятствиях (опционально)
            if obstacles and frame_count % 30 == 0:
                all_obstacles.append({
                    'frame': frame_count,
                    'obstacles': [
                        {'x': obs['center_real'][0], 'y': obs['center_real'][1],
                         'area': obs['area']} for obs in obstacles
                    ]
                })

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
                          f"позиция: ({center_real[0]:.2f}, {center_real[1]:.2f}), "
                          f"препятствий: {len(obstacles)}")
            else:
                if frame_count % 100 == 0:
                    print(f"  Кадр {frame_count}: Робот не найден, препятствий: {len(obstacles)}")

            out.write(rectified)
            processed += 1

            if processed % 100 == 0:
                print(f"  Прогресс: {processed}/{total_frames} ({processed * 100 / total_frames:.1f}%)")

        cap.release()
        out.release()

        print(f"\n✅ Обработка завершена!")
        print(f"   Обработано: {processed} кадров")
        print(f"   Обнаружений робота: {len(self.robot_trajectory)}")
        print(f"   Результат: {self.output_path}")

        return {
            'processed_frames': processed,
            'total_frames': total_frames,
            'fps': fps,
            'robot_detections': len(self.robot_trajectory),
            'obstacles_data': all_obstacles
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

    def save_obstacles_data(self, output_file: str = "obstacles_data.json"):
        """Сохранить данные о препятствиях в JSON файл"""
        if not hasattr(self, 'obstacles_data') or not self.obstacles_data:
            print("✗ Нет данных о препятствиях для сохранения")
            return

        with open(output_file, 'w') as f:
            json.dump(self.obstacles_data, f, indent=2)
        print(f"✓ Данные о препятствиях сохранены в {output_file}")

