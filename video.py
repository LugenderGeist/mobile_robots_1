import cv2
import cv2.aruco as aruco
import numpy as np
from pathlib import Path
import json

class FieldRectifier:
    def __init__(self, video_path: str, output_path: str = "output_video.mp4"):
        self.safety_mask = None
        self.robot_exclusion_radius = None
        self.obstacle_threshold_v = None
        self.obstacle_safety_margin = None
        self.obstacle_max_area = None
        self.obstacle_min_area = None
        self.edge_margin = None
        self.H_inv = None
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

    @staticmethod
    def set_corners_manually(frame: np.ndarray) -> np.ndarray:
        corners = []

        h, w = frame.shape[:2]
        scale = min(1000 / w, 700 / h, 1.0)

        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            display_frame = cv2.resize(frame, (new_w, new_h))
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
        self.H_inv, _ = cv2.findHomography(dst_corners, self.corners)
        return self.H

    def transform_coordinates(self, x_pixel: float, y_pixel: float) -> tuple:
        scale_x = self.field_width / self.output_size[0]
        scale_y = self.field_height / self.output_size[1]

        real_x = x_pixel * scale_x
        real_y = (self.output_size[1] - y_pixel) * scale_y

        return real_x, real_y

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

            center_x_rect = point_rect[0][0][0]
            center_y_rect = point_rect[0][0][1]

            real_x, real_y = self.transform_coordinates(center_x_rect, center_y_rect)

            return True, marker_id, (point_rect[0][0][0], point_rect[0][0][1]), (real_x, real_y), marker_corners

        return False, -1, (0, 0), (0, 0), None

    def draw_axes_2d(self, frame: np.ndarray, marker_corners: np.ndarray, marker_id: int, axis_length: float = 60):
        if len(marker_corners.shape) == 3:
            marker_corners = marker_corners[0]

        pts = marker_corners.astype(np.float32).reshape(-1, 1, 2)
        rectified_corners = cv2.perspectiveTransform(pts, self.H)
        rectified_corners = rectified_corners.reshape(-1, 2)

        center = np.mean(rectified_corners, axis=0)
        cx, cy = int(center[0]), int(center[1])

        dx = rectified_corners[1][0] - rectified_corners[0][0]
        dy = rectified_corners[1][1] - rectified_corners[0][1]
        length = np.sqrt(dx * dx + dy * dy)
        if length > 0:
            dx = dx / length * axis_length
            dy = dy / length * axis_length

        ux = rectified_corners[3][0] - rectified_corners[0][0]
        uy = rectified_corners[3][1] - rectified_corners[0][1]
        length = np.sqrt(ux * ux + uy * uy)
        if length > 0:
            ux = ux / length * axis_length
            uy = uy / length * axis_length

        x_end = (cx + int(dx), cy + int(dy))
        cv2.arrowedLine(frame, (cx, cy), x_end, (0, 0, 255), 2, tipLength=0.3)
        cv2.putText(frame, "X", (x_end[0] + 5, x_end[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        y_end = (cx - int(ux), cy - int(uy))
        cv2.arrowedLine(frame, (cx, cy), y_end, (0, 255, 0), 2, tipLength=0.3)
        cv2.putText(frame, "Y", (y_end[0] + 5, y_end[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.circle(frame, (cx, cy), 4, (0, 0, 0), -1)

        return frame

    def draw_coordinate_axes(self, frame: np.ndarray, margin: int = 20, axis_length: int = 50) -> np.ndarray:
        h, w = frame.shape[:2]

        origin_x = margin
        origin_y = h - margin

        x_end = (origin_x + axis_length, origin_y)
        cv2.arrowedLine(frame, (origin_x, origin_y), x_end, (0, 0, 255), 2, tipLength=0.2)
        cv2.putText(frame, "X", (x_end[0] + 5, x_end[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        y_end = (origin_x, origin_y - axis_length)
        cv2.arrowedLine(frame, (origin_x, origin_y), y_end, (0, 255, 0), 2, tipLength=0.2)
        cv2.putText(frame, "Y", (y_end[0] + 5, y_end[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.circle(frame, (origin_x, origin_y), 4, (0, 0, 0), -1)

        return frame

    def set_obstacle_params(self, edge_margin: int = 20,
                            robot_exclusion_radius: int = 60,
                            min_area: int = 500,
                            max_area: int = 5000,
                            safety_margin: int = 20,
                            threshold_v: int = 100):
        self.edge_margin = edge_margin
        self.robot_exclusion_radius = robot_exclusion_radius
        self.obstacle_min_area = min_area
        self.obstacle_max_area = max_area
        self.obstacle_safety_margin = safety_margin
        self.obstacle_threshold_v = threshold_v

    def detect_obstacles(self, rectified_frame: np.ndarray, robot_center: tuple = None) -> list:
        edge_margin = getattr(self, 'edge_margin', 20)
        robot_exclusion_radius = getattr(self, 'robot_exclusion_radius', 60)
        min_area = getattr(self, 'obstacle_min_area', 500)
        max_area = getattr(self, 'obstacle_max_area', 5000)
        safety_margin = getattr(self, 'obstacle_safety_margin', 20)
        threshold_v = getattr(self, 'obstacle_threshold_v', 100)

        # Переданный кадр уже выровнен
        gray = cv2.cvtColor(rectified_frame, cv2.COLOR_BGR2GRAY)

        # Пороговая обработка
        _, mask = cv2.threshold(gray, threshold_v, 255, cv2.THRESH_BINARY_INV)

        # Морфология
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Очищаем края
        h, w = mask.shape
        mask[0:edge_margin, :] = 0
        mask[h - edge_margin:h, :] = 0
        mask[:, 0:edge_margin] = 0
        mask[:, w - edge_margin:w] = 0

        # Вырезаем зону робота
        if robot_center is not None:
            cx = int(robot_center[0])
            cy = int(robot_center[1])
            cv2.circle(mask, (cx, cy), robot_exclusion_radius, 0, -1)
            cv2.circle(mask, (cx, cy), robot_exclusion_radius + 10, 0, -1)

        # Поиск контуров
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"  Найдено контуров: {len(contours)}")

        obstacles = []
        safety_mask = np.zeros_like(mask)

        for contour in contours:
            area = cv2.contourArea(contour)
            print(f"    Контур: area = {area} пикселей")

            # Фильтрация по площади
            if area < min_area or area > max_area:
                print(f"      → отброшен (min={min_area}, max={max_area})")
                continue

            # Центр препятствия
            center = cv2.moments(contour)
            if center["m00"] != 0:
                center_x_rect = center["m10"] / center["m00"]
                center_y_rect = center["m01"] / center["m00"]
            else:
                continue

            # Вычисляем радиус в пикселях
            radius = int(np.sqrt(area / np.pi))

            # Безопасная зона
            cv2.circle(safety_mask, (int(center_x_rect), int(center_y_rect)),
                       radius + safety_margin, 255, -1)

            # Реальные координаты (только для информации, не для фильтрации)
            scale_x = self.field_width / self.output_size[0]
            scale_y = self.field_height / self.output_size[1]

            real_x = center_x_rect * scale_x
            real_y = (self.output_size[1] - center_y_rect) * scale_y

            obstacles.append({
                'center_pixel': (center_x_rect, center_y_rect),
                'center_real': (real_x, real_y),
                'area': area,
                'radius': radius,
                'radius_with_safety': radius + safety_margin,
                'contour': contour
            })

        self.safety_mask = safety_mask
        return obstacles

    @staticmethod
    def draw_obstacles(frame: np.ndarray, obstacles: list, robot_center: tuple = None,
                       robot_radius: int = 50, edge_margin: int = 20) -> np.ndarray:
        h, w = frame.shape[:2]

        cv2.rectangle(frame, (edge_margin, edge_margin),
                      (w - edge_margin, h - edge_margin),
                      (0, 0, 0), 2)

        if robot_center is not None:
            cx, cy = int(robot_center[0]), int(robot_center[1])
            cv2.circle(frame, (cx, cy), robot_radius, (255, 0, 0), 2)

        for obs in obstacles:
            cx = int(obs['center_pixel'][0])
            cy = int(obs['center_pixel'][1])
            radius = obs['radius']
            safety_radius = obs['radius_with_safety']

            cv2.circle(frame, (cx, cy), radius, (255, 0, 0), 2)
            cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (255, 0, 0), 2)
            cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), safety_radius, (0, 255, 0), 2)

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

        cv2.namedWindow("Processing Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Processing Preview", 800, 800)

        # Переменные для хранения точки пользователя
        user_point = None  # (x, y) в выровненных пикселях
        user_point_real = None  # (x, y) в реальных координатах

        def mouse_callback(event, x, y, flags, param):
            nonlocal user_point, user_point_real
            if event == cv2.EVENT_LBUTTONDOWN:
                user_point = (x, y)
                real_x, real_y = self.transform_coordinates(x, y)
                user_point_real = (real_x, real_y)
                print(f"📍 Точка: экран ({x}, {y}) → поле ({real_x:.1f}, {real_y:.1f})")

        cv2.setMouseCallback("Processing Preview", mouse_callback)

        print(f"\n🔄 Обработка... (окно предпросмотра открыто)")
        print("   Управление:")
        print("   • Левый клик мыши - поставить точку")
        print("   • 'q' - досрочный выход")
        print("   • Пробел - пауза/продолжение\n")

        frame_count = 0
        processed = 0
        self.robot_trajectory = []
        paused = False

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

            rectified = cv2.warpPerspective(frame, self.H, self.output_size)
            rectified = self.draw_coordinate_axes(rectified, margin=20, axis_length=60)

            found, robot_id, center_pixel, center_real, marker_corners = self.detect_robot(frame)

            if found:
                robot_real_x, robot_real_y = self.transform_coordinates(center_pixel[0], center_pixel[1])
                self.robot_trajectory.append({
                    'frame': frame_count,
                    'x_pixel': center_pixel[0],
                    'y_pixel': center_pixel[1],
                    'x_real': robot_real_x,
                    'y_real': robot_real_y
                })

            if found:
                obstacles = self.detect_obstacles(rectified, robot_center=center_pixel)
            else:
                obstacles = self.detect_obstacles(rectified, robot_center=None)

            # Отрисовка препятствий
            for obs in obstacles:
                cx = int(obs['center_pixel'][0])
                cy = int(obs['center_pixel'][1])
                radius = obs['radius']
                safety_radius = obs['radius_with_safety']

                cv2.circle(rectified, (cx, cy), radius, (255, 0, 0), 2)
                cv2.line(rectified, (cx - 10, cy), (cx + 10, cy), (255, 0, 0), 2)
                cv2.line(rectified, (cx, cy - 10), (cx, cy + 10), (255, 0, 0), 2)
                cv2.circle(rectified, (cx, cy), safety_radius, (0, 255, 0), 2)

            # Отрисовка робота
            if found:
                rectified = self.draw_axes_2d(rectified, marker_corners, robot_id, axis_length=50)

            # Рисуем пользовательскую точку (фиолетовый круг)
            if user_point is not None:
                cv2.circle(rectified, user_point, 8, (255, 0, 255), -1)
                cv2.circle(rectified, user_point, 12, (255, 0, 255), 2)

            # ИНФОРМАЦИЯ В ЛЕВОМ ВЕРХНЕМ УГЛУ
            info_y = 25
            line_height = 25

            # Строка 1: количество препятствий
            cv2.putText(rectified, f"Obstacles: {len(obstacles)}", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            info_y += line_height

            # Строка 2: координаты робота (если найден)
            if found:
                cv2.putText(rectified, f"Robot: ({robot_real_x:.1f}, {robot_real_y:.1f})",
                            (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                info_y += line_height

            # Строка 3: координаты целевой точки (если поставлена)
            if user_point_real is not None:
                cv2.putText(rectified, f"Target: ({user_point_real[0]:.1f}, {user_point_real[1]:.1f})",
                            (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            cv2.imshow("Processing Preview", rectified)
            out.write(rectified)
            processed += 1

            key = cv2.waitKey(1 if not paused else 0) & 0xFF

            if key == ord('q'):
                print(f"\n⏹️ Обработка прервана на кадре {frame_count}")
                break
            elif key == ord(' '):
                paused = not paused
                print("⏸️ Пауза" if paused else "▶️ Продолжение")

            if processed % 100 == 0:
                print(f"  Прогресс: {processed}/{total_frames} ({processed * 100 / total_frames:.1f}%)")

            if found and frame_count % 50 == 0:
                print(f"  Кадр {frame_count}: Робот ID={robot_id}, "
                      f"позиция: ({robot_real_x:.1f}, {robot_real_y:.1f}), "
                      f"препятствий: {len(obstacles)}")

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        return {
            'processed_frames': processed,
            'total_frames': total_frames,
            'fps': fps,
            'robot_detections': len(self.robot_trajectory)
        }

    def save_trajectory(self, output_file: str = "robot_trajectory.json"):
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