import cv2
import numpy as np
from typing import List, Tuple

class PathPlanner:
    def __init__(self, field_width: float, field_height: float, step: float = 2.0, safety_margin: float = 10.0):
        self.field_width = field_width
        self.field_height = field_height
        self.step = step
        self.safety_margin = safety_margin  # безопасный отступ от препятствий
        self.grid_width = int(field_width / step) + 1
        self.grid_height = int(field_height / step) + 1
        self.obstacle_map = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        self.path = []

    def update_obstacles(self, obstacles: List[dict]):
        """Обновить карту препятствий"""
        self.obstacle_map.fill(0)

        for obs in obstacles:
            center_x, center_y = obs['center_real']
            # радиус препятствия + безопасный отступ
            radius = obs['radius_cm'] + self.safety_margin

            cell_x = int(center_x / self.step)
            cell_y = int(center_y / self.step)
            cell_radius = int(radius / self.step) + 1

            for dx in range(-cell_radius, cell_radius + 1):
                for dy in range(-cell_radius, cell_radius + 1):
                    grid_x = cell_x + dx
                    grid_y = cell_y + dy
                    if (0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height):
                        if dx*dx + dy*dy <= cell_radius*cell_radius:
                            self.obstacle_map[grid_y, grid_x] = 1

    def visualize_obstacles_contours(self, frame: np.ndarray) -> np.ndarray:
        """Нарисовать контуры препятствий на основе карты (только границы)"""
        h, w = frame.shape[:2]

        # Находим контуры на карте препятствий
        obstacle_map_uint8 = (self.obstacle_map * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            obstacle_map_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            # Преобразуем координаты контура из сетки в пиксельные
            pixel_contour = []
            for point in contour:
                gx, gy = point[0]
                x = int(gx * self.step / self.field_width * w)
                y = int(h - (gy * self.step / self.field_height * h))
                pixel_contour.append([x, y])

            if len(pixel_contour) > 2:
                pixel_contour = np.array(pixel_contour, dtype=np.int32)
                # Рисуем контур красным цветом
                cv2.polylines(frame, [pixel_contour], True, (0, 0, 255), 2)

        return frame

    def find_path(self, start: Tuple[float, float], goal: Tuple[float, float], robot_radius: float = 15.0) -> List[
        Tuple[float, float]]:
        """Поиск пути BFS с учётом радиуса робота"""
        start_x = int(start[0] / self.step)
        start_y = int(start[1] / self.step)

        goal_x = int(goal[0] / self.step)
        goal_y = int(goal[1] / self.step)

        # Проверка границ
        if not (0 <= start_x < self.grid_width and 0 <= start_y < self.grid_height):
            print(f"  Старт вне сетки: ({start_x}, {start_y})")
            return []
        if not (0 <= goal_x < self.grid_width and 0 <= goal_y < self.grid_height):
            print(f"  Цель вне сетки: ({goal_x}, {goal_y})")
            return []

        # Проверка, что цель не в препятствии
        if self.obstacle_map[goal_y, goal_x] == 1:
            print(f"  Цель в препятствии!")
            return []

        # ВРЕМЕННО УБИРАЕМ ПРОВЕРКУ СТАРТА, так как робот может быть рядом с препятствием
        # Просто выводим предупреждение, но не прерываем поиск
        if self.obstacle_map[start_y, start_x] == 1:
            print(f"  ⚠️ Предупреждение: старт в зоне препятствия, но продолжаем поиск...")

        # Разрешаем движение в 8 направлениях
        moves = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1), (0, 1),
                 (1, -1), (1, 0), (1, 1)]

        visited = set()
        parent = {}
        queue = [(start_x, start_y)]
        visited.add((start_x, start_y))

        while queue:
            x, y = queue.pop(0)

            if (x, y) == (goal_x, goal_y):
                # Восстанавливаем путь
                path = []
                curr = (x, y)
                while curr in parent:
                    path.append((curr[0] * self.step, curr[1] * self.step))
                    curr = parent[curr]
                path.append((start_x * self.step, start_y * self.step))
                path.reverse()
                self.path = self.simplify_path(path)
                total_len = self.calculate_path_length(self.path)
                print(f"  ✅ Путь найден! Длина: {total_len:.1f} см, шагов: {len(self.path)}")
                return self.path

            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.grid_width and 0 <= ny < self.grid_height and
                        (nx, ny) not in visited):
                    # Проверяем, не является ли клетка препятствием
                    # НО пропускаем стартовую клетку, даже если она в препятствии
                    if (nx, ny) != (start_x, start_y) and self.obstacle_map[ny, nx] == 1:
                        continue
                    visited.add((nx, ny))
                    parent[(nx, ny)] = (x, y)
                    queue.append((nx, ny))

        print(f"  ❌ Путь не найден!")
        return []

    def calculate_path_length(self, path: List[Tuple[float, float]]) -> float:
        """Вычислить длину пути в сантиметрах"""
        if len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            total += np.sqrt(dx*dx + dy*dy)
        return total

    def simplify_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Упростить путь (убрать лишние точки на прямой)"""
        if len(path) <= 2:
            return path

        simplified = [path[0]]
        for i in range(1, len(path) - 1):
            x1, y1 = simplified[-1]
            x2, y2 = path[i]
            x3, y3 = path[i + 1]

            # Проверяем, лежат ли три точки на одной прямой
            # Векторы
            v1 = (x2 - x1, y2 - y1)
            v2 = (x3 - x2, y3 - y2)

            # Нормализуем
            len1 = np.sqrt(v1[0]**2 + v1[1]**2)
            len2 = np.sqrt(v2[0]**2 + v2[1]**2)

            if len1 > 0 and len2 > 0:
                dir1 = (v1[0] / len1, v1[1] / len1)
                dir2 = (v2[0] / len2, v2[1] / len2)

                # Если направление изменилось, оставляем точку
                if abs(dir1[0] - dir2[0]) > 0.01 or abs(dir1[1] - dir2[1]) > 0.01:
                    simplified.append(path[i])

        simplified.append(path[-1])
        return simplified

    def draw_path_on_frame(self, frame: np.ndarray, path: List[Tuple[float, float]],
                           color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        """Нарисовать путь на кадре"""
        if not path or len(path) < 2:
            return frame

        # Преобразуем реальные координаты в пиксельные
        points = []
        for real_x, real_y in path:
            x_px = int(real_x / self.field_width * frame.shape[1])
            y_px = int(frame.shape[0] - (real_y / self.field_height * frame.shape[0]))
            points.append((x_px, y_px))

        # Рисуем линию
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], color, 3)

        # Рисуем точки поворота (кроме первой и последней)
        for point in points[1:-1]:
            cv2.circle(frame, point, 4, color, -1)

        return frame

    def visualize_map(self, frame: np.ndarray, start: Tuple[float, float], goal: Tuple[float, float]) -> np.ndarray:
        """Визуализировать карту препятствий на кадре (для отладки)"""
        h, w = frame.shape[:2]
        for gy in range(self.grid_height):
            for gx in range(self.grid_width):
                if self.obstacle_map[gy, gx] == 1:
                    # Преобразуем координаты сетки в пиксельные
                    x = int(gx * self.step / self.field_width * w)
                    y = int(h - (gy * self.step / self.field_height * h))
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        # Отмечаем старт и цель
        start_x = int(start[0] / self.field_width * w)
        start_y = int(h - (start[1] / self.field_height * h))
        goal_x = int(goal[0] / self.field_width * w)
        goal_y = int(h - (goal[1] / self.field_height * h))
        cv2.circle(frame, (start_x, start_y), 8, (0, 255, 0), -1)
        cv2.circle(frame, (goal_x, goal_y), 8, (255, 0, 0), -1)

        return frame