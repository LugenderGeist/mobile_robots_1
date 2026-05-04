import cv2
import numpy as np
from typing import List, Tuple
import math
from collections import deque

class PathPlanner:
    def __init__(self, field_width: float, field_height: float, step: float = 2.0,
                 robot_radius: float = 15.0, obstacle_safety: float = 5.0,
                 edge_limit_cm: float = 15.0):
        self.field_width = field_width
        self.field_height = field_height
        self.step = step
        self.robot_radius = robot_radius
        self.obstacle_safety = obstacle_safety
        self.edge_limit_cm = edge_limit_cm  # ← добавить
        self.grid_width = int(field_width / step) + 1
        self.grid_height = int(field_height / step) + 1
        self.obstacles = []
        self.obstacle_map = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        self.path = []

    def update_obstacles(self, obstacles: List[dict]):
        self.obstacles = obstacles

        # Строим карту для визуализации
        self.obstacle_map.fill(0)

        for obs in obstacles:
            center_x, center_y = obs['center_real']
            radius = obs['radius_cm']

            cell_x = int(center_x / self.step)
            cell_y = int(center_y / self.step)
            cell_radius = int(radius / self.step) + 1

            for dx in range(-cell_radius, cell_radius + 1):
                for dy in range(-cell_radius, cell_radius + 1):
                    grid_x = cell_x + dx
                    grid_y = cell_y + dy
                    if (0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height):
                        if dx * dx + dy * dy <= cell_radius * cell_radius:
                            self.obstacle_map[grid_y, grid_x] = 1

    def update_planning_map(self, obstacles: List[dict]):

        self.obstacles = obstacles

        self.obstacle_map.fill(0)

        for obs in obstacles:
            center_x, center_y = obs['center_real']

            # Радиус для планирования = физический радиус препятствия + радиус робота + доп. запас
            total_radius = obs['radius_cm'] + self.robot_radius + self.obstacle_safety

            cell_x = int(center_x / self.step)
            cell_y = int(center_y / self.step)
            cell_radius = int(total_radius / self.step) + 1

            for dx in range(-cell_radius, cell_radius + 1):
                for dy in range(-cell_radius, cell_radius + 1):
                    grid_x = cell_x + dx
                    grid_y = cell_y + dy
                    if (0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height):
                        if dx * dx + dy * dy <= cell_radius * cell_radius:
                            self.obstacle_map[grid_y, grid_x] = 1

    def is_cell_safe(self, grid_x: int, grid_y: int) -> bool:

        # Центр клетки в реальных координатах
        cx = (grid_x + 0.5) * self.step
        cy = (grid_y + 0.5) * self.step

        # Получаем расстояние от края (то же, что и в draw_edge_limit)
        edge_limit_cm = getattr(self, 'edge_limit_cm', 15.0)

        # Проверка границ поля — центр робота не может быть ближе, чем edge_limit_cm
        if (cx < edge_limit_cm or
                cx > self.field_width - edge_limit_cm or
                cy < edge_limit_cm or
                cy > self.field_height - edge_limit_cm):
            return False

        # Проверка расстояния до препятствий
        for obs in self.obstacles:
            obs_x, obs_y = obs['center_real']
            obs_radius = obs['radius_cm']
            dist = math.hypot(cx - obs_x, cy - obs_y)
            if dist < obs_radius + self.robot_radius:
                return False

        return True

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        grid_x = int(x / self.step)
        grid_y = int(y / self.step)
        grid_x = max(0, min(grid_x, self.grid_width - 1))
        grid_y = max(0, min(grid_y, self.grid_height - 1))
        return grid_x, grid_y

    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        return (grid_x + 0.5) * self.step, (grid_y + 0.5) * self.step

    def find_path(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:

        # Преобразуем в клетки сетки
        start_grid = self.world_to_grid(start[0], start[1])
        goal_grid = self.world_to_grid(goal[0], goal[1])

        print(f"    Сетка: {self.grid_width} x {self.grid_height}, шаг {self.step} см")
        print(f"    Старт: клетка ({start_grid[0]}, {start_grid[1]})")
        print(f"    Цель: клетка ({goal_grid[0]}, {goal_grid[1]})")

        # Проверка стартовой позиции
        if not self.is_cell_safe(start_grid[0], start_grid[1]):
            print(" Стартовая позиция небезопасна (препятствие или край)")
            return []

        # Проверка целевой позиции
        if not self.is_cell_safe(goal_grid[0], goal_grid[1]):
            print(" Целевая позиция небезопасна (препятствие или край)")
            return []

        # BFS
        moves = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1), (0, 1),
                 (1, -1), (1, 0), (1, 1)]

        visited = set()
        parent = {}
        queue = deque([(start_grid[0], start_grid[1])])
        visited.add((start_grid[0], start_grid[1]))

        iterations = 0
        max_iterations = self.grid_width * self.grid_height

        while queue and iterations < max_iterations:
            iterations += 1
            x, y = queue.popleft()

            if (x, y) == (goal_grid[0], goal_grid[1]):
                # Восстанавливаем путь
                path = []
                curr = (x, y)
                while curr in parent:
                    path.append(self.grid_to_world(curr[0], curr[1]))
                    curr = parent[curr]
                path.append(self.grid_to_world(start_grid[0], start_grid[1]))
                path.reverse()

                # Упрощаем путь
                self.path = self.simplify_path(path)
                total_length = self.calculate_path_length(self.path)
                return self.path

            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.grid_width and 0 <= ny < self.grid_height and
                        (nx, ny) not in visited and self.is_cell_safe(nx, ny)):
                    visited.add((nx, ny))
                    parent[(nx, ny)] = (x, y)
                    queue.append((nx, ny))
        return []

    def calculate_path_length(self, path: List[Tuple[float, float]]) -> float:
        if len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            total += math.hypot(dx, dy)
        return total

    def simplify_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(path) <= 2:
            return path

        simplified = [path[0]]
        for i in range(1, len(path) - 1):
            x1, y1 = simplified[-1]
            x2, y2 = path[i]
            x3, y3 = path[i + 1]

            area = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

            if area > 5.0:  # Если площадь > 5 см², точка нужна
                simplified.append(path[i])

        simplified.append(path[-1])
        return simplified

    def draw_planning_contours(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]

        # Рисуем контуры на основе obstacle_map
        obstacle_map_uint8 = (self.obstacle_map * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            obstacle_map_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if len(contour) < 4:
                continue

            pixel_contour = []
            for point in contour:
                gx, gy = point[0]
                x = int(gx * self.step / self.field_width * w)
                y = int(h - (gy * self.step / self.field_height * h))
                pixel_contour.append([x, y])

            if len(pixel_contour) > 2:
                pixel_contour = np.array(pixel_contour, dtype=np.int32)
                # Жёлтый контур (для планирования)
                cv2.polylines(frame, [pixel_contour], True, (0, 255, 255), 2)
        return frame

    def draw_path_on_frame(self, frame: np.ndarray, path: List[Tuple[float, float]],
                           color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        if not path or len(path) < 2:
            return frame

        h, w = frame.shape[:2]
        points = []
        for real_x, real_y in path:
            x_px = int(real_x / self.field_width * w)
            y_px = int(h - (real_y / self.field_height * h))
            points.append((x_px, y_px))

        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], color, 3)

        for point in points[1:-1]:
            cv2.circle(frame, point, 4, color, -1)
        return frame