import cv2
import numpy as np
from typing import List, Tuple
import math
from collections import deque


class PathPlanner:
    def __init__(self, field_width: float, field_height: float, step: float = 2.0,
                 robot_radius: float = 15.0, edge_margin: float = 30.0):
        """
        Args:
            field_width: ширина поля в см
            field_height: высота поля в см
            step: шаг дискретизации в см
            robot_radius: радиус робота в см
            edge_margin: отступ от края поля в см
        """
        self.field_width = field_width
        self.field_height = field_height
        self.step = step
        self.robot_radius = robot_radius
        self.edge_margin = edge_margin
        self.grid_width = int(field_width / step) + 1
        self.grid_height = int(field_height / step) + 1
        self.obstacles = []  # список препятствий (точные)
        self.obstacle_map = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        self.path = []

    def update_obstacles(self, obstacles: List[dict]):
        """Обновить список препятствий (все препятствия, без обрезки по краям)"""
        self.obstacles = obstacles

        # Строим карту для визуализации (все препятствия)
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

    def is_cell_safe(self, grid_x: int, grid_y: int) -> bool:
        """
        Проверить, может ли робот находиться в этой клетке
        """
        # Центр клетки в реальных координатах
        cx = (grid_x + 0.5) * self.step
        cy = (grid_y + 0.5) * self.step

        # 1. Проверка границ поля (с учётом радиуса робота)
        if (cx < self.robot_radius or
                cx > self.field_width - self.robot_radius or
                cy < self.robot_radius or
                cy > self.field_height - self.robot_radius):
            return False

        # 2. Проверка расстояния до препятствий
        for obs in self.obstacles:
            obs_x, obs_y = obs['center_real']
            obs_radius = obs['radius_cm']
            dist = math.hypot(cx - obs_x, cy - obs_y)
            if dist < obs_radius + self.robot_radius:
                return False

        return True

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Перевести реальные координаты в клетки сетки"""
        grid_x = int(x / self.step)
        grid_y = int(y / self.step)
        grid_x = max(0, min(grid_x, self.grid_width - 1))
        grid_y = max(0, min(grid_y, self.grid_height - 1))
        return grid_x, grid_y

    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Перевести клетки сетки в реальные координаты (центр клетки)"""
        return (grid_x + 0.5) * self.step, (grid_y + 0.5) * self.step

    def find_path(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        BFS поиск пути с учётом радиуса робота
        """
        print(f"\n  🔍 BFS поиск пути (радиус робота={self.robot_radius} см): {start} → {goal}")

        # Преобразуем в клетки сетки
        start_grid = self.world_to_grid(start[0], start[1])
        goal_grid = self.world_to_grid(goal[0], goal[1])

        print(f"    Сетка: {self.grid_width} x {self.grid_height}, шаг {self.step} см")
        print(f"    Старт: клетка ({start_grid[0]}, {start_grid[1]})")
        print(f"    Цель: клетка ({goal_grid[0]}, {goal_grid[1]})")

        # Проверка стартовой позиции
        if not self.is_cell_safe(start_grid[0], start_grid[1]):
            print("    ❌ Стартовая позиция небезопасна (препятствие или край)")
            return []

        # Проверка целевой позиции
        if not self.is_cell_safe(goal_grid[0], goal_grid[1]):
            print("    ❌ Целевая позиция небезопасна (препятствие или край)")
            return []

        # BFS с 8 направлениями
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
                print(f"    ✅ Путь найден! Длина: {total_length:.1f} см, шагов: {len(self.path)}")
                print(f"    Посещено клеток: {len(visited)} из {self.grid_width * self.grid_height}")
                return self.path

            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.grid_width and 0 <= ny < self.grid_height and
                        (nx, ny) not in visited and self.is_cell_safe(nx, ny)):
                    visited.add((nx, ny))
                    parent[(nx, ny)] = (x, y)
                    queue.append((nx, ny))

        print(f"    ❌ Путь не найден! Посещено клеток: {len(visited)}")
        return []

    def calculate_path_length(self, path: List[Tuple[float, float]]) -> float:
        """Вычислить длину пути в сантиметрах"""
        if len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            total += math.hypot(dx, dy)
        return total

    def simplify_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Упростить путь, убирая лишние точки на прямой"""
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

    def visualize_obstacles_contours(self, frame: np.ndarray) -> np.ndarray:
        """Нарисовать контуры всех препятствий (красным) и их центры (зелёным)"""
        h, w = frame.shape[:2]

        # Рисуем красные контуры ВСЕХ препятствий
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
                cv2.polylines(frame, [pixel_contour], True, (0, 0, 255), 2)

        # Рисуем центры препятствий (зелёные крестики)
        for obs in self.obstacles:
            real_x, real_y = obs['center_real']
            x = int(real_x / self.field_width * w)
            y = int(h - (real_y / self.field_height * h))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.line(frame, (x - 10, y), (x + 10, y), (0, 255, 0), 2)
            cv2.line(frame, (x, y - 10), (x, y + 10), (0, 255, 0), 2)

        return frame

    def draw_path_on_frame(self, frame: np.ndarray, path: List[Tuple[float, float]],
                           color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        """Нарисовать путь на кадре"""
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