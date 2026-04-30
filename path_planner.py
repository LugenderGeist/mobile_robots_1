import cv2
import numpy as np
from typing import List, Tuple
import math
from collections import deque


class PathPlanner:
    def __init__(self, field_width: float, field_height: float, step: float = 1.0,
                 safety_margin: float = 10.0, edge_margin: float = 30.0):
        """
        Args:
            field_width: ширина поля в см
            field_height: высота поля в см
            step: шаг дискретизации в см
            safety_margin: безопасный отступ от препятствий в см
            edge_margin: отступ от края поля в см
        """
        self.field_width = field_width
        self.field_height = field_height
        self.step = step
        self.safety_margin = safety_margin
        self.edge_margin = edge_margin
        self.grid_width = int(field_width / step) + 1
        self.grid_height = int(field_height / step) + 1
        self.obstacle_map = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        self.path = []

    def update_obstacles(self, obstacles: List[dict]):
        """Обновить карту препятствий"""
        self.obstacle_map.fill(0)

        # Сохраняем центры препятствий для отладки
        self.obstacle_centers = []

        for obs in obstacles:
            center_x, center_y = obs['center_real']
            radius = obs['radius_cm'] + self.safety_margin
            self.obstacle_centers.append((center_x, center_y))

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

        # Отступ от края
        edge_cells = int(self.edge_margin / self.step)
        if edge_cells > 0:
            self.obstacle_map[0:edge_cells, :] = 1
            self.obstacle_map[self.grid_height - edge_cells:self.grid_height, :] = 1
            self.obstacle_map[:, 0:edge_cells] = 1
            self.obstacle_map[:, self.grid_width - edge_cells:self.grid_width] = 1

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
        BFS поиск кратчайшего пути
        """
        print(f"\n  🔍 BFS поиск пути: {start} → {goal}")

        # Преобразуем в клетки сетки
        start_grid = self.world_to_grid(start[0], start[1])
        goal_grid = self.world_to_grid(goal[0], goal[1])

        print(f"    Сетка: {self.grid_width} x {self.grid_height}, шаг {self.step} см")
        print(f"    Старт: клетка ({start_grid[0]}, {start_grid[1]})")
        print(f"    Цель: клетка ({goal_grid[0]}, {goal_grid[1]})")

        # Проверка, что старт и цель не в препятствиях
        if self.obstacle_map[start_grid[1], start_grid[0]] == 1:
            print("    ❌ Старт внутри препятствия!")
            return []

        if self.obstacle_map[goal_grid[1], goal_grid[0]] == 1:
            print("    ❌ Цель внутри препятствия!")
            return []

        # BFS
        # 8 направлений движения (включая диагонали)
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
                        self.obstacle_map[ny, nx] == 0 and (nx, ny) not in visited):
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

            # Проверяем, лежат ли три точки на одной прямой
            # Площадь треугольника должна быть близка к 0
            area = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

            if area > 5.0:  # Если площадь > 5 см², точка нужна
                simplified.append(path[i])

        simplified.append(path[-1])
        return simplified

    def visualize_obstacles_contours(self, frame: np.ndarray, obstacles: List[dict] = None) -> np.ndarray:
        """Нарисовать контуры препятствий и их реальные центры"""
        h, w = frame.shape[:2]

        # 1. Рисуем серую зону по краям
        edge_pixels = int(self.edge_margin / self.field_width * w)
        if edge_pixels > 0:
            cv2.rectangle(frame, (0, 0), (w, edge_pixels), (128, 128, 128), -1)
            cv2.rectangle(frame, (0, h - edge_pixels), (w, h), (128, 128, 128), -1)
            cv2.rectangle(frame, (0, 0), (edge_pixels, h), (128, 128, 128), -1)
            cv2.rectangle(frame, (w - edge_pixels, 0), (w, h), (128, 128, 128), -1)

        # 2. Убираем края из карты препятствий
        obstacles_only = self.obstacle_map.copy()
        edge_cells = int(self.edge_margin / self.step)
        if edge_cells > 0:
            obstacles_only[0:edge_cells, :] = 0
            obstacles_only[self.grid_height - edge_cells:self.grid_height, :] = 0
            obstacles_only[:, 0:edge_cells] = 0
            obstacles_only[:, self.grid_width - edge_cells:self.grid_width] = 0

        # 3. Рисуем красные контуры
        obstacle_map_uint8 = (obstacles_only * 255).astype(np.uint8)
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

        # 4. Рисуем РЕАЛЬНЫЕ центры препятствий (зелёные крестики)
        if obstacles:
            for obs in obstacles:
                real_x, real_y = obs['center_real']
                x = int(real_x / self.field_width * w)
                y = int(h - (real_y / self.field_height * h))
                # Зелёный кружок с крестиком
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

        # Рисуем линию
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], color, 3)

        # Рисуем точки поворота
        for point in points[1:-1]:
            cv2.circle(frame, point, 4, color, -1)

        return frame