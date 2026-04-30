import cv2
import numpy as np
from typing import List, Tuple


class PathPlanner:
    def __init__(self, field_width: float, field_height: float, step: float = 5.0):
        """
        Args:
            field_width: ширина поля в САНТИМЕТРАХ
            field_height: высота поля в САНТИМЕТРАХ
            step: шаг дискретизации в САНТИМЕТРАХ
        """
        self.field_width = field_width
        self.field_height = field_height
        self.step = step

        self.grid_width = int(field_width / step) + 1
        self.grid_height = int(field_height / step) + 1
        self.obstacle_map = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        self.path = []

    def update_obstacles(self, obstacles: List[dict]):
        """Обновить карту препятствий"""
        self.obstacle_map.fill(0)

        for obs in obstacles:
            center_x, center_y = obs['center_real']
            radius = obs['radius_with_safety']

            cell_x = int(center_x / self.step)
            cell_y = int(center_y / self.step)
            cell_radius = int(radius / self.step) + 1

            for dx in range(-cell_radius, cell_radius + 1):
                for dy in range(-cell_radius, cell_radius + 1):
                    grid_x = cell_x + dx
                    grid_y = cell_y + dy

                    if (0 <= grid_x < self.grid_width and
                            0 <= grid_y < self.grid_height):
                        dist = np.sqrt(dx ** 2 + dy ** 2)
                        if dist <= cell_radius:
                            self.obstacle_map[grid_y, grid_x] = 1

        # Границы поля
        self.obstacle_map[0, :] = 1
        self.obstacle_map[self.grid_height - 1, :] = 1
        self.obstacle_map[:, 0] = 1
        self.obstacle_map[:, self.grid_width - 1] = 1

    def find_path(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Поиск пути с обходом препятствий (BFS для кратчайшего пути)
        Это даст прямой путь без "липнутия" к стенам
        """
        start_x = int(start[0] / self.step)
        start_y = int(start[1] / self.step)
        goal_x = int(goal[0] / self.step)
        goal_y = int(goal[1] / self.step)

        # Проверка границ
        if not (0 <= start_x < self.grid_width and 0 <= start_y < self.grid_height):
            print(f"  ⚠️ Старт вне границ: ({start_x}, {start_y})")
            return []

        if not (0 <= goal_x < self.grid_width and 0 <= goal_y < self.grid_height):
            print(f"  ⚠️ Цель вне границ: ({goal_x}, {goal_y})")
            return []

        # BFS
        moves = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # вверх, вправо, вниз, влево
        visited = set()
        parent = {}
        queue = [(start_x, start_y)]
        visited.add((start_x, start_y))

        print(f"  Поиск пути от ({start[0]:.1f}, {start[1]:.1f}) до ({goal[0]:.1f}, {goal[1]:.1f})")

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

                # Упрощаем путь (убираем лишние точки на прямой)
                simplified_path = self.simplify_path(path)
                self.path = simplified_path

                distance = self.calculate_path_length(simplified_path)
                print(f"  ✅ Путь найден! Длина: {distance:.1f} см, точек: {len(simplified_path)}")
                return simplified_path

            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.grid_width and 0 <= ny < self.grid_height and
                        self.obstacle_map[ny, nx] == 0 and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    parent[(nx, ny)] = (x, y)
                    queue.append((nx, ny))

        print("  ❌ Путь не найден!")
        return []

    def simplify_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Упростить путь: удалить промежуточные точки на прямой линии
        """
        if len(path) <= 2:
            return path

        simplified = [path[0]]

        for i in range(1, len(path) - 1):
            # Проверяем, лежит ли точка на прямой между соседними
            x1, y1 = simplified[-1]
            x2, y2 = path[i]
            x3, y3 = path[i + 1]

            # Векторы
            v1 = (x2 - x1, y2 - y1)
            v2 = (x3 - x2, y3 - y2)

            # Нормализуем для сравнения направлений
            len1 = np.sqrt(v1[0] ** 2 + v1[1] ** 2)
            len2 = np.sqrt(v2[0] ** 2 + v2[1] ** 2)

            if len1 > 0 and len2 > 0:
                dir1 = (v1[0] / len1, v1[1] / len1)
                dir2 = (v2[0] / len2, v2[1] / len2)

                # Если направление изменилось, добавляем точку
                if abs(dir1[0] - dir2[0]) > 0.1 or abs(dir1[1] - dir2[1]) > 0.1:
                    simplified.append(path[i])
            else:
                simplified.append(path[i])

        simplified.append(path[-1])
        return simplified

    def calculate_path_length(self, path: List[Tuple[float, float]]) -> float:
        """Вычислить длину пути"""
        if len(path) < 2:
            return 0.0

        total = 0.0
        for i in range(len(path) - 1):
            dx = path[i + 1][0] - path[i][0]
            dy = path[i + 1][1] - path[i][1]
            total += np.sqrt(dx ** 2 + dy ** 2)
        return total

    def find_path_with_obstacle_avoidance(self, start: Tuple[float, float],
                                          goal: Tuple[float, float],
                                          obstacles: List[dict]) -> List[Tuple[float, float]]:
        """
        Прямой путь с обходом препятствий (без сетки, для голономного робота)
        Это упрощённая версия - просто идём к цели, при необходимости огибаем препятствия
        """
        # Обновляем карту препятствий
        self.update_obstacles(obstacles)

        # Используем BFS для гарантированного пути
        return self.find_path(start, goal)

    def draw_path_on_frame(self, frame: np.ndarray,
                           path: List[Tuple[float, float]],
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

        # Рисуем точки поворота
        for point in points[1:-1]:
            cv2.circle(frame, point, 5, color, -1)

        return frame