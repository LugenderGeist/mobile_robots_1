import cv2
import numpy as np
from typing import List, Tuple
import math


def create_planner(field_width: float, field_height: float, step: float,
                   robot_radius: float, obstacle_safety: float,
                   edge_limit_cm: float) -> dict:
    grid_width = int(field_width / step) + 1
    grid_height = int(field_height / step) + 1

    planner = {
        'field_width': field_width,
        'field_height': field_height,
        'step': step,
        'robot_radius': robot_radius,
        'obstacle_safety': obstacle_safety,
        'edge_limit_cm': edge_limit_cm,
        'grid_width': grid_width,
        'grid_height': grid_height,
        'obstacles': [],
        'obstacle_map': np.zeros((grid_height, grid_width), dtype=np.uint8),
        'path': [],
        'path_locked': False,  # Флаг, что путь уже построен и не требует обновления
        'current_goal': None  # Текущая цель, для которой построен путь
    }
    return planner


def update_obstacles(planner: dict, obstacles: List[dict]):
    planner['obstacles'] = obstacles
    # Если путь заблокирован, не обновляем obstacle_map
    if planner.get('path_locked', False):
        return
    planner['obstacle_map'].fill(0)

    step = planner['step']
    grid_width = planner['grid_width']
    grid_height = planner['grid_height']
    field_width = planner['field_width']
    field_height = planner['field_height']

    rectified_width = 720
    rectified_height = 720

    for obs in obstacles:
        if 'expanded_contour' in obs:
            expanded_contour = obs['expanded_contour']

            # Преобразуем пиксельные координаты rectified в реальные см
            contour_real = []
            for point in expanded_contour:
                if len(point) == 2:
                    px, py = point
                else:
                    px, py = point[0]

                real_x = px * (field_width / rectified_width)
                real_y = (rectified_height - py) * (field_height / rectified_height)
                contour_real.append([real_x, real_y])

            if len(contour_real) >= 3:
                grid_points = []
                for point in contour_real:
                    gx = int(point[0] / step)
                    gy = int(point[1] / step)
                    if 0 <= gx < grid_width and 0 <= gy < grid_height:
                        grid_points.append([gx, gy])

                if len(grid_points) >= 3:
                    grid_points = np.array(grid_points, dtype=np.int32)

                    # Создаем временную маску
                    temp_mask = np.zeros_like(planner['obstacle_map'])
                    # Закрашиваем внутренность контура
                    cv2.fillPoly(temp_mask, [grid_points], 1)
                    # Вычитаем сам конур (делаем его свободным)
                    cv2.polylines(temp_mask, [grid_points], True, 0, 1)
                    # Объединяем с основной картой
                    planner['obstacle_map'] = np.maximum(planner['obstacle_map'], temp_mask)


def reset_path(planner: dict):
    """Сбрасывает путь и разблокирует планировщик"""
    planner['path'] = []
    planner['path_locked'] = False
    planner['current_goal'] = None


def is_cell_safe(planner: dict, grid_x: int, grid_y: int) -> bool:
    if not (0 <= grid_x < planner['grid_width'] and 0 <= grid_y < planner['grid_height']):
        return False

    # Проверка по obstacle_map (контуры)
    if planner['obstacle_map'][grid_y, grid_x] == 1:
        return False

    cx = (grid_x + 0.5) * planner['step']
    cy = (grid_y + 0.5) * planner['step']

    # Проверка границ поля
    if (cx < planner['edge_limit_cm'] or
            cx > planner['field_width'] - planner['edge_limit_cm'] or
            cy < planner['edge_limit_cm'] or
            cy > planner['field_height'] - planner['edge_limit_cm']):
        return False

    return True


def heuristic(x: int, y: int, goal_x: int, goal_y: int, step: float) -> float:
    dx = (x - goal_x) * step
    dy = (y - goal_y) * step
    return math.hypot(dx, dy)


def world_to_grid(planner: dict, x: float, y: float) -> Tuple[int, int]:
    step = planner['step']
    grid_x = int(x / step)
    grid_y = int(y / step)
    grid_x = max(0, min(grid_x, planner['grid_width'] - 1))
    grid_y = max(0, min(grid_y, planner['grid_height'] - 1))
    return grid_x, grid_y


def grid_to_world(planner: dict, grid_x: int, grid_y: int) -> Tuple[float, float]:
    step = planner['step']
    return (grid_x + 0.5) * step, (grid_y + 0.5) * step


def find_path(planner: dict, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
    # Если путь уже заблокирован и цель та же, возвращаем существующий путь
    if planner.get('path_locked', False) and planner.get('current_goal') == goal:
        return planner['path']

    # Новая цель - сбрасываем блокировку
    planner['path_locked'] = False

    start_grid = world_to_grid(planner, start[0], start[1])
    goal_grid = world_to_grid(planner, goal[0], goal[1])

    if not is_cell_safe(planner, start_grid[0], start_grid[1]):
        print(" Стартовая точка небезопасна")
        return []

    if not is_cell_safe(planner, goal_grid[0], goal_grid[1]):
        print(" Целевая точка небезопасна")
        return []

    moves = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1), (0, 1),
             (1, -1), (1, 0), (1, 1)]

    # Стоимость движения (диагональные дороже)
    def get_move_cost(dx, dy):
        if dx != 0 and dy != 0:
            return math.sqrt(2) * planner['step']
        return planner['step']

    # Используем A*
    import heapq

    INF = float('inf')
    g_score = {}
    parent = {}

    start_node = (start_grid[0], start_grid[1])
    g_score[start_node] = 0

    h = heuristic(start_grid[0], start_grid[1], goal_grid[0], goal_grid[1], planner['step'])
    pq = [(h, start_grid[0], start_grid[1])]

    visited = set()

    while pq:
        _, x, y = heapq.heappop(pq)
        current = (x, y)

        if current in visited:
            continue
        visited.add(current)

        if current == (goal_grid[0], goal_grid[1]):
            # Восстанавливаем путь
            path = []
            curr = current
            while curr in parent:
                path.append(grid_to_world(planner, curr[0], curr[1]))
                curr = parent[curr]
            path.append(grid_to_world(planner, start_grid[0], start_grid[1]))
            path.reverse()

            # ========== ИНТЕРПОЛЯЦИЯ ПУТИ ==========
            # Добавляем промежуточные точки на прямых участках
            interpolated_path = []

            # Расстояние между интерполированными точками (меньше шага сетки)
            interpolation_step = planner['step'] / 2  # 0.5 см между точками

            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]

                # Вычисляем расстояние между точками
                dist = math.hypot(x2 - x1, y2 - y1)

                if dist < interpolation_step:
                    # Если расстояние маленькое, просто добавляем первую точку
                    if i == 0:
                        interpolated_path.append((x1, y1))
                else:
                    # Добавляем первую точку
                    if i == 0:
                        interpolated_path.append((x1, y1))

                    # Вычисляем количество промежуточных точек
                    num_points = int(dist / interpolation_step)

                    # Добавляем промежуточные точки
                    for j in range(1, num_points + 1):
                        t = j / (num_points + 1)
                        ix = x1 + t * (x2 - x1)
                        iy = y1 + t * (y2 - y1)
                        interpolated_path.append((ix, iy))

            # Добавляем последнюю точку
            interpolated_path.append(path[-1])

            # ========== ОПЦИОНАЛЬНО: СГЛАЖИВАНИЕ (не удаляем точки, а только сглаживаем) ==========
            # Применяем небольшое сглаживание для более плавного движения
            smoothed_path = [interpolated_path[0]]
            smoothing_factor = 0.3  # Сила сглаживания

            for i in range(1, len(interpolated_path) - 1):
                prev = smoothed_path[-1]
                curr = interpolated_path[i]
                nxt = interpolated_path[i + 1]

                # Если точки почти на одной прямой, оставляем как есть
                cross = abs((curr[1] - prev[1]) * (nxt[0] - curr[0]) -
                            (curr[0] - prev[0]) * (nxt[1] - curr[1]))

                if cross < 0.5:  # Почти прямая линия
                    smoothed_path.append(curr)
                else:
                    # Сглаживаем угол
                    smooth_x = curr[0] * (1 - smoothing_factor) + (prev[0] + nxt[0]) / 2 * smoothing_factor
                    smooth_y = curr[1] * (1 - smoothing_factor) + (prev[1] + nxt[1]) / 2 * smoothing_factor
                    smoothed_path.append((smooth_x, smooth_y))

            smoothed_path.append(interpolated_path[-1])

            planner['path'] = smoothed_path
            planner['path_locked'] = True
            planner['current_goal'] = goal
            return smoothed_path

        # Сортируем соседей
        neighbors = []
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            neighbor = (nx, ny)

            if not (0 <= nx < planner['grid_width'] and 0 <= ny < planner['grid_height']):
                continue
            if not is_cell_safe(planner, nx, ny):
                continue

            move_cost = get_move_cost(dx, dy)
            tentative_g = g_score[current] + move_cost

            if tentative_g < g_score.get(neighbor, INF):
                parent[neighbor] = current
                g_score[neighbor] = tentative_g
                h = heuristic(nx, ny, goal_grid[0], goal_grid[1], planner['step'])
                neighbors.append((tentative_g + h, nx, ny))

        for priority, nx, ny in neighbors:
            heapq.heappush(pq, (priority, nx, ny))

    print(" Путь не найден")
    return []

def get_velocities(planner: dict, current_x: float, current_y: float,
                   max_speed: float, kp: float, acc_speed_error: float) -> Tuple[float, float]:
    path = planner['path']
    if not path or len(path) < 2:
        return 0.0, 0.0

    min_dist = float('inf')
    nearest_idx = 0
    for i, point in enumerate(path):
        px, py = point
        dist = math.hypot(px - current_x, py - current_y)
        if dist < min_dist:
            min_dist = dist
            nearest_idx = i

    target_idx = min(nearest_idx + 4, len(path) - 1)
    target_x, target_y = path[target_idx]

    error_x = target_x - current_x
    error_y = target_y - current_y
    error_distance = math.hypot(error_x, error_y)

    min_speed_ms = 0.05

    max_speed_cm = max_speed * 100.0
    speed_cm = min(kp * error_distance, max_speed_cm)

    final_goal = path[-1]
    dist_to_final = math.hypot(final_goal[0] - current_x, final_goal[1] - current_y)
    if dist_to_final > acc_speed_error:
        speed_cm = max(speed_cm, min_speed_ms * 100.0)

    if error_distance > 0:
        vx = (error_x / error_distance) * (speed_cm / 100.0)
        vy = (error_y / error_distance) * (speed_cm / 100.0)
    else:
        vx, vy = 0.0, 0.0

    return vx, -vy


def draw_planning_contours(planner: dict, frame: np.ndarray) -> np.ndarray:
    for obs in planner['obstacles']:
        if 'expanded_contour' in obs:
            expanded_contour = obs['expanded_contour']
            if len(expanded_contour) > 2:
                cv2.polylines(frame, [expanded_contour], True, (255, 0, 0), 2)
    return frame


def draw_path_on_frame(planner: dict, frame: np.ndarray, path: List[Tuple[float, float]],
                       color: Tuple[int, int, int] = (255, 0, 255)) -> np.ndarray:
    if not path or len(path) < 2:
        return frame

    h, w = frame.shape[:2]
    field_width = planner['field_width']
    field_height = planner['field_height']

    points = []
    for real_x, real_y in path:
        x_px = int(real_x / field_width * w)
        y_px = int(h - (real_y / field_height * h))
        points.append((x_px, y_px))

    for i in range(len(points) - 1):
        cv2.line(frame, points[i], points[i + 1], color, 3)
    return frame