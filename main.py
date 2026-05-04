import os
from pathlib import Path
from video import FieldRectifier

# ========== НАСТРОЙКИ ==========
VIDEO_FILE = "video_2.mp4"
OUTPUT_DIR = "output"
OUTPUT_FILE = "rectified_video.mp4"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

# РЕАЛЬНЫЕ РАЗМЕРЫ ПОЛЯ (в сантиметрах)
FIELD_WIDTH = 220.0
FIELD_HEIGHT = 220.0

# НАСТРОЙКИ ОБНАРУЖЕНИЯ ПРЕПЯТСТВИЙ
EDGE_MARGIN = 5                    # отступ от края для детекции (пиксели)
OBSTACLE_MIN_AREA = 800            # минимальная площадь препятствия (пиксели²)
OBSTACLE_MAX_AREA = 500000         # максимальная площадь препятствия (пиксели²)
OBSTACLE_THRESHOLD_V = 220         # порог яркости для детекции (0-255)

# НАСТРОЙКИ РОБОТА И ПЛАНИРОВЩИКА ПУТИ
ROBOT_RADIUS = 30.0                # см - радиус робота (для планирования и исключения зоны)
OBSTACLE_SAFETY_MARGIN = 0.0       # см - дополнительный запас вокруг препятствий (жёлтый контур)
PLANNING_STEP = 2.0                # см - шаг дискретизации для планирования пути (чем меньше, тем точнее, но медленнее)

# НАСТРОЙКИ ВИЗУАЛИЗАЦИИ
SHOW_ROBOT_ZONE = True             # показывать зону робота (серый круг)
SHOW_PLANNING_CONTOURS = True      # показывать жёлтые контуры для планирования
EDGE_LIMIT_CM = 15.0             # показывать чёрную пунктирную линию границы

CORNERS_FILE = "field_corners.json"
FORCE_RESELECT = False

# ===============================================

def main():
    # Проверяем видео
    if not os.path.exists(VIDEO_FILE):
        print(f" Ошибка: файл '{VIDEO_FILE}' не найден!")
        return

    # Создаём папку для результатов
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    try:
        # Создаём обработчик
        rectifier = FieldRectifier(VIDEO_FILE, OUTPUT_PATH)
        rectifier.set_field_dimensions(FIELD_WIDTH, FIELD_HEIGHT)

        # Настройки детекции препятствий
        rectifier.set_obstacle_params(
            edge_margin=EDGE_MARGIN,
            min_area=OBSTACLE_MIN_AREA,
            max_area=OBSTACLE_MAX_AREA,
            threshold_v=OBSTACLE_THRESHOLD_V
        )

        # Настройки робота и планировщика пути
        rectifier.set_robot_params(
            robot_radius=ROBOT_RADIUS,
            obstacle_safety_margin=OBSTACLE_SAFETY_MARGIN,
            planning_step=PLANNING_STEP,
            show_robot_zone=SHOW_ROBOT_ZONE,
            show_planning_contours=SHOW_PLANNING_CONTOURS,
            edge_limit_cm=EDGE_LIMIT_CM
        )

        # Загружаем сохранённые углы
        if not FORCE_RESELECT and os.path.exists(CORNERS_FILE):
            rectifier.load_corners(CORNERS_FILE)

        # Запускаем обработку
        stats = rectifier.process_video()

        # Сохраняем углы и траекторию
        rectifier.save_corners(CORNERS_FILE)
        rectifier.save_trajectory(os.path.join(OUTPUT_DIR, "robot_trajectory.json"))

    except Exception as e:
        print(f"\n Ошибка: {e}")

if __name__ == "__main__":
    main()