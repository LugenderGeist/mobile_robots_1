import os
from pathlib import Path
from video import FieldRectifier

# ========== НАСТРОЙКИ ==========
VIDEO_FILE = "video_5.mp4"
OUTPUT_DIR = "output"
OUTPUT_FILE = "rectified_video.mp4"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

# РЕАЛЬНЫЕ РАЗМЕРЫ ПОЛЯ
FIELD_WIDTH = 220.0
FIELD_HEIGHT = 220.0

# НАСТРОЙКИ ОБНАРУЖЕНИЯ ПРЕПЯТСТВИЙ
EDGE_MARGIN = 0                     # не отсекаем объекты в detect_obstacles
ROBOT_EXCLUSION_RADIUS = 100
OBSTACLE_MIN_AREA = 800
OBSTACLE_MAX_AREA = 500000
OBSTACLE_THRESHOLD_V = 220

# НАСТРОЙКИ ПЛАНИРОВЩИКА ПУТИ
PATH_SAFETY_MARGIN = 10.0           # контур препятствия
PATH_EDGE_MARGIN = 15.0             # отступ от края

CORNERS_FILE = "field_corners.json"
FORCE_RESELECT = False

# ===============================================

def main():

    # Проверяем видео
    if not os.path.exists(VIDEO_FILE):
        print(f"❌ Ошибка: файл '{VIDEO_FILE}' не найден!")
        return

    # Создаём папку для результатов
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    try:
        # Создаём обработчик
        rectifier = FieldRectifier(VIDEO_FILE, OUTPUT_PATH)
        rectifier.set_field_dimensions(FIELD_WIDTH, FIELD_HEIGHT)

        rectifier.set_obstacle_params(
            edge_margin=EDGE_MARGIN,
            robot_exclusion_radius=ROBOT_EXCLUSION_RADIUS,
            min_area=OBSTACLE_MIN_AREA,
            max_area=OBSTACLE_MAX_AREA,
            threshold_v=OBSTACLE_THRESHOLD_V
        )

        rectifier.set_path_params(
            safety_margin=PATH_SAFETY_MARGIN,
            edge_margin=PATH_EDGE_MARGIN
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
        print(f"\n❌ Ошибка: {e}")

if __name__ == "__main__":
    main()