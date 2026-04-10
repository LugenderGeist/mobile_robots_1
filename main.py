import os
from pathlib import Path
from video import FieldRectifier

# ========== НАСТРОЙКИ ==========
VIDEO_FILE = "video_4.mp4"
OUTPUT_DIR = "output"
OUTPUT_FILE = "rectified_video.mp4"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

# РЕАЛЬНЫЕ РАЗМЕРЫ ПОЛЯ
FIELD_WIDTH = 220.0  # ← ИЗМЕНИТЕ
FIELD_HEIGHT = 220.0  # ← ИЗМЕНИТЕ

# НАСТРОЙКИ ОБНАРУЖЕНИЯ ПРЕПЯТСТВИЙ
EDGE_MARGIN = 10            # количество пикселей от края
ROBOT_EXCLUSION_RADIUS = 125  # радиус зоны робота
OBSTACLE_MIN_AREA = 500  # минимальная площадь препятствия
OBSTACLE_THRESHOLD_V = 180  # порог яркости для обнаружения объектов

CORNERS_FILE = "field_corners.json"
FORCE_RESELECT = False

# ===============================================

def main():
    print("=" * 50)
    print("ОТСЛЕЖИВАНИЕ РОБОТА И ПРЕПЯТСТВИЙ НА ПОЛЕ")
    print("=" * 50)

    # Проверяем видео
    if not os.path.exists(VIDEO_FILE):
        print(f"❌ Ошибка: файл '{VIDEO_FILE}' не найден!")
        return

    # Создаём папку для результатов
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"\n📹 Видео: {VIDEO_FILE}")
    print(f"📏 Размеры поля: {FIELD_WIDTH} x {FIELD_HEIGHT}")
    print(f"💾 Результат: {OUTPUT_PATH}")
    print(f"\n⚙️ Настройки обнаружения:")
    print(f"   Отступ от края: {EDGE_MARGIN} пикселей")
    print(f"   Зона исключения робота: {ROBOT_EXCLUSION_RADIUS} пикселей")
    print(f"   Мин. площадь препятствия: {OBSTACLE_MIN_AREA} пикселей")
    print(f"   Порог яркости: {OBSTACLE_THRESHOLD_V}")

    try:
        # Создаём обработчик
        rectifier = FieldRectifier(VIDEO_FILE, OUTPUT_PATH)
        rectifier.set_field_dimensions(FIELD_WIDTH, FIELD_HEIGHT)

        # Передаём настройки обнаружения препятствий
        rectifier.set_obstacle_params(
             edge_margin=EDGE_MARGIN,
            robot_exclusion_radius=ROBOT_EXCLUSION_RADIUS,
            min_area=OBSTACLE_MIN_AREA,
            threshold_v=OBSTACLE_THRESHOLD_V
        )

        # Загружаем сохранённые углы
        if not FORCE_RESELECT and os.path.exists(CORNERS_FILE):
            rectifier.load_corners(CORNERS_FILE)

        # Запускаем обработку
        stats = rectifier.process_video()

        # Сохраняем углы и траекторию
        rectifier.save_corners(CORNERS_FILE)
        rectifier.save_trajectory(os.path.join(OUTPUT_DIR, "robot_trajectory.json"))

        print(f"\n✅ Готово!")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")

if __name__ == "__main__":
    main()