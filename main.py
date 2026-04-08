import os
from pathlib import Path
from video import FieldRectifier

# ========== НАСТРОЙКИ (МЕНЯЙТЕ ЗДЕСЬ) ==========
VIDEO_FILE = "video_2.mp4"
OUTPUT_DIR = "output"
OUTPUT_FILE = "rectified_video.mp4"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

# РЕАЛЬНЫЕ РАЗМЕРЫ ПОЛЯ
FIELD_WIDTH = 10.0  # ← ИЗМЕНИТЕ
FIELD_HEIGHT = 5.0  # ← ИЗМЕНИТЕ

CORNERS_FILE = "field_corners.json"


# ===============================================

def main():
    print("=" * 50)
    print("ОТСЛЕЖИВАНИЕ РОБОТА ПО ARUCO МЕТКЕ")
    print("=" * 50)

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

        # Загружаем сохранённые углы
        if os.path.exists(CORNERS_FILE):
            rectifier.load_corners(CORNERS_FILE)

        # Запускаем обработку
        stats = rectifier.process_video()

        # Сохраняем углы и траекторию
        rectifier.save_corners(CORNERS_FILE)
        rectifier.save_trajectory(os.path.join(OUTPUT_DIR, "robot_trajectory.json"))

        print(f"\n✅ Готово!")
        print(f"   Видео: {OUTPUT_PATH}")
        print(f"   Траектория: {OUTPUT_DIR}/robot_trajectory.json")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")


if __name__ == "__main__":
    main()