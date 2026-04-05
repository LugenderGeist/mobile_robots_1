import sys
import os
from video import FieldRectifier

VIDEO_FILE = "video_1.mp4"
OUTPUT_FILE = "rectified_video.mp4"

FIELD_WIDTH = 10.0  # ← ИЗМЕНИТЕ
FIELD_HEIGHT = 5.0  # ← ИЗМЕНИТЕ

CORNERS_FILE = "field_corners.json"

def main():
    print(f"\n📹 Обработка видео: {VIDEO_FILE}")
    print(f"📏 Размеры поля: {FIELD_WIDTH} x {FIELD_HEIGHT}")
    print(f"💾 Результат: {OUTPUT_FILE}\n")

    if not os.path.exists(VIDEO_FILE):
        print(f"❌ Ошибка: файл {VIDEO_FILE} не найден!")
        sys.exit(1)

    try:
        rectifier = FieldRectifier(VIDEO_FILE, OUTPUT_FILE)
        rectifier.set_field_dimensions(FIELD_WIDTH, FIELD_HEIGHT)

        # Автоматически загружаем углы если есть
        if os.path.exists(CORNERS_FILE):
            print("📂 Загружаем сохранённые углы...")
            rectifier.load_corners(CORNERS_FILE)

        rectifier.process_video_to_video()
        rectifier.save_corners(CORNERS_FILE)

        print(f"\n✅ Готово! Результат: {OUTPUT_FILE}")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()