import sys
import os
from pathlib import Path
from video import FieldRectifier

# ========== НАСТРОЙКИ (МЕНЯЙТЕ ЗДЕСЬ) ==========
VIDEO_FILE = "video_1.mp4"  # Имя видео файла
OUTPUT_DIR = "output"  # Папка для результатов
OUTPUT_FILE = "rectified_video.mp4"  # Имя выходного видео
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

# РЕАЛЬНЫЕ РАЗМЕРЫ ПОЛЯ (в метрах или сантиметрах)
FIELD_WIDTH = 10.0  # ← ИЗМЕНИТЕ НА ВАШУ ШИРИНУ
FIELD_HEIGHT = 5.0  # ← ИЗМЕНИТЕ НА ВАШУ ВЫСОТУ

CORNERS_FILE = "field_corners.json"  # Файл с углами
FORCE_RESELECT = False  # True - перевыбрать углы


# ==============================================

def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════════╗
║           ОТСЛЕЖИВАНИЕ РОБОТА НА ПОЛЕ                       ║
║           Выравнивание + ArUco детекция                     ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    print_banner()

    # Проверяем видео
    if not os.path.exists(VIDEO_FILE):
        print(f"\n❌ Ошибка: файл '{VIDEO_FILE}' не найден!")
        print(f"   Убедитесь, что видео находится в той же папке")
        sys.exit(1)

    # Создаём папку для результатов
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"\n📹 Видео: {VIDEO_FILE}")
    print(f"📏 Размеры поля: {FIELD_WIDTH} x {FIELD_HEIGHT}")
    print(f"💾 Результат: {OUTPUT_PATH}\n")

    try:
        # Создаём объект для обработки
        rectifier = FieldRectifier(VIDEO_FILE, OUTPUT_PATH)
        rectifier.set_field_dimensions(FIELD_WIDTH, FIELD_HEIGHT)

        # Загружаем сохранённые углы (если есть и не нужно перевыбирать)
        if not FORCE_RESELECT and os.path.exists(CORNERS_FILE):
            use_saved = input(f"Найден файл с углами. Использовать? (y/n) [y]: ").lower()
            if use_saved != 'n':
                rectifier.load_corners(CORNERS_FILE)

        # Запускаем обработку
        stats = rectifier.process_video()

        # Сохраняем углы для следующего раза
        rectifier.save_corners(CORNERS_FILE)

        # Сохраняем траекторию робота
        rectifier.save_trajectory(os.path.join(OUTPUT_DIR, "robot_trajectory.json"))

        # Выводим итоговую статистику
        print("\n" + "=" * 60)
        print("📊 ИТОГОВАЯ СТАТИСТИКА")
        print("=" * 60)
        print(f"  ✅ Выровненное видео: {OUTPUT_PATH}")
        print(f"  🤖 Робот обнаружен в {stats['robot_detections']} кадрах из {stats['processed_frames']}")

        detection_rate = (stats['robot_detections'] / stats['processed_frames']) * 100
        print(f"  📈 Процент обнаружения: {detection_rate:.1f}%")

        if stats['robot_detections'] > 0:
            print(f"\n  📁 Траектория сохранена: {OUTPUT_DIR}/robot_trajectory.json")

    except KeyboardInterrupt:
        print("\n\n⚠️ Прервано пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()