#!/usr/bin/env python3
"""
Настройка порога для обнаружения препятствий с навигацией по кадрам
"""

import cv2
import numpy as np
import json

VIDEO_FILE = "video_2.mp4"
CORNERS_FILE = "field_corners.json"


def nothing(x):
    pass


def load_homography():
    """Загрузить матрицу гомографии для выравнивания поля"""
    try:
        with open(CORNERS_FILE, 'r') as f:
            data = json.load(f)
            corners = np.array(data['corners'], dtype=np.float32)
            dst = np.array([[0, 0], [720, 0], [720, 720], [0, 720]], dtype=np.float32)
            H, _ = cv2.findHomography(corners, dst)
            print(f"✓ Загружены углы поля")
            return H
    except Exception as e:
        print(f"✗ Не удалось загрузить углы: {e}")
        return None


def main():
    print("=" * 60)
    print("НАСТРОЙКА ОБНАРУЖЕНИЯ ПРЕПЯТСТВИЙ")
    print("=" * 60)

    # Открываем видео
    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        print(f"❌ Не удалось открыть видео: {VIDEO_FILE}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Видео: {total_frames} кадров, {fps:.2f} FPS")

    # Загружаем гомографию
    H = load_homography()

    # Создаём окна
    cv2.namedWindow("Obstacle Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Obstacle Detection", 800, 800)

    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mask", 400, 400)

    cv2.namedWindow("Parameters", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Parameters", 400, 300)

    # Создаём трекбары
    cv2.createTrackbar("Threshold V", "Parameters", 100, 255, nothing)
    cv2.createTrackbar("Min Area", "Parameters", 500, 5000, nothing)
    cv2.createTrackbar("Blur", "Parameters", 5, 20, nothing)

    # Трекбар для навигации по кадрам
    cv2.createTrackbar("Frame", "Parameters", 0, total_frames - 1, nothing)

    print("\n📌 ИНСТРУКЦИЯ:")
    print("   Управление:")
    print("     • Ползунок 'Frame' - переключение между кадрами")
    print("     • Стрелка ВЛЕВО/ВПРАВО - предыдущий/следующий кадр (+/- 30)")
    print("     • ПРОБЕЛ - пауза/воспроизведение")
    print("     • 's' - сохранить текущие параметры")
    print("     • 'q' - выход")
    print("")
    print("   Параметры:")
    print("     • Threshold V - чем выше, тем больше объектов считается тёмными")
    print("     • Min Area - минимальная площадь препятствия")
    print("     • Blur - размытие для удаления шума")
    print("=" * 60)

    current_frame = 0
    auto_play = False
    best_params = None

    # Загружаем сохранённые параметры
    try:
        with open("obstacle_params.json", "r") as f:
            saved = json.load(f)
            cv2.setTrackbarPos("Threshold V", "Parameters", saved.get('threshold_v', 100))
            cv2.setTrackbarPos("Min Area", "Parameters", saved.get('min_area', 500))
            cv2.setTrackbarPos("Blur", "Parameters", saved.get('blur', 5))
            print("✓ Загружены сохранённые параметры")
    except:
        pass

    while True:
        # Получаем позицию из трекбара (если не в режиме автовоспроизведения)
        if not auto_play:
            current_frame = cv2.getTrackbarPos("Frame", "Parameters")

        # Читаем кадр
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        if not ret:
            print(f"Не удалось прочитать кадр {current_frame}")
            break

        # Выравниваем поле
        if H is not None:
            frame = cv2.warpPerspective(frame, H, (720, 720))

        # Получаем параметры с трекбаров
        thresh_v = cv2.getTrackbarPos("Threshold V", "Parameters")
        min_area = cv2.getTrackbarPos("Min Area", "Parameters")
        blur_size = cv2.getTrackbarPos("Blur", "Parameters")
        if blur_size % 2 == 0:
            blur_size += 1

        # Обработка для обнаружения препятствий
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]

        # Размытие
        if blur_size > 1:
            v_channel = cv2.GaussianBlur(v_channel, (blur_size, blur_size), 0)

        # Пороговая обработка (инвертированная)
        _, mask = cv2.threshold(v_channel, thresh_v, 255, cv2.THRESH_BINARY_INV)

        # Морфологическая очистка
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Поиск контуров
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Рисуем результат
        result = frame.copy()
        obstacle_count = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                cv2.drawContours(result, [contour], -1, (0, 0, 255), 2)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(result, (cx, cy), 6, (0, 255, 0), -1)
                    cv2.circle(result, (cx, cy), 10, (0, 255, 0), 2)
                obstacle_count += 1

        # Добавляем информацию на кадр
        info_y = 30
        cv2.putText(result, f"Frame: {current_frame + 1}/{total_frames}", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        info_y += 25
        cv2.putText(result, f"Threshold V: {thresh_v}, Min Area: {min_area}, Blur: {blur_size}",
                    (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        info_y += 25
        cv2.putText(result, f"Obstacles found: {obstacle_count}",
                    (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if obstacle_count > 0 else (0, 0, 255), 1)

        if auto_play:
            cv2.putText(result, "AUTO PLAY: ON", (result.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Показываем маску для наглядности
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(mask_colored, f"Mask (white = detected as dark)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Отображаем
        cv2.imshow("Obstacle Detection", result)
        cv2.imshow("Mask", mask_colored)

        # Управление
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n👋 Выход")
            break

        elif key == ord('s'):
            best_params = {
                'threshold_v': thresh_v,
                'min_area': min_area,
                'blur': blur_size
            }
            with open("obstacle_params.json", "w") as f:
                json.dump(best_params, f, indent=2)
            print(f"\n✅ Параметры сохранены: Threshold V={thresh_v}, Min Area={min_area}, Blur={blur_size}")

        elif key == ord(' '):  # Пробел
            auto_play = not auto_play
            print("▶️ Автовоспроизведение" if auto_play else "⏸️ Пауза")

        elif key == 81 or key == 2424832:  # Стрелка влево
            current_frame = max(0, current_frame - 30)
            cv2.setTrackbarPos("Frame", "Parameters", current_frame)
            auto_play = False
            print(f"← Кадр {current_frame + 1}")

        elif key == 83 or key == 2555904:  # Стрелка вправо
            current_frame = min(total_frames - 1, current_frame + 30)
            cv2.setTrackbarPos("Frame", "Parameters", current_frame)
            auto_play = False
            print(f"→ Кадр {current_frame + 1}")

        # Автовоспроизведение
        if auto_play:
            current_frame = (current_frame + 1) % total_frames
            cv2.setTrackbarPos("Frame", "Parameters", current_frame)

    cap.release()
    cv2.destroyAllWindows()

    if best_params:
        print("\n💾 Для использования в основном коде добавьте в detect_obstacles:")
        print(f"   threshold_v = {best_params['threshold_v']}")
        print(f"   min_area = {best_params['min_area']}")
        print(f"   blur_size = {best_params['blur']}")


if __name__ == "__main__":
    main()