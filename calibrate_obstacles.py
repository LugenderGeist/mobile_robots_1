import cv2
import numpy as np
import json
import os

# ========== НАСТРОЙКИ ==========
CAMERA_ID = 1  # ID камеры (0 - встроенная, 1 - USB)
CORNERS_FILE = "field_corners.json"  # Файл с углами для камеры
PARAMS_FILE = "obstacle_params_camera.json"  # Файл с параметрами для камеры
# =================================

def nothing(x):
    pass

def load_homography():
    try:
        with open(CORNERS_FILE, 'r') as f:
            data = json.load(f)
            corners = np.array(data['corners'], dtype=np.float32)
            dst = np.array([[0, 0], [720, 0], [720, 720], [0, 720]], dtype=np.float32)
            H, _ = cv2.findHomography(corners, dst)
            print(f"✓ Загружены углы поля из {CORNERS_FILE}")
            return H, corners
    except Exception as e:
        print(f"✗ Не удалось загрузить углы: {e}")
        return None, None

def main():
    print("=" * 60)
    print("НАСТРОЙКА ОБНАРУЖЕНИЯ ПРЕПЯТСТВИЙ ДЛЯ КАМЕРЫ")
    print("Ищем всё, что темнее белого фона")
    print("=" * 60)

    # Открываем камеру
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"❌ Не удалось открыть камеру {CAMERA_ID}")
        print("   Проверьте подключение камеры и правильность ID")
        return

    # Получаем FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    print(f"📷 Камера: ID={CAMERA_ID}, FPS={fps:.1f}")

    # Загружаем гомографию
    H, corners = load_homography()
    if H is None:
        print("\n⚠️ Нет калибровки для камеры!")
        print("   Сначала запустите calibrate_camera.py для настройки углов поля")
        cap.release()
        return

    # Создаём окна
    cv2.namedWindow("Obstacle Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Obstacle Detection", 800, 800)

    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mask", 400, 400)

    cv2.namedWindow("Parameters", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Parameters", 400, 300)

    # Создаём трекбары
    cv2.createTrackbar("Threshold White", "Parameters", 220, 255, nothing)
    cv2.createTrackbar("Min Area", "Parameters", 500, 5000, nothing)
    cv2.createTrackbar("Blur", "Parameters", 5, 20, nothing)
    cv2.createTrackbar("Edge Margin", "Parameters", 20, 100, nothing)

    # Дополнительные параметры для камеры
    cv2.createTrackbar("Brightness", "Parameters", 0, 100, nothing)
    cv2.createTrackbar("Contrast", "Parameters", 100, 200, nothing)

    print("\n📌 ИНСТРУКЦИЯ:")
    print("   Настройте параметры с помощью ползунков")
    print("   • 's' - сохранить параметры")
    print("   • 'p' - пауза/продолжение")
    print("   • 'q' - выход\n")

    # Загружаем сохранённые параметры
    try:
        with open(PARAMS_FILE, "r") as f:
            saved = json.load(f)
            cv2.setTrackbarPos("Threshold White", "Parameters", saved.get('threshold_white', 220))
            cv2.setTrackbarPos("Min Area", "Parameters", saved.get('min_area', 500))
            cv2.setTrackbarPos("Blur", "Parameters", saved.get('blur', 5))
            cv2.setTrackbarPos("Edge Margin", "Parameters", saved.get('edge_margin', 20))
            cv2.setTrackbarPos("Brightness", "Parameters", saved.get('brightness', 0))
            cv2.setTrackbarPos("Contrast", "Parameters", saved.get('contrast', 100))
            print("✓ Загружены сохранённые параметры")
    except:
        pass

    paused = False
    frame_count = 0
    best_params = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Не удалось получить кадр с камеры")
                break
            frame_count += 1

        # Получаем параметры
        threshold_white = cv2.getTrackbarPos("Threshold White", "Parameters")
        min_area = cv2.getTrackbarPos("Min Area", "Parameters")
        blur_size = cv2.getTrackbarPos("Blur", "Parameters")
        edge_margin = cv2.getTrackbarPos("Edge Margin", "Parameters")
        brightness = cv2.getTrackbarPos("Brightness", "Parameters") - 50
        contrast = cv2.getTrackbarPos("Contrast", "Parameters") / 100.0

        if blur_size % 2 == 0:
            blur_size += 1

        # Предобработка изображения
        processed = frame.copy()

        # Яркость и контраст
        processed = cv2.convertScaleAbs(processed, alpha=contrast, beta=brightness)

        # Выравниваем поле
        if H is not None:
            processed = cv2.warpPerspective(processed, H, (720, 720))

        # Обработка
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        # Размытие
        if blur_size > 1:
            gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

        # Пороговая обработка
        _, mask = cv2.threshold(gray, threshold_white, 255, cv2.THRESH_BINARY_INV)

        # Морфология
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Убираем края
        h, w = mask.shape
        mask[0:edge_margin, :] = 0
        mask[h - edge_margin:h, :] = 0
        mask[:, 0:edge_margin] = 0
        mask[:, w - edge_margin:w] = 0

        # Поиск контуров
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Рисуем результат
        result = processed.copy()
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

        # Рисуем жёлтую рамку
        cv2.rectangle(result, (edge_margin, edge_margin),
                      (w - edge_margin, h - edge_margin), (0, 255, 255), 2)

        # Информация на кадре
        info_y = 30
        cv2.putText(result, f"Frame: {frame_count}", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        info_y += 25
        cv2.putText(result, f"Threshold: {threshold_white}, Min Area: {min_area}, Blur: {blur_size}",
                    (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        info_y += 20
        cv2.putText(result, f"Brightness: {brightness}, Contrast: {contrast:.1f}, Edge Margin: {edge_margin}",
                    (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        info_y += 25
        cv2.putText(result, f"Obstacles found: {obstacle_count}",
                    (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if obstacle_count > 0 else (0, 0, 255), 1)

        if paused:
            cv2.putText(result, "PAUSED", (result.shape[1] - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Показываем маску
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(mask_colored, f"Mask (white = obstacles)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(mask_colored, f"Obstacles: {obstacle_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if obstacle_count > 0 else (0, 0, 255), 1)

        cv2.imshow("Obstacle Detection", result)
        cv2.imshow("Mask", mask_colored)

        key = cv2.waitKey(1 if not paused else 0) & 0xFF

        if key == ord('q'):
            print("\n Выход")
            break

        elif key == ord('s'):
            best_params = {
                'threshold_white': threshold_white,
                'min_area': min_area,
                'blur': blur_size,
                'edge_margin': edge_margin,
                'brightness': brightness,
                'contrast': contrast
            }
            with open(PARAMS_FILE, "w") as f:
                json.dump(best_params, f, indent=2)
            print(f"\n Параметры сохранены в {PARAMS_FILE}")
            print(f"   Threshold: {threshold_white}, Min Area: {min_area}, Blur: {blur_size}")
            print(f"   Brightness: {brightness}, Contrast: {contrast:.1f}, Edge Margin: {edge_margin}")

        elif key == ord('p'):
            paused = not paused
            print("⏸ Пауза" if paused else "▶ Продолжение")

    cap.release()
    cv2.destroyAllWindows()

    if best_params:
        print("\n Рекомендуемые параметры для video.py:")
        print(f"   OBSTACLE_THRESHOLD_V = {best_params['threshold_white']}")
        print(f"   OBSTACLE_MIN_AREA = {best_params['min_area']}")
        print(f"   EDGE_MARGIN = {best_params['edge_margin']}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Программа остановлена")
    except Exception as e:
        print(f"\n Ошибка: {e}")