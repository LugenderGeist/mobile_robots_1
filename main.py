import os
import cv2
import time
import math
from pathlib import Path
import video as vp
from robotino import (
    connect_to_robotino,
    get_odometry,
    send_velocity,
    stop_robot
)
from planners.greedy_planner import (
    create_planner,
    update_obstacles,
    draw_planning_contours,
    find_path,
    draw_path_on_frame,
    get_velocities
)

# ========== НАСТРОЙКИ ==========
FIELD_WIDTH = 220.0
FIELD_HEIGHT = 220.0

EDGE_MARGIN = 5
OBSTACLE_MIN_AREA = 800
OBSTACLE_MAX_AREA = 500000
OBSTACLE_THRESHOLD_V = 150
ROBOT_RADIUS = 32.0
OBSTACLE_SAFETY_MARGIN = 0.0
PLANNING_STEP = 2.0

SHOW_ROBOT_ZONE = True
SHOW_PLANNING_CONTOURS = True
EDGE_LIMIT_CM = 15.0

# НАСТРОЙКИ УПРАВЛЕНИЯ
MAX_SPEED = 0.2
KP = 0.4
LOOKAHEAD_DISTANCE = 15.0
GOAL_TOLERANCE = 5.0

CORNERS_FILE = "field_corners.json"
OUTPUT_DIR = "output"
# =================================


def init_video_processor():
    vp.set_field_dimensions(FIELD_WIDTH, FIELD_HEIGHT)
    vp.set_obstacle_params(EDGE_MARGIN, OBSTACLE_MIN_AREA, OBSTACLE_MAX_AREA, OBSTACLE_THRESHOLD_V)
    vp.set_robot_params(ROBOT_RADIUS, OBSTACLE_SAFETY_MARGIN, PLANNING_STEP,
                        SHOW_ROBOT_ZONE, SHOW_PLANNING_CONTOURS, EDGE_LIMIT_CM)

    if os.path.exists(CORNERS_FILE):
        vp.load_corners(CORNERS_FILE)


def mode_video_file():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    files = [f for f in os.listdir('.') if f.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]
    if not files:
        print("Нет видеофайлов!")
        return

    print("\nДоступные видеофайлы:")
    for i, f in enumerate(files):
        print(f"   {i + 1}. {f}")

    try:
        choice = int(input(f"\nВыберите файл (1-{len(files)}): ")) - 1
        if 0 <= choice < len(files):
            video_file = files[choice]
        else:
            print("Неверный выбор!")
            return
    except ValueError:
        print("Введите число!")
        return

    output_path = os.path.join(OUTPUT_DIR, f"processed_{video_file}")

    init_video_processor()
    vp.reset_trajectory()
    stats = vp.process_video(video_file, output_path)

    vp.save_corners(CORNERS_FILE)
    vp.save_trajectory(os.path.join(OUTPUT_DIR, "robot_trajectory.json"))

    print(f"\nОбработка завершена!")
    print(f"   Обработано кадров: {stats['processed_frames']}")
    print(f"   Обнаружений робота: {stats['robot_detections']}")
    print(f"   Результат: {output_path}")


def mode_camera():
    init_video_processor()
    print("\nРежим камеры")
    print("   Управление: Левый клик - цель, Пробел - пауза, 'q' - выход\n")
    vp.process_camera_feed(camera_id=1)


def mode_robot():
    """Режим управления роботом с камерой (обратная связь по видео)"""
    print("\n" + "=" * 50)
    print("РЕЖИМ: УПРАВЛЕНИЕ РОБОТОМ ПО ВИДЕО")
    print("=" * 50)

    # Проверяем подключение к роботу
    if not connect_to_robotino():
        print("Не удалось подключиться к Robotino!")
        return

    # Инициализируем обработчик видео
    init_video_processor()

    # Открываем камеру
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Не удалось открыть камеру!")
        return

    # Переменные
    target_point = None
    planner = None
    moving = False
    current_robot_pos = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal target_point, planner, moving
        if event == cv2.EVENT_LBUTTONDOWN:
            real_x, real_y = vp.transform_coordinates(x, y)
            target_point = (real_x, real_y)
            moving = False
            planner = None
            print(f"Цель: ({real_x:.1f}, {real_y:.1f}) см")

    cv2.namedWindow("Robot Control", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Robot Control", 800, 800)
    cv2.setMouseCallback("Robot Control", mouse_callback)

    print("\nИНСТРУКЦИЯ:")
    print("   1. Нажмите на целевую точку на поле")
    print("   2. Робот начнёт движение к цели (обратная связь по видео)")
    print("   3. 's' - остановка, 'q' - выход\n")

    from planners.greedy_planner import (
        create_planner, update_obstacles, draw_planning_contours,
        find_path, draw_path_on_frame, get_velocities
    )

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Потерян кадр с камеры")
            break

        frame_count += 1

        # Выравниваем поле
        rectified = cv2.warpPerspective(frame, vp._state['H'], vp._state['output_size'])
        rectified = vp.draw_coordinate_axes(rectified, margin=20, axis_length=60)

        if vp._state['edge_limit_cm'] > 0:
            rectified = vp.draw_edge_limit(rectified)

        # Детекция робота (получаем позицию из видео)
        found, robot_id, center_pixel, center_real, marker_corners = vp.detect_robot(frame)

        # ОБНОВЛЯЕМ ПОЗИЦИЮ РОБОТА ИЗ ВИДЕО
        if found:
            robot_x, robot_y = vp.transform_coordinates(center_pixel[0], center_pixel[1])
            current_robot_pos = (robot_x, robot_y)
            # Выводим позицию раз в 30 кадров (не спамим)
            if frame_count % 30 == 0:
                print(f"Робот на видео: ({robot_x:.1f}, {robot_y:.1f}) см")
        else:
            # Если робот не найден, останавливаем движение
            if moving:
                print("Робот потерян на видео! Останавливаем движение")
                stop_robot()
                moving = False
            current_robot_pos = None

        # Детекция препятствий
        if found:
            obstacles = vp.detect_obstacles(rectified, robot_center=center_pixel)
        else:
            obstacles = vp.detect_obstacles(rectified, robot_center=None)

        # Планировщик
        if planner is None:
            planner = create_planner(
                field_width=FIELD_WIDTH,
                field_height=FIELD_HEIGHT,
                step=PLANNING_STEP,
                robot_radius=ROBOT_RADIUS,
                obstacle_safety=OBSTACLE_SAFETY_MARGIN,
                edge_limit_cm=EDGE_LIMIT_CM
            )

        # Обновляем карту препятствий и рисуем жёлтые контуры
        update_obstacles(planner, obstacles)
        rectified = draw_planning_contours(planner, rectified)

        # Отрисовка робота
        if found:
            rectified = vp.draw_axes_2d(rectified, marker_corners, robot_id, axis_length=50)
            if SHOW_ROBOT_ZONE:
                robot_radius_px = int(ROBOT_RADIUS / FIELD_WIDTH * rectified.shape[1])
                cv2.circle(rectified, (int(center_pixel[0]), int(center_pixel[1])), robot_radius_px, (100, 100, 100), 1)

        # Планирование пути (когда есть цель и робот)
        if target_point is not None and current_robot_pos is not None and not moving:
            print(f"\nЦель: ({target_point[0]:.1f}, {target_point[1]:.1f}) см")
            print(f"Робот: ({current_robot_pos[0]:.1f}, {current_robot_pos[1]:.1f}) см")

            path = find_path(planner, current_robot_pos, target_point)

            if path:
                print(f"Путь найден! Всего {len(path)} точек")
                rectified = draw_path_on_frame(planner, rectified, path, (0, 255, 255))
                moving = True
            else:
                print("Путь не найден!")

        # УПРАВЛЕНИЕ РОБОТОМ (обратная связь по видео)
        if moving and planner and planner.get('path') and current_robot_pos is not None:
            # Используем позицию из видео
            robot_x_cm, robot_y_cm = current_robot_pos

            # Получаем скорости от планировщика
            vx, vy = get_velocities(
                planner,
                robot_x_cm, robot_y_cm,
                max_speed=MAX_SPEED,
                lookahead_distance=LOOKAHEAD_DISTANCE,
                kp=KP,
                goal_tolerance=GOAL_TOLERANCE
            )

            # Отправляем роботу (Y инвертируем, если нужно)
            send_velocity(vx, vy, 0.0)

            # Проверка достижения цели
            goal = planner['path'][-1]
            dx = goal[0] - robot_x_cm
            dy = goal[1] - robot_y_cm
            dist_to_goal = math.hypot(dx, dy)

            # Выводим отладку раз в 30 кадров
            if frame_count % 30 == 0:
                print(f"  🏃 Скорости: vx={vx:.3f}, vy={vy:.3f}, до цели: {dist_to_goal:.1f} см")

            if dist_to_goal < GOAL_TOLERANCE:
                stop_robot()
                print("Цель достигнута!")
                moving = False

        # Отрисовка цели
        if target_point is not None:
            tx = int(target_point[0] / FIELD_WIDTH * rectified.shape[1])
            ty = int(rectified.shape[0] - (target_point[1] / FIELD_HEIGHT * rectified.shape[0]))
            cv2.circle(rectified, (tx, ty), 8, (255, 0, 255), -1)
            cv2.circle(rectified, (tx, ty), 12, (255, 0, 255), 2)

        # Информация
        info_y = 25
        cv2.putText(rectified, f"Obstacles: {len(obstacles)}", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        info_y += 25
        if current_robot_pos:
            cv2.putText(rectified, f"Robot: ({current_robot_pos[0]:.1f}, {current_robot_pos[1]:.1f})",
                        (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        info_y += 25
        if target_point:
            cv2.putText(rectified, f"Target: ({target_point[0]:.1f}, {target_point[1]:.1f})",
                        (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow("Robot Control", rectified)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_robot()
            break
        elif key == ord('s'):
            stop_robot()
            moving = False
            print("Робот остановлен")

    cap.release()
    cv2.destroyAllWindows()


def main():
    print("=" * 50)
    print("ВЫБЕРИТЕ РЕЖИМ")
    print("=" * 50)
    print("1. Обработка видеофайла")
    print("2. Реальная камера")
    print("3. Управление роботом")
    print("0. Выход")

    choice = input("\nВаш выбор (0-3): ").strip()

    if choice == '1':
        mode_video_file()
    elif choice == '2':
        mode_camera()
    elif choice == '3':
        mode_robot()
    else:
        print("Выход")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрограмма остановлена")
        stop_robot()
    except Exception as e:
        print(f"\nОшибка: {e}")
        stop_robot()