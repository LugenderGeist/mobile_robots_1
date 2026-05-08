import os
import cv2
import math
import video as vp
from robotino import (connect_to_robotino,send_velocity,stop_robot)

# РАЗМЕРЫ ПОЛЯ
FIELD_WIDTH = 220.0
FIELD_HEIGHT = 220.0

# ПРЕПЯТСТВИЯ
EDGE_MARGIN = 5
OBSTACLE_MIN_AREA = 800
OBSTACLE_MAX_AREA = 500000
THRESHOLD = 145
ROBOT_RADIUS = 30.0
OBSTACLE_SAFETY_MARGIN = 0.0
PLANNING_STEP = 2.0

# УПРАВЛЕНИЕ
MAX_SPEED = 0.25
SPEED_KP = 1.2
ACC_SPEED_ERROR = 5.0
MAX_OMEGA = 0.5
ANGLE_KP = 0.5
ACC_ANGLE_ERROR = 10.0
REFERENCE_ANGLE = -90.0

EDGE_LIMIT_CM = 15.0

CORNERS_FILE = "field_corners.json"

def init_video_processor():
    vp.set_field_dimensions(FIELD_WIDTH, FIELD_HEIGHT)
    vp.set_obstacle_params(EDGE_MARGIN, OBSTACLE_MIN_AREA, OBSTACLE_MAX_AREA, THRESHOLD)
    vp.set_robot_params(ROBOT_RADIUS, OBSTACLE_SAFETY_MARGIN, PLANNING_STEP, EDGE_LIMIT_CM)

    if os.path.exists(CORNERS_FILE):
        vp.load_corners(CORNERS_FILE)

def mode_camera():
    init_video_processor()
    vp.process_camera_feed(camera_id=1)

def mode_robot():
    if not connect_to_robotino():
        print("Не удалось подключиться к Robotino!")
        return

    init_video_processor()

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Не удалось открыть камеру!")
        return

    target_point = None
    planner = None
    moving = False
    rotating = False
    pending_target = None
    current_robot_pos = None

    from planners.greedy_planner import (
        create_planner, update_obstacles, draw_planning_contours,
        find_path, draw_path_on_frame, get_velocities
    )

    def get_robot_angle(marker_corners) -> float:
        if marker_corners is None:
            return 0.0
        corner_points = marker_corners.reshape(4, 2)
        dx = corner_points[1][0] - corner_points[0][0]
        dy = corner_points[1][1] - corner_points[0][1]
        return math.degrees(math.atan2(dy, dx))

    def rotate_to_reference_angle(current_angle: float) -> bool:
        delta = REFERENCE_ANGLE - current_angle
        while delta > 180:
            delta -= 360
        while delta < -180:
            delta += 360

        print(f"current={current_angle:.1f}°, target={REFERENCE_ANGLE:.1f}°, delta={delta:.1f}°")

        if abs(delta) < ACC_ANGLE_ERROR:
            return True

        omega = -math.radians(delta) * ANGLE_KP
        max_omega = MAX_OMEGA
        omega = max(-max_omega, min(omega, max_omega))

        if abs(omega) < 0.05 and abs(delta) > ACC_ANGLE_ERROR:
            omega = 0.2 if delta > 0 else -0.2

        send_velocity(0.0, 0.0, omega)
        return False

    def mouse_callback(event, x, y, _flags, _param):
        nonlocal target_point, planner, moving, rotating, pending_target
        if event == cv2.EVENT_LBUTTONDOWN:
            real_x, real_y = vp.transform_coordinates(x, y)
            target_point = (real_x, real_y)
            moving = False
            rotating = False
            pending_target = None
            planner = None
            print(f"Новая цель: ({real_x:.1f}, {real_y:.1f}) см")

    cv2.namedWindow("Robot Control", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Robot Control", 800, 800)
    cv2.setMouseCallback("Robot Control", mouse_callback)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        frame_count += 1

        h_matrix = vp._state['H']
        output_size = vp._state['output_size']
        rectified = cv2.warpPerspective(frame, h_matrix, output_size)
        rectified = vp.draw_coordinate_axes(rectified, margin=20, axis_length=60)

        if vp._state['edge_limit_cm'] > 0:
            rectified = vp.draw_edge_limit(rectified)

        found, robot_id, center_pixel, _, corners = vp.detect_robot(frame)

        if found:
            robot_x, robot_y = vp.transform_coordinates(center_pixel[0], center_pixel[1])
            current_robot_pos = (robot_x, robot_y)
        else:
            current_robot_pos = None
            if moving or rotating:
                print("Робот потерян! Останавливаем движение")
                stop_robot()
                moving = False
                rotating = False
                pending_target = None

        if found:
            obstacles = vp.detect_obstacles(rectified, robot_center=center_pixel)
        else:
            obstacles = vp.detect_obstacles(rectified, robot_center=None)

        if planner is None:
            planner = create_planner(
                field_width=FIELD_WIDTH,
                field_height=FIELD_HEIGHT,
                step=PLANNING_STEP,
                robot_radius=ROBOT_RADIUS,
                obstacle_safety=OBSTACLE_SAFETY_MARGIN,
                edge_limit_cm=EDGE_LIMIT_CM
            )

        update_obstacles(planner, obstacles)
        rectified = draw_planning_contours(planner, rectified)

        if found:
            rectified = vp.draw_axes_2d(rectified, corners, axis_length=50)
            robot_radius_px = int(ROBOT_RADIUS / FIELD_WIDTH * rectified.shape[1])
            cv2.circle(rectified, (int(center_pixel[0]), int(center_pixel[1])), robot_radius_px, (100, 100, 100), 1)

        if target_point is not None and current_robot_pos is not None and not moving and not rotating:
            current_angle = get_robot_angle(corners)

            delta = abs(REFERENCE_ANGLE - current_angle)
            while delta > 180:
                delta = 360 - delta

            if delta > ACC_ANGLE_ERROR:
                rotating = True
                pending_target = target_point
            else:
                path = find_path(planner, current_robot_pos, target_point)
                if path:
                    rectified = draw_path_on_frame(planner, rectified, path, (0, 255, 255))
                    moving = True

        if rotating and current_robot_pos is not None:
            current_angle = get_robot_angle(corners)
            if rotate_to_reference_angle(current_angle):
                path = find_path(planner, current_robot_pos, pending_target)
                if path:
                    print("Путь найден!")
                    rectified = draw_path_on_frame(planner, rectified, path, (0, 255, 255))
                    moving = True

                rotating = False
                pending_target = None
            continue

        if moving and planner and planner.get('path') and current_robot_pos is not None:
            robot_x_cm, robot_y_cm = current_robot_pos

            final_goal = planner['path'][-1]
            dist_to_final = math.hypot(final_goal[0] - robot_x_cm, final_goal[1] - robot_y_cm)

            if dist_to_final < ACC_SPEED_ERROR:
                stop_robot()
                moving = False
                planner = None
                target_point = None
                continue

            vx, vy = get_velocities(
                planner,
                robot_x_cm, robot_y_cm,
                max_speed=MAX_SPEED,
                kp=SPEED_KP,
                acceptable_error=ACC_SPEED_ERROR
            )
            send_velocity(vx, -vy, 0.0)

            if frame_count % 30 == 0:
                print(f"vx={vx:.3f}, vy={vy:.3f}, до цели={dist_to_final:.1f} см")

        if target_point is not None:
            tx = int(target_point[0] / FIELD_WIDTH * rectified.shape[1])
            ty = int(rectified.shape[0] - (target_point[1] / FIELD_HEIGHT * rectified.shape[0]))
            cv2.circle(rectified, (tx, ty), 8, (255, 0, 255), -1)
            cv2.circle(rectified, (tx, ty), 12, (255, 0, 255), 2)

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
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("1. Реальная камера")
    print("2. Управление роботом")

    choice = input("\n1 или 2? ").strip()

    if choice == '1':
        mode_camera()
    elif choice == '2':
        mode_robot()
    else:
        print("Выход")

if __name__ == "__main__":
    main()