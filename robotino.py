import time
import math
import requests
from typing import Tuple, Optional, Dict

# ========== НАСТРОЙКИ ==========
IP_ADDRESS = '192.168.0.1'  # IP робота
BASE_URL = f"http://{IP_ADDRESS}"
# ===============================

def connect_to_robotino() -> bool:
    try:
        response = requests.get(f"{BASE_URL}/data/odometry", timeout=0.5)
        if response.status_code == 200:
            print("✅ Успешное соединение с Robotino!")
            return True
        else:
            print(f"❌ Ошибка соединения: статус {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ошибка соединения: {e}")
        return False

def get_odometry() -> Optional[Dict]:
    try:
        url = f"{BASE_URL}/data/odometry"
        response = requests.get(url, timeout=0.1)

        if response.status_code == 200:
            data = response.json()
            if len(data) == 7:
                return {
                    'x': data[0],  # метры
                    'y': data[1],  # метры
                    'angle': data[2],  # радианы
                    'vx': data[3],  # м/с
                    'vy': data[4],  # м/с
                    'omega': data[5],  # рад/с
                    'timestamp': data[6]  # секунды
                }
            else:
                print(f" Неожиданный формат одометрии: {len(data)} значений")
        else:
            print(f" Ошибка получения одометрии: статус {response.status_code}")
    except Exception as e:
        print(f" Ошибка получения одометрии: {e}")

    return None

def get_proximity_sensor_values() -> Optional[list]:
    try:
        url = f"{BASE_URL}/data/distancesensorarray"
        response = requests.get(url, timeout=0.1)

        if response.status_code == 200:
            sensor_values = response.json()
            if len(sensor_values) == 9:
                return sensor_values
            else:
                print(f" Неожиданное количество датчиков: {len(sensor_values)}")
        else:
            print(f" Ошибка получения датчиков: статус {response.status_code}")
    except Exception as e:
        print(f" Ошибка получения датчиков: {e}")

    return None


def send_velocity(vx: float, vy: float, omega: float = 0.0) -> bool:
    url = f"{BASE_URL}/data/omnidrive"
    data = [vx, vy, omega]

    # Отладка
    print(f"  📤 Sending: vx={vx:.3f}, vy={vy:.3f}, omega={omega:.3f}")

    try:
        response = requests.post(url, json=data, timeout=0.1)
        if response.status_code == 200:
            return True
        else:
            print(f" Ошибка отправки скорости: статус {response.status_code}")
            return False
    except Exception as e:
        print(f" Ошибка отправки скорости: {e}")
        return False

def stop_robot() -> bool:
    return send_velocity(0.0, 0.0, 0.0)

def follow_path(planner: dict,
                goal_tolerance: float = 5.0,
                max_speed: float = 0.5,
                kp: float = 0.8,
                lookahead_distance: float = 15.0,
                control_freq: float = 20.0) -> bool:
    """
    Следовать по спланированному пути
    """
    from planners.greedy_planner import  get_velocities

    if not planner.get('path') or len(planner['path']) < 2:
        print("  ❌ Нет спланированного пути!")
        return False

    print(f"\n  🚀 Начинаем движение по пути из {len(planner['path'])} точек")
    print(f"  📍 Цель: {planner['path'][-1]}")
    print(f"  ⚙️  Параметры: max_speed={max_speed} м/с, kp={kp}, точность={goal_tolerance} см")
    print("  ⚠️ Нажмите Ctrl+C для остановки\n")

    loop_delay = 1.0 / control_freq
    step_count = 0

    try:
        while True:
            # 1. Получаем текущую позицию
            odom = get_odometry()
            if odom is None:
                print("  ⚠️ Не удалось получить одометрию, пробуем ещё раз...")
                time.sleep(loop_delay)
                continue

            # Переводим метры в сантиметры для планировщика
            current_x_cm = odom['x'] * 100.0
            current_y_cm = odom['y'] * 100.0

            # 2. Получаем скорости от планировщика
            vx, vy = get_velocities(
                planner,
                current_x_cm, current_y_cm,
                max_speed=max_speed,
                lookahead_distance=lookahead_distance,
                kp=kp,
                goal_tolerance=goal_tolerance
            )

            # ОТЛАДКА: выводим информацию о скоростях
            step_count += 1
            if step_count % 10 == 0:  # Каждые 10 итераций
                print(f"  📊 [step {step_count}] pos: ({current_x_cm:.1f}, {current_y_cm:.1f}) см")
                print(f"     vx={vx:.3f} м/с, vy={vy:.3f} м/с")
                print(f"     target: ({planner['path'][-1][0]:.1f}, {planner['path'][-1][1]:.1f}) см")
                print(f"     dist_to_goal: {math.hypot(planner['path'][-1][0] - current_x_cm, planner['path'][-1][1] - current_y_cm):.1f} см")

            # 3. Отправляем команду роботу
            success = send_velocity(vx, vy, 0.0)
            if not success and step_count % 20 == 0:
                print("  ⚠️ Проблема с отправкой команды")

            # 4. Проверяем, достигли ли цели
            goal = planner['path'][-1]
            dx = goal[0] - current_x_cm
            dy = goal[1] - current_y_cm
            dist_to_goal = math.hypot(dx, dy)

            if dist_to_goal < goal_tolerance:
                stop_robot()
                print(f"\n  ✅ Цель достигнута! Ошибка: {dist_to_goal:.1f} см")
                return True

            time.sleep(loop_delay)

    except KeyboardInterrupt:
        print("\n  ⏹️ Движение остановлено пользователем")
        stop_robot()
        return False
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        stop_robot()
        return False