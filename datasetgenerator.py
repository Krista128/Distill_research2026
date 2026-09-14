import pybullet as p
import time 
import pybullet_data
import numpy as np
import pickle
import matplotlib.pyplot as plt
from math import sqrt, atan2, sin, cos

class RoboticManipulator():
    def __init__(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.plane = p.loadURDF('plane.urdf')
        self.manipulator = p.loadURDF('reacher.urdf', [0, 0, 0], useFixedBase=True)
        self.target = p.loadURDF('target.urdf', [-0.5, 0, -0.001], useFixedBase=True)
        
        self.joint1 = 0
        self.joint2 = 1

        self.camera_pos = [0, -1.5, 2.5]
        self.camera_target = [0,0,0]
        self.camera_up = [0,0,1]
        self.img_width = 84
        self.img_height = 84

    def get_observation(self):
        # Получаем изображение
        view_matrix = p.computeViewMatrix(self.camera_pos, self.camera_target, self.camera_up)
        proj_matrix = p.computeProjectionMatrixFOV(60, 1, 0.1, 100)
        _, _, rgb, _, _ = p.getCameraImage(self.img_width, self.img_height,
                                           viewMatrix=view_matrix,
                                           projectionMatrix=proj_matrix,
                                           renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb = np.reshape(rgb, (self.img_height, self.img_width, 4))[:,:,:3]  # RGBA -> RGB
        return rgb 
    
    def reset(self):
        p.resetSimulation()
        p.setGravity(0,0,-10)
        self.plane = p.loadURDF("plane.urdf")
        self.manipulator = p.loadURDF("reacher.urdf", [0,0,0], useFixedBase=True)
        self.target = p.loadURDF('target.urdf', [0, 0, 0], useFixedBase=True)  # ← ДОБАВИТЬ
    
    # Случайная позиция цели
        angle = np.random.uniform(0, 2*np.pi)
        radius = np.random.uniform(0.3, 0.9)
        target_x = radius * np.cos(angle)
        target_y = radius * np.sin(angle)
        p.resetBasePositionAndOrientation(self.target, [target_x, target_y, 0.02], [0,0,0,1])
        # Начальные углы
        p.setJointMotorControl2(self.manipulator, self.joint1, p.POSITION_CONTROL, targetPosition=0, force=1)
        p.setJointMotorControl2(self.manipulator, self.joint2, p.POSITION_CONTROL, targetPosition=0, force=1)
        for _ in range(10): p.stepSimulation()
        return self.get_observation()

    
    def start_simulation(self):
        for i in range(100000):
            p.stepSimulation()
            time.sleep(1./240.)

def ik(x, y, L1=0.5, L2=0.5):
    c2 = (x**2 + y**2 - L1**2 - L2**2) / (2*L1*L2)
    s2 = sqrt(1 - c2**2) 
    theta2 = atan2(s2, c2)
    theta1 = atan2(y, x) - atan2(L2*sin(theta2), L1 + L2*cos(theta2))
    return theta1, theta2

# ========== НАСТРОЙКИ СБОРА ДАННЫХ ==========
success_threshold = 0.13
Kp = 8
keep_failed_ratio = 0.3          # Сохраняем 30% неудачных, чтобы показать модели, как НЕ надо
max_episodes_to_save = 300      # Сколько эпизодов собрать (не кадров!)

# Датасет будет списком эпизодов
dataset_episodes = []  # Каждый элемент: {'obs': (T,84,84,3), 'actions': (T,2), 'targets': (T,2), 'success': bool}

reacher = RoboticManipulator()
episode_count = 0
saved_count = 0

print(f"Цель: собрать {max_episodes_to_save} эпизодов...")

while saved_count < max_episodes_to_save:
    episode_count += 1
    
    # ----- СБРОС -----
    obs = reacher.reset()
    
    # Локальные списки ДЛЯ ТЕКУЩЕЙ ПОСЛЕДОВАТЕЛЬНОСТИ
    seq_obs = []
    seq_actions = []
    seq_targets = []
    episode_success = False
    
    # ----- СИМУЛЯЦИЯ (до 100 шагов) -----
    for step in range(100):
        # 1. Сохраняем ТЕКУЩЕЕ наблюдение (кадр) в последовательность
        seq_obs.append(obs.copy())
        
        # 2. Цель
        target_pos, _ = p.getBasePositionAndOrientation(reacher.target)
        x, y = target_pos[0], target_pos[1]
        seq_targets.append([x, y])
        
        # 3. Управление (IK + П-регулятор)
        theta1_curr, _, _, _ = p.getJointState(reacher.manipulator, reacher.joint1)
        theta2_curr, _, _, _ = p.getJointState(reacher.manipulator, reacher.joint2)
        theta1_des, theta2_des = ik(x, y)
        
        action1 = Kp * (theta1_des - theta1_curr)
        action2 = Kp * (theta2_des - theta2_curr)
        seq_actions.append([action1, action2])
        
        # 4. Применяем действие
        p.setJointMotorControl2(reacher.manipulator, reacher.joint1, p.TORQUE_CONTROL, force=action1)
        p.setJointMotorControl2(reacher.manipulator, reacher.joint2, p.TORQUE_CONTROL, force=action2)
        
        for _ in range(5):
            p.stepSimulation()
        
        # 5. Получаем новое состояние для следующего шага
        obs = reacher.get_observation()
        
        # 6. Проверка касания
        end_effector_pos = p.getLinkState(reacher.manipulator, 2)[0]
        distance = np.linalg.norm(np.array(end_effector_pos[:2]) - np.array([x, y]))
        
        if distance < success_threshold:
            episode_success = True
            break  # Эпизод завершен успешно
    
    # ----- РЕШЕНИЕ: СОХРАНЯТЬ ЭПИЗОД ИЛИ НЕТ? -----
    save_episode = False
    if episode_success:
        save_episode = True
    else:
        # Неудачные сохраняем с вероятностью keep_failed_ratio
        if np.random.random() < keep_failed_ratio:
            save_episode = True
    
    if save_episode:
        # Превращаем списки в numpy-массивы для удобства
        episode_data = {
            'observations': np.array(seq_obs, dtype=np.uint8),  # (T, 84, 84, 3)
            'actions': np.array(seq_actions, dtype=np.float32), # (T, 2)
            'targets': np.array(seq_targets, dtype=np.float32), # (T, 2)
            'success': episode_success,
            'length': len(seq_obs)
        }
        dataset_episodes.append(episode_data)
        saved_count += 1
        
        status = "✅ SUCCESS" if episode_success else "❌ FAILED (kept)"
        print(f"[{saved_count}/{max_episodes_to_save}] Ep.{episode_count}: {status}, frames={len(seq_obs)}")


with open('reacher_episodes_val.pkl', 'wb') as f:
    pickle.dump(dataset_episodes, f)

print("Датасет сохранён в 'reacher_episodes_val.pkl'")