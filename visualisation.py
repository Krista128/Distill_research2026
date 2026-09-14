import numpy as np
import pybullet as p
import torch
from datasetgenerator import RoboticManipulator as robomanip
from model import TeacherModel as teacher
import matplotlib.pyplot as plt

def plot_trajectory(model, env, max_steps=100):
    model.eval()
    obs = env.reset()
    trajectory = []
    device = next(model.parameters()).device
    target_pos, _ = p.getBasePositionAndOrientation(env.target)
    x_target, y_target = target_pos[0], target_pos[1]
    
    print(f"Target position: ({x_target}, {y_target})")  # Отладка

    for step in range(max_steps):
        end_eff = p.getLinkState(env.manipulator, 2)[0]
        trajectory.append([end_eff[0], end_eff[1]])

        goal = torch.tensor([[x_target, y_target]], dtype=torch.float32).to(device)
        img = torch.from_numpy(obs).permute(2, 0, 1).float() / 255.0
        img = torch.norm(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action = model(img, goal).cpu().numpy()[0]

        p.setJointMotorControl2(env.manipulator, env.joint1, p.TORQUE_CONTROL, force=action[0])
        p.setJointMotorControl2(env.manipulator, env.joint2, p.TORQUE_CONTROL, force=action[1])
        for _ in range(5): 
            p.stepSimulation()
        obs = env.get_observation()

        dist = np.linalg.norm(np.array([end_eff[0], end_eff[1]]) - np.array([x_target, y_target]))
        print(f"Step {step}: distance = {dist:.4f}")  # Отладка
        
        if dist < 0.13:
            print(f"Target reached at step {step}")
            break

    trajectory = np.array(trajectory)
    print(f"Trajectory shape: {trajectory.shape}")  # Отладка
    
    plt.figure(figsize=(6, 6))
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-o', label='Схват')
    plt.plot(x_target, y_target, 'r*', markersize=20, label='Цель')
    plt.plot(0, 0, 'ks', markersize=10, label='База')
    plt.axis('equal')
    plt.legend()
    plt.title('Траектория схвата')
    plt.grid(True)
    
    # Сохраните график в файл
    plt.savefig('trajectory.png', dpi=150, bbox_inches='tight')
    print("Graph saved as trajectory.png")
    
    plt.show()

# Вызовите функцию

env = robomanip()
model = teacher()
model.load_state_dict(torch.load('teacher_resnet18.pth', map_location='gpu'))
model.eval()

plot_trajectory(model, env)
