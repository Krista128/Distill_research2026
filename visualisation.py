import numpy as np
import pybullet as p
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from datasetgenerator import RoboticManipulator as robomanip
from model import TeacherModel as student
import pickle


def plot_trajectory(model, env, actions_mean, actions_std,
                    targets_mean, targets_std, max_steps=200):
    model.eval()
    obs = env.reset()
    trajectory = []
    device = next(model.parameters()).device

    target_pos, _ = p.getBasePositionAndOrientation(env.target)
    x_target, y_target = target_pos[0], target_pos[1]

    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    episode_success = False
    dist = 999.0   

    for step in range(max_steps):
        img = torch.from_numpy(obs).permute(2, 0, 1).float() / 255.0
        img = transform(img).unsqueeze(0).to(device)

        goal_raw = np.array([x_target, y_target], dtype=np.float32)
        goal_norm = (goal_raw - targets_mean) / targets_std
        goal = torch.tensor(goal_norm, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            action_norm = model(img, goal).cpu().numpy()[0]

        action_real = action_norm * actions_std + actions_mean

        p.setJointMotorControl2(env.manipulator, env.joint1,
                                p.TORQUE_CONTROL, force=action_real[0])
        p.setJointMotorControl2(env.manipulator, env.joint2,
                                p.TORQUE_CONTROL, force=action_real[1])
        for _ in range(5):
            p.stepSimulation()

        end_eff = p.getLinkState(env.manipulator, 2)[0]
        trajectory.append([end_eff[0], end_eff[1]])

        dist = np.linalg.norm(np.array(end_eff[:2]) - goal_raw)
        

        obs = env.get_observation()

        if dist < 0.13:
            episode_success = True
            break

    success = 0
    if episode_success:
        success = 1
        print(f"✅ SUCCESS")
    else:
        print(f"❌ FAILED")


    return episode_success, dist, success


with open('reacher_episodes_full.pkl', 'rb') as f:
    frames = pickle.load(f)

observ  = np.concatenate([ep['observations'] for ep in frames], axis=0)
actions = np.concatenate([ep['actions'] for ep in frames], axis=0)
targets = np.concatenate([ep['targets'] for ep in frames], axis=0)

actions_mean = actions.mean(axis=0).astype(np.float32)
actions_std  = actions.std(axis=0).astype(np.float32) + 1e-8
targets_mean = targets.mean(axis=0).astype(np.float32)
targets_std  = targets.std(axis=0).astype(np.float32) + 1e-8


env = robomanip()
model = student()
state_dict = torch.load('teacher.pth', map_location='cuda', weights_only=False)
model.load_state_dict(state_dict['state_dict'])
model.eval()
count = 0
for i in range (50):
    success, final_dist, res = plot_trajectory(
        model, env, actions_mean, actions_std, targets_mean, targets_std
    )
    count += res
p.disconnect()
print(f"\nИтог: success={success}, final_dist={final_dist:.4f}, res={count}")