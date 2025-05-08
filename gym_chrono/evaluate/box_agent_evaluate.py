import gymnasium as gym
from stable_baselines3 import PPO
from gym_chrono.envs.agent.box_agent import box_agent
import os

render = True
agent_render = True
if agent_render:
    env = box_agent(additional_render_mode='agent_pov')
else:
    env = box_agent()

# Update this directory/path to point to your saved box_agent PPO checkpoints.
checkpoint_dir = '../train/box_ppo_checkpoints'
# Load the model checkpoint (e.g., "ppo_checkpoint0")
loaded_model = PPO.load(os.path.join(checkpoint_dir, "ppo_checkpoint99"), env)

sim_time = 20
timeStep = 0.1
totalSteps = int(sim_time / timeStep)

# Optionally, if your box_agent has a method for setting a nicer mesh for rendering:
# if hasattr(env, "set_nice_vehicle_mesh"):
#     env.set_nice_vehicle_mesh()

obs, _ = env.reset(seed=1)
if render:
    env.render('follow')

for step in range(totalSteps):
    action, _states = loaded_model.predict(obs, deterministic=True)
    print(f"Step {step + 1}")
    print("Action: ", action)
    obs, reward, terminated, truncated, info = env.step(action)
    print("obs =", obs, "reward =", reward, "done =", (terminated or truncated))
    if render:
        env.render('follow')
    if terminated or truncated:
        break
