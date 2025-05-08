import gymnasium as gym
from gym_chrono.envs.agent.box_agent import box_agent
import time
from PIL import Image
import numpy as np
import torch
render = True
additional_render = True


def save_combined_image(observation, filename):
        """
        Save a combined image of the RGB camera data and depth data side-by-side.
        
        Parameters:
        observation (dict): Dictionary with keys "image" and "depth". 
            - "image" is expected to be an array/tensor of shape (3, H, W).
            - "depth" is expected to be an array/tensor of shape (H, W) with values in [0,1].
        filename (str): Path (including name and extension) to save the image.
        """
        # Process RGB image:
        rgb_img = observation["image"]
        # If the image is a torch tensor, convert to numpy:
        if isinstance(rgb_img, torch.Tensor):
            rgb_img = rgb_img.cpu().numpy()
        # The RGB image is in (C, H, W) format, so transpose it to (H, W, C)
        rgb_img = np.transpose(rgb_img, (1, 2, 0))
        # Ensure values are in uint8 (if they already are, this is a no-op)
        rgb_img = rgb_img.astype(np.uint8)
        
        # Process depth image:
        depth_img = observation["depth"]
        # If the depth is a torch tensor, convert to numpy:
        if isinstance(depth_img, torch.Tensor):
            depth_img = depth_img.cpu().numpy()
        # Assuming depth values are normalized between 0 and 1, scale to 0-255 and convert to uint8.
        depth_img = np.clip(depth_img, 0, 1)  # ensure in range
        depth_img = (depth_img * 255).astype(np.uint8)
        # Optionally, convert the single-channel depth image to a 3-channel image for compatibility.
        depth_img_color = np.stack([depth_img] * 3, axis=-1)
        
        # Combine the images side-by-side (horizontally)
        combined_img = np.concatenate((rgb_img, depth_img_color), axis=1)
        
        # Save the combined image using PIL
        im = Image.fromarray(combined_img)
        im.save(filename)


if __name__ == '__main__':
    # Add the agent POV as a render mode
    if additional_render:
        env = box_agent(additional_render_mode='agent_pov')
    else:
        env = box_agent()
    # check_env(env)
    # Set the mode to render for nice viz
    # env.set_nice_vehicle_mesh()
    obs, _ = env.reset()
    if render:
        env.render('follow')

    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.sample())
    time.sleep(2)
    print("here")
    # Hardcoded best agent: always go left!
    n_steps = 1000000
    for step in range(n_steps):
        print(f"Step {step + 1}")
        if (step < 10):
            obs, reward, terminated, truncated, info = env.step(4)
        elif (step > 10 and step < 10000):
            obs, reward, terminated, truncated, info = env.step(24)
        elif (step > 100000):
            obs, reward, terminated, truncated, info = env.step(1)

        save_combined_image(obs, "test_imgs/combined_output" + str(step) + ".png")
        print("Terminated=", terminated, "Truncated=", truncated)
        done = terminated or truncated
        if render:
            env.render('follow')
        if done:
            print("reward=", reward)
            break
        # time.sleep(0.5)

