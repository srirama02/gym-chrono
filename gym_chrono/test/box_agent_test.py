import gymnasium as gym
from gym_chrono.envs.agent.box_agent import box_agent

render = True
additional_render = True
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

    # Hardcoded best agent: always go left!
    n_steps = 1000000
    for step in range(n_steps):
        print(f"Step {step + 1}")
        if (step < 1):
            obs, reward, terminated, truncated, info = env.step(1)
        # elif (step > 100):
        #     obs, reward, terminated, truncated, info = env.step(1)
        print("Terminated=", terminated, "Truncated=", truncated)
        done = terminated or truncated
        if render:
            env.render('follow')
        if done:
            print("reward=", reward)
            break
