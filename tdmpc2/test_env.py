from envs import make_maniskill3_env
import hydra
import time


@hydra.main(config_name="config", config_path=".")
def run(cfg: dict):
    cfg["render_mode"] = "human"
    cfg["task"] = "pick-ycb-multiview"
    env = make_maniskill3_env(cfg)
    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    env.reset(seed=0)  # reset with a seed for determinism
    for _ in range(10):
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            time.sleep(0.01)
            env.render()  # a display is required to render
    env.close()


if __name__ == "__main__":
    run()
