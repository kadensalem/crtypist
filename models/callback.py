import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from tqdm import tqdm

class TrainingCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, check_freq: int, log_dir: str, save_path, verbose=1):
        super(TrainingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        self.best_time_steps = 0
        self.pbar = None

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        # Initialize progress bar
        # Remove timesteps that were done in previous training sessions
        self.pbar = tqdm(total=self.locals["total_timesteps"] - self.model.num_timesteps)

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last episodes
                mean_reward = np.mean(y[-50:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} at {self.best_time_steps} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.best_time_steps = self.num_timesteps
                    # Example for saving best model
                    #   if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)
        # Update progress bar, we do num_envs steps per call to `env.step()`
        self.pbar.update(self.training_env.num_envs)
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        # Flush and close progress bar
        self.pbar.refresh()
        self.pbar.close()