import numpy as np
  
class ScheduledOptimizer():
    def __init__(self, optimizer, initial_lr, n_warmup_steps, current_steps):

        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = current_steps
        self.init_lr = initial_lr

    def _get_lr_scale(self):

        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self, iteration):
    
        self.n_current_steps = iteration if iteration > 0 else 1
        lr = self.init_lr * self._get_lr_scale()

        for i, param_group in enumerate(self._optimizer.param_groups):
            param_group['lr'] = lr

    def step(self, iteration):
    
        self._update_learning_rate(iteration)

    def get_learning_rate(self):
        
        for param_group in self._optimizer.param_groups:
            learning_rate = param_group['lr']

        return learning_rate

    def state_dict(self):

        return self._optimizer.state_dict()

    def load_state_dict(self, dict):

        self._optimizer.load_state_dict(dict)

