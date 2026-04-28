from dataclasses import dataclass
import pickle

@dataclass
class TrainingConfig:
    batch_size: int
    max_cpu_count: int
    epochs: int
    learning_rate: float
    weight_decay: float
    loss_fn: callable
    eval_fn: callable
    optimizer: str
    mask: bool = False
    clip: bool = False
    max_distance: float = 200.0
    
    @property
    def num_workers(self) -> int:
        return min(self.max_cpu_count, self.batch_size)
        
    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(filepath: str) -> 'TrainingConfig':
        with open(filepath, 'rb') as f:
            return pickle.load(f)