from abc import ABC, abstractmethod

class ISensorFusion(ABC):

    @abstractmethod
    def predict(self, dt: float): ...
    @abstractmethod
    def update(self, meas: dict): ...
    @abstractmethod
    def get_state(self) -> dict: ...
