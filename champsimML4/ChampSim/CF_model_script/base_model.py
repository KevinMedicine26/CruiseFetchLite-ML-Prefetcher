from abc import ABC, abstractmethod

class MLPrefetchModel(ABC):
    """
    Abstract base class for prefetch models. For HW-based approaches,
    you can directly add your prediction code. For ML models, you may want
    to use it as a wrapper, but alternative approaches are fine so long as
    the behavior described below is respected.
    """

    @abstractmethod
    def load(self, path):
        """
        Loads your model from the filepath path
        """
        pass

    @abstractmethod
    def save(self, path):
        """
        Saves your model to the filepath path
        """
        pass

    @abstractmethod
    def train(self, data):
        """
        Train your model here. No return value. The data parameter is in the
        same format as the load traces. Namely,
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        """
        pass

    @abstractmethod
    def generate(self, data):
        """
        Generate your prefetches here. Remember to limit yourself to 2 prefetches
        for each instruction ID and to not look into the future :).

        The return format for this will be a list of tuples containing the
        unique instruction ID and the prefetch. For example,
        [
            (A, A1),
            (A, A2),
            (C, C1),
            ...
        ]

        where A, B, and C are the unique instruction IDs and A1, A2 and C1 are
        the prefetch addresses.
        """
        pass
