from abc import ABC, abstractmethod


# --- Base Class for Option Pricing ---
class OptionPricing(ABC):
    """
    Abstract base class for option pricing models.
    """

    @abstractmethod
    def calculate_option_price(self, T: float = 1) -> float:
        """
        Abstract method to calculate the option price.
        Must be implemented by subclasses.
        """
        pass
