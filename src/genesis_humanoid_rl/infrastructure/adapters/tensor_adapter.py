"""
Tensor Compatibility Adapter.
Isolates PyTorch tensor dependencies from domain layer.
"""

import numpy as np
from typing import Union, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# Type aliases for clarity
DomainNumericValue = Union[float, int, np.ndarray, List[float]]
TensorType = Any  # Will be torch.Tensor when PyTorch is available


class TensorAdapter:
    """
    Adapter for converting between domain numeric values and framework tensors.

    Provides a clean boundary between domain logic and PyTorch dependencies,
    allowing the domain to work with standard Python/NumPy types while
    infrastructure can use optimized tensor operations.
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize tensor adapter.

        Args:
            device: Target device for tensor operations ('cpu' or 'cuda')
        """
        self.device = device
        self._torch_available = False

        # Try to import PyTorch
        try:
            import torch

            self._torch = torch
            self._torch_available = True
            logger.info(f"PyTorch available, using device: {device}")
        except ImportError:
            logger.warning("PyTorch not available, using NumPy fallback")
            self._torch = None

    def to_tensor(self, value: DomainNumericValue) -> TensorType:
        """
        Convert domain numeric value to tensor.

        Args:
            value: Domain numeric value (float, int, numpy array, or list)

        Returns:
            Tensor representation suitable for framework operations
        """
        if not self._torch_available:
            # Fallback to NumPy
            return np.asarray(value, dtype=np.float32)

        try:
            # Convert to PyTorch tensor
            if isinstance(value, (int, float)):
                return self._torch.tensor(
                    value, dtype=self._torch.float32, device=self.device
                )
            elif isinstance(value, np.ndarray):
                return self._torch.from_numpy(value.astype(np.float32)).to(self.device)
            elif isinstance(value, list):
                return self._torch.tensor(
                    value, dtype=self._torch.float32, device=self.device
                )
            elif hasattr(value, "to"):  # Already a tensor
                return value.to(self.device)
            else:
                # Try generic conversion
                return self._torch.tensor(
                    value, dtype=self._torch.float32, device=self.device
                )

        except Exception as e:
            logger.error(f"Failed to convert to tensor: {e}")
            # Fallback to NumPy
            return np.asarray(value, dtype=np.float32)

    def to_domain(self, tensor: TensorType) -> DomainNumericValue:
        """
        Convert tensor back to domain-friendly format.

        Args:
            tensor: Framework tensor

        Returns:
            Domain numeric value (typically numpy array or scalar)
        """
        if not self._torch_available:
            # Already NumPy
            return tensor

        try:
            if hasattr(tensor, "cpu") and hasattr(tensor, "numpy"):
                # PyTorch tensor
                return tensor.cpu().detach().numpy()
            elif hasattr(tensor, "item"):
                # Scalar tensor
                return tensor.item()
            else:
                # Already in domain format or unknown
                return tensor

        except Exception as e:
            logger.error(f"Failed to convert from tensor: {e}")
            return tensor

    def sqrt(self, value: DomainNumericValue) -> DomainNumericValue:
        """
        Compute square root with tensor compatibility.

        Args:
            value: Input value

        Returns:
            Square root in domain format
        """
        if not self._torch_available:
            return np.sqrt(np.asarray(value))

        try:
            tensor_value = self.to_tensor(value)
            result_tensor = self._torch.sqrt(tensor_value)
            return self.to_domain(result_tensor)
        except Exception as e:
            logger.error(f"Failed tensor sqrt operation: {e}")
            # Fallback to NumPy
            return np.sqrt(np.asarray(value))

    def sum(self, value: DomainNumericValue, dim: int = None) -> DomainNumericValue:
        """
        Compute sum with tensor compatibility.

        Args:
            value: Input value
            dim: Dimension to sum over (None for all)

        Returns:
            Sum in domain format
        """
        if not self._torch_available:
            return np.sum(np.asarray(value), axis=dim)

        try:
            tensor_value = self.to_tensor(value)
            if dim is not None:
                result_tensor = self._torch.sum(tensor_value, dim=dim)
            else:
                result_tensor = self._torch.sum(tensor_value)
            return self.to_domain(result_tensor)
        except Exception as e:
            logger.error(f"Failed tensor sum operation: {e}")
            return np.sum(np.asarray(value), axis=dim)

    def mean(self, value: DomainNumericValue, dim: int = None) -> DomainNumericValue:
        """
        Compute mean with tensor compatibility.

        Args:
            value: Input value
            dim: Dimension to compute mean over (None for all)

        Returns:
            Mean in domain format
        """
        if not self._torch_available:
            return np.mean(np.asarray(value), axis=dim)

        try:
            tensor_value = self.to_tensor(value)
            if dim is not None:
                result_tensor = self._torch.mean(tensor_value, dim=dim)
            else:
                result_tensor = self._torch.mean(tensor_value)
            return self.to_domain(result_tensor)
        except Exception as e:
            logger.error(f"Failed tensor mean operation: {e}")
            return np.mean(np.asarray(value), axis=dim)

    def clip(
        self, value: DomainNumericValue, min_val: float, max_val: float
    ) -> DomainNumericValue:
        """
        Clip values with tensor compatibility.

        Args:
            value: Input value
            min_val: Minimum value
            max_val: Maximum value

        Returns:
            Clipped value in domain format
        """
        if not self._torch_available:
            return np.clip(np.asarray(value), min_val, max_val)

        try:
            tensor_value = self.to_tensor(value)
            result_tensor = self._torch.clamp(tensor_value, min_val, max_val)
            return self.to_domain(result_tensor)
        except Exception as e:
            logger.error(f"Failed tensor clip operation: {e}")
            return np.clip(np.asarray(value), min_val, max_val)

    def is_tensor_available(self) -> bool:
        """Check if tensor operations are available."""
        return self._torch_available

    def get_device(self) -> str:
        """Get current device."""
        return self.device


# Global adapter instance
default_adapter = TensorAdapter()


def to_tensor(value: DomainNumericValue) -> TensorType:
    """Convenience function for tensor conversion."""
    return default_adapter.to_tensor(value)


def to_domain(tensor: TensorType) -> DomainNumericValue:
    """Convenience function for domain conversion."""
    return default_adapter.to_domain(tensor)


def safe_sqrt(value: DomainNumericValue) -> DomainNumericValue:
    """Safe square root operation."""
    return default_adapter.sqrt(value)


def safe_sum(value: DomainNumericValue, dim: int = None) -> DomainNumericValue:
    """Safe sum operation."""
    return default_adapter.sum(value, dim)


def safe_mean(value: DomainNumericValue, dim: int = None) -> DomainNumericValue:
    """Safe mean operation."""
    return default_adapter.mean(value, dim)


def safe_clip(
    value: DomainNumericValue, min_val: float, max_val: float
) -> DomainNumericValue:
    """Safe clip operation."""
    return default_adapter.clip(value, min_val, max_val)
