"""Secure JSON handling with size limits and validation."""

import json
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class JSONSecurityError(Exception):
    """Exception raised for JSON security violations."""

    pass


class SafeJSONHandler:
    """Secure JSON parser with configurable limits and validation."""

    def __init__(
        self,
        max_size: int = 100000,  # 100KB default
        max_depth: int = 20,
        max_string_length: int = 10000,
    ):
        """Initialize secure JSON handler.

        Args:
            max_size: Maximum JSON payload size in bytes
            max_depth: Maximum nesting depth
            max_string_length: Maximum string value length
        """
        self.max_size = max_size
        self.max_depth = max_depth
        self.max_string_length = max_string_length

    def loads(
        self, json_string: str, field_name: str = "json_data"
    ) -> Union[Dict, List, Any]:
        """Safely parse JSON string with security validation.

        Args:
            json_string: JSON string to parse
            field_name: Field name for error messages

        Returns:
            Parsed JSON object

        Raises:
            JSONSecurityError: If JSON violates security policies
        """
        if not isinstance(json_string, str):
            raise JSONSecurityError(
                f"{field_name} must be a string, got {type(json_string)}"
            )

        # Check size limit
        if len(json_string) > self.max_size:
            logger.warning(
                f"JSON payload too large: {len(json_string)} bytes > {self.max_size}"
            )
            raise JSONSecurityError(
                f"{field_name} payload too large: {len(json_string)} bytes exceeds limit of {self.max_size}"
            )

        # Check for obviously malicious patterns
        if json_string.count("{") > 1000 or json_string.count("[") > 1000:
            logger.warning("Potential JSON bomb detected: excessive nesting structures")
            raise JSONSecurityError(
                f"{field_name} contains excessive nesting structures"
            )

        # Parse JSON with error handling
        try:
            parsed_data = json.loads(json_string)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON format: {str(e)}")
            raise JSONSecurityError(f"{field_name} is not valid JSON: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error parsing JSON: {str(e)}")
            raise JSONSecurityError(f"Failed to parse {field_name}: {str(e)}")

        # Validate parsed structure
        self._validate_parsed_data(parsed_data, field_name)

        return parsed_data

    def dumps(self, data: Any, ensure_ascii: bool = True) -> str:
        """Safely serialize data to JSON string.

        Args:
            data: Data to serialize
            ensure_ascii: Whether to escape non-ASCII characters

        Returns:
            JSON string

        Raises:
            JSONSecurityError: If serialization fails or violates limits
        """
        try:
            json_string = json.dumps(
                data, ensure_ascii=ensure_ascii, separators=(",", ":")
            )
        except (TypeError, ValueError) as e:
            logger.error(f"JSON serialization failed: {str(e)}")
            raise JSONSecurityError(f"Failed to serialize data: {str(e)}")

        # Check result size
        if len(json_string) > self.max_size:
            logger.warning(f"Serialized JSON too large: {len(json_string)} bytes")
            raise JSONSecurityError(
                f"Serialized JSON too large: {len(json_string)} bytes exceeds limit of {self.max_size}"
            )

        return json_string

    def _validate_parsed_data(
        self, data: Any, field_name: str, current_depth: int = 0
    ) -> None:
        """Recursively validate parsed JSON data.

        Args:
            data: Parsed data to validate
            field_name: Field name for error messages
            current_depth: Current nesting depth

        Raises:
            JSONSecurityError: If data violates security policies
        """
        # Check depth limit
        if current_depth > self.max_depth:
            logger.warning(
                f"JSON depth limit exceeded: {current_depth} > {self.max_depth}"
            )
            raise JSONSecurityError(
                f"{field_name} exceeds maximum nesting depth of {self.max_depth}"
            )

        if isinstance(data, dict):
            # Validate dictionary
            if len(data) > 1000:  # Prevent huge objects
                raise JSONSecurityError(
                    f"{field_name} contains too many keys: {len(data)}"
                )

            for key, value in data.items():
                # Validate key
                if not isinstance(key, str):
                    raise JSONSecurityError(
                        f"{field_name} contains non-string key: {type(key)}"
                    )

                if len(key) > 100:  # Reasonable key length limit
                    raise JSONSecurityError(
                        f"{field_name} contains key that is too long: {len(key)} chars"
                    )

                # Recursively validate value
                self._validate_parsed_data(
                    value, f"{field_name}.{key}", current_depth + 1
                )

        elif isinstance(data, list):
            # Validate array
            if len(data) > 10000:  # Prevent huge arrays
                raise JSONSecurityError(
                    f"{field_name} contains too many items: {len(data)}"
                )

            for i, item in enumerate(data):
                # Recursively validate item
                self._validate_parsed_data(
                    item, f"{field_name}[{i}]", current_depth + 1
                )

        elif isinstance(data, str):
            # Validate string length
            if len(data) > self.max_string_length:
                raise JSONSecurityError(
                    f"{field_name} string too long: {len(data)} chars exceeds limit of {self.max_string_length}"
                )

        elif isinstance(data, (int, float)):
            # Validate numeric ranges
            if isinstance(data, float):
                if abs(data) > 1e15:  # Prevent extremely large numbers
                    raise JSONSecurityError(
                        f"{field_name} numeric value too large: {data}"
                    )

                if data != data:  # Check for NaN
                    raise JSONSecurityError(f"{field_name} contains NaN value")

        # Allow None, bool, and validated types
        elif data is not None and not isinstance(data, bool):
            logger.warning(f"Unexpected data type in JSON: {type(data)}")
            raise JSONSecurityError(
                f"{field_name} contains unsupported data type: {type(data)}"
            )


# Global safe JSON handler instance
default_json_handler = SafeJSONHandler()


def safe_loads(
    json_string: str, field_name: str = "json_data"
) -> Union[Dict, List, Any]:
    """Convenience function for safe JSON parsing."""
    return default_json_handler.loads(json_string, field_name)


def safe_dumps(data: Any) -> str:
    """Convenience function for safe JSON serialization."""
    return default_json_handler.dumps(data)
