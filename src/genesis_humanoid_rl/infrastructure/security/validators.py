"""Security validation framework for input sanitization and validation."""

import re
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception raised for security validation failures."""

    pass


@dataclass
class ValidationRule:
    """Defines a security validation rule."""

    name: str
    pattern: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    allowed_types: Optional[tuple] = None
    required: bool = False


class SecurityValidator:
    """Comprehensive security validator for input sanitization."""

    # Security patterns
    SQL_INJECTION_PATTERNS = [
        r"('|(\\')|(;|(\s*;\s*))|(--)|(\s*--\s*))",
        r"(\b(select|union|insert|update|delete|drop|create|alter|exec|execute)\b)",
        r"(\b(script|javascript|vbscript|onload|onerror)\b)",
    ]

    # Safe patterns
    UUID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    SKILL_TYPE_PATTERN = r"^[a-z_]+$"
    NUMERIC_PATTERN = r"^-?\d+(\.\d+)?$"

    def __init__(self, max_string_length: int = 1000, max_json_size: int = 100000):
        """Initialize security validator with limits.

        Args:
            max_string_length: Maximum allowed string length
            max_json_size: Maximum allowed JSON payload size in bytes
        """
        self.max_string_length = max_string_length
        self.max_json_size = max_json_size

        # Compile patterns for performance
        self.sql_injection_regexes = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.SQL_INJECTION_PATTERNS
        ]
        self.uuid_regex = re.compile(self.UUID_PATTERN, re.IGNORECASE)
        self.skill_type_regex = re.compile(self.SKILL_TYPE_PATTERN)
        self.numeric_regex = re.compile(self.NUMERIC_PATTERN)

    def validate_string_input(self, value: str, field_name: str = "input") -> str:
        """Validate and sanitize string input.

        Args:
            value: String to validate
            field_name: Name of field for error messages

        Returns:
            Sanitized string value

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string, got {type(value)}")

        # Check length limits
        if len(value) > self.max_string_length:
            raise ValidationError(
                f"{field_name} exceeds maximum length of {self.max_string_length} characters"
            )

        # Check for SQL injection patterns
        for regex in self.sql_injection_regexes:
            if regex.search(value):
                logger.warning(
                    f"SQL injection attempt detected in {field_name}: {value[:50]}..."
                )
                raise ValidationError(
                    f"{field_name} contains potentially malicious content"
                )

        # Sanitize and return
        return value.strip()

    def validate_uuid(self, value: str, field_name: str = "id") -> str:
        """Validate UUID format.

        Args:
            value: UUID string to validate
            field_name: Name of field for error messages

        Returns:
            Validated UUID string

        Raises:
            ValidationError: If UUID format is invalid
        """
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string, got {type(value)}")

        if not self.uuid_regex.match(value):
            raise ValidationError(f"{field_name} is not a valid UUID format")

        return value.lower()

    def validate_skill_type(self, value: str, field_name: str = "skill_type") -> str:
        """Validate skill type format.

        Args:
            value: Skill type string to validate
            field_name: Name of field for error messages

        Returns:
            Validated skill type string

        Raises:
            ValidationError: If skill type format is invalid
        """
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string, got {type(value)}")

        if not self.skill_type_regex.match(value):
            raise ValidationError(f"{field_name} contains invalid characters")

        if len(value) > 50:  # Reasonable limit for skill types
            raise ValidationError(
                f"{field_name} exceeds maximum length of 50 characters"
            )

        return value

    def validate_numeric(
        self,
        value: Union[int, float, str],
        field_name: str = "number",
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> float:
        """Validate numeric input.

        Args:
            value: Numeric value to validate
            field_name: Name of field for error messages
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Validated numeric value as float

        Raises:
            ValidationError: If validation fails
        """
        # Convert to float if string
        if isinstance(value, str):
            if not self.numeric_regex.match(value):
                raise ValidationError(f"{field_name} is not a valid number format")
            try:
                value = float(value)
            except ValueError:
                raise ValidationError(f"{field_name} cannot be converted to number")

        if not isinstance(value, (int, float)):
            raise ValidationError(f"{field_name} must be a number, got {type(value)}")

        # Check for NaN and infinity
        if isinstance(value, float) and (value != value):  # NaN check
            raise ValidationError(f"{field_name} cannot be NaN")

        if isinstance(value, float) and abs(value) == float("inf"):
            raise ValidationError(f"{field_name} cannot be infinite")

        # Check bounds
        if min_value is not None and value < min_value:
            raise ValidationError(f"{field_name} must be >= {min_value}, got {value}")

        if max_value is not None and value > max_value:
            raise ValidationError(f"{field_name} must be <= {max_value}, got {value}")

        return float(value)

    def validate_proficiency_score(self, value: Union[int, float, str]) -> float:
        """Validate proficiency score (0.0 to 1.0).

        Args:
            value: Proficiency score to validate

        Returns:
            Validated proficiency score

        Raises:
            ValidationError: If score is out of range
        """
        return self.validate_numeric(value, "proficiency_score", 0.0, 1.0)

    def validate_confidence_level(self, value: Union[int, float, str]) -> float:
        """Validate confidence level (0.0 to 1.0).

        Args:
            value: Confidence level to validate

        Returns:
            Validated confidence level

        Raises:
            ValidationError: If confidence is out of range
        """
        return self.validate_numeric(value, "confidence_level", 0.0, 1.0)

    def validate_velocity(self, value: Union[int, float, str]) -> float:
        """Validate robot velocity (-10.0 to 10.0 m/s).

        Args:
            value: Velocity to validate

        Returns:
            Validated velocity

        Raises:
            ValidationError: If velocity is out of reasonable range
        """
        return self.validate_numeric(value, "velocity", -10.0, 10.0)

    def validate_dictionary(
        self, data: Dict[str, Any], rules: Dict[str, ValidationRule]
    ) -> Dict[str, Any]:
        """Validate a dictionary against defined rules.

        Args:
            data: Dictionary to validate
            rules: Validation rules for each field

        Returns:
            Validated and sanitized dictionary

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(data, dict):
            raise ValidationError(f"Expected dictionary, got {type(data)}")

        result = {}

        # Check required fields
        for field_name, rule in rules.items():
            if rule.required and field_name not in data:
                raise ValidationError(f"Required field '{field_name}' is missing")

        # Validate present fields
        for field_name, value in data.items():
            if field_name not in rules:
                logger.warning(f"Unknown field '{field_name}' in input data")
                continue  # Skip unknown fields for security

            rule = rules[field_name]

            # Type validation
            if rule.allowed_types and not isinstance(value, rule.allowed_types):
                raise ValidationError(
                    f"Field '{field_name}' must be of type {rule.allowed_types}, got {type(value)}"
                )

            # String validation
            if isinstance(value, str):
                validated_value = self.validate_string_input(value, field_name)

                # Length validation
                if rule.min_length and len(validated_value) < rule.min_length:
                    raise ValidationError(
                        f"Field '{field_name}' must be at least {rule.min_length} characters"
                    )

                if rule.max_length and len(validated_value) > rule.max_length:
                    raise ValidationError(
                        f"Field '{field_name}' must be at most {rule.max_length} characters"
                    )

                # Pattern validation
                if rule.pattern:
                    pattern_regex = re.compile(rule.pattern)
                    if not pattern_regex.match(validated_value):
                        raise ValidationError(
                            f"Field '{field_name}' does not match required pattern"
                        )

                result[field_name] = validated_value
            else:
                result[field_name] = value

        return result

    def sanitize_log_message(self, message: str) -> str:
        """Sanitize log messages to prevent log injection.

        Args:
            message: Log message to sanitize

        Returns:
            Sanitized log message
        """
        if not isinstance(message, str):
            return str(message)

        # Remove newlines and carriage returns to prevent log injection
        sanitized = message.replace("\n", " ").replace("\r", " ")

        # Limit length to prevent log flooding
        if len(sanitized) > 500:
            sanitized = sanitized[:497] + "..."

        return sanitized


# Global validator instance
default_validator = SecurityValidator()


def validate_string(value: str, field_name: str = "input") -> str:
    """Convenience function for string validation."""
    return default_validator.validate_string_input(value, field_name)


def validate_uuid(value: str, field_name: str = "id") -> str:
    """Convenience function for UUID validation."""
    return default_validator.validate_uuid(value, field_name)


def validate_proficiency(value: Union[int, float, str]) -> float:
    """Convenience function for proficiency score validation."""
    return default_validator.validate_proficiency_score(value)
