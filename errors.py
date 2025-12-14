"""
Error handling system for SI Data Generator application.

This module provides comprehensive error handling, logging, and retry mechanisms
for the application, including custom exception classes, error codes, and an
ErrorHandler class for centralized error management.
"""

import json
import random
import re
import time
import traceback
from enum import Enum
from typing import Any, Callable, Dict, Optional


class ErrorSeverity(Enum):
    """Error severity levels for classification and logging"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ErrorCode(Enum):
    """Error code classifications for different failure scenarios"""
    CORTEX_UNAVAILABLE = "CORTEX_001"
    CORTEX_TIMEOUT = "CORTEX_002"
    CORTEX_INVALID_RESPONSE = "CORTEX_003"
    DATA_GENERATION_FAILED = "DATA_004"
    SCHEMA_CREATION_FAILED = "DATA_005"
    TABLE_CREATION_FAILED = "DATA_006"
    SEARCH_SERVICE_FAILED = "SEARCH_007"
    SEMANTIC_VIEW_FAILED = "VIEW_008"
    AGENT_CREATION_FAILED = "AGENT_009"
    VALIDATION_FAILED = "VALID_010"
    PERMISSION_DENIED = "AUTH_011"
    NETWORK_ERROR = "NET_012"
    SCHEMA_VALIDATION_FAILED = "VALID_013"
    UNKNOWN_ERROR = "UNKNOWN_999"


class RecoverableError(Exception):
    """Error that can be recovered from with retry logic"""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        details: Optional[Dict] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)


class FatalError(Exception):
    """Error that cannot be recovered from and requires intervention"""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        details: Optional[Dict] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)


class CortexServiceError(RecoverableError):
    """Cortex-specific service errors that may be transient"""
    pass


class DataGenerationError(RecoverableError):
    """Data generation errors that may succeed on retry"""
    pass


class ErrorHandler:
    """
    Central error handling and logging system.
    
    Provides methods for logging errors to Snowflake tables, converting error
    codes to user-friendly messages, and managing error state throughout the
    application lifecycle.
    """
    
    def __init__(self, session=None):
        """
        Initialize error handler with optional Snowflake session.
        
        Args:
            session: Snowflake session for error logging (optional)
        """
        self.session = session
        self.error_log_table = "SI_DEMOS.APPLICATIONS.ERROR_LOGS"
        self.ensure_error_table_exists()
    
    def ensure_error_table_exists(self):
        """Create error logging table if it doesn't exist"""
        if self.session is None:
            return
        
        try:
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS SI_DEMOS.APPLICATIONS.ERROR_LOGS (
                log_id NUMBER AUTOINCREMENT,
                timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                error_code STRING,
                error_type STRING,
                severity STRING,
                message STRING,
                stack_trace STRING,
                user_context VARIANT,
                session_id STRING,
                function_name STRING,
                retry_count NUMBER DEFAULT 0
            )
            """
            self.session.sql(create_table_sql).collect()
        except Exception:
            pass
    
    def log_error(
        self,
        error_code: ErrorCode,
        error_type: str,
        severity: ErrorSeverity,
        message: str,
        stack_trace: Optional[str] = None,
        user_context: Optional[Dict] = None,
        session_id: Optional[str] = None,
        function_name: Optional[str] = None,
        retry_count: int = 0
    ):
        """
        Log error to Snowflake event table.
        
        Args:
            error_code: Error code classification
            error_type: Type of error (e.g., exception class name)
            severity: Error severity level
            message: Human-readable error message
            stack_trace: Full stack trace (optional)
            user_context: Additional context information (optional)
            session_id: Session identifier (optional)
            function_name: Name of function where error occurred (optional)
            retry_count: Number of retry attempts (optional)
        """
        if self.session is None:
            return
        
        try:
            log_entry = {
                'error_code': error_code.value,
                'error_type': error_type,
                'severity': severity.value,
                'message': message,
                'stack_trace': stack_trace or '',
                'user_context': json.dumps(user_context or {}),
                'session_id': session_id or '',
                'function_name': function_name or '',
                'retry_count': retry_count
            }
            
            insert_sql = f"""
            INSERT INTO {self.error_log_table} 
            (error_code, error_type, severity, message, stack_trace, 
             user_context, session_id, function_name, retry_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            self.session.sql(
                insert_sql,
                params=[
                    log_entry['error_code'],
                    log_entry['error_type'],
                    log_entry['severity'],
                    log_entry['message'],
                    log_entry['stack_trace'],
                    log_entry['user_context'],
                    log_entry['session_id'],
                    log_entry['function_name'],
                    log_entry['retry_count']
                ]
            ).collect()
        except Exception:
            pass
    
    def get_user_friendly_message(
        self,
        error_code: ErrorCode,
        original_error: str = ""
    ) -> Dict[str, str]:
        """
        Convert error codes to user-friendly messages with actionable guidance.
        
        Args:
            error_code: Error code to translate
            original_error: Original error message for context (optional)
            
        Returns:
            Dictionary with 'title', 'message', and 'action' keys
        """
        messages = {
            ErrorCode.CORTEX_UNAVAILABLE: {
                "title": "AI Service Temporarily Unavailable",
                "message": (
                    "The Snowflake Cortex AI service is currently "
                    "unavailable. Don't worry - we'll use template data "
                    "to continue."
                ),
                "action": (
                    "Your demo will be created with standard templates. "
                    "You can regenerate later for AI-customized content."
                )
            },
            ErrorCode.CORTEX_TIMEOUT: {
                "title": "AI Response Taking Longer Than Expected",
                "message": (
                    "The AI service is responding slowly. We're trying "
                    "again with optimized settings."
                ),
                "action": (
                    "This usually resolves in a few seconds. If it "
                    "persists, try reducing the number of records."
                )
            },
            ErrorCode.CORTEX_INVALID_RESPONSE: {
                "title": "AI Response Format Issue",
                "message": (
                    "The AI service returned data in an unexpected format. "
                    "Using fallback templates."
                ),
                "action": (
                    "Your demo will still be created successfully with "
                    "high-quality template data."
                )
            },
            ErrorCode.DATA_GENERATION_FAILED: {
                "title": "Data Generation Issue",
                "message": "There was a problem generating the demo data.",
                "action": (
                    "Please try again. If the issue persists, try with "
                    "fewer records or contact support."
                )
            },
            ErrorCode.SCHEMA_CREATION_FAILED: {
                "title": "Schema Creation Issue",
                "message": "Unable to create the database schema.",
                "action": (
                    "Please check that you have necessary permissions and "
                    "try a different schema name."
                )
            },
            ErrorCode.TABLE_CREATION_FAILED: {
                "title": "Table Creation Issue",
                "message": (
                    "Unable to create one or more tables in the database."
                ),
                "action": (
                    "Verify you have CREATE TABLE permissions in the "
                    "selected database and schema."
                )
            },
            ErrorCode.SEARCH_SERVICE_FAILED: {
                "title": "Search Service Creation Issue",
                "message": (
                    "The Cortex Search service could not be created."
                ),
                "action": (
                    "The demo will work without search. Ensure you have "
                    "Cortex Search privileges to enable this feature."
                )
            },
            ErrorCode.SEMANTIC_VIEW_FAILED: {
                "title": "Semantic View Creation Issue",
                "message": (
                    "The semantic view could not be created, but your "
                    "tables are ready to use."
                ),
                "action": (
                    "You can still query the tables directly. Try "
                    "recreating the semantic view separately."
                )
            },
            ErrorCode.AGENT_CREATION_FAILED: {
                "title": "Agent Creation Issue",
                "message": (
                    "The AI agent could not be created automatically."
                ),
                "action": (
                    "Your demo data is ready. You can create an agent "
                    "manually using the generated tables."
                )
            },
            ErrorCode.PERMISSION_DENIED: {
                "title": "Permission Required",
                "message": (
                    "You don't have the necessary permissions for this "
                    "operation."
                ),
                "action": (
                    "Contact your Snowflake administrator to grant the "
                    "required privileges."
                )
            },
            ErrorCode.NETWORK_ERROR: {
                "title": "Network Connection Issue",
                "message": (
                    "There was a problem connecting to Snowflake services."
                ),
                "action": (
                    "Check your network connection and try again in a moment."
                )
            },
            ErrorCode.SCHEMA_VALIDATION_FAILED: {
                "title": "Schema Validation Issue",
                "message": "Could not validate the database schema.",
                "action": "The operation will continue with default settings."
            },
            ErrorCode.UNKNOWN_ERROR: {
                "title": "Unexpected Issue",
                "message": (
                    f"An unexpected error occurred: {original_error[:100]}"
                ),
                "action": (
                    "Please try again. If the problem persists, contact "
                    "support with the error details."
                )
            }
        }
        
        return messages.get(error_code, messages[ErrorCode.UNKNOWN_ERROR])


def retry_with_exponential_backoff(
    func: Callable,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    error_handler: Optional[ErrorHandler] = None,
    function_name: Optional[str] = None
) -> Any:
    """
    Retry a function with exponential backoff strategy.
    
    Implements exponential backoff with optional jitter to prevent thundering
    herd problem. Logs retry attempts if error_handler is provided.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff calculation
        jitter: Add random jitter to delay
        error_handler: Optional error handler for logging
        function_name: Name of function for logging
        
    Returns:
        Result of successful function call
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            result = func()
            
            if attempt > 0 and error_handler:
                error_handler.log_error(
                    error_code=ErrorCode.CORTEX_TIMEOUT,
                    error_type="RecoverableError",
                    severity=ErrorSeverity.INFO,
                    message=f"Function succeeded after {attempt} retries",
                    function_name=function_name or func.__name__,
                    retry_count=attempt
                )
            
            return result
            
        except Exception as e:
            last_exception = e
            
            if attempt == max_retries:
                if error_handler:
                    error_handler.log_error(
                        error_code=ErrorCode.CORTEX_TIMEOUT,
                        error_type=type(e).__name__,
                        severity=ErrorSeverity.ERROR,
                        message=(
                            f"Function failed after {max_retries} retries: "
                            f"{str(e)}"
                        ),
                        stack_trace=traceback.format_exc(),
                        function_name=function_name or func.__name__,
                        retry_count=attempt
                    )
                break
            
            delay = min(
                initial_delay * (exponential_base ** attempt),
                max_delay
            )
            
            if jitter:
                delay = delay * (0.5 + random.random())
            
            if error_handler:
                error_handler.log_error(
                    error_code=ErrorCode.CORTEX_TIMEOUT,
                    error_type=type(e).__name__,
                    severity=ErrorSeverity.WARNING,
                    message=(
                        f"Retry attempt {attempt + 1}/{max_retries} after "
                        f"{delay:.2f}s delay: {str(e)}"
                    ),
                    function_name=function_name or func.__name__,
                    retry_count=attempt
                )
            
            time.sleep(delay)
    
    raise last_exception


def check_cortex_availability(session) -> bool:
    """
    Check if Cortex service is available and responding.
    
    Args:
        session: Snowflake session to test Cortex availability
        
    Returns:
        True if Cortex is available, False otherwise
    """
    try:
        test_query = """
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            'llama3.1-8b',
            'test'
        ) as response
        """
        result = session.sql(test_query).collect()
        return result is not None and len(result) > 0
    except Exception:
        return False


def safe_json_parse(json_str: str, fallback: Any = None) -> Any:
    """
    Safely parse JSON string with fallback on error.
    
    Handles common LLM response formats including markdown code blocks
    and extracts JSON objects or arrays from text.
    
    Args:
        json_str: String potentially containing JSON
        fallback: Value to return if parsing fails
        
    Returns:
        Parsed JSON object/array or fallback value
    """
    try:
        # Try to extract JSON from markdown code block
        json_match = re.search(r'```json\n(.*?)\n```', json_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object
            json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            # Try to find JSON array
            elif re.search(r'\[.*\]', json_str, re.DOTALL):
                json_match = re.search(r'\[.*\]', json_str, re.DOTALL)
                json_str = json_match.group(0)
        
        return json.loads(json_str)
    except (json.JSONDecodeError, AttributeError, TypeError):
        return fallback

"""
Error handling decorators for SI Data Generator.

This module provides reusable error handling decorators to replace
repetitive try-catch blocks throughout the codebase, improving
code readability and consistency.
"""

import functools
import streamlit as st
import traceback
from typing import Callable, Any, Optional

from errors import ErrorHandler, ErrorCode, ErrorSeverity


def handle_errors(
    error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    user_message: Optional[str] = None,
    return_on_error: Any = None,
    show_in_ui: bool = True
):
    """
    Decorator to handle errors with consistent logging and user feedback.
    
    This decorator wraps functions with standard error handling, eliminating
    the need for repetitive try-catch blocks. Errors are logged to the error
    handler and optionally displayed in the Streamlit UI.
    
    Args:
        error_code: ErrorCode to log when exception occurs
        severity: ErrorSeverity level for logging
        user_message: Optional custom message to display to user
        return_on_error: Value to return if error occurs (default: None)
        show_in_ui: Whether to display error in Streamlit UI (default: True)
    
    Returns:
        Decorator function
    
    Example:
        @handle_errors(
            error_code=ErrorCode.DATA_GENERATION_FAILED,
            severity=ErrorSeverity.ERROR,
            user_message="Failed to generate data",
            return_on_error=[]
        )
        def generate_data(schema, num_records):
            # ... function implementation
            return data
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Try to get error_handler from args/kwargs (common pattern)
                error_handler = None
                for arg in args:
                    if isinstance(arg, ErrorHandler):
                        error_handler = arg
                        break
                if not error_handler and 'error_handler' in kwargs:
                    error_handler = kwargs['error_handler']
                
                # Log error if handler available
                if error_handler:
                    error_handler.log_error(
                        error_code=error_code,
                        error_type=type(e).__name__,
                        severity=severity,
                        message=f"Error in {func.__name__}: {str(e)}",
                        stack_trace=traceback.format_exc(),
                        function_name=func.__name__
                    )
                
                # Display in UI if requested
                if show_in_ui:
                    if user_message:
                        st.error(f"{user_message}: {str(e)[:100]}")
                    else:
                        if error_handler:
                            msg = error_handler.get_user_friendly_message(error_code)
                            st.error(f"{msg['title']}: {str(e)[:100]}")
                        else:
                            st.error(f"Error in {func.__name__}: {str(e)[:100]}")
                
                return return_on_error
        
        return wrapper
    return decorator


def handle_llm_errors(
    return_on_error: Any = None,
    show_warning: bool = True
):
    """
    Specialized decorator for LLM/Cortex operations.
    
    Handles common LLM errors like timeouts, invalid responses, and rate limits
    with appropriate user messaging.
    
    Args:
        return_on_error: Value to return if LLM call fails
        show_warning: Whether to show warning in UI (default: True)
    
    Returns:
        Decorator function
    
    Example:
        @handle_llm_errors(return_on_error={})
        def call_llm(prompt, session):
            response = session.sql(
                "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?)",
                params=[model, prompt]
            ).collect()
            return parse_response(response)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e).lower()
                
                # Categorize LLM errors
                if 'timeout' in error_msg or 'timed out' in error_msg:
                    if show_warning:
                        st.warning(f"⏱️ LLM request timed out in {func.__name__}. Retrying...")
                elif 'rate limit' in error_msg or 'too many requests' in error_msg:
                    if show_warning:
                        st.warning(f"🚦 Rate limit reached. Slowing down requests...")
                elif 'invalid' in error_msg or 'parse' in error_msg:
                    if show_warning:
                        st.warning(f"⚠️ Invalid LLM response in {func.__name__}. Using fallback...")
                else:
                    if show_warning:
                        st.warning(f"⚠️ LLM error in {func.__name__}: {str(e)[:100]}")
                
                return return_on_error
        
        return wrapper
    return decorator


def handle_database_errors(
    return_on_error: Any = None,
    show_error: bool = True
):
    """
    Specialized decorator for database operations.
    
    Handles common database errors like connection issues, permission errors,
    and query failures with appropriate user messaging.
    
    Args:
        return_on_error: Value to return if database operation fails
        show_error: Whether to show error in UI (default: True)
    
    Returns:
        Decorator function
    
    Example:
        @handle_database_errors(return_on_error=False)
        def create_table(session, schema_name, table_name):
            session.sql(f"CREATE TABLE {schema_name}.{table_name} ...").collect()
            return True
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e).lower()
                
                # Categorize database errors
                if 'does not exist' in error_msg or 'not found' in error_msg:
                    if show_error:
                        st.error(f"❌ Object not found in {func.__name__}: {str(e)[:150]}")
                elif 'permission' in error_msg or 'access denied' in error_msg:
                    if show_error:
                        st.error(f"🔒 Permission denied in {func.__name__}. Check your role privileges.")
                elif 'already exists' in error_msg:
                    if show_error:
                        st.warning(f"ℹ️ Object already exists in {func.__name__}. Continuing...")
                    # For "already exists", might want to return success
                    return return_on_error if return_on_error is not None else True
                elif 'connection' in error_msg or 'network' in error_msg:
                    if show_error:
                        st.error(f"🌐 Connection error in {func.__name__}. Check your network.")
                else:
                    if show_error:
                        st.error(f"❌ Database error in {func.__name__}: {str(e)[:150]}")
                
                return return_on_error
        
        return wrapper
    return decorator


def suppress_exceptions(
    exceptions: tuple = (Exception,),
    return_value: Any = None,
    log_to_console: bool = False
):
    """
    Decorator to suppress specific exceptions and return a default value.
    
    Useful for non-critical operations where failures should be silent.
    
    Args:
        exceptions: Tuple of exception types to suppress
        return_value: Value to return when exception is suppressed
        log_to_console: Whether to print exception to console (default: False)
    
    Returns:
        Decorator function
    
    Example:
        @suppress_exceptions(exceptions=(ValueError, KeyError), return_value=[])
        def get_optional_data():
            # ... might raise ValueError or KeyError
            return data
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if log_to_console:
                    print(f"Suppressed exception in {func.__name__}: {type(e).__name__}: {str(e)}")
                return return_value
        
        return wrapper
    return decorator


def retry_on_failure(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    exceptions: tuple = (Exception,),
    show_retry_message: bool = True
):
    """
    Decorator to retry function on failure with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay_seconds: Initial delay between retries (doubles each attempt)
        exceptions: Tuple of exception types to retry on
        show_retry_message: Whether to show retry messages in UI
    
    Returns:
        Decorator function
    
    Example:
        @retry_on_failure(max_attempts=3, delay_seconds=2.0)
        def flaky_api_call():
            response = requests.get("https://api.example.com/data")
            return response.json()
    """
    import time
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay_seconds
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts:
                        if show_retry_message:
                            st.warning(
                                f"⚠️ Attempt {attempt}/{max_attempts} failed for {func.__name__}. "
                                f"Retrying in {current_delay:.1f}s..."
                            )
                        time.sleep(current_delay)
                        current_delay *= 2  # Exponential backoff
                    else:
                        if show_retry_message:
                            st.error(
                                f"❌ All {max_attempts} attempts failed for {func.__name__}: "
                                f"{str(e)[:100]}"
                            )
            
            # All attempts failed, raise the last exception
            raise last_exception
        
        return wrapper
    return decorator


def validate_inputs(**validators):
    """
    Decorator to validate function inputs before execution.
    
    Args:
        **validators: Keyword arguments mapping parameter names to validation functions
                      Each validation function should take the parameter value and return
                      (is_valid: bool, error_message: str)
    
    Returns:
        Decorator function
    
    Example:
        def validate_positive(value):
            if value > 0:
                return True, ""
            return False, "Value must be positive"
        
        @validate_inputs(num_records=validate_positive)
        def generate_data(num_records, schema):
            # ... function implementation
            return data
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function parameter names
            import inspect
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            
            # Build args dict
            args_dict = {}
            for i, arg in enumerate(args):
                if i < len(param_names):
                    args_dict[param_names[i]] = arg
            args_dict.update(kwargs)
            
            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in args_dict:
                    value = args_dict[param_name]
                    is_valid, error_msg = validator(value)
                    
                    if not is_valid:
                        st.error(f"❌ Invalid input for {param_name} in {func.__name__}: {error_msg}")
                        raise ValueError(f"Invalid {param_name}: {error_msg}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

