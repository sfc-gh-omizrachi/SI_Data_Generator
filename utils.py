"""
Core utilities and constants for SI Data Generator application.

This module consolidates configuration constants, session management,
and core utility functions used throughout the application.
"""

import json
import streamlit as st
import traceback
from pathlib import Path
from typing import Optional, Dict, List
from snowflake.snowpark import Session
from snowflake.snowpark.context import get_active_session
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from metrics import timeit

from errors import (
    ErrorHandler,
    ErrorCode,
    ErrorSeverity,
    retry_with_exponential_backoff,
    safe_json_parse
)
from prompts import get_demo_generation_prompt


# ============================================================================
# CONSTANTS
# ============================================================================

LLM_MODEL = "claude-4-sonnet"
"""Default LLM model for Cortex operations"""

SNOWFLAKE_COLORS = {
    "primary": "#29B5E8",
    "secondary": "#056fb7",
    "success": "#0EA5E9",
    "warning": "#FFC107",
    "error": "#DC3545",
    "info": "#17a2b8"
}
"""Snowflake brand colors used throughout the UI"""

# Data Generation Constants
MAX_FACTS_PER_TABLE = 5
"""Maximum number of fact columns per table"""

MAX_TOTAL_FACTS = 8
"""Maximum total number of facts across all tables"""

TABLE_JOIN_OVERLAP_PERCENTAGE = 0.7
"""Percentage of entity IDs that overlap between joined tables (0.7 = 70%)"""

MAX_DIMENSIONS_PER_TABLE = 6
"""Maximum number of dimension columns per table"""

MAX_TOTAL_DIMENSIONS = 10
"""Maximum total number of dimensions across all tables"""

# Question Generation Constants
DEFAULT_QUESTION_COUNT = 12
"""Default number of demo questions to generate"""

MAX_FOLLOW_UP_QUESTIONS = 3
"""Maximum number of follow-up questions per primary question"""

MAX_QUESTION_CHAINS = 6
"""Maximum number of question chains to build"""

# LLM and API Constants
MAX_RETRY_ATTEMPTS = 3
"""Maximum number of retry attempts for LLM calls"""

MAX_PARALLEL_LLM_CALLS = 3
"""Maximum number of parallel LLM calls to avoid overwhelming Cortex"""

CACHE_TTL_COMPANY_URL = 3600
"""Cache TTL in seconds for company URL analysis (1 hour)"""

CACHE_TTL_DEMO_IDEAS = 1800
"""Cache TTL in seconds for demo idea generation (30 minutes)"""

# Data Sampling Constants
SAMPLE_DATA_LIMIT = 10
"""Maximum number of sample rows to retrieve from tables"""

DEFAULT_NUMERIC_RANGE_MAX = 1000
"""Default maximum value for numeric columns"""

DATE_LOOKBACK_DAYS = 365
"""Number of days to look back when generating date values"""

# Unstructured Data Constants
MIN_CHUNKS_PER_DOCUMENT = 3
"""Minimum number of chunks per document"""

MAX_CHUNKS_PER_DOCUMENT = 8
"""Maximum number of chunks per document"""

CHUNK_TEXT_MIN_LENGTH = 50
"""Minimum characters in a text chunk"""

CHUNK_TEXT_MAX_LENGTH = 500
"""Maximum characters in a text chunk"""

# Database Query Constants
BATCH_SIZE_DESCRIBE_TABLES = 50
"""Maximum tables to describe in a single batch operation"""

BATCH_SIZE_GRANTS = 100
"""Maximum grant statements to execute in a single batch"""


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

@st.cache_resource
def create_local_session():
    """
    Create Snowflake session with proper authentication.
    
    Attempts to create session in this order:
    1. Native App mode (with snowflake.permissions)
    2. Streamlit in Snowflake mode (get_active_session)
    3. Local mode (using connection_config.json)
    
    Returns:
        Snowflake Session object
    """
    try:
        import snowflake.permissions as permissions
        session = get_active_session()
        st.session_state["streamlit_mode"] = "NativeApp"
        return session
    except:
        try:
            session = get_active_session()
            st.session_state["streamlit_mode"] = "SiS"
            return session
        except:
            config_path = Path(__file__).parent / "connection_config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            connection_params = {
                "account": config["account"],
                "user": config["user"],
            }
            
            if "private_key_path" in config and config["private_key_path"]:
                with open(config["private_key_path"], "rb") as key_file:
                    private_key = serialization.load_pem_private_key(
                        key_file.read(),
                        password=None,
                        backend=default_backend()
                    )
            
                pkb = private_key.private_bytes(
                    encoding=serialization.Encoding.DER,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
            
                connection_params["private_key"] = pkb
            else:
                if "_password" in config:
                    connection_params["password"] = config["_password"]
                else:
                    st.error(
                        "No authentication method configured. Please set "
                        "either private_key_path or _password in "
                        "connection_config.json"
                    )
                    st.stop()
            
            if "runtime" in config:
                runtime = config["runtime"]
                if "role" in runtime:
                    connection_params["role"] = runtime["role"]
                if "warehouse" in runtime:
                    connection_params["warehouse"] = runtime["warehouse"]
                if "database" in runtime:
                    connection_params["database"] = runtime["database"]
                if "schema" in runtime:
                    connection_params["schema"] = runtime["schema"]
            
            try:
                session = Session.builder.configs(connection_params).create()
            
                if "runtime" in config and "warehouse" in config["runtime"]:
                    warehouse = config["runtime"]["warehouse"]
                    session.sql(f"USE WAREHOUSE {warehouse}").collect()
            
                if "runtime" in config:
                    if "database" in config["runtime"]:
                        session.sql(
                            f"USE DATABASE {config['runtime']['database']}"
                        ).collect()
               
                    if "schema" in config["runtime"]:
                        session.sql(
                            f"USE SCHEMA {config['runtime']['schema']}"
                        ).collect()
                
                return session
            except Exception as e:
                st.error(f"Failed to connect to Snowflake: {str(e)}")
                st.stop()


def call_cortex_with_retry(
    prompt: str,
    session,
    error_handler: ErrorHandler,
    model: str = None
) -> Optional[str]:
    """
    Call Snowflake Cortex LLM with retry logic and error handling.
    
    Args:
        prompt: Prompt to send to LLM
        session: Snowflake session
        error_handler: ErrorHandler instance for logging
        model: LLM model to use (defaults to LLM_MODEL constant)
        
    Returns:
        LLM response text or None if all attempts fail
    """
    if model is None:
        model = LLM_MODEL
    
    def cortex_call():
        result = session.sql(
            "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) as response",
            params=[model, prompt]
        ).collect()
        return result[0]['RESPONSE']
    
    try:
        response = retry_with_exponential_backoff(
            func=cortex_call,
            max_retries=3,
            initial_delay=1.0,
            error_handler=error_handler,
            function_name="call_cortex"
        )
        return response
    except Exception as e:
        error_handler.log_error(
            error_code=ErrorCode.CORTEX_UNAVAILABLE,
            error_type="CortexServiceError",
            severity=ErrorSeverity.WARNING,
            message=f"Cortex call failed: {str(e)}",
            stack_trace=traceback.format_exc(),
            function_name="call_cortex_with_retry"
        )
        
        user_message = error_handler.get_user_friendly_message(
            ErrorCode.CORTEX_UNAVAILABLE, str(e)
        )
        return None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_company_info_from_url(company_url: str) -> Dict[str, str]:
    """
    Extract basic company information from URL.
    
    Args:
        company_url: Company website URL
        
    Returns:
        Dictionary with name, domain, and url keys
    """
    domain = (
        company_url.replace("https://", "")
        .replace("http://", "")
        .replace("www.", "")
        .split("/")[0]
        .split(".")[0]
    )
    company_name = domain.upper()
    
    return {
        "name": company_name,
        "domain": domain,
        "url": company_url
    }


@st.cache_data(ttl=1800, show_spinner=False)
@timeit
def generate_demo_ideas_with_cortex(
    company_name: str,
    team_members: str,
    use_cases: str,
    _session,
    _error_handler,
    num_ideas: int = 3,
    target_questions: Optional[List[str]] = None,
    advanced_mode: bool = False
) -> Optional[List[Dict]]:
    """
    Generate demo ideas using Cortex LLM.
    
    CACHED FUNCTION: Results cached for 30 minutes to avoid redundant LLM calls
    for the same input parameters.
    
    Args:
        company_name: Name of the company
        team_members: Target audience description
        use_cases: Use cases to address
        _session: Snowflake session (prefixed with _ to skip hashing)
        _error_handler: ErrorHandler instance (prefixed with _ to skip hashing)
        num_ideas: Number of demo ideas to generate
        target_questions: Optional list of questions to answer
        advanced_mode: Whether to generate advanced schemas
        
    Returns:
        List of demo idea dictionaries or None if generation fails
    """
    # Restore original parameter names for use in function body
    session = _session
    error_handler = _error_handler
    # Use prompt from prompts module
    prompt = get_demo_generation_prompt(
        company_name=company_name,
        team_members=team_members,
        use_cases=use_cases,
        num_ideas=num_ideas,
        target_questions=target_questions,
        advanced_mode=advanced_mode
    )

    try:
        response = call_cortex_with_retry(prompt, session, error_handler)
        
        if response:
            parsed = safe_json_parse(response)
            if parsed and "demos" in parsed:
                demos = parsed.get("demos", [])
                
                # Debug: Log the number of demos returned
                st.info(f"🔍 LLM generated {len(demos)} demo scenario(s)")
                
                # Add target audience and customization to each demo
                for demo in demos:
                    demo['target_audience'] = (
                        f"Designed for presentation to: {team_members}"
                    )
                    if use_cases:
                        demo['customization'] = f"Tailored for: {use_cases}"
                
                # Ensure we have the requested number of demos
                if len(demos) >= num_ideas:
                    return demos[:num_ideas]  # Return exactly num_ideas demos
                elif len(demos) > 0:
                    st.warning(
                        f"⚠️ Only {len(demos)} demo(s) generated. "
                        f"Expected {num_ideas}. Using what we got."
                    )
                    return demos  # Return what we got if it's at least 1
            else:
                st.warning(
                    "⚠️ Could not parse LLM response or missing 'demos' key"
                )
    except Exception as e:
        error_handler.log_error(
            error_code=ErrorCode.CORTEX_INVALID_RESPONSE,
            error_type=type(e).__name__,
            severity=ErrorSeverity.ERROR,
            message=f"Error generating demo ideas: {str(e)}",
            stack_trace=traceback.format_exc(),
            function_name="generate_demo_ideas_with_cortex"
        )
        st.error(f"Error in generate_demo_ideas_with_cortex: {str(e)}")
    
    return None

"""
Parallel LLM execution utilities for SI Data Generator.

This module provides helpers for executing multiple LLM calls concurrently
using ThreadPoolExecutor to significantly reduce total execution time.
"""

import functools
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Dict, List, Tuple, Callable, Any, Optional
import streamlit as st


def execute_parallel_llm_calls(
    tasks: Dict[str, Tuple[Callable, List[Any]]],
    max_workers: int = 3
) -> Dict[str, Any]:
    """
    Execute multiple LLM calls in parallel using ThreadPoolExecutor.
    
    This function enables concurrent execution of independent LLM calls,
    reducing total execution time from sequential (sum of all calls) to
    parallel (max of all calls).
    
    Args:
        tasks: Dictionary mapping task keys to (function, args) tuples
               Example: {'table1': (generate_schema, ['TABLE1', 'desc', ...]), ...}
        max_workers: Maximum number of concurrent workers (default: 3)
                    Limited to avoid overwhelming Snowflake Cortex
    
    Returns:
        Dictionary mapping task keys to their results
        
    Example:
        >>> tasks = {
        ...     'table1': (generate_schema_for_table, ['TABLE1', 'desc', company, session, handler]),
        ...     'table2': (generate_schema_for_table, ['TABLE2', 'desc', company, session, handler])
        ... }
        >>> results = execute_parallel_llm_calls(tasks, max_workers=2)
        >>> # Both schemas generated concurrently
    """
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_key = {
            executor.submit(func, *args): key
            for key, (func, args) in tasks.items()
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception as e:
                # Store exception as result so caller can handle it
                results[key] = e
                st.warning(f"⚠️ Error in parallel task '{key}': {str(e)[:100]}")
    
    return results


def execute_parallel_llm_calls_with_progress(
    tasks: Dict[str, Tuple[Callable, List[Any]]],
    progress_callback: Optional[Callable[[str], None]] = None,
    max_workers: int = 3
) -> Dict[str, Any]:
    """
    Execute multiple LLM calls in parallel with progress updates.
    
    Similar to execute_parallel_llm_calls but calls progress_callback
    when each task completes, useful for updating UI progress indicators.
    
    Args:
        tasks: Dictionary mapping task keys to (function, args) tuples
        progress_callback: Optional callback function called with task key on completion
        max_workers: Maximum number of concurrent workers
    
    Returns:
        Dictionary mapping task keys to their results
    """
    results = {}
    completed_count = 0
    total_count = len(tasks)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {
            executor.submit(func, *args): key
            for key, (func, args) in tasks.items()
        }
        
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            completed_count += 1
            
            try:
                results[key] = future.result()
                if progress_callback:
                    progress_callback(f"Completed {key} ({completed_count}/{total_count})")
            except Exception as e:
                results[key] = e
                st.warning(f"⚠️ Error in parallel task '{key}': {str(e)[:100]}")
                if progress_callback:
                    progress_callback(f"Failed {key} ({completed_count}/{total_count})")
    
    return results


def parallelize_function_calls(
    func: Callable,
    args_list: List[List[Any]],
    max_workers: int = 3
) -> List[Any]:
    """
    Execute the same function multiple times with different arguments in parallel.
    
    Simpler interface when calling the same function repeatedly with different args.
    
    Args:
        func: Function to call multiple times
        args_list: List of argument lists for each call
        max_workers: Maximum number of concurrent workers
    
    Returns:
        List of results in same order as args_list
        
    Example:
        >>> results = parallelize_function_calls(
        ...     generate_follow_up,
        ...     [['question1'], ['question2'], ['question3']],
        ...     max_workers=3
        ... )
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(func, *args) for args in args_list]
        results = [future.result() for future in futures]
    
    return results


def safe_parallel_execute(
    func: Callable,
    args_list: List[List[Any]],
    max_workers: int = 3,
    fallback_value: Any = None
) -> List[Any]:
    """
    Execute function calls in parallel with safe exception handling.
    
    Returns fallback_value for any call that raises an exception instead
    of propagating the exception.
    
    Args:
        func: Function to call
        args_list: List of argument lists
        max_workers: Maximum concurrent workers
        fallback_value: Value to return if a call fails (default: None)
    
    Returns:
        List of results (or fallback values for failed calls)
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(func, *args) for args in args_list]
        
        for i, future in enumerate(futures):
            try:
                results.append(future.result())
            except Exception as e:
                st.warning(f"⚠️ Call {i+1} failed: {str(e)[:100]}")
                results.append(fallback_value)
    
    return results

"""
Helper functions to eliminate code duplication in SI Data Generator.

This module provides reusable helper functions for common operations
that were previously duplicated throughout the codebase.
"""

import json
import re
import streamlit as st
from typing import Dict, Any, Optional, List

from utils import LLM_MODEL
from errors import ErrorCode, ErrorSeverity, safe_json_parse


def call_cortex_and_parse(
    prompt: str,
    session,
    error_handler,
    expected_key: Optional[str] = None,
    return_on_error: Any = None,
    model: str = LLM_MODEL
) -> Optional[Dict[str, Any]]:
    """
    Call Snowflake Cortex LLM and parse JSON response.
    
    This helper eliminates the repetitive pattern of:
    1. Calling Cortex with a prompt
    2. Extracting the response
    3. Parsing JSON from the response
    4. Handling errors
    
    Args:
        prompt: LLM prompt text
        session: Snowflake session
        error_handler: ErrorHandler instance
        expected_key: Optional key that must be present in parsed JSON
        return_on_error: Value to return if call fails
        model: LLM model to use (default: LLM_MODEL constant)
    
    Returns:
        Parsed JSON dictionary or return_on_error if failed
    
    Example:
        result = call_cortex_and_parse(
            prompt="Generate a list of items...",
            session=session,
            error_handler=error_handler,
            expected_key="items",
            return_on_error={"items": []}
        )
    """
    try:
        # Call Cortex
        response = session.sql(
            "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) as response",
            params=[model, prompt]
        ).collect()
        
        if not response or len(response) == 0:
            return return_on_error
        
        response_text = response[0]['RESPONSE']
        
        # Try to parse JSON
        parsed = safe_json_parse(response_text)
        
        if not parsed:
            # Try to extract JSON from response
            json_match = re.search(
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', 
                response_text, 
                re.DOTALL
            )
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                except Exception:
                    return return_on_error
            else:
                return return_on_error
        
        # Validate expected key if provided
        if expected_key and expected_key not in parsed:
            return return_on_error
        
        return parsed
        
    except Exception as e:
        if error_handler:
            error_handler.log_error(
                error_code=ErrorCode.CORTEX_API_ERROR,
                error_type=type(e).__name__,
                severity=ErrorSeverity.WARNING,
                message=f"Cortex call failed: {str(e)}",
                function_name="call_cortex_and_parse"
            )
        return return_on_error


def call_cortex_and_parse_list(
    prompt: str,
    session,
    error_handler,
    max_items: Optional[int] = None,
    return_on_error: List = None,
    model: str = LLM_MODEL
) -> List[Any]:
    """
    Call Snowflake Cortex LLM and parse list/array response.
    
    Similar to call_cortex_and_parse but expects a list response.
    
    Args:
        prompt: LLM prompt text
        session: Snowflake session
        error_handler: ErrorHandler instance
        max_items: Optional maximum number of items to return
        return_on_error: Value to return if call fails (default: [])
        model: LLM model to use
    
    Returns:
        Parsed list or return_on_error if failed
    
    Example:
        questions = call_cortex_and_parse_list(
            prompt="Generate 5 questions about...",
            session=session,
            error_handler=error_handler,
            max_items=5,
            return_on_error=[]
        )
    """
    if return_on_error is None:
        return_on_error = []
    
    try:
        # Call Cortex
        response = session.sql(
            "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) as response",
            params=[model, prompt]
        ).collect()
        
        if not response or len(response) == 0:
            return return_on_error
        
        response_text = response[0]['RESPONSE']
        
        # Try to extract list from response
        list_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if list_match:
            try:
                parsed_list = json.loads(list_match.group(0))
                
                if max_items:
                    return parsed_list[:max_items]
                return parsed_list
            except Exception:
                return return_on_error
        
        return return_on_error
        
    except Exception as e:
        if error_handler:
            error_handler.log_error(
                error_code=ErrorCode.CORTEX_API_ERROR,
                error_type=type(e).__name__,
                severity=ErrorSeverity.WARNING,
                message=f"Cortex list call failed: {str(e)}",
                function_name="call_cortex_and_parse_list"
            )
        return return_on_error


def grant_with_feedback(
    session,
    grant_type: str,
    object_name: str,
    role: str,
    show_success: bool = False,
    return_on_error: bool = False
) -> bool:
    """
    Execute a GRANT statement with user feedback.
    
    This helper eliminates the repetitive pattern of:
    1. Executing GRANT statement
    2. Catching exceptions
    3. Displaying appropriate feedback
    
    Args:
        session: Snowflake session
        grant_type: Type of grant (e.g., 'USAGE', 'SELECT', 'ALL PRIVILEGES')
        object_name: Full object name (e.g., 'SCHEMA mydb.myschema', 'TABLE mytable')
        role: Role to grant to
        show_success: Whether to show success message (default: False)
        return_on_error: Value to return if grant fails (default: False)
    
    Returns:
        True if grant succeeded, return_on_error if failed
    
    Example:
        success = grant_with_feedback(
            session=session,
            grant_type='SELECT',
            object_name='VIEW mydb.myschema.myview',
            role='SYSADMIN',
            show_success=True
        )
    """
    try:
        grant_sql = f"GRANT {grant_type} ON {object_name} TO ROLE {role}"
        session.sql(grant_sql).collect()
        
        if show_success:
            st.success(f"✓ Granted {grant_type} on {object_name} to {role}")
        
        return True
        
    except Exception as e:
        error_msg = str(e).lower()
        
        # Check if it's a benign "already granted" error
        if 'already' in error_msg or 'exists' in error_msg:
            if show_success:
                st.info(f"ℹ️ {grant_type} on {object_name} already granted to {role}")
            return True
        
        # Real error
        st.warning(f"⚠️ Could not grant {grant_type} on {object_name} to {role}: {str(e)[:100]}")
        return return_on_error


def execute_sql_with_retry(
    session,
    sql: str,
    max_attempts: int = 3,
    show_errors: bool = True
) -> Optional[List]:
    """
    Execute SQL with automatic retry on failure.
    
    Args:
        session: Snowflake session
        sql: SQL statement to execute
        max_attempts: Maximum retry attempts (default: 3)
        show_errors: Whether to show error messages (default: True)
    
    Returns:
        Query results or None if all attempts failed
    
    Example:
        results = execute_sql_with_retry(
            session=session,
            sql="SELECT * FROM my_table LIMIT 10",
            max_attempts=3
        )
    """
    import time
    
    last_exception = None
    delay = 1.0
    
    for attempt in range(1, max_attempts + 1):
        try:
            return session.sql(sql).collect()
        except Exception as e:
            last_exception = e
            
            if attempt < max_attempts:
                if show_errors:
                    st.warning(f"⚠️ SQL execution failed (attempt {attempt}/{max_attempts}). Retrying...")
                time.sleep(delay)
                delay *= 2
            else:
                if show_errors:
                    st.error(f"❌ SQL execution failed after {max_attempts} attempts: {str(e)[:150]}")
    
    return None


def extract_json_from_text(
    text: str,
    expected_type: str = 'dict'
) -> Optional[Any]:
    """
    Extract and parse JSON from text that may contain other content.
    
    This helper handles the common pattern of LLM responses that include
    JSON embedded in explanatory text.
    
    Args:
        text: Text containing JSON
        expected_type: Expected JSON type ('dict' or 'list')
    
    Returns:
        Parsed JSON object or None if extraction failed
    
    Example:
        text = "Here is the data: {\"name\": \"John\", \"age\": 30}"
        data = extract_json_from_text(text, expected_type='dict')
        # Returns: {"name": "John", "age": 30}
    """
    if not text:
        return None
    
    try:
        # First try direct parsing
        return json.loads(text)
    except Exception:
        pass
    
    # Try to extract based on expected type
    if expected_type == 'dict':
        # Extract object pattern
        match = re.search(
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
            text,
            re.DOTALL
        )
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    
    elif expected_type == 'list':
        # Extract array pattern
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    
    return None


def validate_schema_name(schema_name: str) -> tuple[bool, str]:
    """
    Validate Snowflake schema name.
    
    Args:
        schema_name: Schema name to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    
    Example:
        is_valid, error = validate_schema_name("MY_SCHEMA")
        if not is_valid:
            st.error(error)
    """
    if not schema_name:
        return False, "Schema name cannot be empty"
    
    if len(schema_name) > 255:
        return False, "Schema name too long (max 255 characters)"
    
    # Check for SQL injection patterns
    dangerous_patterns = [';', '--', '/*', '*/', 'DROP', 'DELETE', 'TRUNCATE']
    schema_upper = schema_name.upper()
    
    for pattern in dangerous_patterns:
        if pattern in schema_upper:
            return False, f"Schema name contains dangerous pattern: {pattern}"
    
    # Check for valid characters (alphanumeric, underscore, dollar)
    if not re.match(r'^[A-Za-z0-9_$\.]+$', schema_name):
        return False, "Schema name contains invalid characters"
    
    return True, ""


def validate_table_name(table_name: str) -> tuple[bool, str]:
    """
    Validate Snowflake table name.
    
    Args:
        table_name: Table name to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not table_name:
        return False, "Table name cannot be empty"
    
    if len(table_name) > 255:
        return False, "Table name too long (max 255 characters)"
    
    # Check for SQL injection patterns
    dangerous_patterns = [';', '--', '/*', '*/', 'DROP', 'DELETE', 'TRUNCATE']
    table_upper = table_name.upper()
    
    for pattern in dangerous_patterns:
        if pattern in table_upper:
            return False, f"Table name contains dangerous pattern: {pattern}"
    
    # Check for valid characters
    if not re.match(r'^[A-Za-z0-9_$]+$', table_name):
        return False, "Table name contains invalid characters"
    
    return True, ""


def format_large_number(number: int) -> str:
    """
    Format large numbers with commas for display.
    
    Args:
        number: Number to format
    
    Returns:
        Formatted string (e.g., "1,234,567")
    
    Example:
        st.metric("Total Records", format_large_number(1234567))
        # Displays: "1,234,567"
    """
    return f"{number:,}"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length (default: 100)
        suffix: Suffix to add if truncated (default: "...")
    
    Returns:
        Truncated text
    
    Example:
        short_desc = truncate_text(long_description, max_length=50)
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def build_progress_message(
    current_step: int,
    total_steps: int,
    operation: str
) -> str:
    """
    Build consistent progress message format.
    
    Args:
        current_step: Current step number
        total_steps: Total number of steps
        operation: Operation description
    
    Returns:
        Formatted progress message
    
    Example:
        msg = build_progress_message(5, 10, "Generating schemas")
        progress_placeholder.progress(5/10, text=msg)
    """
    return f"Step {current_step}/{total_steps}: {operation}..."


def display_success_expander(
    title: str,
    details: Dict[str, Any],
    container,
    expanded: bool = False
):
    """
    Display success message with expandable details.
    
    This helper eliminates the repetitive pattern of creating expanders
    for successful operations.
    
    Args:
        title: Success message title
        details: Dictionary of details to display
        container: Streamlit container to display in
        expanded: Whether expander is initially expanded (default: False)
    
    Example:
        display_success_expander(
            title="✅ Table CUSTOMERS created (100 records)",
            details={
                "Columns": "ID, NAME, EMAIL, CREATED_DATE",
                "Primary Key": "ID",
                "Join Overlap": "70%"
            },
            container=status_container
        )
    """
    with container:
        with st.expander(title, expanded=expanded):
            for key, value in details.items():
                st.caption(f"**{key}:** {value}")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Value to return if denominator is zero (default: 0.0)
    
    Returns:
        Result of division or default
    
    Example:
        percentage = safe_divide(completed, total, default=0.0) * 100
    """
    if denominator == 0:
        return default
    return numerator / denominator

"""
Streamlit best practices utilities for SI Data Generator.

This module provides utilities for following Streamlit best practices including
AppState management, proper widget keys, and UI component wrappers.
"""

import streamlit as st
from typing import Any, Optional, Dict, List, Callable
from dataclasses import dataclass, field
from datetime import datetime


# ============================================================================
# APP STATE MANAGEMENT
# ============================================================================

class AppState:
    """
    Centralized application state manager.
    
    This class provides a clean interface for managing st.session_state,
    eliminating the need for direct st.session_state['key'] access throughout
    the codebase.
    
    Example usage:
        state = AppState()
        
        # Set values
        state.set('company_name', 'Acme Corp')
        state.set('num_records', 100)
        
        # Get values
        company = state.get('company_name', default='')
        num_records = state.get('num_records', default=100)
        
        # Check existence
        if state.has('demo_results'):
            results = state.get('demo_results')
    """
    
    def __init__(self):
        """Initialize AppState manager."""
        # Initialize session state if not exists
        if 'initialized' not in st.session_state:
            st.session_state['initialized'] = True
            st.session_state['creation_time'] = datetime.now()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from session state.
        
        Args:
            key: State key to retrieve
            default: Default value if key doesn't exist
        
        Returns:
            Value from session state or default
        """
        return st.session_state.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set value in session state.
        
        Args:
            key: State key to set
            value: Value to store
        """
        st.session_state[key] = value
    
    def has(self, key: str) -> bool:
        """
        Check if key exists in session state.
        
        Args:
            key: State key to check
        
        Returns:
            True if key exists, False otherwise
        """
        return key in st.session_state
    
    def delete(self, key: str) -> None:
        """
        Delete key from session state.
        
        Args:
            key: State key to delete
        """
        if key in st.session_state:
            del st.session_state[key]
    
    def clear_all(self) -> None:
        """Clear all session state (use with caution)."""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get all session state as dictionary.
        
        Returns:
            Dictionary of all session state
        """
        return dict(st.session_state)
    
    def set_multiple(self, values: Dict[str, Any]) -> None:
        """
        Set multiple values at once.
        
        Args:
            values: Dictionary of key-value pairs to set
        """
        for key, value in values.items():
            st.session_state[key] = value
    
    def update(self, key: str, updater: Callable[[Any], Any], default: Any = None) -> None:
        """
        Update a value using an updater function.
        
        Args:
            key: State key to update
            updater: Function that takes current value and returns new value
            default: Default value if key doesn't exist
        
        Example:
            # Increment counter
            state.update('counter', lambda x: x + 1, default=0)
            
            # Append to list
            state.update('items', lambda x: x + [new_item], default=[])
        """
        current = self.get(key, default)
        new_value = updater(current)
        self.set(key, new_value)
    
    # Convenience methods for common patterns
    def increment(self, key: str, amount: int = 1) -> None:
        """Increment a numeric value."""
        self.update(key, lambda x: x + amount, default=0)
    
    def append(self, key: str, item: Any) -> None:
        """Append item to a list."""
        self.update(key, lambda x: x + [item], default=[])
    
    def toggle(self, key: str) -> None:
        """Toggle a boolean value."""
        self.update(key, lambda x: not x, default=False)


# ============================================================================
# WIDGET KEY MANAGEMENT
# ============================================================================

class WidgetKeys:
    """
    Centralized widget key management.
    
    This class generates consistent, unique widget keys to prevent
    Streamlit DuplicateWidgetID errors.
    
    Example usage:
        keys = WidgetKeys(prefix='demo_')
        
        schema_name = st.text_input(
            "Schema Name",
            key=keys.get('schema_name')
        )
    """
    
    def __init__(self, prefix: str = ""):
        """
        Initialize widget key manager.
        
        Args:
            prefix: Optional prefix for all keys (e.g., 'demo_', 'config_')
        """
        self.prefix = prefix
        self._counter = 0
    
    def get(self, name: str) -> str:
        """
        Get a widget key.
        
        Args:
            name: Base name for the key
        
        Returns:
            Full widget key with prefix
        """
        return f"{self.prefix}{name}"
    
    def get_unique(self, name: str) -> str:
        """
        Get a unique widget key (with counter).
        
        Args:
            name: Base name for the key
        
        Returns:
            Unique widget key
        """
        self._counter += 1
        return f"{self.prefix}{name}_{self._counter}"
    
    def get_indexed(self, name: str, index: int) -> str:
        """
        Get an indexed widget key.
        
        Args:
            name: Base name for the key
            index: Index number
        
        Returns:
            Indexed widget key
        """
        return f"{self.prefix}{name}_{index}"


# ============================================================================
# UI COMPONENT WRAPPERS
# ============================================================================

def create_status_container(title: str, expanded: bool = True):
    """
    Create a status container using st.status().
    
    This is the preferred way to show progress in Streamlit instead of
    multiple st.info() calls.
    
    Args:
        title: Status title
        expanded: Whether container is initially expanded
    
    Returns:
        Status container context manager
    
    Example:
        with create_status_container("Generating demo...") as status:
            st.write("Step 1: Creating schema...")
            # ... do work ...
            st.write("Step 2: Generating data...")
            # ... do work ...
    """
    return st.status(title, expanded=expanded)


def create_form_container(form_key: str, clear_on_submit: bool = True):
    """
    Create a form container with proper key management.
    
    Args:
        form_key: Unique key for the form
        clear_on_submit: Whether to clear form after submission
    
    Returns:
        Form container context manager
    
    Example:
        with create_form_container("demo_config_form"):
            schema_name = st.text_input("Schema Name")
            num_records = st.number_input("Number of Records")
            
            submitted = st.form_submit_button("Generate Demo")
            if submitted:
                # Process form
                pass
    """
    return st.form(key=form_key, clear_on_submit=clear_on_submit)


def create_columns_layout(ratios: List[int]):
    """
    Create a columns layout with specified ratios.
    
    Args:
        ratios: List of column width ratios
    
    Returns:
        Tuple of column containers
    
    Example:
        col1, col2, col3 = create_columns_layout([2, 1, 1])
        with col1:
            st.metric("Total Records", 1000)
        with col2:
            st.metric("Tables", 3)
        with col3:
            st.metric("Time", "45s")
    """
    return st.columns(ratios)


def create_expander(title: str, expanded: bool = False, icon: str = ""):
    """
    Create an expander with consistent formatting.
    
    Args:
        title: Expander title
        expanded: Whether initially expanded
        icon: Optional icon to prepend to title
    
    Returns:
        Expander container context manager
    
    Example:
        with create_expander("Advanced Options", icon="⚙️"):
            enable_search = st.checkbox("Enable Search Service")
            enable_agent = st.checkbox("Enable AI Agent")
    """
    full_title = f"{icon} {title}" if icon else title
    return st.expander(full_title, expanded=expanded)


# ============================================================================
# PROGRESS DISPLAY HELPERS
# ============================================================================

def show_metric_row(metrics: List[Dict[str, Any]]):
    """
    Display a row of metrics with consistent formatting.
    
    Args:
        metrics: List of metric dictionaries with keys: label, value, delta (optional)
    
    Example:
        show_metric_row([
            {"label": "Total Records", "value": 1000, "delta": "+100"},
            {"label": "Tables Created", "value": 3},
            {"label": "Time Elapsed", "value": "45s"}
        ])
    """
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        with col:
            st.metric(
                label=metric['label'],
                value=metric['value'],
                delta=metric.get('delta')
            )


def show_success_with_details(
    title: str,
    details: Dict[str, Any],
    expanded: bool = False
):
    """
    Show success message with expandable details.
    
    Args:
        title: Success message title
        details: Dictionary of detail key-value pairs
        expanded: Whether details are initially expanded
    
    Example:
        show_success_with_details(
            title="✅ Table CUSTOMERS created successfully",
            details={
                "Records": 100,
                "Columns": "ID, NAME, EMAIL, CREATED_DATE",
                "Primary Key": "ID"
            }
        )
    """
    with st.expander(title, expanded=expanded):
        for key, value in details.items():
            st.caption(f"**{key}:** {value}")


def show_error_with_action(
    error_message: str,
    action_label: str,
    action_callback: Callable,
    key: str
):
    """
    Show error message with action button.
    
    Args:
        error_message: Error message to display
        action_label: Label for action button
        action_callback: Function to call when button clicked
        key: Unique key for button
    
    Example:
        def retry_generation():
            st.session_state['retry'] = True
        
        show_error_with_action(
            error_message="Generation failed",
            action_label="Retry",
            action_callback=retry_generation,
            key="retry_button"
        )
    """
    st.error(error_message)
    if st.button(action_label, key=key):
        action_callback()


# ============================================================================
# INPUT COMPONENTS WITH VALIDATION
# ============================================================================

def validated_text_input(
    label: str,
    key: str,
    validator: Optional[Callable] = None,
    default: str = "",
    help_text: Optional[str] = None,
    placeholder: Optional[str] = None
) -> Optional[str]:
    """
    Text input with optional validation.
    
    Args:
        label: Input label
        key: Widget key
        validator: Optional validation function (value) -> (is_valid, error_message)
        default: Default value
        help_text: Optional help text
        placeholder: Optional placeholder text
    
    Returns:
        Input value or None if invalid
    
    Example:
        def validate_schema(value):
            if len(value) < 3:
                return False, "Too short"
            return True, ""
        
        schema = validated_text_input(
            label="Schema Name",
            key="schema_input",
            validator=validate_schema
        )
    """
    value = st.text_input(
        label,
        value=default,
        key=key,
        help=help_text,
        placeholder=placeholder
    )
    
    if validator and value:
        is_valid, error = validator(value)
        if not is_valid:
            st.error(f"❌ {error}")
            return None
    
    return value


def validated_number_input(
    label: str,
    key: str,
    validator: Optional[Callable] = None,
    default: int = 0,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
    step: int = 1,
    help_text: Optional[str] = None
) -> Optional[int]:
    """
    Number input with optional validation.
    
    Args:
        label: Input label
        key: Widget key
        validator: Optional validation function
        default: Default value
        min_value: Minimum value
        max_value: Maximum value
        step: Step size
        help_text: Optional help text
    
    Returns:
        Input value or None if invalid
    """
    value = st.number_input(
        label,
        value=default,
        min_value=min_value,
        max_value=max_value,
        step=step,
        key=key,
        help=help_text
    )
    
    if validator:
        is_valid, error = validator(value)
        if not is_valid:
            st.error(f"❌ {error}")
            return None
    
    return value


# ============================================================================
# CONTAINER MANAGEMENT
# ============================================================================

class ContainerManager:
    """
    Manager for organizing UI components into containers.
    
    This helps keep the UI organized and makes it easier to update
    specific sections without re-rendering everything.
    
    Example usage:
        containers = ContainerManager()
        
        # Create containers
        containers.create('header', st.container())
        containers.create('config', st.container())
        containers.create('results', st.container())
        
        # Use containers
        with containers.get('header'):
            st.title("SI Data Generator")
        
        with containers.get('config'):
            schema_name = st.text_input("Schema Name")
    """
    
    def __init__(self):
        """Initialize container manager."""
        self._containers: Dict[str, Any] = {}
    
    def create(self, name: str, container=None):
        """
        Create or register a container.
        
        Args:
            name: Container name
            container: Optional container object (creates new if None)
        
        Returns:
            Container object
        """
        if container is None:
            container = st.container()
        self._containers[name] = container
        return container
    
    def get(self, name: str):
        """
        Get a container by name.
        
        Args:
            name: Container name
        
        Returns:
            Container object or None if not found
        """
        return self._containers.get(name)
    
    def has(self, name: str) -> bool:
        """Check if container exists."""
        return name in self._containers
    
    def clear(self, name: str):
        """Clear a container (recreate it)."""
        if name in self._containers:
            self._containers[name] = st.container()


# ============================================================================
# LOADING STATES
# ============================================================================

def show_loading(message: str = "Loading..."):
    """
    Show loading spinner with message.
    
    Args:
        message: Loading message
    
    Returns:
        Spinner context manager
    
    Example:
        with show_loading("Generating schemas..."):
            schemas = generate_schemas()
    """
    return st.spinner(message)


def show_progress_bar(progress: float, text: str = ""):
    """
    Show progress bar with text.
    
    Args:
        progress: Progress value (0.0 to 1.0)
        text: Progress text
    
    Returns:
        Progress placeholder
    
    Example:
        progress_bar = st.empty()
        for i in range(100):
            show_progress_bar(i/100, f"Processing {i}/100")
            time.sleep(0.1)
    """
    return st.progress(progress, text=text)


# ============================================================================
# CALLBACK HELPERS
# ============================================================================

def create_callback(func: Callable, *args, **kwargs) -> Callable:
    """
    Create a callback function for Streamlit widgets.
    
    Args:
        func: Function to call
        *args: Positional arguments for function
        **kwargs: Keyword arguments for function
    
    Returns:
        Callback function
    
    Example:
        def update_config(key, value):
            st.session_state[key] = value
        
        st.button(
            "Update",
            on_click=create_callback(update_config, 'schema_name', 'NEW_SCHEMA')
        )
    """
    def callback():
        func(*args, **kwargs)
    return callback


# ============================================================================
# SESSION STATE PERSISTENCE
# ============================================================================

def save_state_to_history(key: str):
    """
    Save current state to history for undo functionality.
    
    Args:
        key: Key to identify this history entry
    """
    if 'state_history' not in st.session_state:
        st.session_state['state_history'] = []
    
    state_copy = dict(st.session_state)
    st.session_state['state_history'].append({
        'key': key,
        'state': state_copy,
        'timestamp': datetime.now()
    })
    
    # Keep only last 10 history entries
    if len(st.session_state['state_history']) > 10:
        st.session_state['state_history'] = st.session_state['state_history'][-10:]


def restore_state_from_history(index: int = -1):
    """
    Restore state from history.
    
    Args:
        index: History index to restore (default: -1 for most recent)
    """
    if 'state_history' in st.session_state and st.session_state['state_history']:
        history_entry = st.session_state['state_history'][index]
        for key, value in history_entry['state'].items():
            st.session_state[key] = value

"""
Multi-language support for SI Data Generator application.

This module provides internationalization (i18n) capabilities including
language configuration, content validation, and prompt enhancement for
generating content in multiple languages.
"""

import re
from typing import Dict, List, Optional


# Supported languages with validation patterns and configuration
SUPPORTED_LANGUAGES = {
    "en": {
        "name": "English",
        "code": "en",
        "native_name": "English",
        "cortex_instruction": "Generate content in English",
        "encoding": "utf-8",
        "sample_validation": r"[a-zA-Z]+",
        "char_validation": r"[a-zA-Z\s\d\.,;:!?'\"-]+",
        "direction": "ltr"
    },
    "es": {
        "name": "Spanish",
        "code": "es",
        "native_name": "Español",
        "cortex_instruction": (
            "Generate content in Spanish (Español). Use proper Spanish "
            "grammar, vocabulary, and sentence structure."
        ),
        "encoding": "utf-8",
        "sample_validation": r"[a-zA-Záéíóúñ]+",
        "char_validation": r"[a-zA-Záéíóúñü¿¡\s\d\.,;:!?'\"-]+",
        "direction": "ltr"
    },
    "fr": {
        "name": "French",
        "code": "fr",
        "native_name": "Français",
        "cortex_instruction": (
            "Generate content in French (Français). Use proper French "
            "grammar, vocabulary, and sentence structure."
        ),
        "encoding": "utf-8",
        "sample_validation": r"[a-zA-Zàâäéèêëïîôùûüÿç]+",
        "char_validation": r"[a-zA-Zàâäæéèêëïîôœùûüÿç\s\d\.,;:!?'\"-]+",
        "direction": "ltr"
    },
    "de": {
        "name": "German",
        "code": "de",
        "native_name": "Deutsch",
        "cortex_instruction": (
            "Generate content in German (Deutsch). Use proper German "
            "grammar, vocabulary, and sentence structure."
        ),
        "encoding": "utf-8",
        "sample_validation": r"[a-zA-ZäöüßÄÖÜ]+",
        "char_validation": r"[a-zA-ZäöüßÄÖÜ\s\d\.,;:!?'\"-]+",
        "direction": "ltr"
    },
    "ja": {
        "name": "Japanese",
        "code": "ja",
        "native_name": "日本語",
        "cortex_instruction": (
            "Generate content in Japanese (日本語). Use proper Japanese "
            "grammar and natural Japanese expressions with appropriate use "
            "of hiragana, katakana, and kanji."
        ),
        "encoding": "utf-8",
        "sample_validation": r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+",
        "char_validation": (
            r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\s\d\.,;:!?"
            r"（）「」『』【】、。]+"
        ),
        "direction": "ltr"
    },
    "zh": {
        "name": "Chinese",
        "code": "zh",
        "native_name": "中文",
        "cortex_instruction": (
            "Generate content in Chinese (中文 - Simplified Chinese). "
            "Use proper Chinese grammar and natural Chinese expressions."
        ),
        "encoding": "utf-8",
        "sample_validation": r"[\u4E00-\u9FFF]+",
        "char_validation": (
            r"[\u4E00-\u9FFF\s\d\.,;:!?（）「」『』【】、。]+"
        ),
        "direction": "ltr"
    }
}


def get_language_config(language_code: str) -> Optional[Dict]:
    """
    Get language configuration for a language code.
    
    Args:
        language_code: ISO 639-1 language code (e.g., 'en', 'es', 'fr')
        
    Returns:
        Language configuration dictionary or None if not supported
    """
    return SUPPORTED_LANGUAGES.get(language_code)


def get_language_display_name(language_code: str) -> str:
    """
    Get human-readable display name for a language.
    
    Returns the language name with native name in parentheses if they differ.
    
    Args:
        language_code: ISO 639-1 language code
        
    Returns:
        Display name string (e.g., "Spanish (Español)")
    """
    config = get_language_config(language_code)
    if not config:
        return "English"
    
    if config['name'] == config['native_name']:
        return config['name']
    else:
        return f"{config['name']} ({config['native_name']})"


def validate_language_content(
    text: str,
    language_code: str
) -> tuple[bool, Optional[str]]:
    """
    Validate that generated text matches the expected language.
    
    Uses regex patterns to verify the text contains characters appropriate
    for the specified language and can be encoded properly.
    
    Args:
        text: Text content to validate
        language_code: Expected language code
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    language_config = get_language_config(language_code)
    if not language_config:
        return False, f"Unsupported language: {language_code}"
    
    if not text or len(text.strip()) == 0:
        return False, "Text is empty"
    
    # Check for language-specific character patterns
    pattern = language_config['sample_validation']
    matches = re.findall(pattern, text)
    
    if not matches:
        return (
            False,
            f"Generated text does not appear to be in "
            f"{language_config['name']}"
        )
    
    # Validate encoding
    try:
        text.encode(language_config['encoding'])
    except UnicodeEncodeError as e:
        return (
            False,
            f"Text encoding issue for {language_config['name']}: {str(e)}"
        )
    
    return True, None


def enhance_prompt_with_language(
    base_prompt: str,
    language_code: str
) -> str:
    """
    Enhance a prompt with language-specific instructions.
    
    Appends detailed language requirements to the base prompt to guide
    the LLM in generating content in the specified language.
    
    Args:
        base_prompt: Original prompt text
        language_code: Target language code
        
    Returns:
        Enhanced prompt with language instructions
    """
    language_config = get_language_config(language_code)
    if not language_config or language_code == "en":
        return base_prompt
    
    language_instruction = (
        f"\n\nIMPORTANT LANGUAGE REQUIREMENT:\n"
        f"{language_config['cortex_instruction']}\n"
    )
    language_instruction += (
        f"ALL generated text content must be in {language_config['name']} "
        f"({language_config['native_name']}).\n"
    )
    language_instruction += (
        f"Use natural, fluent {language_config['name']} that a native "
        f"speaker would use.\n"
    )
    language_instruction += (
        "Do not translate word-by-word; instead, express ideas naturally "
        "in the target language.\n"
    )
    
    return base_prompt + language_instruction


def add_language_metadata_to_chunks(
    chunks_data: List[Dict],
    language_code: str
) -> List[Dict]:
    """
    Add language metadata fields to unstructured data chunks.
    
    Enriches each chunk with language code, name, and native name for
    better searchability and filtering in Cortex Search.
    
    Args:
        chunks_data: List of chunk dictionaries
        language_code: Language code for the chunks
        
    Returns:
        Enhanced chunks list with language metadata
    """
    language_config = get_language_config(language_code)
    if not language_config:
        language_config = SUPPORTED_LANGUAGES["en"]
    
    for chunk in chunks_data:
        chunk['LANGUAGE'] = language_code
        chunk['LANGUAGE_NAME'] = language_config['name']
        chunk['LANGUAGE_NATIVE'] = language_config['native_name']
    
    return chunks_data


def get_supported_language_codes() -> List[str]:
    """
    Get list of all supported language codes.
    
    Returns:
        List of ISO 639-1 language codes
    """
    return list(SUPPORTED_LANGUAGES.keys())


def get_supported_languages_display() -> List[Dict[str, str]]:
    """
    Get list of supported languages with display information.
    
    Returns:
        List of dictionaries with 'code', 'name', and 'display_name' keys
    """
    return [
        {
            'code': code,
            'name': config['name'],
            'native_name': config['native_name'],
            'display_name': get_language_display_name(code)
        }
        for code, config in SUPPORTED_LANGUAGES.items()
    ]

