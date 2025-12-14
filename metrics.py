"""
Performance metrics and telemetry for SI Data Generator.

This module provides decorators and utilities for tracking performance metrics
throughout the application, enabling performance monitoring and optimization.
"""

import functools
import time
import streamlit as st
from typing import Callable, Any


def timeit(func: Callable) -> Callable:
    """
    Decorator to measure and store execution time of functions.
    
    Execution times are stored in st.session_state['metrics'] for later display
    and analysis. This enables performance monitoring across the application.
    
    Usage:
        @timeit
        def my_function(arg1, arg2):
            # ... function implementation
            return result
    
    Args:
        func: Function to time
    
    Returns:
        Wrapped function that tracks execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        # Store in session state for display
        if 'metrics' not in st.session_state:
            st.session_state['metrics'] = {}
        
        # Accumulate times for functions called multiple times (e.g., generate_data_from_schema)
        if func.__name__ in st.session_state['metrics']:
            st.session_state['metrics'][func.__name__] += elapsed_time
        else:
            st.session_state['metrics'][func.__name__] = elapsed_time
        
        return result
    
    return wrapper


def display_performance_summary():
    """
    Display a summary of performance metrics collected during execution.
    
    Shows timing information for all instrumented functions, sorted by
    execution time. Useful for identifying performance bottlenecks.
    """
    if 'metrics' in st.session_state and st.session_state['metrics']:
        st.subheader("⏱️ Performance Summary")
        
        metrics = st.session_state['metrics']
        
        # Sort by execution time (descending)
        sorted_metrics = sorted(
            metrics.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Calculate total time
        total_time = sum(metrics.values())
        
        # Display summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Time", f"{total_time:.2f}s")
        with col2:
            st.metric("Functions Tracked", len(metrics))
        with col3:
            slowest_func = sorted_metrics[0][0] if sorted_metrics else "N/A"
            st.metric("Slowest Function", slowest_func)
        
        # Display detailed breakdown (no expander, show directly)
        st.markdown("**⏱️ Detailed Performance Breakdown**")
        for func_name, elapsed_time in sorted_metrics:
            percentage = (elapsed_time / total_time * 100) if total_time > 0 else 0
            st.caption(f"**{func_name}**: {elapsed_time:.2f}s ({percentage:.1f}%)")


def get_metric(func_name: str) -> float:
    """
    Get the execution time metric for a specific function.
    
    Args:
        func_name: Name of the function to retrieve metric for
    
    Returns:
        Execution time in seconds, or 0.0 if not found
    """
    if 'metrics' not in st.session_state:
        return 0.0
    
    return st.session_state['metrics'].get(func_name, 0.0)


def clear_metrics():
    """Clear all stored performance metrics."""
    if 'metrics' in st.session_state:
        st.session_state['metrics'] = {}


def add_custom_metric(name: str, value: float):
    """
    Manually add a custom metric (for operations not wrapped by @timeit).
    
    Args:
        name: Metric name
        value: Metric value (typically time in seconds)
    """
    if 'metrics' not in st.session_state:
        st.session_state['metrics'] = {}
    
    st.session_state['metrics'][name] = value

"""
Progress tracking utilities for SI Data Generator.

This module provides a ProgressTracker class to replace manual progress
calculation throughout the codebase, improving code readability and
reducing errors in progress tracking logic.
"""

import streamlit as st
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum


class ProgressPhase(Enum):
    """Enum for different phases of demo generation."""
    INITIALIZATION = "Initialization"
    SCHEMA_CREATION = "Schema Creation"
    DATA_GENERATION = "Data Generation"
    TABLE_CREATION = "Table Creation"
    INFRASTRUCTURE = "Infrastructure"
    COMPLETION = "Completion"


@dataclass
class ProgressStep:
    """Data class representing a single progress step."""
    name: str
    phase: ProgressPhase
    weight: float = 1.0  # Relative weight of this step
    completed: bool = False


class ProgressTracker:
    """
    Manages progress tracking for complex multi-step operations.
    
    This class eliminates manual progress calculation by automatically
    tracking steps and computing progress percentages.
    
    Example usage:
        tracker = ProgressTracker(progress_placeholder)
        
        # Define steps
        tracker.add_step("Create schema", ProgressPhase.INITIALIZATION)
        tracker.add_step("Generate schemas", ProgressPhase.SCHEMA_CREATION, weight=2.0)
        tracker.add_step("Generate data", ProgressPhase.DATA_GENERATION, weight=3.0)
        
        # Update progress
        tracker.start_step("Create schema", "Creating schema...")
        # ... do work ...
        tracker.complete_step("Create schema")
        
        tracker.start_step("Generate schemas", "Generating AI schemas...")
        # ... do work ...
        tracker.complete_step("Generate schemas")
    """
    
    def __init__(
        self,
        progress_placeholder,
        show_phase_info: bool = True,
        show_time_estimate: bool = False
    ):
        """
        Initialize ProgressTracker.
        
        Args:
            progress_placeholder: Streamlit placeholder for progress bar
            show_phase_info: Whether to show current phase in progress text
            show_time_estimate: Whether to show estimated time remaining
        """
        self.progress_placeholder = progress_placeholder
        self.show_phase_info = show_phase_info
        self.show_time_estimate = show_time_estimate
        
        self.steps: List[ProgressStep] = []
        self.current_step: Optional[str] = None
        self.step_map: Dict[str, int] = {}  # name -> index
        
        self.start_time: Optional[float] = None
        self.step_times: Dict[str, float] = {}
        
    def add_step(
        self,
        name: str,
        phase: ProgressPhase = ProgressPhase.INITIALIZATION,
        weight: float = 1.0
    ):
        """
        Add a step to the progress tracker.
        
        Args:
            name: Unique name for the step
            phase: Phase this step belongs to
            weight: Relative weight of this step (default: 1.0)
        """
        step = ProgressStep(name=name, phase=phase, weight=weight)
        self.step_map[name] = len(self.steps)
        self.steps.append(step)
    
    def add_steps_batch(self, step_configs: List[Dict]):
        """
        Add multiple steps at once.
        
        Args:
            step_configs: List of step config dicts with keys: name, phase, weight
        
        Example:
            tracker.add_steps_batch([
                {"name": "Step 1", "phase": ProgressPhase.INITIALIZATION},
                {"name": "Step 2", "phase": ProgressPhase.DATA_GENERATION, "weight": 2.0}
            ])
        """
        for config in step_configs:
            self.add_step(
                name=config['name'],
                phase=config.get('phase', ProgressPhase.INITIALIZATION),
                weight=config.get('weight', 1.0)
            )
    
    def start_step(self, name: str, message: Optional[str] = None):
        """
        Mark a step as started and update progress display.
        
        Args:
            name: Name of the step to start
            message: Optional custom message to display
        """
        import time
        
        if name not in self.step_map:
            raise ValueError(f"Unknown step: {name}")
        
        self.current_step = name
        
        if self.start_time is None:
            self.start_time = time.time()
        
        # Update progress
        self._update_progress(message)
    
    def complete_step(self, name: str):
        """
        Mark a step as completed.
        
        Args:
            name: Name of the step to complete
        """
        import time
        
        if name not in self.step_map:
            raise ValueError(f"Unknown step: {name}")
        
        idx = self.step_map[name]
        self.steps[idx].completed = True
        
        # Record step time
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.step_times[name] = elapsed
        
        self.current_step = None
    
    def skip_step(self, name: str):
        """
        Mark a step as completed without actually doing it (for optional steps).
        
        Args:
            name: Name of the step to skip
        """
        if name in self.step_map:
            idx = self.step_map[name]
            self.steps[idx].completed = True
    
    def get_progress(self) -> float:
        """
        Calculate current progress as a percentage (0.0 to 1.0).
        
        Returns:
            Progress percentage
        """
        if not self.steps:
            return 0.0
        
        total_weight = sum(step.weight for step in self.steps)
        completed_weight = sum(
            step.weight for step in self.steps if step.completed
        )
        
        # Add partial credit for current step
        if self.current_step and self.current_step in self.step_map:
            idx = self.step_map[self.current_step]
            if not self.steps[idx].completed:
                completed_weight += self.steps[idx].weight * 0.5
        
        return completed_weight / total_weight if total_weight > 0 else 0.0
    
    def get_completed_count(self) -> int:
        """Get number of completed steps."""
        return sum(1 for step in self.steps if step.completed)
    
    def get_total_count(self) -> int:
        """Get total number of steps."""
        return len(self.steps)
    
    def get_current_phase(self) -> Optional[ProgressPhase]:
        """Get the phase of the current step."""
        if self.current_step and self.current_step in self.step_map:
            idx = self.step_map[self.current_step]
            return self.steps[idx].phase
        return None
    
    def get_estimated_time_remaining(self) -> Optional[float]:
        """
        Estimate time remaining based on completed steps.
        
        Returns:
            Estimated seconds remaining, or None if not enough data
        """
        import time
        
        if not self.start_time or not self.steps:
            return None
        
        completed_count = self.get_completed_count()
        if completed_count == 0:
            return None
        
        elapsed = time.time() - self.start_time
        avg_time_per_step = elapsed / completed_count
        remaining_steps = len(self.steps) - completed_count
        
        return avg_time_per_step * remaining_steps
    
    def _update_progress(self, message: Optional[str] = None):
        """
        Update the progress bar display.
        
        Args:
            message: Optional custom message
        """
        progress = self.get_progress()
        completed = self.get_completed_count()
        total = self.get_total_count()
        
        # Build progress message
        if message:
            progress_text = message
        elif self.current_step:
            progress_text = self.current_step
        else:
            progress_text = "Processing..."
        
        # Add step counter
        progress_text = f"Step {completed + 1}/{total}: {progress_text}"
        
        # Add phase info if enabled
        if self.show_phase_info:
            phase = self.get_current_phase()
            if phase:
                progress_text = f"[{phase.value}] {progress_text}"
        
        # Add time estimate if enabled
        if self.show_time_estimate:
            time_remaining = self.get_estimated_time_remaining()
            if time_remaining:
                mins = int(time_remaining // 60)
                secs = int(time_remaining % 60)
                if mins > 0:
                    progress_text += f" (~{mins}m {secs}s remaining)"
                else:
                    progress_text += f" (~{secs}s remaining)"
        
        # Update progress bar
        self.progress_placeholder.progress(progress, text=progress_text)
    
    def update_message(self, message: str):
        """
        Update progress message without changing step.
        
        Args:
            message: New message to display
        """
        self._update_progress(message)
    
    def reset(self):
        """Reset all progress tracking."""
        for step in self.steps:
            step.completed = False
        self.current_step = None
        self.start_time = None
        self.step_times = {}
    
    def get_summary(self) -> Dict:
        """
        Get progress summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        import time
        
        total_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            'total_steps': len(self.steps),
            'completed_steps': self.get_completed_count(),
            'progress_percentage': self.get_progress() * 100,
            'total_time_seconds': total_time,
            'current_phase': self.get_current_phase().value if self.get_current_phase() else None,
            'estimated_time_remaining': self.get_estimated_time_remaining(),
            'step_times': self.step_times.copy()
        }
    
    def display_summary(self, container=None):
        """
        Display progress summary in Streamlit.
        
        Args:
            container: Optional Streamlit container to display in
        """
        summary = self.get_summary()
        
        display_fn = container if container else st
        
        col1, col2, col3 = display_fn.columns(3)
        with col1:
            display_fn.metric(
                "Completed Steps",
                f"{summary['completed_steps']}/{summary['total_steps']}"
            )
        with col2:
            display_fn.metric(
                "Progress",
                f"{summary['progress_percentage']:.1f}%"
            )
        with col3:
            mins = int(summary['total_time_seconds'] // 60)
            secs = int(summary['total_time_seconds'] % 60)
            display_fn.metric("Time Elapsed", f"{mins}m {secs}s")


def create_demo_generation_tracker(
    progress_placeholder,
    num_structured_tables: int = 2,
    num_unstructured_tables: int = 1,
    enable_semantic_view: bool = True,
    enable_search_service: bool = True,
    enable_agent: bool = True,
    has_target_questions: bool = False
) -> ProgressTracker:
    """
    Create a pre-configured ProgressTracker for demo generation.
    
    This factory function creates a tracker with all the standard steps
    for demo generation, eliminating the need to manually configure steps.
    
    Args:
        progress_placeholder: Streamlit progress placeholder
        num_structured_tables: Number of structured tables to generate
        num_unstructured_tables: Number of unstructured tables
        enable_semantic_view: Whether semantic view creation is enabled
        enable_search_service: Whether search service creation is enabled
        enable_agent: Whether agent creation is enabled
        has_target_questions: Whether target questions need analysis
    
    Returns:
        Configured ProgressTracker instance
    
    Example:
        tracker = create_demo_generation_tracker(
            progress_placeholder,
            num_structured_tables=3,
            enable_agent=True
        )
        
        tracker.start_step("Create schema", "Creating database schema...")
        # ... do work ...
        tracker.complete_step("Create schema")
    """
    tracker = ProgressTracker(progress_placeholder, show_phase_info=True)
    
    # Phase 1: Initialization
    if has_target_questions:
        tracker.add_step("Analyze questions", ProgressPhase.INITIALIZATION, weight=1.0)
    tracker.add_step("Create schema", ProgressPhase.INITIALIZATION, weight=0.5)
    
    # Phase 2: Schema Creation
    tracker.add_step(
        "Generate schemas",
        ProgressPhase.SCHEMA_CREATION,
        weight=2.0  # Higher weight as this is a major operation
    )
    
    # Phase 3: Data Generation
    for i in range(num_structured_tables):
        tracker.add_step(
            f"Generate data for table {i+1}",
            ProgressPhase.DATA_GENERATION,
            weight=1.5
        )
    
    # Phase 4: Table Creation
    for i in range(num_structured_tables):
        tracker.add_step(
            f"Save table {i+1}",
            ProgressPhase.TABLE_CREATION,
            weight=1.0
        )
    
    for i in range(num_unstructured_tables):
        tracker.add_step(
            f"Create unstructured table {i+1}",
            ProgressPhase.TABLE_CREATION,
            weight=1.5
        )
    
    # Phase 5: Infrastructure
    if enable_semantic_view:
        tracker.add_step("Create semantic view", ProgressPhase.INFRASTRUCTURE, weight=1.5)
    
    if enable_search_service:
        for i in range(num_unstructured_tables):
            tracker.add_step(
                f"Create search service {i+1}",
                ProgressPhase.INFRASTRUCTURE,
                weight=1.5
            )
    
    tracker.add_step("Generate questions", ProgressPhase.INFRASTRUCTURE, weight=2.0)
    
    if enable_agent:
        tracker.add_step("Create agent", ProgressPhase.INFRASTRUCTURE, weight=2.0)
    
    # Phase 6: Completion
    tracker.add_step("Finalize", ProgressPhase.COMPLETION, weight=0.5)
    
    return tracker

