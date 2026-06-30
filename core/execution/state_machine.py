"""
State Machine for PHTN.AI Sub-Agent Framework

Provides state machine functionality for complex agent workflows
with support for custom states, transitions, and guards.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class State:
    """Represents a state in the state machine."""
    name: str
    is_initial: bool = False
    is_final: bool = False
    on_enter: Optional[Callable] = None
    on_exit: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Transition:
    """Represents a transition between states."""
    name: str
    from_state: str
    to_state: str
    trigger: str
    guard: Optional[Callable[[Dict[str, Any]], bool]] = None
    action: Optional[Callable[[Dict[str, Any]], Any]] = None
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateHistoryEntry:
    """Entry in state history."""
    from_state: str
    to_state: str
    trigger: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)


class StateMachine:
    """
    State machine for managing agent execution flow.
    
    Features:
    - Custom states and transitions
    - Guard conditions for transitions
    - Entry/exit actions for states
    - Transition actions
    - State history tracking
    - Hierarchical states (nested state machines)
    """
    
    def __init__(
        self,
        name: str = "agent_state_machine",
        initial_state: Optional[str] = None,
    ):
        """
        Initialize StateMachine.
        
        Args:
            name: State machine name
            initial_state: Initial state name
        """
        self.name = name
        self._states: Dict[str, State] = {}
        self._transitions: List[Transition] = []
        self._current_state: Optional[str] = None
        self._initial_state = initial_state
        self._history: List[StateHistoryEntry] = []
        self._context: Dict[str, Any] = {}
        
        logger.debug(f"StateMachine created: {name}")
    
    @property
    def current_state(self) -> Optional[str]:
        """Get current state name."""
        return self._current_state
    
    @property
    def history(self) -> List[StateHistoryEntry]:
        """Get state history."""
        return self._history.copy()
    
    @property
    def context(self) -> Dict[str, Any]:
        """Get state machine context."""
        return self._context.copy()
    
    def add_state(
        self,
        name: str,
        is_initial: bool = False,
        is_final: bool = False,
        on_enter: Optional[Callable] = None,
        on_exit: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "StateMachine":
        """
        Add a state to the state machine.
        
        Args:
            name: State name
            is_initial: Whether this is the initial state
            is_final: Whether this is a final state
            on_enter: Callback when entering state
            on_exit: Callback when exiting state
            metadata: Additional state metadata
            
        Returns:
            Self for chaining
        """
        state = State(
            name=name,
            is_initial=is_initial,
            is_final=is_final,
            on_enter=on_enter,
            on_exit=on_exit,
            metadata=metadata or {},
        )
        
        self._states[name] = state
        
        if is_initial:
            self._initial_state = name
        
        logger.debug(f"State added: {name}")
        return self
    
    def add_transition(
        self,
        name: str,
        from_state: str,
        to_state: str,
        trigger: str,
        guard: Optional[Callable[[Dict[str, Any]], bool]] = None,
        action: Optional[Callable[[Dict[str, Any]], Any]] = None,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "StateMachine":
        """
        Add a transition to the state machine.
        
        Args:
            name: Transition name
            from_state: Source state
            to_state: Target state
            trigger: Trigger event name
            guard: Guard condition function
            action: Action to execute during transition
            priority: Transition priority (higher = evaluated first)
            metadata: Additional transition metadata
            
        Returns:
            Self for chaining
        """
        transition = Transition(
            name=name,
            from_state=from_state,
            to_state=to_state,
            trigger=trigger,
            guard=guard,
            action=action,
            priority=priority,
            metadata=metadata or {},
        )
        
        self._transitions.append(transition)
        self._transitions.sort(key=lambda t: -t.priority)
        
        logger.debug(f"Transition added: {name} ({from_state} -> {to_state})")
        return self
    
    def start(self, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Start the state machine.
        
        Args:
            context: Initial context
        """
        if not self._initial_state:
            initial_states = [s for s in self._states.values() if s.is_initial]
            if initial_states:
                self._initial_state = initial_states[0].name
            elif self._states:
                self._initial_state = next(iter(self._states.keys()))
            else:
                raise ValueError("No states defined in state machine")
        
        self._context = context or {}
        self._current_state = self._initial_state
        
        state = self._states.get(self._current_state)
        if state and state.on_enter:
            state.on_enter(self._context)
        
        logger.info(f"StateMachine started in state: {self._current_state}")
    
    def trigger(self, event: str, context_update: Optional[Dict[str, Any]] = None) -> bool:
        """
        Trigger an event to potentially cause a transition.
        
        Args:
            event: Event name
            context_update: Context updates
            
        Returns:
            True if transition occurred, False otherwise
        """
        if not self._current_state:
            raise RuntimeError("State machine not started")
        
        if context_update:
            self._context.update(context_update)
        
        for transition in self._transitions:
            if transition.from_state != self._current_state:
                continue
            if transition.trigger != event:
                continue
            
            if transition.guard and not transition.guard(self._context):
                continue
            
            return self._execute_transition(transition)
        
        logger.debug(f"No valid transition for event '{event}' in state '{self._current_state}'")
        return False
    
    def _execute_transition(self, transition: Transition) -> bool:
        """Execute a transition."""
        from_state = self._states.get(transition.from_state)
        to_state = self._states.get(transition.to_state)
        
        if not to_state:
            logger.error(f"Target state not found: {transition.to_state}")
            return False
        
        if from_state and from_state.on_exit:
            from_state.on_exit(self._context)
        
        if transition.action:
            transition.action(self._context)
        
        self._history.append(StateHistoryEntry(
            from_state=transition.from_state,
            to_state=transition.to_state,
            trigger=transition.trigger,
            timestamp=datetime.utcnow(),
            context=self._context.copy(),
        ))
        
        self._current_state = transition.to_state
        
        if to_state.on_enter:
            to_state.on_enter(self._context)
        
        logger.info(f"Transition: {transition.from_state} -> {transition.to_state} (trigger: {transition.trigger})")
        return True
    
    def can_trigger(self, event: str) -> bool:
        """
        Check if an event can trigger a transition.
        
        Args:
            event: Event name
            
        Returns:
            True if event can trigger a transition
        """
        if not self._current_state:
            return False
        
        for transition in self._transitions:
            if transition.from_state != self._current_state:
                continue
            if transition.trigger != event:
                continue
            if transition.guard and not transition.guard(self._context):
                continue
            return True
        
        return False
    
    def get_available_triggers(self) -> List[str]:
        """Get list of available triggers from current state."""
        if not self._current_state:
            return []
        
        triggers = set()
        for transition in self._transitions:
            if transition.from_state == self._current_state:
                if not transition.guard or transition.guard(self._context):
                    triggers.add(transition.trigger)
        
        return list(triggers)
    
    def is_in_final_state(self) -> bool:
        """Check if state machine is in a final state."""
        if not self._current_state:
            return False
        
        state = self._states.get(self._current_state)
        return state.is_final if state else False
    
    def reset(self) -> None:
        """Reset state machine to initial state."""
        self._current_state = None
        self._history.clear()
        self._context.clear()
        logger.info("StateMachine reset")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state machine to dictionary."""
        return {
            "name": self.name,
            "current_state": self._current_state,
            "initial_state": self._initial_state,
            "states": list(self._states.keys()),
            "transitions": [
                {
                    "name": t.name,
                    "from": t.from_state,
                    "to": t.to_state,
                    "trigger": t.trigger,
                }
                for t in self._transitions
            ],
            "history_length": len(self._history),
            "context": self._context,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "StateMachine":
        """
        Create state machine from configuration.
        
        Args:
            config: State machine configuration
            
        Returns:
            StateMachine instance
        """
        sm = cls(
            name=config.get("name", "state_machine"),
            initial_state=config.get("initial_state"),
        )
        
        for state_config in config.get("states", []):
            if isinstance(state_config, str):
                sm.add_state(state_config)
            else:
                sm.add_state(
                    name=state_config["name"],
                    is_initial=state_config.get("is_initial", False),
                    is_final=state_config.get("is_final", False),
                    metadata=state_config.get("metadata"),
                )
        
        for trans_config in config.get("transitions", []):
            sm.add_transition(
                name=trans_config.get("name", f"{trans_config['from']}_to_{trans_config['to']}"),
                from_state=trans_config["from"],
                to_state=trans_config["to"],
                trigger=trans_config["trigger"],
                priority=trans_config.get("priority", 0),
                metadata=trans_config.get("metadata"),
            )
        
        return sm
