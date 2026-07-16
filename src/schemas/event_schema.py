from enum import Enum, auto


class VisionEvent(Enum):

    """Application-level events (semantic, not keyboard-specific)."""
    
    NONE = auto()
    EXIT = auto()
    CREATE_FOLDER = auto()
    SAVE_POSE = auto()
    TRAIN_MODEL = auto()
    TYPE_CHAR = auto()   
    BACKSPACE = auto()   


class KeyCode(Enum):
    
    """Raw keyboard codes. Kept separate from VisionEvent on purpose —
    mixing semantic events and raw key codes in one Enum breaks int comparisons."""

    CLOSE_KEY = 27
    ENTER_KEY = 13
    SAVE_KEY = 9
    TRAIN_KEY = 61
    BACKSPACE_KEY = 8