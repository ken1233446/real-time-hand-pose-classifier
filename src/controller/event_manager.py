from src.schemas.event_schema import VisionEvent, KeyCode

class EventManager:

    """
    EventManager Contract

    What:

        Converts raw keyboard input
        into application events.

    Why:
        Separates input handling from
        application behavior.

    Input:
        Keyboard ASCII value.

    Process:
        Read key.
        Convert key into event.

    Output:
        VisionEvent.

    """

    # Keyboard mapping.
    CLOSE_KEY = 27
    ENTER_KEY = 13
    SAVE_KEY = 9
    TRAIN_KEY = 61

    def __init__(self) -> None:
            pass

    def process_key(self, key: int) -> VisionEvent:

        if key == -1:  # no key pressed this frame
            return VisionEvent.NONE

        if key == KeyCode.CLOSE_KEY.value:
            return VisionEvent.EXIT

        if key == KeyCode.ENTER_KEY.value:
            return VisionEvent.CREATE_FOLDER

        if key == KeyCode.SAVE_KEY.value:
            return VisionEvent.SAVE_POSE

        if key == KeyCode.TRAIN_KEY.value:
            return VisionEvent.TRAIN_MODEL

        if key == KeyCode.BACKSPACE_KEY.value:
            return VisionEvent.BACKSPACE

        if 32 <= key <= 126:  # printable ASCII, includes letters
            return VisionEvent.TYPE_CHAR

        return VisionEvent.NONE