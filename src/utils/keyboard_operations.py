

"""
What:
    All keyboard operation will be in this module.
"""
class KeyboardOps:
    
    #Use to handle keyboard input.
    def update_text_input(self, key: int, text: str) -> str:

        #CONSTANTS.
        TAB_KEY = 9
        ESC_KEY = 27

        #When Backspace is entered it subtract the text string.
        if key == 8: 
            text = text[:-1]

        #All keys except the conditions will update the text variable.
        #Tab key and ESC key should NOT MODITY the text.
        elif 32 <= key <= 126 and key != TAB_KEY and key != ESC_KEY:
            text += chr(key)

        return text
