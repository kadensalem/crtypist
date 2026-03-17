CHARS = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'c',
         'v', 'b', 'n', 'm', ' ']  # characters to be typed
KEYS = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'c',
        'v', 'b', 'n', 'm', '<', '>', ' ']  # keys on the soft keyboard
PLACES = ['input_box', 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l',
          'z', 'x', 'c', 'v', 'b', 'n', 'm', '<', '>', ' ']  # places to look at on the screen

CHARS_FIN = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'c',
         'v', 'b', 'n', 'm', ' ', 'å', 'ö', 'ä']  # characters to be typed
KEYS_FIN = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'c',
        'v', 'b', 'n', 'm', '<', '>', ' ', 'å', 'ö', 'ä']  # keys on the soft keyboard
PLACES_FIN = ['input_box', 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l',
          'z', 'x', 'c', 'v', 'b', 'n', 'm', '<', '>', ' ', 'å', 'ö', 'ä']  # places to look at on the screen

CHARS_KALQ = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'c',
         'v', 'b', 'n', 'm', ' ']  # characters to be typed
KEYS_KALQ = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'z', 'x', 'c',
        'v', 'b', 'n', 'm', '<', '>', ' ', '  ']  # keys on the soft keyboard
PLACES_KALQ = ['input_box', 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l',
          'z', 'x', 'c', 'v', 'b', 'n', 'm', '<', '>', ' ', '  ']  # places to look at on the screen

NUM_ONE_KEY = {'1': 'q', '2': 'w', '3': 'e', '4': 'r', '5': 't', '6': 'y', '7': 'u', '8': 'i', '9': 'o', '0': 'p'}

"""
backspace('<'), enter('>') and space(' ') 
for kalq keyboard, ' ' is left space, '  'is right space
"""

keys_for_labeling = ['screenshot_name', 'keyboard_name', 'keyboard_index',
                     # basic info about the screenshot
                     'text', 'theme', 'border', 'mode', 'number_row', 'word_prediction',
                     # coordinate of texts
                     'input_box', 'predictive_text_1', 'predictive_text_2', 'predictive_text_3',
                     # coordinate of characters
                     'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p',
                     'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l',
                     'z', 'x', 'c', 'v', 'b', 'n', 'm',
                     # coordinate of numbers
                     '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
                     # coordinate of symbols
                     'backspace', 'space', 'enter', 'shift', 'symbol',
                     # punctuations
                     ",", ".", "!", "?", "-", "'"
                     ]
