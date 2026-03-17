import os.path as osp

DEFAULT_ROOT_DIR = osp.join((osp.abspath(osp.dirname(__file__))))
# Currently the screenshots should be in ROOT_DIR/screenshots/keyboard_dataset
DEFAULT_SCREENSHOTS_DIR = osp.join(DEFAULT_ROOT_DIR, 'screenshots')
DEFAULT_KEYBOARD_DATASET_DIR = osp.join(DEFAULT_ROOT_DIR, 'screenshots', 'keyboard_dataset')
DEFAULT_UTILS_DIR = osp.dirname(__file__)
DEFAULT_VALIDATION_DIR = osp.join(DEFAULT_SCREENSHOTS_DIR, 'validation_images')
DEFAULT_ORIGINAL_DATASET_DIR = osp.join(DEFAULT_SCREENSHOTS_DIR, 'origin_keyboard_dataset')
KEYBOARD_VALIDATION = osp.join(DEFAULT_SCREENSHOTS_DIR, 'keyboard_validation')
SCREENSHOT_VALIDATION = osp.join(DEFAULT_SCREENSHOTS_DIR, 'screenshot_validation')
KBD1K_PATH = osp.join(DEFAULT_ROOT_DIR, 'kbd1k')
KBD1K_SCREENSHOTS_DIR = osp.join(KBD1K_PATH, 'keyboard_dataset')
LATENT_SPACE_VALIDATION_DIR = osp.join(DEFAULT_SCREENSHOTS_DIR, 'latent_space_validation')
DEFAULT_MODEL_DIR = osp.join(DEFAULT_ROOT_DIR, 'outputs')
DEFAULT_DATA_DIR = osp.join(DEFAULT_ROOT_DIR, 'data')
DEFAULT_ANDROID_TYPING_ENV_DIR = osp.join(DEFAULT_ROOT_DIR, 'android_typing_env')

keyboard_index = {
    'Gboard': '00',
    'Microsoft SwiftKey': '01',
    'Fonts Keyboard': '02',
    'Emojikeyboard': '03',
    'GoKeyboardLite': '04',
    'LED Keyboard': '05',
    'Grammarly': '06',
    'Yandex Keyboard': '07',
    'Design Keyboard': '08',
    'Kika Keyboard': '09',
    'Giphy Keyboard': '10',
    'Fonts Art Keyboard': '11',
    'Stylish Text Keyboard': '12',
    'GoKeyboardPro': '13',
    'Deco Keyboard': '14',
    '2023 Keyboard': '15',
    'iKeyboard GIF Keyboard': '16',
    'Classic Big Keyboard': '17',
    'Stylish Fonts and Keyboard': '18',
    'Neon LED Keyboard': '19',
    'My Photo Keyboard With Themes': '20',
    'Facemoji Emoji Keyboard': '21',
    'Bobble Keyboard': '22',
    'Laban Key': '23',
    'Fast Typing Keyboard': '24',
    'Malayalam Keyboard': '25',
    'Ridmik Keyboard': '26',
    'Hacker\'s Keyboard': '27',
    'Decoration Text Keyboard': '28',
    'Cute Emoji Keyboard': '29',
    'iMore Keyboard': '30',
    'All Arabic Keyboard': '31',
}

keyboard_name = {v: k for k, v in keyboard_index.items()}


def get_keyboard_manual_label():
    """
    manually label

    x coordinate of symbol right,   # -1 means referring to the position of 'z'
    x coordinate of space left end,   # -1 means referring to the position of 'z',
                                      # -2 means referring to the position of 'x'
    x coordinate of space right end,    # -1 means referring to the position of 'n',
                                        # -2 means referring to the position of 'b'
    x coordinate of enter left,     # -1 means referring to the position of 'm',
    predictive_text and first row gap,
    predictive text height,

    coordinate of [translate key x, translate key y]  # translate all the labels of characters

    x coordinate of [predictive_text 1 left, predictive_text 2 left, predictive_text 3 left, predictive_text 3 right]
                                                                # -1 means referring to the position of 'Q', 'R', 'U', image.width

    x coordinate of [comma left, comma right] on character mode # -1 means referring to the position of 'z', next to 'Symbol'
                                                                # -2 means referring to the position of 'x', next to 'Space'
    x coordinate of [period left, period right] on character mode # -1 means referring to (between) the position of 'enter' and 'space'

    if the value is -1 or -2, it means there is no need to use manual label, -3 means there is no such key
    :return:
    """
    keyboards_manual_label = dict()
    keyboards_manual_label[keyboard_name['00']] = [-1, -1, -1, -1, 0, 118, [0, 0], [-1, -1, -1, -1], [-1, -1], [-1, -1]]
    keyboards_manual_label[keyboard_name['01']] = [114, 328, -1, -1, 0, 111, [0, -20], [-1, -1, -1, -1], [222, -2], [-1, -1]]
    keyboards_manual_label[keyboard_name['02']] = [-1, -1, -1, -1, 0, 0, [0, 0], [-1, -1, -1, -1], [-1, -1], [-1, -1]]
    keyboards_manual_label[keyboard_name['03']] = [-1, -1, -1, -1, 0, 114, [0, 0], [-1, -1, -1, -1], [-2, -2], [-1, -1]]
    keyboards_manual_label[keyboard_name['04']] = [-1, -1, -1, -1, 0, 0, [0, 0], [-1, -1, -1, -1], [-2, -2], [-1, -1]]
    keyboards_manual_label[keyboard_name['05']] = [-1, -1, -1, -1, 0, 100, [0, 0], [-1, 408, 764, -1], [-1, -1], [-1, -1]]
    keyboards_manual_label[keyboard_name['06']] = [150, 368, 819, 928, 0, 114, [0, 0], [104, 436, 758, -1], [-1, 259], [-1, -1]]
    keyboards_manual_label[keyboard_name['07']] = [141, 358, -1, 908, 0, 0, [0, 0], [-1, -1, -1, -1], [258, -2], [-1, -1]]
    keyboards_manual_label[keyboard_name['08']] = [147, 360, -1, -1, 0, 0, [0, 0], [-1, -1, -1, -1], [250, -2], [-1, -1]]
    keyboards_manual_label[keyboard_name['09']] = [-1, -1, -1, -1, 0, 108, [0, 0], [-1, 376, 741, -1], [-2, -2], [-1, -1]]
    keyboards_manual_label[keyboard_name['10']] = [-1, -1, -1, -1, 0, 0, [0, 0], [-1, -1, -1, -1], [-1, -1], [-1, -1]]
    keyboards_manual_label[keyboard_name['11']] = [172, 388, -1, -1, 0, 0, [0, 0], [-1, -1, -1, -1], [-1, -1], [-1, -1]]
    keyboards_manual_label[keyboard_name['12']] = [-1, -2, -1, -1, 0, 0, [0, 0], [-1, -1, -1, -1], [-1, -1], [-1, -1]]
    keyboards_manual_label[keyboard_name['13']] = [-1, -1, -1, -1, 0, 0, [0, 0], [-1, -1, -1, -1], [-2, -2], [-1, -1]]
    keyboards_manual_label[keyboard_name['14']] = [147, 360, -1, -1, 0, 0, [0, 0], [-1, -1, -1, -1], [250, -2], [-1, -1]]
    keyboards_manual_label[keyboard_name['15']] = [-1, 384, -1, -1, 0, 102, [0, 0], [-1, 421, 719, -1], [-1, -1], [-1, -1]]
    keyboards_manual_label[keyboard_name['16']] = [-1, -1, -1, -1, 0, 111, [0, 0], [0, 340, 723, -1], [-2, -2], [-1, -1]]
    keyboards_manual_label[keyboard_name['17']] = [105, 321, 732, 949, 0, 77, [0, -5], [0, 360, 720, -1], [214, 322], [732, 841]]
    keyboards_manual_label[keyboard_name['18']] = [-1, 386, 748, 913, 0, 0, [0, 5], [-1, -1, -1, -1], [-3, -3], [-1, -1]]
    keyboards_manual_label[keyboard_name['19']] = [-1, -1, -1, -1, 0, 104, [0, 0], [-1, 336, 602, 863], [-1, -1], [-1, -1]]
    keyboards_manual_label[keyboard_name['20']] = [-1, -2, -1, -1, 0, 0, [0, 0], [-1, -1, -1, -1], [-1, -1], [-1, -1]]
    keyboards_manual_label[keyboard_name['21']] = [-1, -1, -1, -1, 0, 0, [0, -10], [-1, -1, -1, -1], [-2, -2], [-1, -1]]
    keyboards_manual_label[keyboard_name['22']] = [-1, -1, -1, -1, 0, 107, [0, 0], [0, 364, 717, -1], [-1, -1], [-1, -1]]
    keyboards_manual_label[keyboard_name['23']] = [-1, -2, -1, -1, 0, 118, [0, 0], [91, -1, -1, 972], [-1, -1], [-1, -1]]
    keyboards_manual_label[keyboard_name['24']] = [-1, -1, -1, -1, 0, 103, [0, 0], [-1, 421, 718, -1], [-1, -1], [-1, -1]]
    keyboards_manual_label[keyboard_name['25']] = [-1, -1, -1, -1, 0, 0, [0, 0], [-1, -1, -1, -1], [-1, -1], [-1, -1]]
    keyboards_manual_label[keyboard_name['26']] = [-1, -1, -1, -1, 0, 104, [0, 0], [-1, -1, -1, 1007], [-2, -2], [-1, -1]]
    keyboards_manual_label[keyboard_name['27']] = [-1, -1, -2, -1, 0, 0, [0, 0], [-1, -1, -1, -1], [-2, -2], [807, -1]]
    keyboards_manual_label[keyboard_name['28']] = [130, 356, 734, 853, 0, 0, [0, 0], [-1, -1, -1, -1], [237, -2], [-1, -1]]
    keyboards_manual_label[keyboard_name['29']] = [-1, -1, -1, -1, 0, 101, [0, 0], [-1, 431, 756, -1], [-2, -2], [-1, -1]]
    keyboards_manual_label[keyboard_name['30']] = [-1, -1, -1, -1, 16, 105, [0, 0], [-1, -1, -1, 1000], [-1, -1], [-1, -1]]
    keyboards_manual_label[keyboard_name['31']] = [121, 356, 753, 871, 0, 0, [0, 5], [-1, -1, -1, -1], [240, 356], [-1, -1]]
    return keyboards_manual_label
