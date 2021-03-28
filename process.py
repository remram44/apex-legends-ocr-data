import cv2
import logging
import numpy
from PIL import Image, ImageOps
import pytesseract
import sklearn


logger = logging.getLogger('process')


def cv2pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil2cv(img):
    return cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)


player_name_end_icon = cv2.imread('player_name_end_icon.png')


def get_player_name(frame):
    # Get player name area
    player_name = frame.crop((113, 640, 267, 660))
    # Find the icon at the end of the player name
    match = cv2.matchTemplate(
        pil2cv(player_name),
        player_name_end_icon,
        cv2.TM_CCOEFF_NORMED,
    )
    _, max_value, _, max_location = cv2.minMaxLoc(match)
    if max_value < 0.8:
        logger.info("Can't find player name end icon")
    else:
        # Crop the image before the icon
        player_name = player_name.crop((
            0, 0,
            max_location[0], player_name.size[1],
        ))
        # Scaling helps
        player_name = ImageOps.scale(player_name, 4.0, Image.NEAREST)
        player_name_ocr = pytesseract.image_to_string(player_name)
        player_name_ocr = player_name_ocr.rstrip('\r\n\x0C')
        logger.info("Player name: %r", player_name_ocr)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    for frameno in range(835, 1500):
        logger.info(">>> Frame %06d", frameno)
        frame = Image.open('2021-03-27_965657358/%06d.png' % frameno)

        get_player_name(frame)


if __name__ == '__main__':
    main()
