import cv2
import logging
import numpy
from PIL import Image, ImageOps
import pytesseract
import sklearn
import stringdist


logger = logging.getLogger('process')


def cv2pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil2cv(img):
    return cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)


player_name_end_icon = cv2.imread('player_name_end_icon.png')


with open('player_names.txt') as fp:
    known_names = set(n.strip() for n in fp)
known_names.discard('')


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
        # Scale up
        player_name = ImageOps.scale(player_name, 4.0, Image.NEAREST)

        # Threshold, turn black-on-white
        array = numpy.array(player_name)
        new_array = numpy.zeros((array.shape[0], array.shape[1]), dtype=numpy.uint8)
        THRESHOLD = int(0.4 * 255)
        for y in range(array.shape[0]):
            for x in range(array.shape[1]):
                if numpy.mean(array[y][x]) < THRESHOLD:
                    new_array[y][x] = 255
                else:
                    new_array[y][x] = 255 - numpy.mean(array[y][x])
        player_name = Image.fromarray(new_array)

        # OCR
        ocr = pytesseract.image_to_string(player_name)
        ocr = ocr.rstrip('\r\n\x0C')
        logger.info("Player name: %r", ocr)

        # Find closest name
        if ocr not in known_names:
            dists = [
                (
                    stringdist.levenshtein_norm(ocr, candidate),
                    candidate,
                )
                for candidate in known_names
            ]
            best_dist, best_name = min(dists)
            if best_dist < 0.45:
                corrected = best_name
                logger.info("Player name: %r -> %r", ocr, corrected)
            else:
                logger.info("Unknown player: %r (best guess: %r, %.2f)", ocr, best_name, best_dist)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt='%H:%M:%S',
    )

    for frameno in range(835, 1500):
        logger.info("")
        logger.info(">>> Frame %06d", frameno)
        frame = Image.open('2021-03-27_965657358/%06d.png' % frameno)

        get_player_name(frame)


if __name__ == '__main__':
    main()
