import csv
import cv2
import logging
import numpy
from PIL import Image, ImageOps
import pytesseract
import sklearn
import stringdist
import sys


logger = logging.getLogger('process')


def cv2pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil2cv(img):
    return cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)


player_name_end_icon = cv2.imread('player_name_end_icon.png')


with open('weapons.txt') as fp:
    known_weapons = set(n.strip() for n in fp)
known_weapons.discard('')


def get_player_name(frame, known_player_names):
    # Get player name area
    player_name = frame.crop((169, 960, 400, 990))

    # Find the icon at the end of the player name
    match = cv2.matchTemplate(
        pil2cv(player_name),
        player_name_end_icon,
        cv2.TM_CCOEFF_NORMED,
    )
    _, max_value, _, max_location = cv2.minMaxLoc(match)
    if max_value < 0.8:
        logger.info("Can't find player name end icon")
        return None

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
    if not ocr:
        return None

    # Find closest name
    if ocr not in known_player_names:
        dists = [
            (
                stringdist.levenshtein_norm(ocr, candidate),
                candidate,
            )
            for candidate in known_player_names
        ]
        best_dist, best_name = min(dists)
        if best_dist < 0.45:
            logger.info("Player name: %r -> %r", ocr, best_name)
            ocr = best_name
        else:
            logger.info("Unknown player: %r (best guess: %r, %.2f)", ocr, best_name, best_dist)
            return None
    else:
        logger.info("Player name: %r", ocr)

    return ocr


def get_weapons(frame):
    weapons = []

    for i, loc in enumerate([
        (1554, 1035, 1660, 1054),
        (1710, 1035, 1815, 1055),
    ]):
        # Get weapon name area
        weapon = frame.crop(loc)

        # Scale up
        weapon = ImageOps.scale(weapon, 4.0, Image.NEAREST)

        # Threshold, turn black-on-white
        array = numpy.array(weapon)
        new_array = numpy.zeros((array.shape[0], array.shape[1]), dtype=numpy.uint8)
        THRESHOLD = int(0.4 * 255)
        for y in range(array.shape[0]):
            for x in range(array.shape[1]):
                if numpy.mean(array[y][x]) < THRESHOLD:
                    new_array[y][x] = 255
                else:
                    new_array[y][x] = 255 - numpy.mean(array[y][x])
        weapon = Image.fromarray(new_array)

        # FIXME: Running a classifier would be much faster than OCR

        # OCR
        ocr = pytesseract.image_to_string(weapon)
        ocr = ocr.rstrip('\r\n\x0C')
        if not ocr:
            continue

        # Find closest name
        if ocr not in known_weapons:
            dists = [
                (
                    stringdist.levenshtein_norm(ocr, candidate),
                    candidate,
                )
                for candidate in known_weapons
            ]
            best_dist, best_name = min(dists)
            if best_dist < 0.45:
                logger.info("Weapon %d: %r -> %r", i + 1, ocr, best_name)
                ocr = best_name
            else:
                logger.info("Unknown weapon: %r (best guess: %r, %.2f)", ocr, best_name, best_dist)
                continue
        else:
            logger.info("Weapon %d: %r", i + 1, ocr)

        weapons.append(ocr)

    return weapons


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt='%H:%M:%S',
    )

    folder, from_frame, to_frame = args
    from_frame = int(from_frame)
    to_frame = int(to_frame)
    assert from_frame < to_frame

    with open('%s.players.txt' % folder) as fp:
        known_player_names = set(n.strip() for n in fp)
    known_player_names.discard('')

    with open('%06d-%06d.csv' % (from_frame, to_frame), 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['frame', 'player name', 'weapon 1', 'weapon 2'])

        for frameno in range(from_frame, to_frame):
            logger.info("")
            logger.info(">>> Frame %06d", frameno)
            frame = Image.open('%s/%06d.png' % (folder, frameno))

            player_name = get_player_name(frame, known_player_names)
            if not player_name:
                continue

            weapons = get_weapons(frame)

            writer.writerow(
                [
                    frameno,
                    player_name,
                ]
                + weapons + [''] * (2 - len(weapons)),
            )


if __name__ == '__main__':
    main(sys.argv[1:])
