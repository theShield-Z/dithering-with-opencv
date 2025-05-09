"""This program shows the effect of dithering algorithms on video. This is meant mostly as a learning experience for me
 and a tutorial for others, so I added a good amount of documentation. This includes the standard Floyd-Steinberg and
 Ordered Bayer algorithms, as well as a few custom Ordered patterns. If you use main_2(), you can also use the output as
 a virtual webcam (for Zoom, OBS, etc.)

Controls:
m - switch dithering mode
g - toggle between grayscale and color
q - quit the program"""

import numpy as np
import cv2 as cv
import numba  # Decorators that compile functions to machine code
import pyvirtualcam  # Used to cast to virtual webcam; only need to install if using main_2()

# CONSTANTS ############################################################################################################

BAYER_MATRIX_8X8 = (1 / 64) * np.array([
    [0, 48, 12, 60, 3, 51, 15, 63],
    [32, 16, 44, 28, 35, 19, 47, 31],
    [8, 56, 4, 52, 11, 59, 7, 55],
    [40, 24, 36, 20, 43, 27, 39, 23],
    [2, 50, 14, 62, 1, 49, 13, 61],
    [34, 18, 46, 30, 33, 17, 45, 29],
    [10, 58, 6, 54, 9, 57, 5, 53],
    [42, 26, 38, 22, 41, 25, 37, 21]
]) * 255

BAYER_MATRIX_4X4 = (1 / 16) * np.array([
    [0, 12, 3, 15],
    [8, 4, 11, 7],
    [2, 14, 1, 13],
    [10, 6, 9, 5]
]) * 255

BAYER_MATRIX_2X2 = (1 / 4) * np.array([
    [0, 3],
    [2, 1]
]) * 255

HALFTONE_4X4 = (1 / 16) * np.array([
    [15, 11, 7, 13],
    [5, 3, 1, 8],
    [9, 0, 2, 4],
    [12, 6, 10, 14]
]) * 255

DIAGONAL = (1 / 64) * np.array([
    [0, 8, 16, 24, 32, 40, 48, 56],
    [57, 1, 9, 17, 25, 33, 41, 49],
    [50, 58, 2, 10, 18, 26, 34, 42],
    [43, 51, 59, 3, 11, 19, 27, 35],
    [36, 44, 52, 60, 4, 12, 20, 28],
    [29, 37, 45, 53, 61, 5, 13, 21],
    [22, 30, 38, 46, 54, 62, 6, 14],
    [15, 23, 31, 39, 47, 55, 63, 7],
]) * 255

BRICK = (1 / 64) * np.array([
    [0, 16, 28, 38, 41, 33, 22, 8],
    [44, 1, 17, 29, 34, 23, 9, 49],
    [54, 45, 2, 18, 24, 10, 50, 57],
    [60, 42, 35, 3, 11, 30, 39, 62],
    [43, 36, 25, 12, 4, 19, 31, 40],
    [37, 26, 13, 51, 46, 5, 20, 32],
    [27, 14, 52, 58, 55, 47, 6, 21],
    [15, 53, 59, 61, 63, 56, 48, 7],
]) * 255

RHOMBUS = (1 / 64) * np.array([
    [0, 2, 4, 6, 8, 10, 12, 14],
    [22, 24, 26, 28, 30, 32, 16, 43],
    [44, 46, 48, 50, 34, 18, 41, 57],
    [58, 60, 52, 36, 20, 39, 55, 63],
    [62, 54, 38, 21, 37, 53, 61, 9],
    [56, 40, 19, 35, 51, 49, 47, 45],
    [42, 17, 33, 31, 29, 27, 25, 23],
    [15, 13, 11, 9, 7, 5, 3, 1],
]) * 255

HALFTONE = (1 / 64) * np.array([
    [60, 52, 48, 32, 36, 44, 53, 61],
    [56, 40, 28, 16, 20, 24, 41, 57],
    [47, 27, 12,  4,  8, 13, 29, 49],
    [39, 23, 11,  0,  1,  5, 17, 33],
    [35, 19,  7,  2,  3,  9, 21, 37],
    [51, 30, 15, 10,  6, 14, 25, 45],
    [59, 43, 26, 22, 18, 31, 42, 54],
    [63, 55, 46, 38, 34, 50, 58, 62],
]) * 255

HALFTONE_R = 255 - HALFTONE

ZIG_ZAG = (1 / 49) * np.array([
    [26, 22, 17, 15, 12, 30, 45],
    [48, 44, 24, 19,  7, 10, 32],
    [42, 37, 28,  6,  1, 14, 34],
    [39, 35,  4,  0,  3, 36, 40],
    [33, 13,  5,  2, 27, 38, 41],
    [31,  9,  8, 20, 23, 43, 47],
    [46, 29, 11, 16, 18, 21, 25],
]) * 255

MODES = ['floydsteinberg', 'floydsteinberg_g', 'bayer', 'bayer_g', 'halftone', 'halftone_g', 'brick', 'brick_g']


# DITHERING ALGORITHMS #################################################################################################

def ordered_dither(image, grayscale=False, pattern=BAYER_MATRIX_8X8):
    """Ordered dithering using pattern matrix."""
    if grayscale:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY).astype(np.float32)[:, :, np.newaxis]

    # Get image dimensions.
    height, width, _ = image.shape

    # Create Bayer threshold map matching image dimensions.
    threshold_map = np.tile(pattern, (height // pattern.shape[0] + 1, width // pattern.shape[0] + 1))[:height, :width]
    threshold_map = threshold_map[:, :, np.newaxis]

    # Apply dithering.
    dithered_image = (image > threshold_map) * 255
    return dithered_image.astype(np.uint8)


@numba.njit
def floyd_steinberg_dither_fast(img):
    """FS is a very slow algorithm (in Python). numba compiles the function to machine code, which is much faster.
    Brief intro to numba: https://numba.readthedocs.io/en/stable/user/5minguide.html"""
    height, width, ch = img.shape
    for y in range(height):
        for x in range(width):
            for c in range(ch):
                old_pixel = img[y, x, c]
                new_pixel = 0.0 if old_pixel < 128 else 255.0
                img[y, x, c] = new_pixel
                quant_error = old_pixel - new_pixel

                if x + 1 < width:
                    img[y, x + 1, c] += quant_error * 7 / 16
                if x - 1 >= 0 and y + 1 < height:
                    img[y + 1, x - 1, c] += quant_error * 3 / 16
                if y + 1 < height:
                    img[y + 1, x, c] += quant_error * 5 / 16
                if x + 1 < width and y + 1 < height:
                    img[y + 1, x + 1, c] += quant_error * 1 / 16
    return img


def floyd_steinberg_dither(image, grayscale=False):
    """Dither according to Floyd-Steinberg Algorithm.
    Obtained primarily from https://research.cs.wisc.edu/graphics/Courses/559-s2004/docs/floyd-steinberg.pdf"""
    if grayscale:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY).astype(np.float32)[:, :, np.newaxis]
        dithered_image = floyd_steinberg_dither_fast(gray.astype(np.float32).copy())
    else:
        dithered_image = floyd_steinberg_dither_fast(image.astype(np.float32).copy())
    return dithered_image.astype(np.uint8)


def dither(image, mode):
    """Use different dithering algo according to mode."""
    match mode:
        case 'floydsteinberg':
            return floyd_steinberg_dither(image)
        case 'floydsteinberg_g':
            return floyd_steinberg_dither(image, True)
        case 'bayer':
            return ordered_dither(image)
        case 'bayer_g':
            return ordered_dither(image, True)
        case 'halftone':
            return ordered_dither(image, pattern=HALFTONE_R)
        case 'halftone_g':
            return ordered_dither(image, True, HALFTONE_R)
        case 'brick':
            return ordered_dither(image, pattern=BRICK)
        case 'brick_g':
            return ordered_dither(image, True, BRICK)

        case _:
            # Default to no dithering.
            return image


# MAIN #################################################################################################################

def main():
    # Capture video.
    cap = cv.VideoCapture(0)  # Replace 0 w another integer for a different camera, or w a str path for an existing file

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    mode = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        mode_name = MODES[mode]

        if not ret:
            # Frame is not read correctly.
            print("Can't receive frame or stream end. Exiting ...")
            break

        # Process the frame.
        # small = cv.resize(frame, (960, 540))  # May need to resize to improve processing speeds.
        # dithered = dither(small, MODES[mode])
        dithered = dither(frame, mode_name)

        # Display processed frame.
        cv.imshow(f'{mode_name}', dithered)

        # Detect keypresses.
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            # Quit.
            break
        elif key == ord('m'):
            # Switch mode.
            cv.destroyWindow(f'{mode_name}')
            mode = (mode + 1) % len(MODES)
        elif key == ord('g'):
            # Toggle Grayscale.
            cv.destroyWindow(f'Preview: {mode_name}')
            if mode_name.endswith('_g'):
                # In color, switch to grayscale.
                mode = (mode - 1) % len(MODES)
            else:
                # In grayscale, switch to color.
                mode = (mode + 1) % len(MODES)

    # Release capture and Destroy all windows.
    cap.release()
    cv.destroyAllWindows()


def main_2():
    """Cast to virtual webcam."""
    # Capture video.
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    mode = 0
    ret, frame = cap.read()
    height, width, _ = frame.shape

    with pyvirtualcam.Camera(width=width, height=height, fps=30) as cam:
        print(f'Virtual camera started: {cam.device}')

        while True:
            ret, frame = cap.read()
            if not ret:
                # Frame is not read correctly.
                print("Can't receive frame or stream end. Exiting ...")
                break

            mode_name = MODES[mode]

            # Process the frame.
            # small = cv.resize(frame, (960, 540))  # May need to resize to improve processing speeds.
            # dithered = cv.resize(dither(small, MODES[mode]), (width, height))
            dithered = dither(frame, mode_name)

            # pyvirtualcam needs RGB.
            if mode_name.endswith('_g'):
                rgb_frame = cv.cvtColor(dithered, cv.COLOR_GRAY2RGB)
            else:
                rgb_frame = cv.cvtColor(dithered, cv.COLOR_BGR2RGB)

            cam.send(rgb_frame)
            cam.sleep_until_next_frame()

            # Display processed frame.
            cv.imshow(f'Preview: {mode_name}', dithered)

            # Detect keypresses.
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                # Quit.
                break
            elif key == ord('m'):
                # Switch mode.
                cv.destroyWindow(f'Preview: {mode_name}')
                mode = (mode + 2) % len(MODES)
            elif key == ord('g'):
                # Toggle Grayscale.
                cv.destroyWindow(f'Preview: {mode_name}')
                if mode_name.endswith('_g'):
                    # In color, switch to grayscale.
                    mode = (mode - 1) % len(MODES)
                else:
                    # In grayscale, switch to color.
                    mode = (mode + 1) % len(MODES)

    # Release capture and Destroy all windows.
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # main()
    main_2()  # Also cast to virtual webcam (otherwise same functionality as main()).
