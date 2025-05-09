# dithering-with-opencv
Examples of a few different dithering algorithms using opencv-python with a live webcam.

This program shows the effect of dithering algorithms on images. This is meant mostly as a learning experience for me
and a tutorial for others, so I added a good amount of documentation. This includes the standard Floyd-Steinberg and
Ordered Bayer algorithms, as well as a few custom Ordered patterns. If you use main_2(), you can also use the output as
a virtual webcam (for Zoom, OBS, etc.)

Controls:
m - switch dithering mode
g - toggle between grayscale and color
q - quit the program
