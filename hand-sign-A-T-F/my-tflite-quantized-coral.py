# Nov06, 2021
# my-tifile-quantized-coral.py

# 1. Install the edgetpu library following Coral's official instructions
# https://coral.withgoogle.com/docs/edgetpu/api-intro/#install-the-library

# 2. pip install the following packages like so:
# pip3 install Pillow opencv-python opencv-contrib-python

# 3. Download modelfrom TM2

# 4. Use this code snippet to run this model on Edge TPU:
from edgetpu.classification.engine import ClassificationEngine
from PIL import Image
import cv2
import re
import time
import myPixels  # for LED on respeaker 2 mic
# apa102.py should be on the same dir.  myPixels.py imports it.

import board
import digitalio
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306


# the TFLite converted to be used with edgetpu
modelPath = 'converted_tflite_quantized/model.tflite'
# The path to labels.txt that was downloaded with your model
labelPath = 'converted_tflite_quantized/labels.txt'

# OLED dimensions
# dimension
WIDTH = 128
HEIGHT = 64
BORDER = 5


# This function parses the labels.txt and puts it in a python dictionary
def loadLabels(labelPath):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(labelPath, 'r', encoding='utf-8') as labelFile:
        lines = (p.match(line).groups() for line in labelFile.readlines())
        return {int(num): text.strip() for num, text in lines}


# This function takes in a PIL Image and the ClassificationEngine
def classifyImage(image, engine):
    # Classify and ouptut inference
    # classifications = engine.ClassifyWithImage(image)
    classifications = engine.classify_with_image(image)
    return classifications


def createOLED():
    # Define the Reset Pin
    oled_reset = digitalio.DigitalInOut(board.D4)
    # Use for I2C.
    i2c = board.I2C()
    oled = adafruit_ssd1306.SSD1306_I2C(WIDTH, HEIGHT, i2c,
                                        addr=0x3d,
                                        reset=oled_reset)
    return oled


def drawOLED(oled, text):
    # Create blank image for drawing.
    # Make sure to create image with mode '1' for 1-bit color.
    image = Image.new("1", (oled.width, oled.height))

    # Get drawing object to draw on image.
    draw = ImageDraw.Draw(image)

    # Draw a white background
    draw.rectangle((0, 0, oled.width, oled.height), outline=255, fill=255)

    # Draw a smaller inner rectangle
    draw.rectangle(
        (BORDER, BORDER, oled.width - BORDER - 1, oled.height - BORDER - 1),
        outline=0,
        fill=0,
    )

    # Load default font.
    font = ImageFont.load_default()

    # Draw Some Text
    (font_width, font_height) = font.getsize(text)
    draw.text(
        (oled.width // 2 - font_width // 2,
         oled.height // 2 - font_height // 2),
        text,
        font=font,
        fill=255,
    )

    # Display image
    oled.image(image)
    oled.show()


def main():
    # OLED
    oled = createOLED()
    # Clear display.
    oled.fill(0)
    oled.show()

    drawOLED(oled, "Hello.")
    time.sleep(3)
    drawOLED(oled, "watching...")

    # LED on respeaker
    pixels = myPixels.Pixels()
    leds = (pixels.led1, pixels.led2, pixels.led3)

    # Load your model onto your Coral Edgetpu
    engine = ClassificationEngine(modelPath)
    labels = loadLabels(labelPath)

    cap = cv2.VideoCapture(0)

    bucket = []

    # since cv2.waitKey() does not work,I use Ctrl + C
    # this is to do after break wrapping up
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Format the image into a PIL Image so its compatable with Edge TPU
            cv2_im = frame
            cv2_im_rotated = cv2.rotate(cv2_im, cv2.ROTATE_180)
            pil_im = Image.fromarray(cv2_im_rotated)

            # Resize and flip image so its a square and matches training
            pil_im.resize((224, 224))
            pil_im.transpose(Image.FLIP_LEFT_RIGHT)

            # Classify and display image
            results = classifyImage(pil_im, engine)
            # cv2.imshow('frame', cv2_im)
            # print(results)
            call = labels[results[0][0]]
            prob = results[0][1]
            print(call, prob)
            if call != 'none':
                if len(bucket) == 0:
                    bucket.append(call)
                else:
                    if call in bucket:
                        bucket.append(call)
                    else:
                        bucket = [call]
                if len(bucket) >= 10:
                    # OLED
                    msg = call + ': ' + str(prob)
                    drawOLED(oled, msg)
                    # turn LED on
                    # 0: led1, red -> Aircon
                    # 1: led2, green -> TV
                    # 2: led3, blue -> S
                    leds[results[0][0]]()
                    time.sleep(2)
                    drawOLED(oled, "watching...")
                    pixels.off()
                    # TODO: IR blaster
                    bucket = []
            else:
                bucket = []

            # this does not work... just leave it
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        drawOLED(oled, "Ctrl + C...")


if __name__ == '__main__':
    main()
