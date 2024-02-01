import cv2
import pytesseract
import numpy as np
import imutils

# Path to your Tesseract executable
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def extract_text_from_frame(frame):
    try:
        frame = imutils.resize(frame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
        edged = cv2.Canny(bfilter, 30, 200)  # Edge detection

        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        location = None

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break

        mask = np.zeros(gray.shape, np.uint8)

        if location is not None:
            new_image = cv2.drawContours(mask, [location], 0, 255, -1)
            new_image = cv2.bitwise_and(gray, gray, mask=mask)
            (x, y) = np.where(mask == 255)
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
            cropped_image = cv2.bilateralFilter(cropped_image, 11, 17, 17)  # Noise reduction
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            cropped_image = cv2.filter2D(cropped_image, ddepth=-1, kernel=kernel)
            cropped_image = imutils.resize(cropped_image, height=50)

            # Define character whitelist and Tesseract configuration options
            alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            options = "-c tessedit_char_whitelist={}".format(alphanumeric)
            options += " --psm {}".format(13)

            reader = pytesseract.image_to_string(cropped_image, config=options)
            result = reader
            text = "".join([c if ord(c) < 128 else "" for c in result]).strip()
            return frame, text
        else:
            return frame, "Text not found"
    except Exception as e:
        return frame, f"Error: {str(e)}"

# Open a video capture
cap = cv2.VideoCapture(r'C:\Users\Cloudanalogy\Desktop\Parking-system-with-ANPR\vid00.mp4')  # Replace 'your_video.mp4' with the path to your video file

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result_frame, extracted_text = extract_text_from_frame(frame)

    cv2.imshow('Result Frame', result_frame)
    print("Extracted Text:", extracted_text)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
