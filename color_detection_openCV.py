import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Define color ranges (HSV) + Box Color (BGR)
# Note: In OpenCV, HSV ranges are H: 0-180, S: 0-255, V: 0-255
colors = {
    "Red": {
        "ranges": [
            {"lower": [0, 120, 70], "upper": [10, 255, 255]},
            {"lower": [170, 120, 70], "upper": [180, 255, 255]}
        ],
        "box_color": (0, 0, 255)
    },
    "Orange": {
        "ranges": [{"lower": [11, 100, 100], "upper": [24, 255, 255]}],
        "box_color": (0, 165, 255)
    },
    "Yellow": {
        "ranges": [{"lower": [25, 100, 100], "upper": [34, 255, 255]}],
        "box_color": (0, 255, 255)
    },
    "Lime": {
        "ranges": [{"lower": [35, 100, 100], "upper": [45, 255, 255]}],
        "box_color": (50, 205, 50)
    },
    "Green": {
        "ranges": [{"lower": [46, 50, 50], "upper": [75, 255, 255]}],
        "box_color": (0, 255, 0)
    },
    "Teal": {
        "ranges": [{"lower": [76, 100, 80], "upper": [85, 255, 255]}],
        "box_color": (128, 128, 0)
    },
    "Cyan": {
        "ranges": [{"lower": [86, 100, 100], "upper": [100, 255, 255]}],
        "box_color": (255, 255, 0)
    },
    "Blue": {
        "ranges": [{"lower": [101, 150, 70], "upper": [130, 255, 255]}],
        "box_color": (255, 0, 0)
    },
    "Navy Blue": {
        "ranges": [{"lower": [105, 150, 20], "upper": [135, 255, 120]}],
        "box_color": (128, 0, 0)
    },
    "Purple": {
        "ranges": [{"lower": [131, 80, 50], "upper": [145, 255, 255]}],
        "box_color": (255, 0, 128)
    },
    "Magenta": {
        "ranges": [{"lower": [146, 100, 100], "upper": [155, 255, 255]}],
        "box_color": (255, 0, 255)
    },
    "Pink": {
        "ranges": [{"lower": [156, 50, 100], "upper": [169, 255, 255]}],
        "box_color": (203, 192, 255)
    },
    "Brown": {
        "ranges": [{"lower": [8, 80, 20], "upper": [20, 200, 150]}],
        "box_color": (19, 69, 139)
    },
    "Maroon": {
        "ranges": [{"lower": [0, 120, 20], "upper": [10, 255, 120]}],
        "box_color": (0, 0, 128)
    },
    "White": {
        "ranges": [{"lower": [0, 0, 200], "upper": [180, 30, 255]}],
        "box_color": (255, 255, 255)
    },
    "Gray": {
        "ranges": [{"lower": [0, 0, 50], "upper": [180, 40, 180]}],
        "box_color": (128, 128, 128)
    }
}

kernel = np.ones((5, 5), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color_name, color_info in colors.items():
        box_color = color_info["box_color"]
        
        # Combine multiple HSV ranges if required (e.g., for Red)
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for r in color_info["ranges"]:
            lower = np.array(r["lower"])
            upper = np.array(r["upper"])
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Remove noise
        mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(cnt)

                # Center the square box over the contour
                center_x, center_y = x + w // 2, y + h // 2
                size = max(w, h)
                x_square = max(0, center_x - size // 2)
                y_square = max(0, center_y - size // 2)

                cv2.rectangle(frame, (x_square, y_square), (x_square + size, y_square + size), box_color, 2)

                # Label background for visual clarity
                label_y = max(15, y_square - 10)
                cv2.putText(frame, color_name, (x_square, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    cv2.imshow("Multi-Color Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()