import os
import cv2
import pytesseract

# Tesseract binary path
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image Path {image_path} not found or could not be loaded.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 50, 200)
    return edged, image


def detect_rectangles(edged):
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 1000]
    return rectangles


def extract_text_from_rectangles(rectangles, image):
    classes = []
    for rect in rectangles:
        x, y, w, h = rect
        roi = image[y:y + h, x:x + w]
        text = pytesseract.image_to_string(roi, config='--psm 6')
        if text.strip():
            classes.append(text.strip())
    return classes


def clean_text(text):
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        cleaned_line = line.strip()
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    return cleaned_lines


def generate_plantuml(classes):
    plantuml_syntax = "@startuml\n"
    seen_classes = set()
    for class_text in classes:
        lines = clean_text(class_text)
        if lines:
            class_name = lines[0]
            if class_name.startswith('<') and class_name.endswith('>'):
                if len(lines) > 1:
                    class_name = lines[1]
                    lines = lines[1:]
                else:
                    continue
            if len(class_name) == 3 and len(lines) == 1:
                continue
            if class_name in seen_classes:
                continue
            seen_classes.add(class_name)
            plantuml_syntax += f"class {class_name} {{\n"
            for attribute in lines[1:]:
                plantuml_syntax += f"  {attribute}\n"
            plantuml_syntax += "}\n"
    plantuml_syntax += "@enduml"
    return plantuml_syntax


def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    all_classes = []
    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)
        if os.path.isfile(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                edged, image = preprocess_image(image_path)
                rectangles = detect_rectangles(edged)
                classes = extract_text_from_rectangles(rectangles, image)

                if classes:
                    all_classes.append(classes[0])  # Her resim dosyası için sadece bir sınıf

                    # Save text to the output directory
                    text_filename = os.path.splitext(filename)[0] + ".txt"
                    text_path = os.path.join(output_dir, text_filename)
                    with open(text_path, "w") as text_file:
                        text_file.write(classes[0])
                    print(f"Text saved to {text_path}")

            except FileNotFoundError as e:
                print(e)

    return all_classes


def run_text(input_base_dir, output_base_dir):
    if not os.path.isdir(input_base_dir):
        raise FileNotFoundError(f"Input directory {input_base_dir} not found.")

    os.makedirs(output_base_dir, exist_ok=True)  # Creating the base output directory

    for folder in os.listdir(input_base_dir):
        input_dir = os.path.join(input_base_dir, folder)
        if os.path.isdir(input_dir):
            output_dir = os.path.join(output_base_dir, folder, "output")  # Output directory structure
            os.makedirs(output_dir, exist_ok=True)

            all_classes = process_directory(input_dir, output_dir)
            plantuml_syntax = generate_plantuml(all_classes)

            # Save the PlantUML syntax to the output directory
            output_path = os.path.join(output_dir, "output.puml")
            with open(output_path, "w") as f:
                f.write(plantuml_syntax)
            print(f"PlantUML syntax saved to {output_path}")



