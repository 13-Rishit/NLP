import os
import io
import cv2
from google.cloud import vision
from google.cloud import translate_v2 as translate

# Initialize Google Cloud Clients
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"nlp-6942-962654b9d1d9.json"

vision_client = vision.ImageAnnotatorClient()
translate_client = translate.Client()

# Function to capture image from webcam
def capture_image():
    # Use OpenCV to capture image from the webcam
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Press Space to Capture Image', frame)
        
        if cv2.waitKey(1) & 0xFF == ord(' '):
            image_path = 'captured_image.jpg'
            cv2.imwrite(image_path, frame)
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return image_path

# Function to perform OCR using Google Vision API
def extract_text_from_image(image_path):
    # Load image into memory
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    
    # Perform text detection
    response = vision_client.text_detection(image=image)
    texts = response.text_annotations
    
    if response.error.message:
        raise Exception(f'{response.error.message}')
    
    # Extract the detected text
    if texts:
        detected_text = texts[0].description
        print(f"Detected Text: {detected_text}")
        return detected_text
    else:
        print("No text detected.")
        return None

# Function to translate text using Google Translate API
def translate_text(text, target_language='en'):
    translation = translate_client.translate(text, target_language=target_language)
    translated_text = translation['translatedText']
    print(f"Translated Text: {translated_text}")
    return translated_text

# Main workflow
def main():
    print("Starting the Google Lens-like app...")

    # Step 1: Capture image from camera
    image_path = capture_image()
    
    # Step 2: Extract text from the captured image
    detected_text = extract_text_from_image(image_path)
    
    if detected_text:
        # Step 3: Translate the detected text
        target_language = input("Enter target language code (e.g., 'en' for English, 'es' for Spanish): ")
        translated_text = translate_text(detected_text, target_language)

        # Output the translated text
        print(f"Final Translation: {translated_text}")
    else:
        print("No text found to translate.")

if __name__ == "__main__":
    main()
