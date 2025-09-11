Car Colour Detection Model
This Streamlit application analyzes a traffic image to detect cars and people. It applies specific rules for visualizing the detected cars based on their color and provides a count of all detected objects.

Core Features
Object Detection: The application uses the YOLOv3 model (via the cvlib library) to accurately detect common objects like cars and people in an uploaded image.

Conditional Coloring: The model implements a specific visualization rule:

Blue Cars: A red rectangle is drawn around any car identified as blue.

Other Cars: A blue rectangle is drawn around cars of any other color.

Object Counting: After processing the image, the application displays the total number of cars and people detected.

Modern Dark UI: The application features a custom dark theme for a professional look and better readability.

How to Run the Project
Open a Terminal: Launch your command line tool (PowerShell, Command Prompt, etc.).

Activate Virtual Environment: Navigate to the main submission folder and activate the virtual environment:

# Navigate to the main folder
cd path/to/Nullclass Internship Submission

# Activate the environment
.\venv\Scripts\Activate.ps1

Navigate to Project Folder: Move into this project's directory:

cd "Car Colour Detection"

Install Requirements: Install the necessary Python libraries:

pip install -r requirements.txt

Run the App: Start the Streamlit application:

streamlit run "Car Colour Detection.py"

The application will open in your web browser, where you can upload a traffic image for analysis.