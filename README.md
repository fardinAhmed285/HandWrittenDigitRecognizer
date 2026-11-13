# Handwritten Digit Recognizer

A web-based handwritten digit recognition application built with **Python, PyTorch, and Streamlit**. Users can draw a digit from 0 to 9 on an interactive canvas and receive a prediction with its confidence score.

The application also includes a feedback mechanism and PostgreSQL integration to store prediction records and display recent prediction history.

<escape>
</escape>

## Live Application

**[Try the Handwritten Digit Recognizer](https://handwrittendigitrecognizer-production.up.railway.app/)**

**[View Source Code on GitHub](https://github.com/fardinAhmed285/HandWrittenDigitRecognizer)**

## Project Overview

Handwritten digit recognition is a computer vision task that involves identifying numerical digits from images.

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits into one of ten classes (0–9).

A Streamlit interface allows users to draw digits directly in their browser. The drawing is processed into an image format suitable for the trained model, which then returns its predicted digit and confidence score.

Beyond prediction, the application includes a feedback system that records predictions and user-provided correct labels in a PostgreSQL database.

## Key Features

- **Interactive drawing canvas:** Draw handwritten digits directly in the browser.
- **Deep learning predictions:** Classify handwritten digits from 0 to 9 using a CNN.
- **Confidence scores:** Display the model's confidence in its prediction.
- **User feedback:** Submit the correct digit when a prediction is incorrect.
- **Prediction history:** View recent predictions and feedback records.
- **PostgreSQL integration:** Store prediction data with timestamps.
- **Containerisation:** Includes Docker and Docker Compose configuration.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| PyTorch | CNN architecture and inference |
| Streamlit | Interactive web interface |
| NumPy | Numerical and image-array operations |
| OpenCV | Image preprocessing |
| Pillow | Image handling |
| Streamlit Drawable Canvas | Interactive digit drawing |
| PostgreSQL | Prediction and feedback storage |
| psycopg2 | PostgreSQL connectivity |
| Pandas | Displaying prediction history |
| Docker | Application containerisation |
| Docker Compose | Multi-container configuration |

## CNN Architecture

The application uses a custom Convolutional Neural Network implemented with PyTorch.

The architecture consists of:

| Layer | Configuration |
|---|---|
| Input | 1-channel grayscale image |
| Convolutional Layer 1 | 10 filters, 5 × 5 kernel |
| Activation | ReLU |
| Pooling | 2 × 2 Max Pooling |
| Convolutional Layer 2 | 20 filters, 5 × 5 kernel |
| Dropout | 2D Dropout |
| Activation | ReLU |
| Pooling | 2 × 2 Max Pooling |
| Flatten | 320 features |
| Fully Connected Layer 1 | 50 neurons, ReLU |
| Dropout | Dropout regularisation |
| Output Layer | 10 neurons |
| Output Activation | Softmax |

The final output represents the model's predicted probabilities for the ten digit classes.

The trained model is stored in `model.pth` and loaded by the application for inference.

## Image Preprocessing

Before a drawing is passed to the CNN, it is converted into the expected input format.

The preprocessing pipeline performs the following operations:

1. Converts the canvas image from RGBA to grayscale.
2. Resizes the image to 28 × 28 pixels.
3. Converts pixel values to floating-point numbers.
4. Normalises pixel values to the range [0, 1].
5. Converts the image into a PyTorch tensor.
6. Reshapes the tensor to the CNN input format.

The final tensor has the shape:

```python
(1, 1, 28, 28)
```

This represents one grayscale image with a height and width of 28 pixels.

## Application Workflow

The application follows this workflow:

1. The user draws a digit using the Streamlit canvas.
2. The application checks whether the canvas contains a drawing.
3. The image is converted to grayscale and resized.
4. The preprocessed image is passed to the trained CNN.
5. PyTorch generates class probabilities.
6. The application identifies the digit with the highest probability.
7. The predicted digit and confidence score are displayed.
8. The user can submit feedback if the prediction is incorrect.
9. Prediction and feedback information can be stored in PostgreSQL.

## Database Integration

The application uses PostgreSQL to maintain prediction records.

The `predictions` table is designed to store:

| Column | Description |
|---|---|
| `id` | Unique prediction record ID |
| `timestamp` | Date and time of the prediction |
| `predicted_digit` | Digit predicted by the CNN |
| `true_label` | User-provided correct digit |

The application also retrieves the ten most recent records and displays them in a Streamlit dataframe.

This provides a basic feedback and prediction-tracking mechanism.

## Repository Structure

```text
HandWrittenDigitRecognizer/
│
├── main.py
├── model.pth
├── requirements.txt
├── create_table.py
├── Dockerfile
├── docker-compose.yaml
├── ProcFile
└── README.md
```

| File | Description |
|---|---|
| `main.py` | Streamlit application, CNN definition, image preprocessing, prediction, and feedback handling |
| `model.pth` | Saved PyTorch model weights |
| `requirements.txt` | Python dependencies |
| `create_table.py` | PostgreSQL table creation script |
| `Dockerfile` | Instructions for building the application container |
| `docker-compose.yaml` | Configuration for the application and PostgreSQL services |
| `ProcFile` | Deployment process configuration |
| `README.md` | Project documentation |

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/fardinAhmed285/HandWrittenDigitRecognizer.git
```

### 2. Navigate to the project directory

```bash
cd HandWrittenDigitRecognizer
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the database

The application uses PostgreSQL for storing prediction feedback and history.

Configure the database connection using a `DATABASE_URL` environment variable before starting the application. The connection code currently reads this variable directly.

Ensure the `predictions` table exists before using the feedback and history features.

### 6. Run the application

```bash
streamlit run main.py
```

Open the local Streamlit address shown in your terminal to use the application.

## Running with Docker

The repository also includes Docker configuration.

To build and start the application and PostgreSQL services, use:

```bash
docker compose up --build
```

The Streamlit application is configured to use port `8501`, while PostgreSQL uses port `5432`.

Make sure the database connection environment is configured consistently between the application and the PostgreSQL service before relying on database features.

## Model Evaluation

The repository includes the trained model weights (`model.pth`) and the inference application.

A training notebook or training script and reproducible evaluation metrics are not currently included in the repository. Therefore, no test accuracy or other benchmark figures are claimed here.

Adding the training workflow, dataset details, and test-set evaluation results would make the project's machine learning performance easier for recruiters and other developers to assess.

## Future Improvements

- Add a documented model training pipeline.
- Include test accuracy, precision, recall, and F1-score.
- Add a confusion matrix to analyse classification performance.
- Improve preprocessing for different handwriting styles.
- Support image uploads in addition to canvas drawings.
- Add more detailed prediction analytics and visualisations.
- Improve database configuration and error handling.
- Add automated tests for preprocessing, prediction, and database operations.
- Explore model optimisation for faster inference.

## Author

**Fardin Ahmed**

GitHub: [@fardinAhmed285](https://github.com/fardinAhmed285)

## Project Links

- [GitHub Repository](https://github.com/fardinAhmed285/HandWrittenDigitRecognizer)
- [Live Application](https://handwrittendigitrecognizer-production.up.railway.app/)

---

*This project demonstrates practical application of deep learning, computer vision, interactive web development, database integration, and containerisation.*