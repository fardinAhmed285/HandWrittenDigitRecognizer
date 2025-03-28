import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from datetime import datetime
from PIL import Image
import io
import streamlit_drawable_canvas as canvas
import psycopg2
import pandas as pd
import os

def ensure_table_exists():
    try:
        conn = connect_db()
        cur = conn.cursor()

        # Create the table if it doesn't exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                predicted_digit INTEGER,
                true_label INTEGER
            );
        """)

        conn.commit()
        cur.close()
        conn.close()

        print("Table check complete: 'predictions' exists.")
    
    except Exception as e:
        print(f"Error ensuring table exists: {e}")


ensure_table_exists()

# Getting database connection details from environment variables

DB_URL = os.getenv("DATABASE_URL")
DB_HOST = os.getenv("DB_HOST", "db")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "DBMS123")
DB_PORT = os.getenv("DB_PORT", "5432")

# Function to connect to PostgreSQL
def connect_db():
    return psycopg2.connect(DB_URL)



# Function to log data into the database
def log_prediction(predicted_digit, true_label):
    try:
        # Connect to the database
        conn = connect_db()
        cur = conn.cursor()
        
        # Debug print
        print(f"🔹 Logging feedback: Predicted={predicted_digit}, True Label={true_label}")
        
        # Insert the data into the table
        cur.execute(
            "INSERT INTO predictions (predicted_digit, true_label) VALUES (%s, %s)",
            (predicted_digit, true_label),
        )
        
        # Commit the transaction
        conn.commit()
        cur.close()
        conn.close()
        
        # Debug print
        print("✅ Data successfully saved to the database!")
        
    except Exception as e:
        # Print any error
        print(f"❌ Error saving to DB: {e}")


# Function to fetch prediction history
def fetch_history(limit=10):
    try:
        conn = connect_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, timestamp, predicted_digit, true_label FROM predictions ORDER BY timestamp DESC LIMIT %s", 
            (limit,)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        st.error(f"❌ Error fetching history: {e}")
        return []


# Defined the CNN model (same architecture as used during training)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        
        if self.training:  # Only apply dropout during training
            x = F.dropout(x, training=True)

        x = self.fc2(x)
        return F.softmax(x, dim=1)

model = CNN()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

st.title("Handwritten Digit Recognizer")
st.write("Draw a digit (0-9) below and click 'Predict'")

canvas_result = canvas.st_canvas(
    fill_color="black",  # Background color
    stroke_width=15,
    stroke_color="white",  # Drawing color
    background_color="black",  # Canvas background
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

def preprocess_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (28, 28))  # Resize to 28x28
    image = image.astype(np.float32) / 255.0  # Normalize
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)  # Reshape to (1,1,28,28)
    return image
    
# Store prediction to prevent clearing on button press
if "prediction" not in st.session_state:
    st.session_state["prediction"] = None
if "confidence_scores" not in st.session_state:
    st.session_state["confidence_scores"] = None

if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert canvas data to NumPy array
        image_array = (canvas_result.image_data[:, :, :3] * 255).astype(np.uint8)
        
        # Check if the image is completely black (all pixel values are 0)
        if np.sum(image_array) == 0:  
            st.warning("Canvas is empty! Please draw a digit before predicting.")
        else:
            # Convert to PIL Image
            image = Image.fromarray(image_array)
            processed_image = preprocess_image(image)
            
            with torch.no_grad():
                output = model(processed_image)
                
                # Get the confidence for all digits (0-9)
                confidence_scores = output.squeeze().tolist()  # Convert the tensor to a list
                
                # Display the confidence for each digit
               # for i in range(10):
                #    st.write(f"**Digit {i}:** {confidence_scores[i] * 100:.2f}% confidence")

                # Get the predicted digit and its confidence
                predicted_digit = torch.argmax(output, dim=1).item()
                predicted_confidence = torch.max(output).item() * 100  # Convert to percentage
            
            # Save prediction in session state
            st.session_state["prediction"] = predicted_digit
            st.session_state["confidence_scores"] = confidence_scores

           # st.write(f"**Predicted Digit:** {predicted_digit}")
           # st.write(f"**Confidence:** {predicted_confidence}")
           

# Display prediction if available
if st.session_state["prediction"] is not None:
    st.write(f"### **Predicted Digit:** {st.session_state['prediction']}")
    st.write(f"**Confidence in Predicted Digit:** {st.session_state['confidence_scores'][st.session_state['prediction']] * 100:.2f}%")
    # User input for correct digit
    true_label = st.number_input("Enter the correct digit (if incorrect):", min_value=0, max_value=9, step=1)

    # Submit feedback button
    if st.button("Submit Feedback"):
        log_prediction(st.session_state["prediction"], true_label)




#  Display Prediction History
st.subheader("Prediction History")

history = fetch_history(limit=10)

if history:
    df = pd.DataFrame(history, columns=["ID", "Timestamp", "Predicted Digit", "True Label"])
    st.dataframe(df)
else:
    st.write("No history available.")
