import torch
import torch.nn as nn
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Define the ANN model architecture
class StockPriceANN(nn.Module):
    def __init__(self, input_size):
        super(StockPriceANN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load scaler
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load model
input_size = scaler.n_features_in_ 
model = StockPriceANN(input_size)
model.load_state_dict(torch.load("models/StockPriceANN_model.pth", map_location=torch.device("cpu")))
model.eval()  # Set model to evaluation mode

# Initialize FastAPI
app = FastAPI()

# ✅ Enable CORS for all origins (You can restrict it later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change this in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define request schema
class StockFeatures(BaseModel):
    Open: float
    High: float
    Low: float
    Adj_Close: float
    Volume: float
    Year: int
    Month: int
    Day: int
    DayOfWeek: int

@app.post("/predict")
def predict(features: StockFeatures):
    # Convert input to numpy array
    input_data = np.array([[features.Open, features.High, features.Low, features.Adj_Close,
                            features.Volume, features.Year, features.Month, features.Day,
                            features.DayOfWeek]])
    
    # Scale input data
    input_scaled = scaler.transform(input_data)

    # Convert to tensor
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    # Get prediction
    with torch.no_grad():
        prediction = model(input_tensor).item()

    return {"predicted_stock_price": prediction}