# Memory vs Attention Simulator

This project is an interactive Explainer that simulates and contrasts Recurrent Memory and Full-Context Attention approaches in machine learning. It features a React frontend and a FastAPI backend written in Python.

## Prerequisites

Before running the project, please ensure you have the following installed on your machine:
- **Node.js** (v18 or higher) and npm
- **Python** (v3.8 or higher)

---

## Setup Instructions

When you extract this project for the first time, you will need to set up the dependencies for both the frontend and the backend.

### 1. Install Frontend Dependencies
Open a terminal in the root folder of this project (where this README is located) and run:
```bash
npm install
```

### 2. Set Up the Backend Environment
Next, you need to create a Python virtual environment and install the required Python packages. 

Run the following commands in your terminal:
```bash
cd backend
python -m venv venv

# If you are on Windows, activate the virtual environment using:
.\venv\Scripts\activate
# If you are on macOS or Linux, activate the virtual environment using:
# source venv/bin/activate

pip install -r requirements.txt
cd ..
```

---

## Running the Application

Once everything is installed, you can start both the frontend and the backend simultaneously using a single command. 

Ensure you are in the root folder of the project, and simply run:
```bash
npm run dev
```

This command will automatically start:
1. The Vite Frontend server (typically running on `http://localhost:5173`)
2. The Python FastAPI Backend server (typically running on `http://127.0.0.1:8001`)

Open the frontend URL in your browser to interact with the simulation. To stop the application, simply press `Ctrl + C` in your terminal.
