$ErrorActionPreference = "Stop"
cd "d:\Open CV Projects\multi-camera-ai"

Write-Host "Creating Virtual Environment..."
python -m venv venv

Write-Host "Activating venv and installing requirements... (This might take a minute)"
.\venv\Scripts\activate
pip install -r requirements.txt

Write-Host "Starting FastAPI Backend..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'd:\Open CV Projects\multi-camera-ai'; .\venv\Scripts\activate; uvicorn backend.main:app --host 0.0.0.0 --port 8000"

Write-Host "Starting Video Processing Engine..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'd:\Open CV Projects\multi-camera-ai'; .\venv\Scripts\activate; python pipeline/processor.py"

Write-Host "Starting Streamlit Dashboard..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'd:\Open CV Projects\multi-camera-ai'; .\venv\Scripts\activate; streamlit run dashboard/app.py"

Write-Host "All processes started! Three new windows should open showing the live processes."
