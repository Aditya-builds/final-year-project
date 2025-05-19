import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API Key from environment variable or hard-code it here
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-api-key-here")

# Run the Streamlit app
if __name__ == "__main__":
    import subprocess
    subprocess.run(["streamlit", "run", "app.py"])