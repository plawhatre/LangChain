#!/bin/bash

# Update package lists and install curl if not present
sudo apt update && sudo apt install -y curl

# Download and run the Ollama install script
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama -v

echo "Ollama installation complete."

# Download the Mistral model
ollama pull mistral:v0.3

# Create a virtual environment
python3 -m venv env_langchain