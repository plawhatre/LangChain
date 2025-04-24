#!/bin/bash

# Update package lists and install curl if not present
sudo apt update && sudo apt install -y curl

# Download and run the Ollama install script
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama -v

echo "Ollama installation complete."
