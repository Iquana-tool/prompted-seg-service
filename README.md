# Prompted Segmentation
This repo contains the API for the prompted segmentation service of the CORAL tag application.

## Running the service
### 1. Using Docker [recommended]
> Not implemented yet!
### 2. Using the [uv package manager](https://docs.astral.sh/uv/)
1. Install uv following their installation guide
2. Run the following:
```shell
# Install dependencies
uv sync

# Install Pytorch; follow instructions at https://pytorch.org/get-started/locally/
uv add torch torchvision

# Install SAM2 and its dependencies; needs special installation
git clone https://github.com/facebookresearch/sam2.git
cd sam2 
uv pip install -e ".[notebooks]" --no-cache-dir 
cd ..

# Run the server
uv run fastapi run main.py --port 8000
```
