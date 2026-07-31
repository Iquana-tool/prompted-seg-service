# Prompted Segmentation

> [!CAUTION]
> **Deprecated — this service is no longer maintained or deployed.**
>
> It has been merged into the unified **[IQUANA AI service](https://github.com/Iquana-tool/ai-service)**,
> which serves every AI task from one model-centric codebase. The per-task split was dropped because
> models cross task boundaries (SAM3 does prompted segmentation *and* instance suggestion *and* more),
> which forced the same model to be reimplemented in up to three repos.
>
> **Where this service's functionality lives now:** the `prompted-segmentation` task surface of the
> unified service, mounted at `http://<host>:8004/prompted-segmentation`. Relative paths are unchanged,
> so `POST /inference` becomes `POST /prompted-segmentation/inference`. The backend picks this up from a
> single `AI_SERVICE_URL` setting; drop any `PROMPTED_SEGMENTATION_BACKEND_URL` pin from your `.env`.
>
> **Do not add models or features here.** Port them to `ai-service/models/` against the capability-model
> interface instead.
>
> Everything below is kept for historical reference only.

This repo contains the API for the prompted segmentation service of the CORAL tag application.

## Running the service
### 1. Using Docker [recommended]
1. Make sure you have Docker installed on your machine.
2. Build the Docker image:
   1. Without CUDA support:
    ```shell
    docker build -t prompted-seg-service .
    ```
    2. With CUDA support:
    ```shell
    docker build -t prompted-seg-service -f Dockerfile.cuda .
    ```
3. Run the Docker container:
```shell
docker run -d -p 8000:8000 prompted-seg-service
```
### 2. Using the [uv package manager](https://docs.astral.sh/uv/)
1. Install uv following their installation guide
2. Install initial dependencies (independent of CUDA support):
```shell
# Install initial dependencies
uv sync
```
3. Install torch with or without CUDA support depending on your hardware (see: https://pytorch.org/get-started/locally/ for instructions)
```shell
# Install Pytorch; follow instructions at https://pytorch.org/get-started/locally/
uv add torch torchvision
```
4. Install remaining dependencies (like SAM2) according to CUDA availability:
```shell
# Install SAM2 with CUDA support
git clone https://github.com/facebookresearch/sam2.git
cd sam2 
uv pip install . --no-cache-dir 
cd ..
rm -rf sam2
```
5. Finally, run the server:
```shell
# Run the server
uv run fastapi run main.py --port 8000
```
