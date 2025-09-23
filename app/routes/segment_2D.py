from fastapi import APIRouter, UploadFile, File

router = APIRouter("/segment", tags=["segment"])


@router.post("/image_with_prompts")
async def segment_image_with_prompts(prompts: Prompts, image_file: UploadFile = File(...)):
