from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import torch
from torchvision.ops import box_convert
import supervision as sv
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import PIL
from PIL import Image
import gradio as gr
from diffusers import StableDiffusionInpaintPipeline
from rembg import remove
import pyvista as pv

def get_bbox(IMAGE_PATH:str,
             TEXT_PROMPT:str,
             BOX_TRESHOLD = 0.35,
            TEXT_TRESHOLD = 0.25) -> list:
    """
    Based on the image and the prompt, this function outputs the bounding box that surrounds the target in the image

    Args:
    IMAGE_PATH : Path to the broken image
    TEXT_PROMPT : What to point to in the image given
    BOX_THRESHOLD
    TEXT_THRESHOLD
    """

    model = load_model("grounding_dino/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "grounding_dino/GroundingDINO/weights/groundingdino_swint_ogc.pth")
    image_source, image = load_image(IMAGE_PATH)
    print(image.shape)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    #cv2.imwrite("annotated_image.jpg", annotated_frame)
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)
    box_coordinates = detections.xyxy[0]
    return box_coordinates

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def init_sam():
    """
    -> Initializes the model that is used for SAM
    -> If you want to use a different model, you can change it in this function
    """
    sam_checkpoint = "weights_SAM/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = 'cuda'
    return sam_checkpoint, model_type, device
    
def get_mask(IMAGE_PATH:str,
             box_coordinates: list, 
             MASK_IMAGE_PATH:str):

    """
    Based on the image and the bounding box, this function generates a mask of the main object.
    """

    sam_checkpoint, model_type, device = init_sam()
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    image = cv2.imread(IMAGE_PATH)
    #box_coordinates = detections
    input_boxes = np.array(box_coordinates)
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
    point_coords = None,
    point_labels = None, 
    box = input_boxes,
    multimask_output = False,
)
    #assert masks.shape == image.shape
    mask = masks.squeeze()
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_uint8 = cv2.bitwise_not(mask_uint8)

    cv2.imwrite(MASK_IMAGE_PATH, mask_uint8)
    print(f"INFO:: Mask is successfully saved at {MASK_IMAGE_PATH}")
    return MASK_IMAGE_PATH

def init_stable_diffusion_model():
    device = 'cuda'
    model_path = 'runwayml/stable-diffusion-inpainting'
    return device, model_path

def get_complete_image(broken_image_path:str, 
                       mask_image_path :str,
                       PROMPT:str,
                       CORRECTED_img_path:str,
                       guidance_scale = 7.5,
                       num_samples = 1,
                       SEED = 0):
    device, model_path = init_stable_diffusion_model()
    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(model_path,
                                                            torch_dtype=torch.float16).to(device)
    except:
        print("INFO:: The model information you selected is incorrect, please check you model and its corresponding information on HuggingFace")
    
    original_image = Image.open(broken_image_path).convert('RGB')
    original_image_size = original_image.size
    masked_image = Image.open(mask_image_path).convert('RGB')
    original_image = original_image.resize((512, 512))
    masked_image = masked_image.resize((512, 512))
    generator = torch.Generator(device=device).manual_seed(SEED)
    images = pipe(
        prompt=PROMPT,
        #negative_prompt = negative_prompt,
        image=original_image,
        mask_image=masked_image,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=num_samples,
    ).images
    for i, img in enumerate(images):
        img = img.resize(original_image_size)
        print(CORRECTED_img_path)
        img.save(CORRECTED_img_path)
    print('INFO:: SAVED THE CORRECT IMAGE')

def remove_background(IMAGE_PATH:str, 
                      output_path:str):
    input_img = Image.open(IMAGE_PATH)
    output = remove(input_img)
    output.save(output_path)
    print(f"INFO:: Removed background of image: {output_path}")
    return output_path

def remove_not_brokenpart(broken_img_path, correct_img_path, SaveBrokenPartImageat):
    #compare the masks of original image and corrected image
    bb_cords_correct = get_bbox(correct_img_path, 'main object in the image')
    bb_cords_broken = get_bbox(broken_img_path, 'main object in the image')
    correct_image_mask_path = get_mask(correct_img_path, bb_cords_correct, 'tmp/images/correctmask_image.png')
    broken_image_mask_path = get_mask(broken_img_path, bb_cords_broken, 'tmp/images/brokenmask_image.png')
    broken_img_mask = cv2.imread(broken_image_mask_path, cv2.IMREAD_GRAYSCALE)
    correct_image_mask = cv2.imread(correct_image_mask_path, cv2.IMREAD_GRAYSCALE)
    diff_mask = cv2.subtract(broken_img_mask, correct_image_mask)
    cv2.imwrite('tmp/images/brokenpart_mask.png', diff_mask)

    final_image = cv2.imread(correct_img_path)
    mask = cv2.imread('tmp/images/brokenpart_mask.png', cv2.COLOR_BGR2GRAY)
    result = np.dstack((final_image, mask))
    cv2.imwrite(SaveBrokenPartImageat, result)

    print("INFO:: Only the broken part image is generated.")
    return SaveBrokenPartImageat

def remove_broken_part_pyvista(broken_obj_path, 
                               correct_obj_path, 
                               Savestl_only_broken_part= 'OnlyBrokenPart.stl'):
    corr_mesh = pv.read(correct_obj_path)
    corr_mesh = corr_mesh.extract_surface().smooth(n_iter=10) #MAKES THE MESH FINER AND BETTER
    bro_mesh = pv.read(broken_obj_path)
    result = corr_mesh.boolean_difference(bro_mesh)
    result.save(Savestl_only_broken_part)
    print('INFO:: The STL file of the broken part is saved at ')
    

"""
if __name__ == '__main__':
    IMAGE_PATH = "images/broken_images/brokenwineglass.png"
    TEXT_PROMPT = "main part"
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    
    bbox_coordinates = get_bbox(IMAGE_PATH=IMAGE_PATH,
                                TEXT_PROMPT=TEXT_PROMPT,)
    get_mask(IMAGE_PATH=IMAGE_PATH,
             box_coordinates=bbox_coordinates,
             MASK_IMAGE_PATH="Testmask.png")
    get_complete_image(IMAGE_PATH, mask_image_path="Testmask.png", PROMPT= "Complete image of Glass Mug", CORRECTED_img_path='correct_image_path.png')
    remove_background(IMAGE_PATH, "broken_img_noback.png")
    remove_background("correct_image_path.png", "correct_image_noback.png")
    #only_broken_part_path = remove_not_brokenpart(broken_img_path = IMAGE_PATH,
    #                                               correct_img_path = 'correct_image_noback.png', 
    #                                               SaveBrokenPartImageat = 'only_missing_part_broken_img.png')
    #print("Done!")
    correct_image_mask = cv2.imread('tmp/images/correctmask_image.png')
    broken_image_mask = cv2.imread('tmp/images/brokenmask_image.png')
    diff_mask = cv2.subtract(broken_image_mask, correct_image_mask)
    #cv2.imwrite("Image_subtracted.png", diff_mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

"""
