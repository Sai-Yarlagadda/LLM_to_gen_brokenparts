import torch
import numpy as np
import rembg
from PIL import Image
from pytorch_lightning import seed_everything
from einops import rearrange, repeat
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, StableDiffusionInpaintPipeline
from src.utils.infer_util import remove_background, resize_foreground
import os
from torchvision.transforms import v2
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image
import tempfile
from tqdm import tqdm
import imageio
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (FOV_to_intrinsics, get_zero123plus_input_cameras,get_circular_camera_poses,)
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from grounding_dino.GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
from torchvision.ops import box_convert
import supervision as sv
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import PIL
from PIL import Image
import gradio as gr
from rembg import remove
from utils import *

def preprocess(input_image, do_remove_background):
    rembg_session = rembg.new_session() if do_remove_background else None
    if do_remove_background:
        input_image = remove_background(input_image, rembg_session)
        input_image = resize_foreground(input_image, 0.85)
    return input_image

def generate_mvs(input_image, sample_steps, sample_seed, pipeline):
    seed_everything(sample_seed)
    device = torch.device('cuda:1')
    generator = torch.Generator(device=device)
    z123_image = pipeline(
        input_image, 
        num_inference_steps=sample_steps, 
        generator=generator,
    ).images[0]
    show_image = np.asarray(z123_image, dtype=np.uint8)
    show_image = torch.from_numpy(show_image)     # (960, 640, 3)
    show_image = rearrange(show_image, '(n h) (m w) c -> (n m) h w c', n=3, m=2)
    show_image = rearrange(show_image, '(n m) h w c -> (n h) (m w) c', n=2, m=3)
    show_image = Image.fromarray(show_image.numpy())
    return z123_image, show_image

def images_to_video(images, output_path, fps=30):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames = []
    for i in range(images.shape[0]):
        frame = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).clip(0, 255)
        assert frame.shape[0] == images.shape[2] and frame.shape[1] == images.shape[3], \
            f"Frame shape mismatch: {frame.shape} vs {images.shape}"
        assert frame.min() >= 0 and frame.max() <= 255, \
            f"Frame value out of range: {frame.min()} ~ {frame.max()}"
        frames.append(frame)
    imageio.mimwrite(output_path, np.stack(frames), fps=fps, codec='h264')

def get_render_cameras(batch_size=1, M=120, radius=2.5, elevation=10.0, is_flexicubes=False):
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras

def make_mesh(mesh_fpath, planes, model, infer_config):
    mesh_basename = os.path.basename(mesh_fpath).split('.')[0]
    mesh_dirname = os.path.dirname(mesh_fpath)
    mesh_vis_fpath = os.path.join(mesh_dirname, f"{mesh_basename}.glb")
    with torch.no_grad():
        mesh_out = model.extract_mesh(planes, use_texture_map=True, **infer_config,)
        vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
        save_obj_with_mtl(
            vertices.data.cpu().numpy(),
            uvs.data.cpu().numpy(),
            faces.data.cpu().numpy(),
            mesh_tex_idx.data.cpu().numpy(),
            tex_map.permute(1, 2, 0).data.cpu().numpy(),
            mesh_fpath,
        )
        print(f"Mesh with texmap saved to {mesh_fpath}")
    return mesh_fpath

def make3d(images, model, infer_config, IS_FLEXICUBES, part_name):
    device = 'cuda:1' #change the device over here
    images = np.asarray(images, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
    images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)
    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device)
    render_cameras = get_render_cameras(
        batch_size=1, radius=4.5, elevation=20.0, is_flexicubes=IS_FLEXICUBES).to(device)
    images = images.unsqueeze(0).to(device)
    images = v2.functional.resize(images, (320, 320), interpolation=3, antialias=True).clamp(0, 1)
    directory = 'tmp'
    if not os.path.exists(directory):
        os.makedirs(directory)
    tempfile.tempdir = directory
    mesh_fpath = tempfile.NamedTemporaryFile(prefix=f'{part_name}', suffix=f".obj", delete=False).name
    print(mesh_fpath)
    mesh_basename = os.path.basename(mesh_fpath).split('.')[0]
    mesh_dirname = os.path.dirname(mesh_fpath)

    with torch.no_grad():
        planes = model.forward_planes(images, input_cameras)
        chunk_size = 20 if IS_FLEXICUBES else 1
        render_size = 384
        frames = []
        for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
            if IS_FLEXICUBES:
                frame = model.forward_geometry(planes, render_cameras[:, i:i+chunk_size], render_size=render_size,)['img']
            else:
                frame = model.synthesizer(planes, cameras=render_cameras[:, i:i+chunk_size],render_size=render_size,)['images_rgb']
            frames.append(frame)
        frames = torch.cat(frames, dim=1)
        #images_to_video(frames[0], video_fpath, fps=30,)
        #print(f"Video saved to {video_fpath}")
    
    mesh_fpath = make_mesh(mesh_fpath, planes, model, infer_config)
    print(f'INFO:: Mesh file saved at: {mesh_fpath}')
    return mesh_fpath

def get_mv_images(input_image_path, Savemv_imgs_at = 'mv_images_completepart.png'):
    torch.cuda.empty_cache()
    pipeline = DiffusionPipeline.from_pretrained("sudo-ai/zero123plus-v1.2", custom_pipeline="zero123plus",torch_dtype=torch.float16,)
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing='trailing')
    unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
    state_dict = torch.load(unet_ckpt_path, map_location='cpu')
    pipeline.unet.load_state_dict(state_dict, strict=True)
    device = torch.device('cuda:1')
    pipeline = pipeline.to(device) # should be outside the loop to make the code more memory efficient
    seed_everything(0)
    input_image_path = input_image_path #@param {type:"string"}
    input_image = Image.open(input_image_path)
    #processed_image = preprocess(input_image, True)
    #processed_image
    mv_images, mv_show_images = generate_mvs(input_image, 75, 42, pipeline)
    mv_images.save(Savemv_imgs_at)
    print(f'INFO:: Multiview imagem images saved at {Savemv_imgs_at}')

def get_obj_file(TEXT_PROMPT, mv_images_path = 'mv_images_completepart.png'):
    torch.cuda.empty_cache()
    config_path = 'configs/instant-mesh-base.yaml'
    config = OmegaConf.load(config_path)
    config_name = os.path.basename(config_path).replace('.yaml', '')
    model_config = config.model_config
    infer_config = config.infer_config
    model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="instant_mesh_base.ckpt", repo_type="model")
    model = instantiate_from_config(model_config)
    state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
    state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.') and 'source_camera' not in k}
    model.load_state_dict(state_dict, strict=True)
    device = torch.device('cuda:1')
    model = model.to(device)
    IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False
    if IS_FLEXICUBES:
        model.init_flexicubes_geometry(device, fovy=30.0)
    model = model.eval()
    mv_images = Image.open(mv_images_path)
    output_model_obj = make3d(mv_images, model, infer_config, IS_FLEXICUBES, TEXT_PROMPT)
    print('INFO:: The mesh file is created')
    return output_model_obj


def main():
    
    #get bounding box of the broken image
    IMAGE_PATH = "images/broken_images/brokentoy1.png"
    TEXT_PROMPT = "Toy"
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    torch.cuda.empty_cache()
    bbox_coordinates = get_bbox(IMAGE_PATH=IMAGE_PATH,
                                TEXT_PROMPT=TEXT_PROMPT,)
    
    #get mask of the broken part
    torch.cuda.empty_cache()
    get_mask(IMAGE_PATH=IMAGE_PATH,
             box_coordinates=bbox_coordinates,
             MASK_IMAGE_PATH="Testmask.png")
    
    #run the stable diffusion model to generate image
    torch.cuda.empty_cache()
    get_complete_image(IMAGE_PATH, mask_image_path="Testmask.png", PROMPT= "A new wine glass that is not broken", CORRECTED_img_path='correct_image_path.png')
    
    #remove the background and in both the broken image and corrected image
    remove_background(IMAGE_PATH, "broken_img_noback.png")
    

    #remove the complete background of the correct image
    bbox_coordinates = get_bbox(IMAGE_PATH='correct_image_path.png',
                                TEXT_PROMPT=TEXT_PROMPT)
    
    #get mask of the broken part
    torch.cuda.empty_cache()
    get_mask(IMAGE_PATH='correct_image_path.png',
             box_coordinates=bbox_coordinates,
             MASK_IMAGE_PATH="MaskCorrect.png")
    
    corr_img = cv2.imread('correct_image_path.png')
    corr_mask = cv2.imread('MaskCorrect.png', cv2.IMREAD_GRAYSCALE)
    corr_mask = cv2.bitwise_not(corr_mask)
    _, binary_mask = cv2.threshold(corr_mask, 127, 255, cv2.THRESH_BINARY)
    masked_image = cv2.bitwise_and(corr_img, corr_img, mask=binary_mask)
    result_path = 'correct_image_noback.png'
    cv2.imwrite(result_path, masked_image)


    #remove_background("correct_image_path.png", "correct_image_noback.png")

    #get multiple views
    torch.cuda.empty_cache()
    pipeline = DiffusionPipeline.from_pretrained("sudo-ai/zero123plus-v1.2", custom_pipeline="zero123plus",torch_dtype=torch.float16,)
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing='trailing')
    unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
    state_dict = torch.load(unet_ckpt_path, map_location='cpu')
    pipeline.unet.load_state_dict(state_dict, strict=True)
    device = torch.device('cuda:1')
    pipeline = pipeline.to(device)
    seed_everything(0)
    input_image_path = 'images/broken_images/brokentoy1.png' #@param {type:"string"}
    input_image = Image.open(input_image_path)
    #processed_image = preprocess(input_image, True)
    #processed_image
    mv_images, mv_show_images = generate_mvs(input_image, 75, 42, pipeline)
    mv_images.save('mv_images_completepart.png')
    
    #get mesh
    torch.cuda.empty_cache()
    config_path = 'configs/instant-mesh-base.yaml'
    config = OmegaConf.load(config_path)
    config_name = os.path.basename(config_path).replace('.yaml', '')
    model_config = config.model_config
    infer_config = config.infer_config
    model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="instant_mesh_base.ckpt", repo_type="model")
    model = instantiate_from_config(model_config)
    state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
    state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.') and 'source_camera' not in k}
    model.load_state_dict(state_dict, strict=True)
    device = torch.device('cuda:1')
    model = model.to(device)
    IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False
    if IS_FLEXICUBES:
        model.init_flexicubes_geometry(device, fovy=30.0)
    model = model.eval()
    mv_images = Image.open('mv_images_completepart.png')
    output_model_obj = make3d(mv_images, model, infer_config, IS_FLEXICUBES, TEXT_PROMPT)

    #subtract the broken image from the corrected image
    only_broken_part_path = remove_not_brokenpart(broken_img_path = IMAGE_PATH,
                                                   correct_img_path = 'correct_image_noback.png', 
                                                   SaveBrokenPartImageat = 'only_missing_part_broken_img.png')
    
    #get mv of only broken part
    torch.cuda.empty_cache()
    pipeline = DiffusionPipeline.from_pretrained("sudo-ai/zero123plus-v1.2", custom_pipeline="zero123plus",torch_dtype=torch.float16,)
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing='trailing')
    unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
    state_dict = torch.load(unet_ckpt_path, map_location='cpu')
    pipeline.unet.load_state_dict(state_dict, strict=True)
    device = torch.device('cuda:1')
    pipeline = pipeline.to(device)
    seed_everything(0)
    input_image_path = only_broken_part_path #@param {type:"string"}
    input_image = Image.open(input_image_path)
    #processed_image = preprocess(input_image, True)
    #processed_image
    mv_images, mv_show_images = generate_mvs(input_image, 75, 42, pipeline)
    mv_images.save('mv_images_only_brokenpart.png')

    #get 3d model of only brokenpart
    torch.cuda.empty_cache()
    config_path = 'configs/instant-mesh-base.yaml'
    config = OmegaConf.load(config_path)
    config_name = os.path.basename(config_path).replace('.yaml', '')
    model_config = config.model_config
    infer_config = config.infer_config
    model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="instant_mesh_base.ckpt", repo_type="model")
    model = instantiate_from_config(model_config)
    state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
    state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.') and 'source_camera' not in k}
    model.load_state_dict(state_dict, strict=True)
    device = torch.device('cuda:1')
    model = model.to(device)
    IS_FLEXICUBES = True if config_name.startswith('instant-mesh') else False
    if IS_FLEXICUBES:
        model.init_flexicubes_geometry(device, fovy=30.0)
    model = model.eval()
    mv_images = Image.open('mv_images_only_brokenpart.png')
    output_model_obj = make3d(mv_images, model, infer_config, IS_FLEXICUBES, 'BrokenPart')

    return output_model_obj

def get_args():
    pass

if __name__ == '__main__':
    main()
    """IMAGE_PATH = "images/broken_images/brokenwineglass.png"
    get_mv_images(IMAGE_PATH, Savemv_imgs_at = 'mv_images_brokenpart.png')
    output = get_obj_file('Wine_Glass', 'mv_images_brokenpart.png')
    print('Done')"""
