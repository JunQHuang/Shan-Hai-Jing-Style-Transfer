from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
import cv2
from PIL import Image
import numpy as np
from diffusers.utils import load_image
from diffusers import UniPCMultistepScheduler
from diffusers import DPMSolverMultistepScheduler

# canny_controlnet = ControlNetModel.from_pretrained("./models/control_canny.safetensors", torch_dtype=torch.float16)

canny_controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)

controlnet_dict = {
    "canny": canny_controlnet
}

def enhance_sketch(input_image):
    img_array = np.array(input_image)
    # 动态对比度增强（解决儿童简笔画线条模糊问题）
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    # 智能阈值Canny（适应不同画风）
    gray = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)
    avg_intensity = np.mean(gray)
    low_thresh = int(max(0, avg_intensity * 0.3))  # 儿童绘画线条较浅
    high_thresh = int(min(255, avg_intensity * 2.0))
    
    # 方向敏感型边缘检测（保留笔画方向特征）
    edges = cv2.Canny(gray, low_thresh, high_thresh, apertureSize=5, L2gradient=True)
    
    # 笔画连接优化（解决断线问题）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    img = Image.fromarray(edges)
    # print(img)
    return img

def create_pipeline(config=None, controlnet_name_list=['canny']):
    # controlnet_name_list = [r'C:\Users\qq100\Desktop\flask_paint\models\control_canny.safetensors']
    controlnet_list = []
    for controlnet_name in controlnet_name_list:
        controlnet = controlnet_dict.get(controlnet_name)
        if controlnet is not None:
            controlnet_list.append(controlnet)
    print("加载的 ControlNet 模型列表：", len(controlnet_list))  # 调试信息

    # pipe = StableDiffusionControlNetPipeline.from_pretrained(
    # "models/v1-5-pruned-emaonly.safetensors",  # 替换为本地路径
    # controlnet=controlnet_list, 
    # torch_dtype=torch.float16,
    # safety_checker=None
    # )
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet_list, torch_dtype=torch.float16,
        safety_checker=None

    )


    pipe.load_lora_weights("models", weight_name="hjq_2-000010.safetensors", adapter_name="mmongo", adapter_weights=0.55)


    pipe.set_adapters("mmongo")
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # 在create_pipeline函数中添加：
    pipe.enable_attention_slicing()  # 降低显存占用
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="sde-dpmsolver++",  # 同步DPM++2M
        use_karras_sigmas=True            # 同步Karras schedule
    )

    pipe.enable_model_cpu_offload()
    # pipe.enable_xformers_memory_efficient_attention()
    # pipe.unet.set_attn_processor(XFormersAttnProcessor()) 
    return pipe


if __name__ == '__main__':
    image = load_image(
        "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
    )

    image = np.array(image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)


    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=[controlnet], torch_dtype=torch.float16
    )

    prompt = ", best quality, extremely detailed"
    prompt = [t + prompt for t in ["Sandra Oh", "Kim Kardashian", "rihanna", "taylor swift"]]
    generator = [torch.Generator(device="cpu").manual_seed(2) for i in range(len(prompt))]
    print(prompt)
    # output = pipe(
    #     prompt,
    #     [canny_image],
    #     negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 4,
    #     num_inference_steps=20,
    #     generator=generator,
    # )
    #
    # output.images[0].show()
