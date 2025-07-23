from flask import Flask, request, jsonify, render_template
import base64
from io import BytesIO
from PIL import Image
import torch
from pipe_utils import create_pipeline, enhance_sketch
import os
import traceback
import re

# 初始化 ControlNet 管道
pipeline = create_pipeline()

# 初始化 Flask 应用
app = Flask(__name__)

# 首页路由
@app.route('/')
def index():
    return render_template('index.html')


# 图像生成路由
@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        # 接收前端数据
        data = request.get_json()
        
        if not data:
            raise ValueError("请求数据为空！")
        img_data = data.get('img')
        if not img_data:
            raise ValueError("缺少图像数据！")
        # prompt = data.get('prompt', 'traditional media')
        
        # 定义固定prompt模板（可抽离为配置文件）
        # SYSTEM_PROMPT = "​chinese writing" 
        SYSTEM_PROMPT = "​traditional media, no humans" 
        # 智能融合逻辑
        user_prompt = data.get('prompt', '').strip()  # 获取用户输入（允许为空）
        if user_prompt:
            prompt = f"{user_prompt}, {SYSTEM_PROMPT}"  # 用户输入在前
        else:
            prompt = SYSTEM_PROMPT  # 完全使用系统prompt
        

        # 解码 Base64 图像
        img_bytes = base64.b64decode(img_data.split(",")[1])
        input_image = Image.open(BytesIO(img_bytes)).convert("RGB")
        input_image = input_image.resize((512, 512)) 
        input_image.save('raw_demo.png')
        processed_sketch = enhance_sketch(input_image)  # 新增此行
        processed_sketch.save('demo.png')
        control_net_inputs = [processed_sketch]  # 替换原图输入
        # 保存输入图像
        input_image.save("./static/cache/input_image.png")  # 保存接收到的输入图像为 PNG 格式


        # 确保输入图像数量与 ControlNet 模型数量一致
        control_net_inputs = [input_image]  # * len(controlnet_list)

# 处理文件名，移除非法字符
        safe_prompt = re.sub(r'[\\, ]', "_", user_prompt)
        input_image_path = f"./static/cache/{safe_prompt}_input_image.png"
        input_image.save(input_image_path)  # 保存接收到的输入图像为 PNG 格式
        # 调用生成模型
        generator = torch.Generator(device="cpu").manual_seed(-1)
        # output_images = pipeline(
        #     prompt=[prompt],
        #     image=control_net_inputs,
        #     # negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"]
        #     negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality, 3d render, disney style, text, watermark, signature ,blurry, out of focus, extra limbs, mutated hands"  # 修正肢体异常
        #     ],
        #     num_inference_steps=25,
        #     generator=[generator],
        # )
        output_images = pipeline(
            prompt=[prompt],
            image=control_net_inputs,
            # negative_prompt = ["monochrome, lowres, bad anatomy, worst quality, low quality, 3d render, disney style, text, watermark, signature, blurry, out of focus, extra limbs, mutated hands, unreadable text, letters, numbers, words, caption, title, human, person, people"],
            negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"],
            num_inference_steps=30, #
            guidance_scale=7,       
            controlnet_conditioning_scale=1.2,  
            generator=[generator],
            guidance_start=0.0,   
            guidance_end=1.0,      
        )
        output_image = output_images.images[0]
        # 保存生成的输出图像，使用用户输入的 prompt 作为文件名
        output_image_path = f"./static/cache/{safe_prompt}.png"
        output_image.save(output_image_path)  # 保存生成的输出图像为 PNG 格式

        # 转换输出图像为 Base64
        buffered = BytesIO()
        output_image.save(buffered, format="PNG")
        output_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # 返回生成的图像
        return jsonify({"img": f"data:image/png;base64,{output_base64}"})
    except Exception as e:
        print("生成图像时发生错误：", str(e))
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500




# 更新管道（如果需要动态调整）
@app.route('/update', methods=['PUT'])
def update_pipeline():
    global pipeline
    pipeline = create_pipeline()
    return jsonify({"message": "pipeline has been updated"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


