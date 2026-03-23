import streamlit as st
from transformers import pipeline
from PIL import Image
import torch
import os

# 1. 页面配置
st.set_page_config(
    page_title="ISOM5240 Retail AI Assistant", 
    page_icon="🛍️",
    layout="centered"
)

# UI 标题与简介
st.title("🛍️ 智能零售营销助手 (微调版)")
st.write("当前运行：微调后的 ViT (识别) + GPT-2 (营销文案生成)。")

# 2. 加载模型 (使用你的 Hugging Face Model ID)
@st.cache_resource
def load_pipelines():
    vit_model_id = "JescYip/vit-retail-finetuned"
    gpt_model_id = "JescYip/gpt2-ad-finetuned"
    
    # 加载图像识别 Pipeline
    image_classifier = pipeline("image-classification", model=vit_model_id)
    # 加载文本生成 Pipeline
    text_generator = pipeline("text-generation", model=gpt_model_id)
    
    return image_classifier, text_generator

try:
    v_pipe, t_pipe = load_pipelines()
except Exception as e:
    st.error(f"模型加载失败！请确保模型已同步到 Hugging Face。错误详情: {e}")
    st.stop()

# 3. 上传图片组件
uploaded_file = st.file_uploader("上传商品图片以开始营销自动化...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='待处理商品', use_container_width=True)
    
    st.divider()
    
    # --- 第一步：商品识别 (Keyword Identification) ---
    with st.spinner('微调模型正在精准识别...'):
        v_results = v_pipe(image)
        # 获取 Top-1 结果
        top_label = v_results[0]['label']
        confidence = v_results[0]['score']
        
    st.subheader("第一步：商品识别 (Keyword)")
    # 这里解释一下：分类模型输出的是类别标签，作为第二步的输入
    st.success(f"识别类别: **{top_label}** (置信度: {confidence:.2%})")
    
    # --- 第二步：广告生成 (Ad Generation) ---
    with st.spinner('正在基于微调逻辑生成文案...'):
        # 必须匹配训练时的格式: "Product: {label} \n Ad:"
        # 注意：训练时可能在 \n 后面有空格，这里采用最稳健的匹配
        target_prompt = f"Product: {top_label} \n Ad:"
        
        t_results = t_pipe(
            target_prompt, 
            max_length=100,
            num_return_sequences=1, 
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=50256
        )
        
        raw_output = t_results[0]['generated_text']
        
        # 稳健的提取逻辑：截取 "Ad:" 之后的内容
        if "Ad:" in raw_output:
            final_ad = raw_output.split("Ad:")[-1].strip()
            # 移除结束符并只取第一行（防止模型输出一长串乱码）
            final_ad = final_ad.replace("<|endoftext|>", "").split("\n")[0]
        else:
            final_ad = raw_output.replace(target_prompt, "").strip()

    st.subheader("第二步：自动化广告生成")
    if final_ad and len(final_ad) > 5:
        st.info(f"✨ {final_ad}")
    else:
        st.warning("⚠️ 模型当前生成的文案过短或格式不符。")
        with st.expander("查看调试信息 (Debug Info)"):
            st.write(f"Raw Output: {raw_output}")

    # 4. 展示逻辑解题方法 (Logical Approach)
    with st.expander("查看技术闭环 (Technical Business Logic)"):
        st.write("1. **Computer Vision**: 使用微调后的 ViT 提取商品特征，将其映射到 141 个零售特定类别。")
        st.write(f"2. **NLP Bridge**: 将识别出的关键词 '{top_label}' 作为上下文输入给微调后的 GPT-2。")
        st.write("3. **Generative AI**: GPT-2 通过自回归预测生成符合电商风格的营销短语。")
