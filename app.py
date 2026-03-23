import streamlit as st
from transformers import pipeline
from PIL import Image
import torch

# --- 页面配置 ---
st.set_page_config(page_title="ISOM5240 Retail AI Assistant", page_icon="🛍️", layout="wide")

st.title("🛍️ Intelligent Retail Marketing Assistant (Pro Version)")
st.write("Integrated multimodal automatic marketing system with Swin-Transformer, GIT-large, and GPT-2.")

# --- 1. 加载模型 (Pipeline 集成) ---
@st.cache_resource
def load_pipelines():
    # 1. 图像分类 (Swin-Tiny)
    classifier = pipeline("image-classification", model="JescYip/Swin-Tiny")
    
    # 2. 图像描述 (GIT-large) — image-text-to-text (~750MB)
    captioner = pipeline("image-text-to-text", model="microsoft/git-large")
    
    # 3. 广告生成 (GPT-2)
    ad_generator = pipeline("text-generation", model="SCM1120/gpt2-ad-finetuned")

    return classifier, captioner, ad_generator
    
with st.spinner('AI 引擎启动中...'):
    v_classifier, v_captioner, t_generator = load_pipelines()

# --- 2. 侧边栏与上传组件 ---
with st.sidebar:
    st.header("Upload Center")
    uploaded_file = st.file_uploader("Select product image...", type=["jpg", "jpeg", "png"])
    st.info("Recommendation: Use clean background e-commerce product images for best results.")

# --- 3. 主交互逻辑 ---
if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption='Product to be identified', use_container_width=True)

    with col2:
        st.subheader("Step 1: Deep Feature Extraction")
        
        # --- A. 运行 Swin-Tiny 分类 ---
        with st.spinner('Swin-Tiny analyzing category...'):
            cls_results = v_classifier(image)
            top_label = cls_results[0]['label']
            cls_confidence = cls_results[0]['score']
        
        # --- B. 运行 BLIP-2 生成描述 ---
        with st.spinner('GIT-large generating visual description...'):
            cap_results = v_captioner(image, text="")
            # 获取完整描述用于广告生成
            full_description = cap_results[0]['generated_text']
            keywords = ", ".join(full_description.split()[:10]) # 提取前10个词以提供更多上下文

        # 展示第一步结果
        st.success(f"**Product Category**: {top_label}")
        st.write(f"**Visual Description**: `{full_description}`")
        st.caption(f"Classification Confidence: {cls_confidence:.2%}")

        st.divider()

        # --- 第二步：GPT-2 广告生成 ---
        st.subheader("Step 2: Intelligent Copy Creation")
        with st.spinner('GPT-2 crafting advertisement...'):
            # 改进Prompt：使用更自然的语言，避免模板化
            prompt = f"Imagine you're writing a catchy slogan for a {top_label} with these features: {full_description}. Create an exciting and persuasive ad copy that highlights the benefits and makes people want to buy it:"
            
            # 检查机制：如果广告太短，重新生成
            ad_text = ""
            min_words = 10  # 至少10个词
            max_attempts = 5  # 最多尝试5次
            attempts = 0
            
            while len(ad_text.split()) < min_words and attempts < max_attempts:
                ad_results = t_generator(
                    prompt,
                    max_length=150,
                    min_length=50,
                    num_return_sequences=1,  # 每次生成一个
                    truncation=True,
                    temperature=0.8,
                    pad_token_id=50256,
                    do_sample=True,
                    no_repeat_ngram_size=2
                )
                
                ad_text = ad_results[0]['generated_text'].replace(prompt, "").strip()
                # 移除常见的广告模板前缀和后缀
                ad_text = ad_text.replace("Ad:", "").replace("#", "").strip()
                attempts += 1
            
            if len(ad_text.split()) < min_words:
                ad_text = "Sorry, unable to generate a sufficiently long advertisement. Please try a different image or adjust parameters."

        st.info(ad_text if ad_text else "Crafting in progress...")

    # --- 4. 技术架构说明 (符合 ISOM5240 项目要求) ---
    with st.expander("View Project Technical Architecture (Technical Pipeline Logic)"):
        st.markdown(f"""
        1.  **Swin-Tiny (Vision)**: Employs hierarchical Transformer architecture for precise 3-category classification of products (tops/bottoms/shoes).
        2.  **GIT-large (Visual-Language)**: Lightweight captioning model (`microsoft/git-large`, ~750MB), generating visual descriptions via image-text-to-text pipeline within Streamlit Cloud free tier memory limits.
        3.  **GPT-2 (Generative AI)**: Receives `Category + Description` multidimensional input, generating e-commerce compliant marketing copy through autoregression.
        """)
