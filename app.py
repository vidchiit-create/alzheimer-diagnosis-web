import os
import gdown
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from monai.networks.nets import DenseNet121
from captum.attr import GradientShap, LayerGradCam

# --- TỰ ĐỘNG TẢI MODEL CHECKPOINT TỪ GOOGLE DRIVE NẾU CHƯA CÓ ---
CHECKPOINT_PATH = "model_checkpoint.pth"
if not os.path.exists(CHECKPOINT_PATH):
    file_id = "197KRcZsT6UpNWdXb0BNXms7rtkF8o_I"
    url = f"https://drive.google.com/uc?id={file_id}"
    print("Đang tải model checkpoint từ Google Drive...")
    gdown.download(url, CHECKPOINT_PATH, quiet=False)

st.set_page_config(page_title="Hệ thống Chẩn đoán Alzheimer", layout="wide")

# ==========================================
# 1. CÁC HÀM XAI & COUNTERFACTUAL
# ==========================================
def compute_3d_gradcam(model, image_tensor, target_class):
    target_layer = model.features.denseblock4
    grad_cam = LayerGradCam(model, target_layer)
    attr = grad_cam.attribute(image_tensor, target=target_class)
    attr_upsampled = F.interpolate(attr, size=image_tensor.shape[2:], mode='trilinear', align_corners=False)
    heatmap = attr_upsampled.detach().cpu().numpy()[0, 0]
    return np.maximum(heatmap, 0) / (np.maximum(heatmap, 0).max() + 1e-8)

def compute_counterfactual_delta_map(target_img_np, cn_template_np):
    # Delta > 0: Vùng teo mô so với người bình thường
    delta = cn_template_np - target_img_np
    threshold = np.std(delta) * 1.2
    atrophy_map = np.where(delta > threshold, delta, 0)
    return atrophy_map

# ==========================================
# 2. LOAD MÔ HÌNH VÀ TEMPLATE MỘT LẦN (CACHE)
# ==========================================
@st.cache_resource
def load_system():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=3).to(device)
    
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    template_path = "cn_mean_template.npy"
    if os.path.exists(template_path):
        cn_template = np.load(template_path)
    else:
        cn_template = None
        
    return model, device, cn_template

model, device, cn_template = load_system()
target_names = ['CN (Bình thường)', 'MCI (Suy giảm nhẹ)', 'AD (Alzheimer)']

# ==========================================
# 3. GIAO DIỆN CHÍNH
# ==========================================
st.title("🧠 Hệ thống Phân tích MRI 3D chẩn đoán Alzheimer (AD)")

uploaded_file = st.file_uploader("Chọn file ảnh chụp MRI (.npy)", type=["npy"])

if uploaded_file is not None:
    img_np = np.load(uploaded_file)
    
    if len(img_np.shape) == 3:
        input_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    else:
        input_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).to(device)
        img_np = img_np[0]

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_label = outputs.argmax(dim=1).item()
        
    st.markdown(f"### 🎯 Kết quả chẩn đoán: **{target_names[pred_label]}** (Độ tin cậy: {probs[pred_label]*100:.2f}%)")
    
    with st.spinner('Đang tính toán các bản đồ phân tích XAI đa kênh...'):
        # 1. XAI: Grad-CAM & SHAP
        gradcam_map = compute_3d_gradcam(model, input_tensor, pred_label)
        baseline = torch.zeros_like(input_tensor).to(device)
        attr_shap = GradientShap(model).attribute(input_tensor, n_samples=3, stdevs=0.0001, baselines=baseline, target=pred_label)
        shap_map = attr_shap.cpu().detach().numpy()[0, 0]
        vmax_shap = np.max(np.abs(shap_map))
        
        # 2. XAI: Counterfactual Map
        if cn_template is not None:
            atrophy_map = compute_counterfactual_delta_map(img_np, cn_template)
            vmax_delta = np.max(np.abs(atrophy_map)) if np.max(np.abs(atrophy_map)) > 0 else 1.0
        else:
            atrophy_map = np.zeros_like(img_np)
            vmax_delta = 1.0
            st.warning("Không tìm thấy mẫu chuẩn CN, bỏ qua Bản đồ đối chứng.")

    # === THANH TRƯỢT & HIỂN THỊ HÌNH ẢNH ===
    max_slices = img_np.shape[2] - 1
    slice_idx = st.slider('Kéo thanh trượt để xem các lát cắt (Trục Z):', min_value=0, max_value=max_slices, value=max_slices//2, step=1)
    
    fig = plt.figure(figsize=(14, 10))
    
    # Khung 1: Ảnh gốc
    plt.subplot(2, 2, 1)
    plt.imshow(img_np[:, :, slice_idx], cmap='gray')
    plt.title(f"[1] MRI Bệnh nhân (Z={slice_idx})", fontweight='bold')
    plt.axis('off')
    
    # Khung 2: Counterfactual Delta Map
    plt.subplot(2, 2, 2)
    plt.imshow(img_np[:, :, slice_idx], cmap='gray')
    plt.imshow(atrophy_map[:, :, slice_idx], cmap='hot', alpha=0.6, vmin=0, vmax=vmax_delta)
    plt.title("[2] Bản đồ teo dịch não\n(Vàng/Đỏ: Vùng teo mô so với chuẩn)", fontweight='bold')
    plt.axis('off')
    
    # Khung 3: Grad-CAM
    plt.subplot(2, 2, 3)
    plt.imshow(img_np[:, :, slice_idx], cmap='gray')
    plt.imshow(gradcam_map[:, :, slice_idx], cmap='jet', alpha=0.5)
    plt.title("[3] 3D Grad-CAM\n(Vùng mạng CNN chú ý nhất)", fontweight='bold')
    plt.axis('off')
    
    # Khung 4: GradientSHAP
    plt.subplot(2, 2, 4)
    plt.imshow(img_np[:, :, slice_idx], cmap='gray')
    plt.imshow(shap_map[:, :, slice_idx], cmap='bwr', alpha=0.6, vmin=-vmax_shap, vmax=vmax_shap)
    plt.title("[4] GradientSHAP\n(Đỏ: Dấu hiệu bệnh | Xanh: Dấu hiệu khỏe)", fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)