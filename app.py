import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import time
import datetime

# ================= 1. 页面基础设置 =================
st.set_page_config(page_title="建筑工地智能巡检系统", page_icon="👷", layout="wide")

# 侧边栏控制台
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/worker.png", width=80)
    st.title("系统控制台")
    st.info("迈鑫建筑安装有限公司\n\n版本：V1.2 (内测版)")
    
    st.markdown("---")
    # 核心功能：置信度调节
    conf_threshold = st.slider("AI 识别严格度 (置信度)", min_value=0.1, max_value=0.9, value=0.35, step=0.05, 
                               help="数值越高，AI越谨慎，只有非常确定的才会框出来。")
    
    st.markdown("---")
    st.caption("📞 技术支持：IT部实习生")


# ================= 2. 主界面布局 =================
st.title("👷 建筑施工现场智能安全巡检系统")
st.markdown("**基于 AI 视觉技术的本地化安全排查工具** | *保障工地零事故*")
st.divider()

# 模拟顶部数据看板 (Metrics)
col1, col2, col3, col4 = st.columns(4)
col1.metric("今日巡检次数", "128 次", "+12")
col2.metric("已排查照片", "453 张", "+36")
col3.metric("发现疑似违规", "7 次", "-2", delta_color="inverse")
col4.metric("系统运行状态", "正常", "在线")

st.divider()

# ================= 3. 模型加载 =================
@st.cache_resource
def load_model():
    # 注意：这里依然调用 yolov8n.pt，它能认出“人”
    return YOLO("yolov8n.pt") 

model = load_model()

# ================= 4. 图片上传与检测 =================
st.subheader("📸 现场影像排查")
uploaded_file = st.file_uploader("请上传工地监控截图或现场照片 (支持 JPG/PNG 格式)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 使用双列布局，左右对比
    col_left, col_right = st.columns(2)
    
    image = Image.open(uploaded_file)
    with col_left:
        st.info("原始影像 (监控截图)")
        st.image(image, use_container_width=True)
        
    with col_right:
        st.success("AI 识别结果")
        # 占位符，等会放结果图
        result_placeholder = st.empty()
        
    st.divider()

    if st.button("🚀 开始智能排查", type="primary", use_container_width=True):
        with st.spinner("AI 正在扫描画面中的安全隐患..."):
            # 加入一点点延迟，模拟系统正在费力计算，显得更真实
            time.sleep(1.5) 
            
            # 根据侧边栏的滑块设置置信度
            results = model.predict(image, conf=conf_threshold)
            
            res_plotted = results[0].plot()
            with col_right:
                result_placeholder.image(res_plotted, use_container_width=True)
            
            # 处理识别数据
            boxes = results[0].boxes
            if len(boxes) > 0:
                class_counts = {}
                for cls in boxes.cls:
                    class_name = model.names[int(cls)]
                    if class_name == "person":
                        class_name = "现场施工人员"
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                # 显示报表
                st.subheader("📊 排查数据报表")
                df = pd.DataFrame(list(class_counts.items()), columns=["识别目标", "数量"])
                
                # 左右分栏显示表格和警报
                col_t1, col_t2 = st.columns([1, 2])
                with col_t1:
                    st.dataframe(df, use_container_width=True)
                with col_t2:
                    st.markdown("### 🚨 现场预警日志")
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if "现场施工人员" in class_counts:
                        st.error(f"[{current_time}] 警报：画面中发现 {class_counts['现场施工人员']} 名施工人员，请安全员核实是否按规定佩戴安全帽及反光衣！")
                    else:
                        st.warning(f"[{current_time}] 提示：未在画面中检测到人员活动。")
                
                # 导出报告按钮 (模拟)
                csv = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 下载本次排查报告 (CSV)",
                    data=csv,
                    file_name=f"安全排查报告_{current_time}.csv",
                    mime="text/csv",
                )
                
            else:
                st.warning("当前画面未检测到人员或违规目标。")
else:
    # 没上传图片时的占位提示
    st.info("💡 提示：请在上方上传现场照片，系统将自动调用 AI 模型进行安全排查。")