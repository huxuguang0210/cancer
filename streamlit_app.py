"""
肿瘤复发风险预测临床决策支持系统
Clinical Decision Support System for Cancer Recurrence Prediction
===========================================================
中国医科大学附属盛京医院
Shengjing Hospital of China Medical University
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import json
import io
import base64
from datetime import datetime
from typing import Dict
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# ================== 页面配置 ==================
st.set_page_config(
    page_title="肿瘤复发预测系统 | Cancer Recurrence Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================== Logo加载 ==================
def load_logo():
    """加载Logo图片，返回Base64编码"""
    logo_paths = ['logo.png', 'logo.jpg', 'logo.jpeg', 'assets/logo.png']
    
    for path in logo_paths:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    
    # 如果没有找到Logo文件，返回None
    return None

LOGO_BASE64 = load_logo()

# 判断是否有Logo
HAS_LOGO = LOGO_BASE64 is not None

# ================== CSS样式 ==================
st.markdown("""
<style>
    [data-testid="collapsedControl"] {display: none}
    section[data-testid="stSidebar"] {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main .block-container {
        padding: 0.5rem 1.5rem 2rem 1.5rem;
        max-width: 100%;
    }
    
    /* 顶部栏 */
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.8rem 0;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #1a5276;
        background: linear-gradient(90deg, #f8f9fa, #ffffff);
    }
    .logo-section {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .logo-img {
        height: 60px;
        width: auto;
        border-radius: 8px;
    }
    .logo-text h2 {
        margin: 0;
        font-size: 1.2rem;
        color: #1a5276;
        font-weight: 700;
    }
    .logo-text p {
        margin: 0;
        font-size: 0.8rem;
        color: #666;
    }
    
    /* 医院标题头 */
    .hospital-header {
        background: linear-gradient(135deg, #1a5276 0%, #2980b9 50%, #1a5276 100%);
        padding: 1.2rem 2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 25px;
    }
    .header-logo {
        background: white;
        border-radius: 10px;
        padding: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .header-logo img {
        height: 70px;
        width: auto;
        display: block;
    }
    .header-text {
        text-align: center;
    }
    .header-text h1 {
        color: white;
        font-size: 1.5rem;
        margin: 0 0 0.3rem 0;
        font-weight: 600;
    }
    .header-text .subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 0.9rem;
        margin: 0;
    }
    .header-text .hospital-name {
        color: #f1c40f;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }
    
    /* 无Logo时的纯文字头部 */
    .hospital-header-nologo {
        background: linear-gradient(135deg, #1a5276 0%, #2980b9 50%, #1a5276 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        text-align: center;
    }
    .hospital-header-nologo h1 {
        color: white;
        font-size: 1.6rem;
        margin: 0 0 0.3rem 0;
        font-weight: 600;
    }
    .hospital-header-nologo .subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 0.95rem;
        margin: 0;
    }
    .hospital-header-nologo .hospital-name {
        color: #f1c40f;
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.4rem;
    }
    
    /* 模块卡片 */
    .module-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        border: 1px solid #e8e8e8;
    }
    .module-title {
        background: linear-gradient(90deg, #3498db, #2980b9);
        color: white;
        padding: 0.4rem 0.6rem;
        border-radius: 5px;
        margin: -0.8rem -0.8rem 0.6rem -0.8rem;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .module-title.pathology { background: linear-gradient(90deg, #9b59b6, #8e44ad); }
    .module-title.surgery { background: linear-gradient(90deg, #e67e22, #d35400); }
    .module-title.markers { background: linear-gradient(90deg, #1abc9c, #16a085); }
    
    /* 结果区域 */
    .result-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        border: 2px solid #dee2e6;
    }
    .result-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
        text-align: center;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    
    /* 图表容器 */
    .chart-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e8e8e8;
        height: 100%;
    }
    
    /* 建议卡片 */
    .advice-box {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 5px solid;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .advice-box.low { border-color: #28a745; background: linear-gradient(90deg, #f0fff0, white); }
    .advice-box.medium { border-color: #ffc107; background: linear-gradient(90deg, #fffef0, white); }
    .advice-box.high { border-color: #dc3545; background: linear-gradient(90deg, #fff0f0, white); }
    .advice-box h4 { margin: 0 0 0.6rem 0; font-size: 1rem; color: #2c3e50; }
    
    /* 按钮 */
    .stButton > button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 25px;
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4);
    }
    
    /* 选择框 */
    .stSelectbox label { font-weight: 500; color: #2c3e50; font-size: 0.8rem; }
    
    /* 标签页 */
    .stTabs [data-baseweb="tab-list"] { gap: 0; background: #f8f9fa; border-radius: 8px; padding: 3px; }
    .stTabs [data-baseweb="tab"] { background: transparent; border-radius: 5px; padding: 8px 16px; font-weight: 600; font-size: 0.9rem; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #3498db, #2980b9); color: white !important; }
    
    /* 指标卡片 */
    [data-testid="metric-container"] {
        background: white;
        padding: 0.6rem;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        border: 1px solid #e8e8e8;
    }
    
    /* 页脚 */
    .footer {
        background: linear-gradient(135deg, #1a5276, #2980b9);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin-top: 1.5rem;
        text-align: center;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 20px;
    }
    .footer-logo {
        background: white;
        border-radius: 8px;
        padding: 5px;
    }
    .footer-logo img { height: 50px; width: auto; }
    .footer-text .hospital-name { color: #f1c40f; font-weight: 600; font-size: 1rem; }
    .footer-text .version { font-size: 0.85rem; opacity: 0.9; margin-top: 3px; }
</style>
""", unsafe_allow_html=True)

# ================== 语言配置 ==================
LANGUAGES = {"中文": "zh", "English": "en"}

TRANSLATIONS = {
    "title": {"zh": "肿瘤复发风险预测系统", "en": "Cancer Recurrence Prediction System"},
    "subtitle": {"zh": "临床决策支持平台", "en": "Clinical Decision Support Platform"},
    "hospital": {"zh": "中国医科大学附属盛京医院", "en": "Shengjing Hospital of China Medical University"},
    "hospital_short": {"zh": "盛京医院", "en": "Shengjing Hospital"},
    "single_patient": {"zh": "单例预测", "en": "Single Prediction"},
    "batch_prediction": {"zh": "批量预测", "en": "Batch Prediction"},
    "basic_info": {"zh": "基本信息", "en": "Basic Info"},
    "surgical_info": {"zh": "手术信息", "en": "Surgical Info"},
    "pathology_info": {"zh": "病理信息", "en": "Pathology Info"},
    "tumor_markers": {"zh": "肿瘤标志物", "en": "Tumor Markers"},
    "predict_button": {"zh": "开始风险评估", "en": "Start Assessment"},
    "prediction_results": {"zh": "风险评估结果", "en": "Risk Assessment Results"},
    "overall_risk": {"zh": "综合复发风险", "en": "Overall Risk"},
    "risk_level": {"zh": "风险分层", "en": "Risk Level"},
    "low_risk": {"zh": "低危", "en": "Low Risk"},
    "medium_risk": {"zh": "中危", "en": "Intermediate"},
    "high_risk": {"zh": "高危", "en": "High Risk"},
    "survival_curve": {"zh": "无复发生存曲线", "en": "Recurrence-Free Survival"},
    "cumulative_risk_curve": {"zh": "累积复发风险曲线", "en": "Cumulative Risk Curve"},
    "time_risk": {"zh": "各时间点复发风险", "en": "Time-Point Risk"},
    "clinical_advice": {"zh": "临床随访建议", "en": "Follow-up Recommendations"},
    "disclaimer": {"zh": "⚠️ 提示：本系统预测结果仅供临床参考，最终诊疗方案请由主治医师综合判断后确定。", 
                  "en": "⚠️ Note: Predictions are for clinical reference only. Final decisions should be made by physicians."},
    "months": {"zh": "月", "en": "M"},
    "time_months": {"zh": "时间（月）", "en": "Time (Months)"},
    "probability": {"zh": "概率", "en": "Probability"},
    "survival_prob": {"zh": "生存概率", "en": "Survival Probability"},
    "risk_prob": {"zh": "复发概率", "en": "Recurrence Probability"},
    "upload_file": {"zh": "上传患者数据", "en": "Upload Patient Data"},
    "download_template": {"zh": "下载模板", "en": "Download Template"},
    "export_excel": {"zh": "导出Excel", "en": "Export Excel"},
    "export_pdf": {"zh": "导出PDF", "en": "Export PDF"},
    "export_csv": {"zh": "导出CSV", "en": "Export CSV"},
    "patient_id": {"zh": "患者编号", "en": "Patient ID"},
    "total_patients": {"zh": "总例数", "en": "Total"},
    "high_risk_count": {"zh": "高危", "en": "High"},
    "medium_risk_count": {"zh": "中危", "en": "Medium"},
    "low_risk_count": {"zh": "低危", "en": "Low"},
    "risk_distribution": {"zh": "风险分层分布", "en": "Risk Distribution"},
    "processing": {"zh": "正在评估中...", "en": "Assessing..."},
    "export_results": {"zh": "导出报告", "en": "Export Report"},
    "detailed_results": {"zh": "详细结果", "en": "Detailed Results"},
    "step1": {"zh": "步骤1：下载模板", "en": "Step 1: Download Template"},
    "step2": {"zh": "步骤2：上传数据", "en": "Step 2: Upload Data"},
    "preview_template": {"zh": "预览模板", "en": "Preview Template"},
    "preview_data": {"zh": "预览数据", "en": "Preview Data"},
    "loaded_patients": {"zh": "已加载", "en": "Loaded"},
    "patients_unit": {"zh": "例", "en": "cases"},
    "high_risk_attention": {"zh": "高危患者名单", "en": "High-Risk Patients"},
    "month_12": {"zh": "12个月", "en": "12M"},
    "month_36": {"zh": "36个月", "en": "36M"},
    "month_60": {"zh": "60个月", "en": "60M"},
    "advice_low": {
        "zh": "• 常规随访：每6个月复查\n• 影像检查：每年盆腔超声\n• 标志物：每6个月CA125、HE4\n• 健康生活：均衡饮食，适度运动",
        "en": "• Routine follow-up: Every 6 months\n• Imaging: Annual pelvic ultrasound\n• Markers: CA125, HE4 every 6 months\n• Healthy lifestyle recommended"
    },
    "advice_medium": {
        "zh": "• 加强随访：每3-4个月复查\n• 影像检查：每6个月CT/MRI\n• 标志物：每3个月检测\n• 评估辅助治疗必要性\n• 建议遗传咨询",
        "en": "• Enhanced follow-up: Every 3-4 months\n• Imaging: CT/MRI every 6 months\n• Markers: Every 3 months\n• Evaluate adjuvant therapy\n• Genetic counseling recommended"
    },
    "advice_high": {
        "zh": "• 密切随访：每2-3个月复查\n• 影像检查：每3个月CT/MRI\n• 标志物：每6-8周检测\n• 强烈建议辅助化疗\n• 建议MDT多学科会诊\n• 可考虑临床试验",
        "en": "• Close follow-up: Every 2-3 months\n• Imaging: CT/MRI every 3 months\n• Markers: Every 6-8 weeks\n• Adjuvant chemo recommended\n• MDT consultation advised\n• Consider clinical trials"
    }
}

# ================== 输入变量 ==================
INPUT_VARIABLES = {
    "age": {"zh": "年龄", "en": "Age", "type": "number", "min": 18, "max": 100, "default": 50, "unit": {"zh": "岁", "en": "yrs"}},
    "family_cancer_history": {"zh": "家族史", "en": "Family Hx", "type": "select", "options": {"no": {"zh": "无", "en": "No"}, "yes": {"zh": "有", "en": "Yes"}}},
    "sexual_history": {"zh": "性生活史", "en": "Sexual Hx", "type": "select", "options": {"no": {"zh": "无", "en": "No"}, "yes": {"zh": "有", "en": "Yes"}}},
    "parity": {"zh": "孕产次", "en": "Parity", "type": "select", "options": {"0": {"zh": "未育", "en": "0"}, "1": {"zh": "1次", "en": "1"}, "2": {"zh": "2次", "en": "2"}, "3": {"zh": "≥3", "en": "≥3"}}},
    "menopausal_status": {"zh": "月经状态", "en": "Menopause", "type": "select", "options": {"premenopausal": {"zh": "绝经前", "en": "Pre"}, "postmenopausal": {"zh": "绝经后", "en": "Post"}}},
    "comorbidities": {"zh": "合并症", "en": "Comorbidities", "type": "select", "options": {"no": {"zh": "无", "en": "None"}, "hypertension": {"zh": "高血压", "en": "HTN"}, "diabetes": {"zh": "糖尿病", "en": "DM"}, "cardiovascular": {"zh": "心血管", "en": "CVD"}, "multiple": {"zh": "多种", "en": "Multi"}}},
    "smoking_drinking_history": {"zh": "烟酒史", "en": "Smoke/Alcohol", "type": "select", "options": {"no": {"zh": "无", "en": "No"}, "smoking": {"zh": "吸烟", "en": "Smoke"}, "drinking": {"zh": "饮酒", "en": "Alcohol"}, "both": {"zh": "均有", "en": "Both"}}},
    "receive_estrogens": {"zh": "激素暴露", "en": "Hormone", "type": "select", "options": {"no": {"zh": "无", "en": "No"}, "hrt": {"zh": "HRT", "en": "HRT"}, "contraceptive": {"zh": "避孕药", "en": "OCP"}, "other": {"zh": "其他", "en": "Other"}}},
    "ovulation_induction": {"zh": "促排卵史", "en": "Ovul Induc", "type": "select", "options": {"no": {"zh": "无", "en": "No"}, "yes": {"zh": "有", "en": "Yes"}}},
    "presenting_symptom": {"zh": "主诉症状", "en": "Symptom", "type": "select", "options": {"asymptomatic": {"zh": "无症状", "en": "None"}, "abdominal_pain": {"zh": "腹痛", "en": "Pain"}, "bloating": {"zh": "腹胀", "en": "Bloat"}, "mass": {"zh": "包块", "en": "Mass"}, "bleeding": {"zh": "出血", "en": "Bleed"}, "other": {"zh": "其他", "en": "Other"}}},
    "surgical_route": {"zh": "手术途径", "en": "Surgery", "type": "select", "options": {"laparoscopy": {"zh": "腹腔镜", "en": "Lap"}, "laparotomy": {"zh": "开腹", "en": "Open"}, "robotic": {"zh": "机器人", "en": "Robot"}, "conversion": {"zh": "中转", "en": "Conv"}}},
    "tumor_envelope_integrity": {"zh": "包膜完整", "en": "Capsule", "type": "select", "options": {"intact": {"zh": "完整", "en": "Intact"}, "ruptured_before": {"zh": "术前破", "en": "Pre-rupt"}, "ruptured_during": {"zh": "术中破", "en": "Intra-rupt"}}},
    "fertility_sparing_surgery": {"zh": "保留生育", "en": "Fertility", "type": "select", "options": {"no": {"zh": "否", "en": "No"}, "yes": {"zh": "是", "en": "Yes"}}},
    "completeness_of_surgery": {"zh": "手术完整", "en": "Complete", "type": "select", "options": {"incomplete": {"zh": "不完整", "en": "Incomp"}, "complete": {"zh": "完整", "en": "Comp"}}},
    "omentectomy": {"zh": "网膜切除", "en": "Omentectomy", "type": "select", "options": {"no": {"zh": "未切", "en": "No"}, "partial": {"zh": "部分", "en": "Part"}, "total": {"zh": "全切", "en": "Total"}}},
    "lymphadenectomy": {"zh": "淋巴结清扫", "en": "LND", "type": "select", "options": {"no": {"zh": "未清扫", "en": "No"}, "pelvic": {"zh": "盆腔", "en": "Pelv"}, "paraaortic": {"zh": "腹主旁", "en": "PA"}, "both": {"zh": "盆+腹主", "en": "Both"}}},
    "postoperative_adjuvant_therapy": {"zh": "辅助治疗", "en": "Adjuvant", "type": "select", "options": {"no": {"zh": "无", "en": "No"}, "chemotherapy": {"zh": "化疗", "en": "Chemo"}, "targeted": {"zh": "靶向", "en": "Target"}, "combined": {"zh": "联合", "en": "Comb"}}},
    "histological_subtype": {"zh": "组织类型", "en": "Histology", "type": "select", "options": {"serous": {"zh": "浆液性", "en": "Serous"}, "mucinous": {"zh": "黏液性", "en": "Mucin"}, "endometrioid": {"zh": "内膜样", "en": "Endo"}, "clear_cell": {"zh": "透明细胞", "en": "Clear"}, "mixed": {"zh": "混合", "en": "Mixed"}, "other": {"zh": "其他", "en": "Other"}}},
    "micropapillary": {"zh": "微乳头", "en": "Micropap", "type": "select", "options": {"no": {"zh": "无", "en": "No"}, "yes": {"zh": "有", "en": "Yes"}}},
    "microinfiltration": {"zh": "微浸润", "en": "Microinv", "type": "select", "options": {"no": {"zh": "无", "en": "No"}, "yes": {"zh": "有", "en": "Yes"}}},
    "psammoma_bodies_calcification": {"zh": "砂粒体", "en": "Psammoma", "type": "select", "options": {"no": {"zh": "无", "en": "No"}, "yes": {"zh": "有", "en": "Yes"}}},
    "peritoneal_implantation": {"zh": "腹膜种植", "en": "Peritoneal", "type": "select", "options": {"no": {"zh": "无", "en": "No"}, "noninvasive": {"zh": "非浸润", "en": "Non-inv"}, "invasive": {"zh": "浸润", "en": "Inv"}}},
    "ascites_cytology": {"zh": "腹水细胞学", "en": "Ascites", "type": "select", "options": {"no_ascites": {"zh": "无腹水", "en": "None"}, "negative": {"zh": "阴性", "en": "Neg"}, "positive": {"zh": "阳性", "en": "Pos"}}},
    "figo_staging": {"zh": "FIGO分期", "en": "FIGO", "type": "select", "options": {"IA": {"zh": "IA", "en": "IA"}, "IB": {"zh": "IB", "en": "IB"}, "IC1": {"zh": "IC1", "en": "IC1"}, "IC2": {"zh": "IC2", "en": "IC2"}, "IC3": {"zh": "IC3", "en": "IC3"}, "II": {"zh": "II", "en": "II"}, "IIIA": {"zh": "IIIA", "en": "IIIA"}, "IIIB": {"zh": "IIIB", "en": "IIIB"}, "IIIC": {"zh": "IIIC", "en": "IIIC"}}},
    "unilateral_or_bilateral": {"zh": "侧别", "en": "Lateral", "type": "select", "options": {"left": {"zh": "左", "en": "L"}, "right": {"zh": "右", "en": "R"}, "bilateral": {"zh": "双侧", "en": "Bil"}}},
    "tumor_size": {"zh": "肿瘤大小", "en": "Size", "type": "select", "options": {"<=5": {"zh": "≤5cm", "en": "≤5"}, "5-10": {"zh": "5-10cm", "en": "5-10"}, "10-15": {"zh": "10-15cm", "en": "10-15"}, ">15": {"zh": ">15cm", "en": ">15"}}},
    "type_of_lesion": {"zh": "病灶性质", "en": "Lesion", "type": "select", "options": {"cystic": {"zh": "囊性", "en": "Cyst"}, "solid": {"zh": "实性", "en": "Solid"}, "mixed": {"zh": "囊实", "en": "Mix"}}},
    "papillary_area_ratio": {"zh": "乳头占比", "en": "Papillary%", "type": "select", "options": {"<10%": {"zh": "<10%", "en": "<10%"}, "10-30%": {"zh": "10-30%", "en": "10-30%"}, "30-50%": {"zh": "30-50%", "en": "30-50%"}, ">50%": {"zh": ">50%", "en": ">50%"}}},
    "ca125": {"zh": "CA125", "en": "CA125", "type": "select", "options": {"normal": {"zh": "正常", "en": "Norm"}, "mild": {"zh": "轻度↑", "en": "Mild↑"}, "moderate": {"zh": "中度↑", "en": "Mod↑"}, "high": {"zh": "显著↑", "en": "High↑"}}},
    "cea": {"zh": "CEA", "en": "CEA", "type": "select", "options": {"normal": {"zh": "正常", "en": "Norm"}, "elevated": {"zh": "升高", "en": "↑"}}},
    "ca199": {"zh": "CA199", "en": "CA199", "type": "select", "options": {"normal": {"zh": "正常", "en": "Norm"}, "elevated": {"zh": "升高", "en": "↑"}}},
    "afp": {"zh": "AFP", "en": "AFP", "type": "select", "options": {"normal": {"zh": "正常", "en": "Norm"}, "elevated": {"zh": "升高", "en": "↑"}}},
    "ca724": {"zh": "CA724", "en": "CA724", "type": "select", "options": {"normal": {"zh": "正常", "en": "Norm"}, "elevated": {"zh": "升高", "en": "↑"}}},
    "he4": {"zh": "HE4", "en": "HE4", "type": "select", "options": {"normal": {"zh": "正常", "en": "Norm"}, "mild": {"zh": "轻度↑", "en": "Mild↑"}, "elevated": {"zh": "显著↑", "en": "High↑"}}}
}

# ================== 模型类 ==================
class DataPreprocessor:
    def __init__(self, select_k=None):
        self.scaler = StandardScaler()
        self.selector = None
        self.select_k = select_k
    def fit(self, X, y=None):
        self.scaler.fit(X)
        if self.select_k and y is not None:
            self.selector = SelectKBest(f_classif, k=min(self.select_k, X.shape[1]))
            self.selector.fit(self.scaler.transform(X), y)
        return self
    def transform(self, X):
        X_s = self.scaler.transform(X)
        return self.selector.transform(X_s) if self.selector else X_s

class SEBlock(nn.Module):
    def __init__(self, dim, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim, max(dim//r,1)), nn.ReLU(), nn.Linear(max(dim//r,1), dim), nn.Sigmoid())
    def forward(self, x): return x * self.fc(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim, drop=0.3):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(dim,dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(drop), nn.Linear(dim,dim), nn.BatchNorm1d(dim))
        self.se, self.act, self.drop = SEBlock(dim), nn.GELU(), nn.Dropout(drop)
    def forward(self, x): return self.act(x + self.drop(self.se(self.block(x))))

class EnhancedDeepSurv(nn.Module):
    def __init__(self, in_dim, h=[256,128,64], drop=0.3, n_res=2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, h[0]), nn.BatchNorm1d(h[0]), nn.GELU(), nn.Dropout(drop))
        self.res = nn.ModuleList([ResidualBlock(h[0], drop) for _ in range(n_res)])
        self.down = nn.ModuleList([nn.Sequential(nn.Linear(h[i], h[i+1]), nn.BatchNorm1d(h[i+1]), nn.GELU(), nn.Dropout(drop)) for i in range(len(h)-1)])
        self.out = nn.Linear(h[-1], 1)
    def forward(self, x):
        x = self.proj(x)
        for r in self.res: x = r(x)
        for d in self.down: x = d(x)
        return self.out(x).squeeze(-1)

class EnhancedDeepHit(nn.Module):
    def __init__(self, in_dim, h=[256,128], n_dur=10, drop=0.3):
        super().__init__()
        layers, d = [], in_dim
        for hd in h: layers.extend([nn.Linear(d, hd), nn.BatchNorm1d(hd), nn.GELU(), nn.Dropout(drop)]); d = hd
        layers.append(nn.Linear(d, n_dur))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return torch.softmax(self.net(x), dim=1)

class EnhancedDenoisingAE(nn.Module):
    def __init__(self, in_dim, h=[256,128], lat=64, drop=0.2):
        super().__init__()
        enc, d = [], in_dim
        for hd in h: enc.extend([nn.Linear(d, hd), nn.BatchNorm1d(hd), nn.GELU(), nn.Dropout(drop)]); d = hd
        enc.append(nn.Linear(d, lat))
        self.encoder = nn.Sequential(*enc)
    def encode(self, x): return self.encoder(x)

class EnhancedTransformer(nn.Module):
    def __init__(self, lat, n_h=4, ff=256, n_l=2, drop=0.1):
        super().__init__()
        while lat % n_h != 0 and n_h > 1: n_h -= 1
        self.norm = nn.LayerNorm(lat)
        self.trans = nn.TransformerEncoder(nn.TransformerEncoderLayer(lat, n_h, ff, drop, 'gelu', batch_first=True), n_l)
        self.proj = nn.Sequential(nn.Linear(lat, lat), nn.GELU(), nn.Dropout(drop))
    def forward(self, z):
        if z.dim() == 2: z = z.unsqueeze(1)
        return self.proj(self.trans(self.norm(z)).squeeze(1))

class LearnableFusion(nn.Module):
    def __init__(self, in_d=2, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x).squeeze(-1)

# ================== 工具函数 ==================
def get_text(key, lang): return TRANSLATIONS.get(key, {}).get(lang, key)

def encode_option(var, opt):
    opts = INPUT_VARIABLES.get(var, {}).get("options", {})
    try: return float(list(opts.keys()).index(opt))
    except: return 0.0

@st.cache_resource
def load_models(model_dir="results_clinical_enhanced_v3"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, ok = {}, False
    try:
        req = ['model_ae.pt', 'model_trans.pt', 'model_deepsurv.pt', 'model_deephit.pt', 'model_fusion.pt', 'preprocessor.joblib', 'time_cuts.npy', 'ds_min_max.npy', 'best_parameters.json']
        if all(os.path.exists(os.path.join(model_dir, f)) for f in req):
            with open(os.path.join(model_dir, "best_parameters.json")) as f: params = json.load(f)
            prep = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))
            time_cuts = np.load(os.path.join(model_dir, "time_cuts.npy"))
            ds_mm = np.load(os.path.join(model_dir, "ds_min_max.npy"))
            in_dim = prep.scaler.n_features_in_
            if hasattr(prep, 'selector') and prep.selector: in_dim = getattr(prep.selector, 'k', in_dim)
            lat, fused = params.get('ae_latent', 64), params.get('ae_latent', 64) * 2
            ae = EnhancedDenoisingAE(in_dim, [params.get('ae_h1',256), params.get('ae_h2',128)], lat)
            ae.load_state_dict(torch.load(os.path.join(model_dir, "model_ae.pt"), map_location=device)); ae.eval()
            trans = EnhancedTransformer(lat)
            trans.load_state_dict(torch.load(os.path.join(model_dir, "model_trans.pt"), map_location=device)); trans.eval()
            ds = EnhancedDeepSurv(fused, [params.get('ds_h1',256), params.get('ds_h2',128), params.get('ds_h3',64)], params.get('ds_drop',0.3))
            ds.load_state_dict(torch.load(os.path.join(model_dir, "model_deepsurv.pt"), map_location=device)); ds.eval()
            dh = EnhancedDeepHit(fused, [params.get('dh_h1',256), params.get('dh_h2',128)], len(time_cuts)-1)
            dh.load_state_dict(torch.load(os.path.join(model_dir, "model_deephit.pt"), map_location=device)); dh.eval()
            fusion = LearnableFusion()
            fusion.load_state_dict(torch.load(os.path.join(model_dir, "model_fusion.pt"), map_location=device)); fusion.eval()
            models = {'ae': ae.to(device), 'trans': trans.to(device), 'ds': ds.to(device), 'dh': dh.to(device), 'fusion': fusion.to(device), 'prep': prep, 'time_cuts': time_cuts, 'ds_mm': ds_mm, 'device': device}
            ok = True
    except: pass
    if not ok:
        in_dim, lat, fused, n_bins = len(INPUT_VARIABLES), 64, 128, 10
        models = {'ae': EnhancedDenoisingAE(in_dim, [256,128], lat).to(device), 'trans': EnhancedTransformer(lat).to(device), 'ds': EnhancedDeepSurv(fused, [256,128,64]).to(device), 'dh': EnhancedDeepHit(fused, [256,128], n_bins).to(device), 'fusion': LearnableFusion().to(device), 'prep': None, 'time_cuts': np.linspace(0,120,11), 'ds_mm': np.array([-5.,5.]), 'device': device}
        for k in ['ae','trans','ds','dh','fusion']: models[k].eval()
    models['ok'] = ok
    return models

def preprocess(data, models):
    feats = [encode_option(v, data.get(v)) if INPUT_VARIABLES[v]['type']=='select' else float(data.get(v, INPUT_VARIABLES[v].get('default',0))) for v in INPUT_VARIABLES]
    X = np.array(feats).reshape(1, -1)
    if models.get('prep'):
        try: X = models['prep'].transform(X)
        except: X = (X - X.mean()) / (X.std() + 1e-8)
    else: X = (X - X.mean()) / (X.std() + 1e-8)
    return X

def predict(data, models):
    dev = models['device']
    X = torch.tensor(preprocess(data, models), dtype=torch.float32, device=dev)
    with torch.no_grad():
        Z = models['ae'].encode(X)
        T = models['trans'](Z)
        Xf = torch.cat([Z, T], dim=1)
        r_ds = models['ds'](Xf).cpu().numpy(); r_ds = r_ds.item() if r_ds.ndim == 0 else r_ds[0]
        pmf = models['dh'](Xf).cpu().numpy()[0]
        mn, mx = models['ds_mm']
        p_ds = np.clip((r_ds - mn) / (mx - mn + 1e-8), 0, 1)
        cif, surv = np.cumsum(pmf), 1 - np.cumsum(pmf)
        r_dh = cif[len(pmf)//2]
        final = models['fusion'](torch.tensor([[p_ds, r_dh]], dtype=torch.float32, device=dev)).cpu().numpy()
        final = final.item() if final.ndim == 0 else final[0]
    tc = models['time_cuts']
    tp = (tc[:-1] + tc[1:]) / 2
    n = len(cif)
    return {'risk': float(final), 'surv': surv, 'cif': cif, 'tp': tp, 'r12': float(cif[min(int(n*0.1),n-1)]), 'r36': float(cif[min(int(n*0.3),n-1)]), 'r60': float(cif[min(int(n*0.5),n-1)])}

def batch_predict(df, models, lang):
    results = []
    prog = st.progress(0)
    for i, row in df.iterrows():
        data = {}
        for v in INPUT_VARIABLES:
            for lg in ['zh', 'en']:
                col = INPUT_VARIABLES[v][lg]
                if col in row: data[v] = row[col]; break
            if v not in data and v in row: data[v] = row[v]
        try:
            p = predict(data, models)
            lv = get_text("low_risk" if p['risk']<0.3 else ("medium_risk" if p['risk']<0.6 else "high_risk"), lang)
            m = get_text("months", lang)
            results.append({get_text("patient_id",lang): row.get('patient_id', row.get('患者编号', i+1)), get_text("overall_risk",lang): f"{p['risk']*100:.1f}%", f"12{m}": f"{p['r12']*100:.1f}%", f"36{m}": f"{p['r36']*100:.1f}%", f"60{m}": f"{p['r60']*100:.1f}%", get_text("risk_level",lang): lv, '_r': p['risk']})
        except: pass
        prog.progress((i+1)/len(df))
    prog.empty()
    return pd.DataFrame(results)

def make_template(lang):
    cols = [get_text("patient_id", lang)] + [INPUT_VARIABLES[v][lang] for v in INPUT_VARIABLES]
    data = {cols[0]: [1,2,3]}
    for i, (v, info) in enumerate(INPUT_VARIABLES.items()):
        data[cols[i+1]] = [list(info['options'].keys())[0]]*3 if info['type']=='select' else [info.get('default',0)]*3
    return pd.DataFrame(data)

# ================== 图表函数 ==================
def make_gauge(risk, lang):
    if risk < 0.3: col, lv = "#27ae60", get_text("low_risk", lang)
    elif risk < 0.6: col, lv = "#f39c12", get_text("medium_risk", lang)
    else: col, lv = "#e74c3c", get_text("high_risk", lang)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=risk*100,
        number={'suffix':'%', 'font':{'size':64, 'color':col, 'family':'Arial Black'}},
        title={'text': f"<b>{get_text('overall_risk', lang)}</b><br><span style='font-size:26px;color:{col}'>{lv}</span>", 'font':{'size':20}},
        gauge={'axis':{'range':[0,100], 'tickwidth':2, 'tickcolor':'#555', 'tickfont':{'size':16}, 'dtick':25},
               'bar':{'color':col, 'thickness':0.7}, 'bgcolor':'#f0f0f0', 'borderwidth':2, 'bordercolor':'#888',
               'steps':[{'range':[0,30],'color':'rgba(39,174,96,0.2)'}, {'range':[30,60],'color':'rgba(243,156,18,0.2)'}, {'range':[60,100],'color':'rgba(231,76,60,0.2)'}]}
    ))
    fig.update_layout(height=350, margin=dict(l=30,r=30,t=100,b=30), paper_bgcolor='rgba(0,0,0,0)')
    return fig

def make_time_bar(r12, r36, r60, lang):
    labels = [get_text('month_12', lang), get_text('month_36', lang), get_text('month_60', lang)]
    vals = [r12*100, r36*100, r60*100]
    cols = ['#27ae60' if v<30 else ('#f39c12' if v<60 else '#e74c3c') for v in vals]
    fig = go.Figure(data=[go.Bar(x=labels, y=vals, marker_color=cols, text=[f'<b>{v:.1f}%</b>' for v in vals], textposition='outside', textfont=dict(size=20, color='#333'), width=0.5)])
    fig.update_layout(title=dict(text=f"<b>{get_text('time_risk', lang)}</b>", font=dict(size=18), x=0.5),
                     xaxis=dict(tickfont=dict(size=16)), yaxis=dict(title=f"<b>{get_text('risk_prob', lang)} (%)</b>", title_font=dict(size=16), tickfont=dict(size=14), range=[0, max(vals)*1.35 if max(vals)>0 else 100], gridcolor='#e8e8e8'),
                     height=350, margin=dict(l=70,r=30,t=70,b=50), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white')
    return fig

def make_survival_chart(surv, tp, lang):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tp, y=surv, mode='lines+markers', line=dict(color='#3498db', width=3), fill='tozeroy', fillcolor='rgba(52,152,219,0.15)', marker=dict(size=10, color='#3498db', line=dict(width=2, color='white'))))
    fig.update_layout(title=dict(text=f"<b>{get_text('survival_curve', lang)}</b>", font=dict(size=18), x=0.5),
                     xaxis=dict(title=f"<b>{get_text('time_months', lang)}</b>", title_font=dict(size=16), tickfont=dict(size=14), gridcolor='#e8e8e8', dtick=12),
                     yaxis=dict(title=f"<b>{get_text('survival_prob', lang)}</b>", title_font=dict(size=16), tickfont=dict(size=14), range=[0,1.05], gridcolor='#e8e8e8', tickformat='.0%'),
                     height=350, margin=dict(l=70,r=30,t=70,b=60), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white', showlegend=False)
    return fig

def make_cumulative_chart(cif, tp, lang):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tp, y=cif, mode='lines+markers', line=dict(color='#e74c3c', width=3), fill='tozeroy', fillcolor='rgba(231,76,60,0.15)', marker=dict(size=10, color='#e74c3c', symbol='square', line=dict(width=2, color='white'))))
    fig.update_layout(title=dict(text=f"<b>{get_text('cumulative_risk_curve', lang)}</b>", font=dict(size=18), x=0.5),
                     xaxis=dict(title=f"<b>{get_text('time_months', lang)}</b>", title_font=dict(size=16), tickfont=dict(size=14), gridcolor='#e8e8e8', dtick=12),
                     yaxis=dict(title=f"<b>{get_text('risk_prob', lang)}</b>", title_font=dict(size=16), tickfont=dict(size=14), range=[0,1.05], gridcolor='#e8e8e8', tickformat='.0%'),
                     height=350, margin=dict(l=70,r=30,t=70,b=60), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white', showlegend=False)
    return fig

def make_pie(df, lang):
    rc = get_text("risk_level", lang)
    h = len(df[df[rc].str.contains('High|高', case=False, na=False)]) if rc in df.columns else 0
    m = len(df[df[rc].str.contains('Intermediate|中', case=False, na=False)]) if rc in df.columns else 0
    l = len(df) - h - m
    fig = go.Figure(data=[go.Pie(labels=[get_text('low_risk',lang), get_text('medium_risk',lang), get_text('high_risk',lang)], values=[l, m, h], marker_colors=['#27ae60','#f39c12','#e74c3c'], hole=0.45, textinfo='label+percent+value', textfont=dict(size=15), pull=[0, 0, 0.05])])
    fig.update_layout(title=dict(text=f"<b>{get_text('risk_distribution', lang)}</b>", font=dict(size=18), x=0.5), height=380, margin=dict(l=20,r=20,t=70,b=20), paper_bgcolor='rgba(0,0,0,0)', legend=dict(font=dict(size=14), orientation='h', yanchor='bottom', y=-0.12, xanchor='center', x=0.5))
    return fig

# ================== PDF生成 ==================
def make_pdf(df, lang):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    total = len(df)
    rc = get_text("risk_level", lang)
    h = len(df[df[rc].str.contains('High|高', case=False, na=False)]) if rc in df.columns else 0
    m = len(df[df[rc].str.contains('Intermediate|中', case=False, na=False)]) if rc in df.columns else 0
    l = total - h - m
    story = [Paragraph("Cancer Recurrence Risk Report", ParagraphStyle('T', parent=styles['Heading1'], fontSize=18, spaceAfter=20, alignment=1)),
             Paragraph("Shengjing Hospital of China Medical University", styles['Normal']),
             Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']), Spacer(1, 20)]
    data = [["Category", "Count", "%"], ["Total", str(total), "100%"], ["High", str(h), f"{h/total*100:.1f}%" if total else "0%"], ["Medium", str(m), f"{m/total*100:.1f}%" if total else "0%"], ["Low", str(l), f"{l/total*100:.1f}%" if total else "0%"]]
    tbl = Table(data, colWidths=[120, 80, 80])
    tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#3498db')), ('TEXTCOLOR',(0,0),(-1,0),colors.white), ('ALIGN',(0,0),(-1,-1),'CENTER'), ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'), ('GRID',(0,0),(-1,-1),1,colors.black)]))
    story.extend([tbl, Spacer(1, 20), Paragraph("For clinical reference only.", ParagraphStyle('D', fontSize=9, textColor=colors.grey))])
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

def make_single_pdf(res, lang):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    r = res['risk']
    lv = "Low" if r < 0.3 else ("Medium" if r < 0.6 else "High")
    story = [Paragraph("Patient Risk Assessment", ParagraphStyle('T', parent=styles['Heading1'], fontSize=18, spaceAfter=20, alignment=1)),
             Paragraph("Shengjing Hospital", styles['Normal']),
             Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']), Spacer(1, 20)]
    data = [["Item", "Value"], ["Risk", f"{r*100:.1f}%"], ["Level", lv], ["12M", f"{res['r12']*100:.1f}%"], ["36M", f"{res['r36']*100:.1f}%"], ["60M", f"{res['r60']*100:.1f}%"]]
    tbl = Table(data, colWidths=[150, 150])
    tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#3498db')), ('TEXTCOLOR',(0,0),(-1,0),colors.white), ('ALIGN',(0,0),(-1,-1),'CENTER'), ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'), ('GRID',(0,0),(-1,-1),1,colors.black)]))
    story.extend([tbl, Spacer(1, 20), Paragraph("For clinical reference only.", ParagraphStyle('D', fontSize=9, textColor=colors.grey))])
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

# ================== 输入控件 ==================
def sel_widget(v, info, lang, pre=""):
    return st.selectbox(info[lang], list(info['options'].keys()), format_func=lambda x: info['options'][x][lang], key=f"{pre}{v}")

def num_widget(v, info, lang, pre=""):
    lbl = f"{info[lang]} ({info['unit'][lang]})" if 'unit' in info else info[lang]
    return st.number_input(lbl, float(info.get('min',0)), float(info.get('max',100)), float(info.get('default',0)), key=f"{pre}{v}")

# ================== 主函数 ==================
def main():
    models = load_models()
    
    # 顶部栏
    if HAS_LOGO:
        st.markdown(f"""
        <div class="top-bar">
            <div class="logo-section">
                <img src="data:image/png;base64,{LOGO_BASE64}" class="logo-img" alt="Logo">
                <div class="logo-text">
                    <h2>盛京医院 Shengjing Hospital</h2>
                    <p>中国医科大学附属盛京医院</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="top-bar">
            <div class="logo-section">
                <div class="logo-text">
                    <h2>🏥 盛京医院 Shengjing Hospital</h2>
                    <p>中国医科大学附属盛京医院</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # 语言选择（右上角）
    col_space, col_lang = st.columns([10, 1])
    with col_lang:
        lang = LANGUAGES[st.selectbox("🌐", list(LANGUAGES.keys()), label_visibility="collapsed", key="lang")]
    
    # 医院头部
    if HAS_LOGO:
        st.markdown(f"""
        <div class="hospital-header">
            <div class="header-logo">
                <img src="data:image/png;base64,{LOGO_BASE64}" alt="Logo">
            </div>
            <div class="header-text">
                <h1>🏥 {get_text('title', lang)}</h1>
                <p class="subtitle">{get_text('subtitle', lang)}</p>
                <p class="hospital-name">{get_text('hospital', lang)}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="hospital-header-nologo">
            <h1>🏥 {get_text('title', lang)}</h1>
            <p class="subtitle">{get_text('subtitle', lang)}</p>
            <p class="hospital-name">{get_text('hospital', lang)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 标签页
    tab1, tab2 = st.tabs([f"📋 {get_text('single_patient', lang)}", f"📊 {get_text('batch_prediction', lang)}"])
    
    # ========== 单例预测 ==========
    with tab1:
        c1, c2, c3 = st.columns(3)
        data = {}
        
        with c1:
            st.markdown(f'<div class="module-card"><div class="module-title">📝 {get_text("basic_info", lang)}</div>', unsafe_allow_html=True)
            for v in ['age','family_cancer_history','sexual_history','parity','menopausal_status','comorbidities','smoking_drinking_history','receive_estrogens','ovulation_induction']:
                info = INPUT_VARIABLES[v]
                data[v] = num_widget(v, info, lang, "s_") if info['type']=='number' else sel_widget(v, info, lang, "s_")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with c2:
            st.markdown(f'<div class="module-card"><div class="module-title surgery">🔪 {get_text("surgical_info", lang)}</div>', unsafe_allow_html=True)
            for v in ['presenting_symptom','surgical_route','tumor_envelope_integrity','fertility_sparing_surgery','completeness_of_surgery','omentectomy','lymphadenectomy','postoperative_adjuvant_therapy']:
                data[v] = sel_widget(v, INPUT_VARIABLES[v], lang, "s_")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with c3:
            st.markdown(f'<div class="module-card"><div class="module-title pathology">🔬 {get_text("pathology_info", lang)}</div>', unsafe_allow_html=True)
            for v in ['histological_subtype','micropapillary','microinfiltration','psammoma_bodies_calcification','peritoneal_implantation','ascites_cytology','figo_staging','unilateral_or_bilateral','tumor_size','type_of_lesion','papillary_area_ratio']:
                data[v] = sel_widget(v, INPUT_VARIABLES[v], lang, "s_")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="module-card"><div class="module-title markers">🧪 {get_text("tumor_markers", lang)}</div>', unsafe_allow_html=True)
        mc = st.columns(6)
        for i, v in enumerate(['ca125','cea','ca199','afp','ca724','he4']):
            with mc[i]: data[v] = sel_widget(v, INPUT_VARIABLES[v], lang, "s_")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        bc1, bc2, bc3 = st.columns([2, 1, 2])
        with bc2:
            predict_btn = st.button(f"🔮 {get_text('predict_button', lang)}", use_container_width=True, key="pred")
        
        if predict_btn:
            with st.spinner(get_text('processing', lang)):
                res = predict(data, models)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f'<div class="result-section"><div class="result-title">📊 {get_text("prediction_results", lang)}</div>', unsafe_allow_html=True)
                
                # 2x2布局
                row1_c1, row1_c2 = st.columns(2)
                with row1_c1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(make_gauge(res['risk'], lang), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with row1_c2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(make_time_bar(res['r12'], res['r36'], res['r60'], lang), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                row2_c1, row2_c2 = st.columns(2)
                with row2_c1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(make_survival_chart(res['surv'], res['tp'], lang), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with row2_c2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(make_cumulative_chart(res['cif'], res['tp'], lang), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # 临床建议
                r = res['risk']
                if r < 0.3: lv, adv, css = "low_risk", "advice_low", "low"
                elif r < 0.6: lv, adv, css = "medium_risk", "advice_medium", "medium"
                else: lv, adv, css = "high_risk", "advice_high", "high"
                
                st.markdown(f"""
                <div class="advice-box {css}">
                    <h4>💊 {get_text('clinical_advice', lang)} — {get_text(lv, lang)} ({r*100:.1f}%)</h4>
                    <pre style="white-space: pre-wrap; font-family: inherit; margin: 0; line-height: 1.8; font-size: 0.95rem;">{get_text(adv, lang)}</pre>
                </div>
                """, unsafe_allow_html=True)
                
                # 导出
                st.markdown(f"#### 📥 {get_text('export_results', lang)}")
                ec1, ec2, ec3 = st.columns(3)
                with ec1:
                    df_exp = pd.DataFrame({get_text('overall_risk',lang): [f"{res['risk']*100:.1f}%"], get_text('month_12',lang): [f"{res['r12']*100:.1f}%"], get_text('month_36',lang): [f"{res['r36']*100:.1f}%"], get_text('month_60',lang): [f"{res['r60']*100:.1f}%"]})
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine='openpyxl') as w: df_exp.to_excel(w, index=False)
                    st.download_button(f"📊 {get_text('export_excel', lang)}", buf.getvalue(), f"result_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx", use_container_width=True)
                with ec2:
                    st.download_button(f"📄 {get_text('export_pdf', lang)}", make_single_pdf(res, lang), f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", "application/pdf", use_container_width=True)
    
    # ========== 批量预测 ==========
    with tab2:
        st.markdown(f"#### {get_text('step1', lang)}")
        tpl = make_template(lang)
        buf = io.StringIO(); tpl.to_csv(buf, index=False, encoding='utf-8-sig')
        st.download_button(f"📥 {get_text('download_template', lang)}", buf.getvalue(), f"template_{lang}.csv", "text/csv")
        
        with st.expander(get_text('preview_template', lang)):
            st.dataframe(tpl, use_container_width=True)
        
        st.markdown("---")
        st.markdown(f"#### {get_text('step2', lang)}")
        file = st.file_uploader(get_text('upload_file', lang), ['csv', 'xlsx'])
        
        if file:
            try:
                df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
                st.success(f"✅ {get_text('loaded_patients', lang)} {len(df)} {get_text('patients_unit', lang)}")
                
                with st.expander(get_text('preview_data', lang)):
                    st.dataframe(df.head(10), use_container_width=True)
                
                if st.button(f"🔮 {get_text('predict_button', lang)}", key="batch"):
                    with st.spinner(get_text('processing', lang)):
                        res_df = batch_predict(df, models, lang)
                        
                        st.markdown("---")
                        st.markdown(f"### 📊 {get_text('detailed_results', lang)}")
                        
                        total = len(res_df)
                        rc = get_text("risk_level", lang)
                        h = len(res_df[res_df[rc].str.contains('High|高', case=False, na=False)]) if rc in res_df.columns else 0
                        m = len(res_df[res_df[rc].str.contains('Intermediate|中', case=False, na=False)]) if rc in res_df.columns else 0
                        l = total - h - m
                        
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric(get_text('total_patients', lang), total)
                        m2.metric(get_text('high_risk_count', lang), h)
                        m3.metric(get_text('medium_risk_count', lang), m)
                        m4.metric(get_text('low_risk_count', lang), l)
                        
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            st.plotly_chart(make_pie(res_df, lang), use_container_width=True)
                        with cc2:
                            if '_r' in res_df.columns:
                                fig = go.Figure(go.Histogram(x=res_df['_r']*100, nbinsx=20, marker_color='#3498db', opacity=0.8))
                                fig.add_vline(x=30, line_dash="dash", line_color="#27ae60", line_width=2)
                                fig.add_vline(x=60, line_dash="dash", line_color="#e74c3c", line_width=2)
                                fig.update_layout(title=dict(text=f"<b>{get_text('risk_distribution', lang)}</b>", font=dict(size=18), x=0.5), xaxis=dict(title=f"<b>{get_text('risk_prob', lang)} (%)</b>", title_font=dict(size=16), tickfont=dict(size=14)), yaxis=dict(title=f"<b>{get_text('total_patients', lang)}</b>", title_font=dict(size=16), tickfont=dict(size=14)), height=380, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white')
                                st.plotly_chart(fig, use_container_width=True)
                        
                        disp = res_df.drop(columns=[c for c in res_df.columns if c.startswith('_')], errors='ignore')
                        def hl(row):
                            v = str(row.get(rc, ''))
                            if 'High' in v or '高' in v: return ['background-color:#f8d7da']*len(row)
                            if 'Intermediate' in v or '中' in v: return ['background-color:#fff3cd']*len(row)
                            return ['background-color:#d4edda']*len(row)
                        st.dataframe(disp.style.apply(hl, axis=1), use_container_width=True, height=350)
                        
                        st.markdown(f"#### 📥 {get_text('export_results', lang)}")
                        e1, e2, e3 = st.columns(3)
                        with e1:
                            buf = io.StringIO(); disp.to_csv(buf, index=False, encoding='utf-8-sig')
                            st.download_button(f"📋 {get_text('export_csv', lang)}", buf.getvalue(), f"batch_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", use_container_width=True)
                        with e2:
                            buf = io.BytesIO()
                            with pd.ExcelWriter(buf, engine='openpyxl') as w: disp.to_excel(w, index=False)
                            st.download_button(f"📊 {get_text('export_excel', lang)}", buf.getvalue(), f"batch_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx", use_container_width=True)
                        with e3:
                            st.download_button(f"📄 {get_text('export_pdf', lang)}", make_pdf(res_df, lang), f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", "application/pdf", use_container_width=True)
                        
                        if h > 0:
                            st.markdown("---")
                            st.markdown(f"### ⚠️ {get_text('high_risk_attention', lang)}")
                            hdf = disp[disp[rc].str.contains('High|高', case=False, na=False)]
                            st.dataframe(hdf.style.apply(lambda x: ['background-color:#f8d7da']*len(x), axis=1), use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
    
    # 页脚
    st.markdown("---")
    st.info(get_text('disclaimer', lang))
    
    if HAS_LOGO:
        st.markdown(f"""
        <div class="footer">
            <div class="footer-logo">
                <img src="data:image/png;base64,{LOGO_BASE64}" alt="Logo">
            </div>
            <div class="footer-text">
                <p class="hospital-name">{get_text('hospital', lang)}</p>
                <p class="version">Cancer Recurrence Risk Prediction System </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="footer">
            <div class="footer-text">
                <p class="hospital-name">🏥 {get_text('hospital', lang)}</p>
                <p class="version">Cancer Recurrence Risk Prediction System </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
