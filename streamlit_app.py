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
from plotly.subplots import make_subplots
import joblib
import json
import io
from datetime import datetime
from typing import Dict, List
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

# ================== 专业临床界面CSS ==================
st.markdown("""
<style>
    /* 隐藏默认元素 */
    [data-testid="collapsedControl"] {display: none}
    section[data-testid="stSidebar"] {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 主容器 */
    .main .block-container {
        padding: 1rem 2rem 2rem 2rem;
        max-width: 1400px;
    }
    
    /* 医院标题头部 */
    .hospital-header {
        background: linear-gradient(135deg, #1a5276 0%, #2980b9 50%, #1a5276 100%);
        padding: 1.5rem 2rem;
        border-radius: 0 0 20px 20px;
        margin: -1rem -2rem 1.5rem -2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        text-align: center;
        position: relative;
    }
    .hospital-header::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #e74c3c, #f39c12, #27ae60, #3498db, #9b59b6);
    }
    .hospital-header h1 {
        color: white;
        font-size: 1.8rem;
        margin: 0 0 0.3rem 0;
        font-weight: 600;
        letter-spacing: 2px;
    }
    .hospital-header .subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        margin: 0;
    }
    .hospital-header .hospital-name {
        color: #f1c40f;
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 0.5rem;
        letter-spacing: 1px;
    }
    
    /* 系统状态栏 */
    .status-bar {
        background: linear-gradient(90deg, #d5f4e6, #ffeaa7, #dfe6e9);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.85rem;
        border: 1px solid #b8e0d2;
    }
    .status-item {
        display: flex;
        align-items: center;
        gap: 5px;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #27ae60;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* 模块卡片 */
    .module-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #e8e8e8;
    }
    .module-card:hover {
        box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    }
    
    /* 模块标题 */
    .module-title {
        background: linear-gradient(90deg, #3498db, #2980b9);
        color: white;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        margin: -1.2rem -1.2rem 1rem -1.2rem;
        font-weight: 600;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .module-title.pathology {
        background: linear-gradient(90deg, #9b59b6, #8e44ad);
    }
    .module-title.surgery {
        background: linear-gradient(90deg, #e67e22, #d35400);
    }
    .module-title.markers {
        background: linear-gradient(90deg, #1abc9c, #16a085);
    }
    .module-title.results {
        background: linear-gradient(90deg, #e74c3c, #c0392b);
    }
    
    /* 结果展示区 */
    .result-panel {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid #dee2e6;
    }
    
    /* 风险等级卡片 */
    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .risk-low {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
    }
    .risk-medium {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        border: 2px solid #ffc107;
    }
    .risk-high {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid #dc3545;
    }
    .risk-card h2 {
        margin: 0 0 0.5rem 0;
        font-size: 1.3rem;
    }
    .risk-card .risk-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .risk-low .risk-value { color: #155724; }
    .risk-medium .risk-value { color: #856404; }
    .risk-high .risk-value { color: #721c24; }
    
    /* 建议卡片 */
    .advice-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        border-left: 5px solid;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }
    .advice-card.low { border-color: #28a745; }
    .advice-card.medium { border-color: #ffc107; }
    .advice-card.high { border-color: #dc3545; }
    .advice-card h4 {
        margin: 0 0 0.8rem 0;
        color: #2c3e50;
        font-size: 1.1rem;
    }
    .advice-card ul {
        margin: 0;
        padding-left: 1.2rem;
        line-height: 1.8;
    }
    
    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        border: none;
        padding: 0.8rem 2.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 30px;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.4);
        transition: all 0.3s ease;
        letter-spacing: 1px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(52, 152, 219, 0.5);
        background: linear-gradient(135deg, #2980b9 0%, #1a5276 100%);
    }
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* 选择框优化 */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid #ced4da;
    }
    .stSelectbox > div > div:focus-within {
        border-color: #3498db;
        box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
    }
    .stSelectbox label {
        font-weight: 500;
        color: #2c3e50;
        font-size: 0.9rem;
    }
    
    /* 数字输入框优化 */
    .stNumberInput > div > div > input {
        border-radius: 8px;
    }
    .stNumberInput label {
        font-weight: 500;
        color: #2c3e50;
        font-size: 0.9rem;
    }
    
    /* 标签页样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #f8f9fa;
        border-radius: 10px;
        padding: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        color: #6c757d;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white !important;
    }
    
    /* 指标卡片 */
    [data-testid="metric-container"] {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid #e8e8e8;
    }
    [data-testid="metric-container"] label {
        font-weight: 600;
        color: #2c3e50;
    }
    
    /* 文件上传区域 */
    .stFileUploader > div {
        border: 2px dashed #3498db;
        border-radius: 12px;
        padding: 1rem;
        background: #f8f9fa;
    }
    .stFileUploader > div:hover {
        background: #eef5fc;
        border-color: #2980b9;
    }
    
    /* 成功/警告/错误消息 */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* 分隔线 */
    hr {
        margin: 1.5rem 0;
        border: none;
        border-top: 1px solid #dee2e6;
    }
    
    /* 页脚 */
    .footer {
        background: linear-gradient(135deg, #1a5276 0%, #2980b9 100%);
        padding: 1.5rem 2rem;
        border-radius: 15px;
        margin-top: 2rem;
        text-align: center;
        color: white;
    }
    .footer .hospital-name {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f1c40f;
        margin-bottom: 0.5rem;
    }
    .footer .version {
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .footer .copyright {
        font-size: 0.8rem;
        opacity: 0.7;
        margin-top: 0.5rem;
    }
    
    /* 进度条 */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3498db, #2ecc71);
        border-radius: 10px;
    }
    
    /* 表格样式 */
    .dataframe {
        font-size: 0.9rem !important;
    }
    .dataframe th {
        background: #3498db !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ================== 语言配置 ==================
LANGUAGES = {"中文": "zh", "English": "en"}

# ================== 翻译字典 ==================
TRANSLATIONS = {
    "title": {"zh": "肿瘤复发风险预测系统", "en": "Cancer Recurrence Prediction System"},
    "subtitle": {"zh": "临床决策支持平台 · Clinical Decision Support", "en": "Clinical Decision Support Platform"},
    "hospital": {"zh": "中国医科大学附属盛京医院", "en": "Shengjing Hospital of China Medical University"},
    "system_ready": {"zh": "系统就绪", "en": "System Ready"},
    "model_status": {"zh": "模型状态", "en": "Model Status"},
    "current_time": {"zh": "当前时间", "en": "Current Time"},
    "single_patient": {"zh": "单例预测", "en": "Single Prediction"},
    "batch_prediction": {"zh": "批量预测", "en": "Batch Prediction"},
    "basic_info": {"zh": "基本信息", "en": "Basic Information"},
    "surgical_info": {"zh": "手术信息", "en": "Surgical Information"},
    "pathology_info": {"zh": "病理信息", "en": "Pathology Information"},
    "tumor_markers": {"zh": "肿瘤标志物", "en": "Tumor Markers"},
    "predict_button": {"zh": "开始风险评估", "en": "Start Risk Assessment"},
    "prediction_results": {"zh": "风险评估结果", "en": "Risk Assessment Results"},
    "overall_risk": {"zh": "综合复发风险", "en": "Overall Recurrence Risk"},
    "risk_level": {"zh": "风险分层", "en": "Risk Stratification"},
    "low_risk": {"zh": "低危", "en": "Low Risk"},
    "medium_risk": {"zh": "中危", "en": "Intermediate Risk"},
    "high_risk": {"zh": "高危", "en": "High Risk"},
    "survival_curve": {"zh": "生存曲线分析", "en": "Survival Curve Analysis"},
    "time_risk": {"zh": "各时间点复发概率", "en": "Recurrence Probability by Time"},
    "clinical_advice": {"zh": "临床随访建议", "en": "Clinical Follow-up Recommendations"},
    "disclaimer": {
        "zh": "⚠️ 重要提示：本系统预测结果仅供临床参考，最终诊疗方案请结合患者实际情况，由主治医师综合判断后确定。",
        "en": "⚠️ Important: Predictions are for clinical reference only. Final treatment decisions should be made by the attending physician."
    },
    "months": {"zh": "月", "en": "M"},
    "probability": {"zh": "概率", "en": "Probability"},
    "time_months": {"zh": "时间（月）", "en": "Time (Months)"},
    "survival_prob": {"zh": "无复发生存率", "en": "Recurrence-Free Survival"},
    "cumulative_risk": {"zh": "累积复发风险", "en": "Cumulative Recurrence Risk"},
    "upload_file": {"zh": "上传患者数据文件", "en": "Upload Patient Data File"},
    "download_template": {"zh": "下载数据模板", "en": "Download Template"},
    "export_excel": {"zh": "导出Excel", "en": "Export Excel"},
    "export_pdf": {"zh": "导出PDF", "en": "Export PDF"},
    "export_csv": {"zh": "导出CSV", "en": "Export CSV"},
    "patient_id": {"zh": "患者编号", "en": "Patient ID"},
    "total_patients": {"zh": "总例数", "en": "Total"},
    "high_risk_count": {"zh": "高危", "en": "High Risk"},
    "medium_risk_count": {"zh": "中危", "en": "Intermediate"},
    "low_risk_count": {"zh": "低危", "en": "Low Risk"},
    "risk_distribution": {"zh": "风险分层分布", "en": "Risk Stratification Distribution"},
    "processing": {"zh": "正在进行风险评估，请稍候...", "en": "Performing risk assessment, please wait..."},
    "export_results": {"zh": "导出评估报告", "en": "Export Report"},
    "detailed_results": {"zh": "详细评估结果", "en": "Detailed Results"},
    "step1": {"zh": "步骤1：获取数据模板", "en": "Step 1: Get Data Template"},
    "step2": {"zh": "步骤2：上传患者数据", "en": "Step 2: Upload Patient Data"},
    "preview_template": {"zh": "查看模板格式", "en": "View Template Format"},
    "preview_data": {"zh": "预览上传数据", "en": "Preview Uploaded Data"},
    "loaded_patients": {"zh": "成功加载", "en": "Successfully loaded"},
    "patients_unit": {"zh": "例患者数据", "en": "patient records"},
    "high_risk_attention": {"zh": "高危患者（需重点关注）", "en": "High-Risk Patients (Require Attention)"},
    "high_risk_warning": {"zh": "例患者评估为高危，建议加强随访监测", "en": "patients classified as high-risk, enhanced follow-up recommended"},
    "month_12": {"zh": "12个月", "en": "12 Months"},
    "month_36": {"zh": "36个月", "en": "36 Months"},
    "month_60": {"zh": "60个月", "en": "60 Months"},
    "recurrence_risk": {"zh": "复发风险", "en": "Recurrence Risk"},
    "advice_low": {
        "zh": """
• 常规随访：每6个月门诊复查
• 影像学检查：每12个月盆腔超声
• 肿瘤标志物：每6个月检测CA125、HE4
• 生活方式：保持健康饮食和适度运动
        """,
        "en": """
• Routine follow-up: Outpatient review every 6 months
• Imaging: Pelvic ultrasound every 12 months  
• Tumor markers: CA125, HE4 every 6 months
• Lifestyle: Maintain healthy diet and moderate exercise
        """
    },
    "advice_medium": {
        "zh": """
• 加强随访：每3-4个月门诊复查
• 影像学检查：每6个月CT/MRI检查
• 肿瘤标志物：每3个月检测CA125、HE4
• 辅助治疗：评估是否需要辅助化疗
• 基因检测：建议进行BRCA等遗传咨询
        """,
        "en": """
• Enhanced follow-up: Outpatient review every 3-4 months
• Imaging: CT/MRI every 6 months
• Tumor markers: CA125, HE4 every 3 months
• Adjuvant therapy: Evaluate need for chemotherapy
• Genetic testing: BRCA and genetic counseling recommended
        """
    },
    "advice_high": {
        "zh": """
• 密切随访：每2-3个月门诊复查
• 影像学检查：每3个月CT/MRI检查
• 肿瘤标志物：每6-8周检测
• 辅助治疗：强烈建议辅助化疗±靶向治疗
• MDT讨论：建议多学科团队会诊
• 临床试验：可考虑符合条件的临床研究
        """,
        "en": """
• Close follow-up: Outpatient review every 2-3 months
• Imaging: CT/MRI every 3 months
• Tumor markers: Every 6-8 weeks
• Adjuvant therapy: Strongly recommend chemo ± targeted therapy
• MDT: Multidisciplinary team consultation recommended
• Clinical trials: Consider eligible clinical studies
        """
    }
}

# ================== 输入变量配置 ==================
INPUT_VARIABLES = {
    "age": {"zh": "年龄", "en": "Age", "type": "number", "min": 18, "max": 100, "default": 50, "unit": {"zh": "岁", "en": "yrs"}},
    "family_cancer_history": {"zh": "肿瘤家族史", "en": "Family Cancer History", "type": "select",
        "options": {"no": {"zh": "无", "en": "No"}, "yes": {"zh": "有", "en": "Yes"}}},
    "sexual_history": {"zh": "性生活史", "en": "Sexual History", "type": "select",
        "options": {"no": {"zh": "无", "en": "No"}, "yes": {"zh": "有", "en": "Yes"}}},
    "parity": {"zh": "孕产次", "en": "Parity", "type": "select",
        "options": {"0": {"zh": "未育", "en": "Nulliparous"}, "1": {"zh": "1次", "en": "1"}, "2": {"zh": "2次", "en": "2"}, "3": {"zh": "≥3次", "en": "≥3"}}},
    "menopausal_status": {"zh": "月经状态", "en": "Menopausal Status", "type": "select",
        "options": {"premenopausal": {"zh": "绝经前", "en": "Premenopausal"}, "postmenopausal": {"zh": "绝经后", "en": "Postmenopausal"}}},
    "comorbidities": {"zh": "合并症", "en": "Comorbidities", "type": "select",
        "options": {"no": {"zh": "无", "en": "None"}, "hypertension": {"zh": "高血压", "en": "Hypertension"},
                   "diabetes": {"zh": "糖尿病", "en": "Diabetes"}, "cardiovascular": {"zh": "心血管病", "en": "CVD"}, "multiple": {"zh": "多种", "en": "Multiple"}}},
    "smoking_drinking_history": {"zh": "烟酒史", "en": "Smoking/Drinking", "type": "select",
        "options": {"no": {"zh": "无", "en": "No"}, "smoking": {"zh": "吸烟", "en": "Smoking"}, "drinking": {"zh": "饮酒", "en": "Drinking"}, "both": {"zh": "均有", "en": "Both"}}},
    "receive_estrogens": {"zh": "激素暴露史", "en": "Hormone Exposure", "type": "select",
        "options": {"no": {"zh": "无", "en": "No"}, "hrt": {"zh": "HRT", "en": "HRT"}, "contraceptive": {"zh": "避孕药", "en": "OCP"}, "other": {"zh": "其他", "en": "Other"}}},
    "ovulation_induction": {"zh": "促排卵史", "en": "Ovulation Induction", "type": "select",
        "options": {"no": {"zh": "无", "en": "No"}, "yes": {"zh": "有", "en": "Yes"}}},
    "presenting_symptom": {"zh": "主诉症状", "en": "Chief Complaint", "type": "select",
        "options": {"asymptomatic": {"zh": "无症状", "en": "Asymptomatic"}, "abdominal_pain": {"zh": "腹痛", "en": "Abdominal Pain"},
                   "bloating": {"zh": "腹胀", "en": "Bloating"}, "mass": {"zh": "包块", "en": "Mass"}, "bleeding": {"zh": "异常出血", "en": "Bleeding"}, "other": {"zh": "其他", "en": "Other"}}},
    "surgical_route": {"zh": "手术途径", "en": "Surgical Approach", "type": "select",
        "options": {"laparoscopy": {"zh": "腹腔镜", "en": "Laparoscopic"}, "laparotomy": {"zh": "开腹", "en": "Open"}, "robotic": {"zh": "机器人", "en": "Robotic"}, "conversion": {"zh": "中转开腹", "en": "Conversion"}}},
    "tumor_envelope_integrity": {"zh": "包膜完整性", "en": "Capsule Integrity", "type": "select",
        "options": {"intact": {"zh": "完整", "en": "Intact"}, "ruptured_before": {"zh": "术前破裂", "en": "Pre-op Rupture"}, "ruptured_during": {"zh": "术中破裂", "en": "Intra-op Rupture"}}},
    "fertility_sparing_surgery": {"zh": "保留生育", "en": "Fertility Sparing", "type": "select",
        "options": {"no": {"zh": "否", "en": "No"}, "yes": {"zh": "是", "en": "Yes"}}},
    "completeness_of_surgery": {"zh": "手术完整性", "en": "Surgical Completeness", "type": "select",
        "options": {"incomplete": {"zh": "不完整分期", "en": "Incomplete"}, "complete": {"zh": "完整分期", "en": "Complete"}}},
    "omentectomy": {"zh": "大网膜切除", "en": "Omentectomy", "type": "select",
        "options": {"no": {"zh": "未切", "en": "No"}, "partial": {"zh": "部分", "en": "Partial"}, "total": {"zh": "全切", "en": "Total"}}},
    "lymphadenectomy": {"zh": "淋巴结清扫", "en": "Lymphadenectomy", "type": "select",
        "options": {"no": {"zh": "未清扫", "en": "No"}, "pelvic": {"zh": "盆腔", "en": "Pelvic"}, "paraaortic": {"zh": "腹主动脉旁", "en": "Para-aortic"}, "both": {"zh": "盆腔+腹主", "en": "Both"}}},
    "postoperative_adjuvant_therapy": {"zh": "辅助治疗", "en": "Adjuvant Therapy", "type": "select",
        "options": {"no": {"zh": "无", "en": "No"}, "chemotherapy": {"zh": "化疗", "en": "Chemo"}, "targeted": {"zh": "靶向", "en": "Targeted"}, "combined": {"zh": "联合", "en": "Combined"}}},
    "histological_subtype": {"zh": "组织学类型", "en": "Histology", "type": "select",
        "options": {"serous": {"zh": "浆液性", "en": "Serous"}, "mucinous": {"zh": "黏液性", "en": "Mucinous"}, "endometrioid": {"zh": "内膜样", "en": "Endometrioid"},
                   "clear_cell": {"zh": "透明细胞", "en": "Clear Cell"}, "mixed": {"zh": "混合型", "en": "Mixed"}, "other": {"zh": "其他", "en": "Other"}}},
    "micropapillary": {"zh": "微乳头结构", "en": "Micropapillary", "type": "select",
        "options": {"no": {"zh": "无", "en": "No"}, "yes": {"zh": "有", "en": "Yes"}}},
    "microinfiltration": {"zh": "微浸润", "en": "Microinvasion", "type": "select",
        "options": {"no": {"zh": "无", "en": "No"}, "yes": {"zh": "有", "en": "Yes"}}},
    "psammoma_bodies_calcification": {"zh": "砂粒体", "en": "Psammoma Bodies", "type": "select",
        "options": {"no": {"zh": "无", "en": "No"}, "yes": {"zh": "有", "en": "Yes"}}},
    "peritoneal_implantation": {"zh": "腹膜种植", "en": "Peritoneal Implants", "type": "select",
        "options": {"no": {"zh": "无", "en": "No"}, "noninvasive": {"zh": "非浸润", "en": "Non-invasive"}, "invasive": {"zh": "浸润", "en": "Invasive"}}},
    "ascites_cytology": {"zh": "腹水细胞学", "en": "Ascites Cytology", "type": "select",
        "options": {"no_ascites": {"zh": "无腹水", "en": "No Ascites"}, "negative": {"zh": "阴性", "en": "Negative"}, "positive": {"zh": "阳性", "en": "Positive"}}},
    "figo_staging": {"zh": "FIGO分期", "en": "FIGO Stage", "type": "select",
        "options": {"IA": {"zh": "IA", "en": "IA"}, "IB": {"zh": "IB", "en": "IB"}, "IC1": {"zh": "IC1", "en": "IC1"}, "IC2": {"zh": "IC2", "en": "IC2"}, "IC3": {"zh": "IC3", "en": "IC3"},
                   "II": {"zh": "II", "en": "II"}, "IIIA": {"zh": "IIIA", "en": "IIIA"}, "IIIB": {"zh": "IIIB", "en": "IIIB"}, "IIIC": {"zh": "IIIC", "en": "IIIC"}}},
    "unilateral_or_bilateral": {"zh": "侧别", "en": "Laterality", "type": "select",
        "options": {"left": {"zh": "左侧", "en": "Left"}, "right": {"zh": "右侧", "en": "Right"}, "bilateral": {"zh": "双侧", "en": "Bilateral"}}},
    "tumor_size": {"zh": "肿瘤大小", "en": "Tumor Size", "type": "select",
        "options": {"<=5": {"zh": "≤5cm", "en": "≤5cm"}, "5-10": {"zh": "5-10cm", "en": "5-10cm"}, "10-15": {"zh": "10-15cm", "en": "10-15cm"}, ">15": {"zh": ">15cm", "en": ">15cm"}}},
    "type_of_lesion": {"zh": "病灶性质", "en": "Lesion Type", "type": "select",
        "options": {"cystic": {"zh": "囊性", "en": "Cystic"}, "solid": {"zh": "实性", "en": "Solid"}, "mixed": {"zh": "囊实性", "en": "Mixed"}}},
    "papillary_area_ratio": {"zh": "乳头占比", "en": "Papillary Ratio", "type": "select",
        "options": {"<10%": {"zh": "<10%", "en": "<10%"}, "10-30%": {"zh": "10-30%", "en": "10-30%"}, "30-50%": {"zh": "30-50%", "en": "30-50%"}, ">50%": {"zh": ">50%", "en": ">50%"}}},
    "ca125": {"zh": "CA125", "en": "CA125", "type": "select",
        "options": {"normal": {"zh": "正常(<35)", "en": "Normal"}, "mild": {"zh": "轻度↑", "en": "Mild↑"}, "moderate": {"zh": "中度↑", "en": "Mod↑"}, "high": {"zh": "显著↑", "en": "High↑"}}},
    "cea": {"zh": "CEA", "en": "CEA", "type": "select",
        "options": {"normal": {"zh": "正常", "en": "Normal"}, "elevated": {"zh": "升高", "en": "Elevated"}}},
    "ca199": {"zh": "CA199", "en": "CA199", "type": "select",
        "options": {"normal": {"zh": "正常", "en": "Normal"}, "elevated": {"zh": "升高", "en": "Elevated"}}},
    "afp": {"zh": "AFP", "en": "AFP", "type": "select",
        "options": {"normal": {"zh": "正常", "en": "Normal"}, "elevated": {"zh": "升高", "en": "Elevated"}}},
    "ca724": {"zh": "CA724", "en": "CA724", "type": "select",
        "options": {"normal": {"zh": "正常", "en": "Normal"}, "elevated": {"zh": "升高", "en": "Elevated"}}},
    "he4": {"zh": "HE4", "en": "HE4", "type": "select",
        "options": {"normal": {"zh": "正常(<70)", "en": "Normal"}, "mild": {"zh": "轻度↑", "en": "Mild↑"}, "elevated": {"zh": "显著↑", "en": "High↑"}}}
}


# ================== 模型相关类和函数 ==================
class DataPreprocessor:
    def __init__(self, select_k=None):
        self.scaler = StandardScaler()
        self.selector = None
        self.select_k = select_k
    def fit(self, X, y=None):
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        if self.select_k and y is not None:
            self.selector = SelectKBest(f_classif, k=min(self.select_k, X.shape[1]))
            self.selector.fit(X_scaled, y)
        return self
    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        if self.selector:
            X_scaled = self.selector.transform(X_scaled)
        return X_scaled

class SEBlock(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(dim, max(dim//reduction,1)), nn.ReLU(), nn.Linear(max(dim//reduction,1), dim), nn.Sigmoid())
    def forward(self, x): return x * self.fc(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(dim,dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim,dim), nn.BatchNorm1d(dim))
        self.se = SEBlock(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
    def forward(self, x): return self.act(x + self.drop(self.se(self.block(x))))

class EnhancedDeepSurv(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256,128,64], drop_rate=0.3, n_res=2):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dims[0]), nn.BatchNorm1d(hidden_dims[0]), nn.GELU(), nn.Dropout(drop_rate))
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dims[0], drop_rate) for _ in range(n_res)])
        self.down = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.BatchNorm1d(hidden_dims[i+1]), nn.GELU(), nn.Dropout(drop_rate)) for i in range(len(hidden_dims)-1)])
        self.out = nn.Linear(hidden_dims[-1], 1)
    def forward(self, x):
        x = self.input_proj(x)
        for r in self.res_blocks: x = r(x)
        for d in self.down: x = d(x)
        return self.out(x).squeeze(-1)

class EnhancedDeepHit(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256,128], num_dur=10, drop=0.3):
        super().__init__()
        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop)])
            in_d = h
        layers.append(nn.Linear(in_d, num_dur))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return torch.softmax(self.net(x), dim=1)

class EnhancedDenoisingAE(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256,128], latent_dim=64, drop=0.2):
        super().__init__()
        enc, in_d = [], input_dim
        for h in hidden_dims:
            enc.extend([nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop)])
            in_d = h
        enc.append(nn.Linear(in_d, latent_dim))
        self.encoder = nn.Sequential(*enc)
        dec, in_d = [], latent_dim
        for h in reversed(hidden_dims):
            dec.extend([nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop)])
            in_d = h
        dec.append(nn.Linear(in_d, input_dim))
        self.decoder = nn.Sequential(*dec)
    def encode(self, x): return self.encoder(x)
    def forward(self, x): z = self.encoder(x); return self.decoder(z), z

class EnhancedTransformer(nn.Module):
    def __init__(self, latent_dim, n_heads=4, ff_dim=256, n_layers=2, drop=0.1):
        super().__init__()
        while latent_dim % n_heads != 0 and n_heads > 1: n_heads -= 1
        self.norm = nn.LayerNorm(latent_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=drop, activation='gelu', batch_first=True)
        self.trans = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.proj = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU(), nn.Dropout(drop))
    def forward(self, z):
        if z.dim() == 2: z = z.unsqueeze(1)
        return self.proj(self.trans(self.norm(z)).squeeze(1))

class LearnableFusion(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.2),
                                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x).squeeze(-1)


def get_text(key, lang): return TRANSLATIONS.get(key, {}).get(lang, key)
def encode_option(var_name, option_key):
    options = INPUT_VARIABLES.get(var_name, {}).get("options", {})
    keys = list(options.keys())
    try: return float(keys.index(option_key))
    except: return 0.0

@st.cache_resource
def load_models(model_dir="results_clinical_enhanced_v3"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, use_pretrained = {}, False
    try:
        required = ['model_ae.pt', 'model_trans.pt', 'model_deepsurv.pt', 'model_deephit.pt', 'model_fusion.pt', 'preprocessor.joblib', 'time_cuts.npy', 'ds_min_max.npy', 'best_parameters.json']
        if all(os.path.exists(os.path.join(model_dir, f)) for f in required):
            with open(os.path.join(model_dir, "best_parameters.json")) as f: params = json.load(f)
            preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))
            time_cuts = np.load(os.path.join(model_dir, "time_cuts.npy"))
            ds_min_max = np.load(os.path.join(model_dir, "ds_min_max.npy"))
            fcm_path = os.path.join(model_dir, "fcm_centers.npy")
            fcm_centers = np.load(fcm_path) if os.path.exists(fcm_path) else np.array([[0.3,0.3],[0.7,0.7]])
            input_dim = preprocessor.scaler.n_features_in_
            if hasattr(preprocessor, 'selector') and preprocessor.selector: input_dim = getattr(preprocessor.selector, 'k', input_dim)
            latent_dim, fused_dim = params.get('ae_latent', 64), params.get('ae_latent', 64) * 2
            ae = EnhancedDenoisingAE(input_dim, [params.get('ae_h1',256), params.get('ae_h2',128)], latent_dim)
            ae.load_state_dict(torch.load(os.path.join(model_dir, "model_ae.pt"), map_location=device)); ae.eval()
            trans = EnhancedTransformer(latent_dim)
            trans.load_state_dict(torch.load(os.path.join(model_dir, "model_trans.pt"), map_location=device)); trans.eval()
            ds = EnhancedDeepSurv(fused_dim, [params.get('ds_h1',256), params.get('ds_h2',128), params.get('ds_h3',64)], params.get('ds_drop',0.3))
            ds.load_state_dict(torch.load(os.path.join(model_dir, "model_deepsurv.pt"), map_location=device)); ds.eval()
            dh = EnhancedDeepHit(fused_dim, [params.get('dh_h1',256), params.get('dh_h2',128)], len(time_cuts)-1)
            dh.load_state_dict(torch.load(os.path.join(model_dir, "model_deephit.pt"), map_location=device)); dh.eval()
            fusion = LearnableFusion()
            fusion.load_state_dict(torch.load(os.path.join(model_dir, "model_fusion.pt"), map_location=device)); fusion.eval()
            models = {'ae': ae.to(device), 'trans': trans.to(device), 'ds': ds.to(device), 'dh': dh.to(device), 'fusion': fusion.to(device),
                     'preprocessor': preprocessor, 'time_cuts': time_cuts, 'ds_min_max': ds_min_max, 'device': device}
            use_pretrained = True
    except: pass
    if not use_pretrained:
        input_dim, latent_dim, fused_dim, num_bins = len(INPUT_VARIABLES), 64, 128, 10
        models = {'ae': EnhancedDenoisingAE(input_dim, [256,128], latent_dim).to(device), 'trans': EnhancedTransformer(latent_dim).to(device),
                 'ds': EnhancedDeepSurv(fused_dim, [256,128,64]).to(device), 'dh': EnhancedDeepHit(fused_dim, [256,128], num_bins).to(device),
                 'fusion': LearnableFusion().to(device), 'preprocessor': None, 'time_cuts': np.linspace(0,120,11), 'ds_min_max': np.array([-5.,5.]), 'device': device}
        for k in ['ae','trans','ds','dh','fusion']: models[k].eval()
    models['use_pretrained'] = use_pretrained
    return models

def preprocess_input(input_data, models):
    features = []
    for var in INPUT_VARIABLES:
        val = input_data.get(var)
        info = INPUT_VARIABLES[var]
        features.append(encode_option(var, val) if info['type'] == 'select' and val else (float(val) if val and info['type'] == 'number' else float(info.get('default', 0))))
    X = np.array(features).reshape(1, -1)
    if models.get('preprocessor'):
        try: X = models['preprocessor'].transform(X)
        except: X = (X - X.mean()) / (X.std() + 1e-8)
    else: X = (X - X.mean()) / (X.std() + 1e-8)
    return X

def predict_single(input_data, models):
    device = models['device']
    X = torch.tensor(preprocess_input(input_data, models), dtype=torch.float32, device=device)
    with torch.no_grad():
        Z = models['ae'].encode(X)
        T = models['trans'](Z)
        Xf = torch.cat([Z, T], dim=1)
        risk_ds = models['ds'](Xf).cpu().numpy(); risk_ds = risk_ds.item() if risk_ds.ndim == 0 else risk_ds[0]
        pmf = models['dh'](Xf).cpu().numpy()[0]
        min_ds, max_ds = models['ds_min_max']
        prob_ds = np.clip((risk_ds - min_ds) / (max_ds - min_ds + 1e-8), 0, 1)
        cif, survival = np.cumsum(pmf), 1 - np.cumsum(pmf)
        risk_dh = cif[len(pmf)//2]
        final_risk = models['fusion'](torch.tensor([[prob_ds, risk_dh]], dtype=torch.float32, device=device)).cpu().numpy()
        final_risk = final_risk.item() if final_risk.ndim == 0 else final_risk[0]
    time_cuts = models['time_cuts']
    time_points = (time_cuts[:-1] + time_cuts[1:]) / 2
    n = len(cif)
    return {'final_risk': float(final_risk), 'survival': survival, 'cif': cif, 'time_points': time_points,
            'risk_12m': float(cif[min(int(n*0.1), n-1)]), 'risk_36m': float(cif[min(int(n*0.3), n-1)]), 'risk_60m': float(cif[min(int(n*0.5), n-1)])}

def predict_batch(df, models, lang):
    results = []
    progress = st.progress(0)
    for idx, row in df.iterrows():
        input_data = {}
        for var in INPUT_VARIABLES:
            for lg in ['zh', 'en']:
                col = INPUT_VARIABLES[var][lg]
                if col in row: input_data[var] = row[col]; break
            if var not in input_data and var in row: input_data[var] = row[var]
        try:
            pred = predict_single(input_data, models)
            level = get_text("low_risk" if pred['final_risk']<0.3 else ("medium_risk" if pred['final_risk']<0.6 else "high_risk"), lang)
            m = get_text("months", lang)
            results.append({get_text("patient_id",lang): row.get('patient_id', row.get('患者编号', idx+1)), get_text("overall_risk",lang): f"{pred['final_risk']*100:.1f}%",
                           f"12{m}": f"{pred['risk_12m']*100:.1f}%", f"36{m}": f"{pred['risk_36m']*100:.1f}%", f"60{m}": f"{pred['risk_60m']*100:.1f}%",
                           get_text("risk_level",lang): level, '_risk': pred['final_risk']})
        except: pass
        progress.progress((idx+1)/len(df))
    progress.empty()
    return pd.DataFrame(results)

def create_template_csv(lang):
    cols = [get_text("patient_id", lang)] + [INPUT_VARIABLES[v][lang] for v in INPUT_VARIABLES]
    data = {cols[0]: [1,2,3]}
    for i, (v, info) in enumerate(INPUT_VARIABLES.items()):
        data[cols[i+1]] = [list(info['options'].keys())[0]]*3 if info['type']=='select' else [info.get('default',0)]*3
    return pd.DataFrame(data)


# ================== 可视化函数（大字体版本）==================
def create_gauge_chart(risk, lang):
    if risk < 0.3: color, level = "#27ae60", get_text("low_risk", lang)
    elif risk < 0.6: color, level = "#f39c12", get_text("medium_risk", lang)
    else: color, level = "#e74c3c", get_text("high_risk", lang)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=risk*100, domain={'x':[0,1],'y':[0,1]},
        number={'suffix':'%', 'font':{'size':60, 'color':color, 'family':'Arial Black'}},
        title={'text': f"<b>{get_text('overall_risk', lang)}</b><br><span style='font-size:24px;color:{color}'>{level}</span>", 'font':{'size':20}},
        gauge={'axis':{'range':[0,100], 'tickwidth':3, 'tickcolor':'#333', 'tickfont':{'size':16, 'color':'#333'}},
               'bar':{'color':color, 'thickness':0.8},
               'bgcolor':'#f0f0f0', 'borderwidth':3, 'bordercolor':'#333',
               'steps':[{'range':[0,30],'color':'rgba(39,174,96,0.3)'}, {'range':[30,60],'color':'rgba(243,156,18,0.3)'}, {'range':[60,100],'color':'rgba(231,76,60,0.3)'}],
               'threshold':{'line':{'color':'#333','width':5}, 'thickness':0.8, 'value':risk*100}}))
    fig.update_layout(height=320, margin=dict(l=30,r=30,t=80,b=30), paper_bgcolor='rgba(0,0,0,0)', font={'family':'Arial'})
    return fig

def create_survival_curve(survival, time_points, lang):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"<b>{get_text('survival_prob', lang)}</b>", f"<b>{get_text('cumulative_risk', lang)}</b>"),
                       horizontal_spacing=0.12)
    
    # 生存曲线
    fig.add_trace(go.Scatter(x=time_points, y=survival, mode='lines+markers', name=get_text('survival_prob', lang),
                            line=dict(color='#3498db', width=4), fill='tozeroy', fillcolor='rgba(52,152,219,0.15)',
                            marker=dict(size=10, color='#3498db', line=dict(width=2, color='white'))), row=1, col=1)
    
    # 累积风险
    fig.add_trace(go.Scatter(x=time_points, y=1-survival, mode='lines+markers', name=get_text('cumulative_risk', lang),
                            line=dict(color='#e74c3c', width=4), fill='tozeroy', fillcolor='rgba(231,76,60,0.15)',
                            marker=dict(size=10, color='#e74c3c', line=dict(width=2, color='white'))), row=1, col=2)
    
    time_label = get_text('time_months', lang)
    prob_label = get_text('probability', lang)
    
    for col in [1, 2]:
        fig.update_xaxes(title_text=f"<b>{time_label}</b>", title_font=dict(size=18, color='#2c3e50'),
                        tickfont=dict(size=14, color='#2c3e50'), gridcolor='#e0e0e0', gridwidth=1,
                        linecolor='#2c3e50', linewidth=2, row=1, col=col)
        fig.update_yaxes(title_text=f"<b>{prob_label}</b>", title_font=dict(size=18, color='#2c3e50'),
                        tickfont=dict(size=14, color='#2c3e50'), range=[0,1], gridcolor='#e0e0e0', gridwidth=1,
                        linecolor='#2c3e50', linewidth=2, tickformat='.0%', row=1, col=col)
    
    fig.update_layout(height=420, showlegend=False, margin=dict(l=80,r=40,t=80,b=80),
                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white', font={'family':'Arial'})
    fig.update_annotations(font_size=18)
    return fig

def create_time_risk_bar(r12, r36, r60, lang):
    labels = [get_text('month_12', lang), get_text('month_36', lang), get_text('month_60', lang)]
    values = [r12*100, r36*100, r60*100]
    colors = ['#27ae60' if v<30 else ('#f39c12' if v<60 else '#e74c3c') for v in values]
    
    fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=colors, text=[f'{v:.1f}%' for v in values],
                                textposition='outside', textfont=dict(size=18, color='#2c3e50', family='Arial Black'),
                                width=0.6)])
    fig.update_layout(title=dict(text=f"<b>{get_text('time_risk', lang)}</b>", font=dict(size=20, color='#2c3e50'), x=0.5),
                     yaxis=dict(title=f"<b>{get_text('recurrence_risk', lang)} (%)</b>", title_font=dict(size=16), tickfont=dict(size=14), range=[0, max(values)*1.3], gridcolor='#e0e0e0'),
                     xaxis=dict(tickfont=dict(size=16, color='#2c3e50')),
                     height=380, margin=dict(l=80,r=40,t=80,b=60), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white')
    return fig

def create_risk_distribution(results_df, lang):
    risk_col = get_text("risk_level", lang)
    if risk_col in results_df.columns:
        high = len(results_df[results_df[risk_col].str.contains('High|高', case=False, na=False)])
        med = len(results_df[results_df[risk_col].str.contains('Intermediate|中', case=False, na=False)])
        low = len(results_df[results_df[risk_col].str.contains('Low|低', case=False, na=False)])
    else: high, med, low = 0, 0, 0
    
    fig = go.Figure(data=[go.Pie(labels=[get_text('low_risk',lang), get_text('medium_risk',lang), get_text('high_risk',lang)],
                                values=[low, med, high], marker_colors=['#27ae60','#f39c12','#e74c3c'],
                                hole=0.5, textinfo='label+percent+value', textfont=dict(size=14),
                                pull=[0, 0, 0.08])])
    fig.update_layout(title=dict(text=f"<b>{get_text('risk_distribution', lang)}</b>", font=dict(size=20), x=0.5),
                     height=400, margin=dict(l=20,r=20,t=80,b=20), paper_bgcolor='rgba(0,0,0,0)',
                     legend=dict(font=dict(size=14), orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5))
    return fig


# ================== PDF生成 ==================
def generate_pdf(results_df, lang):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph("Cancer Recurrence Risk Assessment Report", ParagraphStyle('T', parent=styles['Heading1'], fontSize=18, spaceAfter=20, alignment=1)),
             Paragraph("Shengjing Hospital of China Medical University", styles['Normal']),
             Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']), Spacer(1, 20)]
    
    total = len(results_df)
    risk_col = get_text("risk_level", lang)
    high = len(results_df[results_df[risk_col].str.contains('High|高', case=False, na=False)]) if risk_col in results_df.columns else 0
    med = len(results_df[results_df[risk_col].str.contains('Intermediate|中', case=False, na=False)]) if risk_col in results_df.columns else 0
    low = total - high - med
    
    data = [["Metric", "Value"], ["Total", str(total)], ["High Risk", f"{high} ({high/total*100:.1f}%)" if total else "0"],
            ["Intermediate", f"{med} ({med/total*100:.1f}%)" if total else "0"], ["Low Risk", f"{low} ({low/total*100:.1f}%)" if total else "0"]]
    tbl = Table(data, colWidths=[200, 200])
    tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#3498db')), ('TEXTCOLOR',(0,0),(-1,0),colors.white),
                            ('ALIGN',(0,0),(-1,-1),'CENTER'), ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'), ('FONTSIZE',(0,0),(-1,0),12),
                            ('BACKGROUND',(0,1),(-1,-1),colors.HexColor('#f8f9fa')), ('GRID',(0,0),(-1,-1),1,colors.black)]))
    story.extend([tbl, Spacer(1, 30), Paragraph("This report is for clinical reference only.", ParagraphStyle('D', fontSize=8, textColor=colors.grey))])
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

def generate_single_pdf(results, lang):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    risk = results['final_risk']
    level = "Low Risk" if risk < 0.3 else ("Intermediate Risk" if risk < 0.6 else "High Risk")
    
    story = [Paragraph("Patient Risk Assessment Report", ParagraphStyle('T', parent=styles['Heading1'], fontSize=18, spaceAfter=20, alignment=1)),
             Paragraph("Shengjing Hospital of China Medical University", styles['Normal']),
             Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']), Spacer(1, 20)]
    
    data = [["Assessment Item", "Result"], ["Overall Risk", f"{risk*100:.1f}%"], ["Risk Level", level],
            ["12-Month Risk", f"{results['risk_12m']*100:.1f}%"], ["36-Month Risk", f"{results['risk_36m']*100:.1f}%"], ["60-Month Risk", f"{results['risk_60m']*100:.1f}%"]]
    tbl = Table(data, colWidths=[200, 200])
    tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#3498db')), ('TEXTCOLOR',(0,0),(-1,0),colors.white),
                            ('ALIGN',(0,0),(-1,-1),'CENTER'), ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                            ('GRID',(0,0),(-1,-1),1,colors.black), ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, colors.HexColor('#f8f9fa')])]))
    story.extend([tbl, Spacer(1, 30), Paragraph("This report is for clinical reference only.", ParagraphStyle('D', fontSize=8, textColor=colors.grey))])
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# ================== 输入控件 ==================
def render_select(var, info, lang, prefix=""):
    return st.selectbox(info[lang], list(info['options'].keys()), format_func=lambda x: info['options'][x][lang], key=f"{prefix}{var}")

def render_number(var, info, lang, prefix=""):
    label = f"{info[lang]} ({info['unit'][lang]})" if 'unit' in info else info[lang]
    return st.number_input(label, min_value=float(info.get('min',0)), max_value=float(info.get('max',100)), value=float(info.get('default',0)), key=f"{prefix}{var}")


# ================== 主应用 ==================
def main():
    models = load_models()
    
    # 语言选择
    c1, c2, c3 = st.columns([1,1,1])
    with c2:
        lang = LANGUAGES[st.selectbox("🌐", list(LANGUAGES.keys()), label_visibility="collapsed")]
    
    # 头部
    st.markdown(f"""
    <div class="hospital-header">
        <h1>🏥 {get_text('title', lang)}</h1>
        <p class="subtitle">{get_text('subtitle', lang)}</p>
        <p class="hospital-name">{get_text('hospital', lang)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 状态栏
    status = "✓ " + (get_text('system_ready', lang))
    model_stat = "AI模型已加载" if lang == 'zh' else "AI Model Loaded"
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.markdown(f"""
    <div class="status-bar">
        <div class="status-item"><div class="status-dot"></div> {status}</div>
        <div class="status-item">🤖 {model_stat}</div>
        <div class="status-item">🕐 {time_now}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 标签页
    tab1, tab2 = st.tabs([f"📋 {get_text('single_patient', lang)}", f"📊 {get_text('batch_prediction', lang)}"])
    
    # ================== 单例预测 ==================
    with tab1:
        col1, col2, col3 = st.columns(3)
        input_data = {}
        
        with col1:
            st.markdown(f'<div class="module-card"><div class="module-title">📝 {get_text("basic_info", lang)}</div>', unsafe_allow_html=True)
            for v in ['age','family_cancer_history','sexual_history','parity','menopausal_status','comorbidities','smoking_drinking_history','receive_estrogens','ovulation_induction']:
                info = INPUT_VARIABLES[v]
                input_data[v] = render_number(v, info, lang, "s_") if info['type']=='number' else render_select(v, info, lang, "s_")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="module-card"><div class="module-title surgery">🔪 {get_text("surgical_info", lang)}</div>', unsafe_allow_html=True)
            for v in ['presenting_symptom','surgical_route','tumor_envelope_integrity','fertility_sparing_surgery','completeness_of_surgery','omentectomy','lymphadenectomy','postoperative_adjuvant_therapy']:
                info = INPUT_VARIABLES[v]
                input_data[v] = render_select(v, info, lang, "s_")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'<div class="module-card"><div class="module-title pathology">🔬 {get_text("pathology_info", lang)}</div>', unsafe_allow_html=True)
            for v in ['histological_subtype','micropapillary','microinfiltration','psammoma_bodies_calcification','peritoneal_implantation','ascites_cytology','figo_staging','unilateral_or_bilateral','tumor_size','type_of_lesion','papillary_area_ratio']:
                info = INPUT_VARIABLES[v]
                input_data[v] = render_select(v, info, lang, "s_")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 肿瘤标志物
        st.markdown(f'<div class="module-card"><div class="module-title markers">🧪 {get_text("tumor_markers", lang)}</div>', unsafe_allow_html=True)
        mcols = st.columns(6)
        for i, v in enumerate(['ca125','cea','ca199','afp','ca724','he4']):
            with mcols[i]:
                input_data[v] = render_select(v, INPUT_VARIABLES[v], lang, "s_")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 预测按钮
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1,1,1])
        with c2:
            if st.button(f"🔮 {get_text('predict_button', lang)}", use_container_width=True, key="predict"):
                with st.spinner(get_text('processing', lang)):
                    results = predict_single(input_data, models)
                    
                    st.markdown("---")
                    st.markdown(f'<div class="module-card"><div class="module-title results">📊 {get_text("prediction_results", lang)}</div>', unsafe_allow_html=True)
                    
                    rc1, rc2 = st.columns([1, 2])
                    with rc1:
                        st.plotly_chart(create_gauge_chart(results['final_risk'], lang), use_container_width=True)
                        st.plotly_chart(create_time_risk_bar(results['risk_12m'], results['risk_36m'], results['risk_60m'], lang), use_container_width=True)
                    with rc2:
                        st.plotly_chart(create_survival_curve(results['survival'], results['time_points'], lang), use_container_width=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 临床建议
                    risk = results['final_risk']
                    if risk < 0.3: level, advice, css = "low_risk", "advice_low", "low"
                    elif risk < 0.6: level, advice, css = "medium_risk", "advice_medium", "medium"
                    else: level, advice, css = "high_risk", "advice_high", "high"
                    
                    st.markdown(f"""
                    <div class="advice-card {css}">
                        <h4>💊 {get_text('clinical_advice', lang)} - {get_text(level, lang)} ({risk*100:.1f}%)</h4>
                        <pre style="white-space: pre-wrap; font-family: inherit; margin: 0;">{get_text(advice, lang)}</pre>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 导出
                    st.markdown(f"#### 📥 {get_text('export_results', lang)}")
                    ec1, ec2 = st.columns(2)
                    with ec1:
                        df = pd.DataFrame({get_text('metric_label' if 'metric_label' in TRANSLATIONS else 'risk_level', lang): [get_text('overall_risk',lang), get_text('month_12',lang), get_text('month_36',lang), get_text('month_60',lang)],
                                          'Value': [f"{results['final_risk']*100:.1f}%", f"{results['risk_12m']*100:.1f}%", f"{results['risk_36m']*100:.1f}%", f"{results['risk_60m']*100:.1f}%"]})
                        buf = io.BytesIO()
                        with pd.ExcelWriter(buf, engine='openpyxl') as w: df.to_excel(w, index=False)
                        st.download_button(get_text('export_excel', lang), buf.getvalue(), f"result_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx")
                    with ec2:
                        st.download_button(get_text('export_pdf', lang), generate_single_pdf(results, lang), f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", "application/pdf")
    
    # ================== 批量预测 ==================
    with tab2:
        st.markdown(f"#### {get_text('step1', lang)}")
        template = create_template_csv(lang)
        buf = io.StringIO()
        template.to_csv(buf, index=False, encoding='utf-8-sig')
        st.download_button(get_text('download_template', lang), buf.getvalue(), f"template_{lang}.csv", "text/csv")
        
        with st.expander(get_text('preview_template', lang)):
            st.dataframe(template, use_container_width=True)
        
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
                        results_df = predict_batch(df, models, lang)
                        
                        st.markdown("---")
                        st.markdown(f"### 📊 {get_text('detailed_results', lang)}")
                        
                        total = len(results_df)
                        risk_col = get_text("risk_level", lang)
                        high = len(results_df[results_df[risk_col].str.contains('High|高', case=False, na=False)]) if risk_col in results_df.columns else 0
                        med = len(results_df[results_df[risk_col].str.contains('Intermediate|中', case=False, na=False)]) if risk_col in results_df.columns else 0
                        low = total - high - med
                        
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric(get_text('total_patients', lang), total)
                        m2.metric(get_text('high_risk_count', lang), high)
                        m3.metric(get_text('medium_risk_count', lang), med)
                        m4.metric(get_text('low_risk_count', lang), low)
                        
                        cc1, cc2 = st.columns(2)
                        with cc1: st.plotly_chart(create_risk_distribution(results_df, lang), use_container_width=True)
                        with cc2:
                            if '_risk' in results_df.columns:
                                fig = go.Figure(go.Histogram(x=results_df['_risk']*100, nbinsx=20, marker_color='#3498db'))
                                fig.add_vline(x=30, line_dash="dash", line_color="#27ae60")
                                fig.add_vline(x=60, line_dash="dash", line_color="#e74c3c")
                                fig.update_layout(title=dict(text=f"<b>{get_text('risk_distribution', lang)}</b>", font=dict(size=18)),
                                                 xaxis=dict(title=f"<b>{get_text('recurrence_risk', lang)} (%)</b>", title_font=dict(size=16), tickfont=dict(size=14)),
                                                 yaxis=dict(title=f"<b>{get_text('total_patients', lang)}</b>", title_font=dict(size=16), tickfont=dict(size=14)),
                                                 height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white')
                                st.plotly_chart(fig, use_container_width=True)
                        
                        display_df = results_df.drop(columns=[c for c in results_df.columns if c.startswith('_')], errors='ignore')
                        def highlight(row):
                            v = str(row.get(risk_col, ''))
                            if 'High' in v or '高' in v: return ['background-color:#f8d7da']*len(row)
                            if 'Intermediate' in v or '中' in v: return ['background-color:#fff3cd']*len(row)
                            return ['background-color:#d4edda']*len(row)
                        st.dataframe(display_df.style.apply(highlight, axis=1), use_container_width=True, height=400)
                        
                        # 导出
                        st.markdown(f"#### 📥 {get_text('export_results', lang)}")
                        e1, e2, e3 = st.columns(3)
                        with e1:
                            buf = io.StringIO()
                            display_df.to_csv(buf, index=False, encoding='utf-8-sig')
                            st.download_button(get_text('export_csv', lang), buf.getvalue(), f"batch_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
                        with e2:
                            buf = io.BytesIO()
                            with pd.ExcelWriter(buf, engine='openpyxl') as w: display_df.to_excel(w, index=False)
                            st.download_button(get_text('export_excel', lang), buf.getvalue(), f"batch_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx")
                        with e3:
                            st.download_button(get_text('export_pdf', lang), generate_pdf(results_df, lang), f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", "application/pdf")
                        
                        if high > 0:
                            st.markdown("---")
                            st.markdown(f"### ⚠️ {get_text('high_risk_attention', lang)}")
                            high_df = display_df[display_df[risk_col].str.contains('High|高', case=False, na=False)]
                            st.dataframe(high_df.style.apply(lambda x: ['background-color:#f8d7da']*len(x), axis=1), use_container_width=True)
                            st.warning(f"⚠️ {high} {get_text('high_risk_warning', lang)}")
            except Exception as e:
                st.error(f"Error: {e}")
    
    # 页脚
    st.markdown("---")
    st.info(get_text('disclaimer', lang))
    st.markdown(f"""
    <div class="footer">
        <p class="hospital-name">{get_text('hospital', lang)}</p>
        <p class="version">Cancer Recurrence Risk Prediction System v3.0</p>
        <p class="copyright">© 2024 All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
