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
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import os
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

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
    return None

LOGO_BASE64 = load_logo()
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
    
    .status-box {
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        font-weight: 500;
    }
    .status-success {
        background: linear-gradient(90deg, #d4edda, #c3e6cb);
        border-left: 5px solid #28a745;
        color: #155724;
    }
    .status-error {
        background: linear-gradient(90deg, #f8d7da, #f5c6cb);
        border-left: 5px solid #dc3545;
        color: #721c24;
    }
    
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
    
    .chart-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e8e8e8;
        height: 100%;
    }
    
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
    
    .stSelectbox label { font-weight: 500; color: #2c3e50; font-size: 0.8rem; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 0; background: #f8f9fa; border-radius: 8px; padding: 3px; }
    .stTabs [data-baseweb="tab"] { background: transparent; border-radius: 5px; padding: 8px 16px; font-weight: 600; font-size: 0.9rem; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #3498db, #2980b9); color: white !important; }
    
    [data-testid="metric-container"] {
        background: white;
        padding: 0.6rem;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        border: 1px solid #e8e8e8;
    }
    
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
    "debug_info": {"zh": "调试信息", "en": "Debug Info"},
    "input_data": {"zh": "输入数据", "en": "Input Data"},
    "processed_features": {"zh": "预处理后特征", "en": "Processed Features"},
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

# ================== 变量顺序定义（与神经网络输入一致）==================
VARIABLE_ORDER = [
    "age",                          # 0
    "family_cancer_history",        # 1
    "sexual_history",               # 2
    "parity",                       # 3
    "menopausal_status",            # 4
    "comorbidities",                # 5
    "presenting_symptom",           # 6
    "surgical_route",               # 7
    "tumor_envelope_integrity",     # 8
    "fertility_sparing_surgery",    # 9
    "completeness_of_surgery",      # 10
    "omentectomy",                  # 11
    "lymphadenectomy",              # 12
    "histological_subtype",         # 13
    "micropapillary",               # 14
    "microinfiltration",            # 15
    "psammoma_bodies_calcification",# 16
    "peritoneal_implantation",      # 17
    "ascites_cytology",             # 18
    "figo_staging",                 # 19
    "unilateral_or_bilateral",      # 20
    "tumor_size",                   # 21
    "ca125",                        # 22
    "cea",                          # 23
    "ca199",                        # 24
    "afp",                          # 25
    "ca724",                        # 26
    "he4",                          # 27
    "smoking_drinking_history",     # 28
    "receive_estrogens",            # 29
    "ovulation_induction",          # 30
    "postoperative_adjuvant_therapy",# 31
    "type_of_lesion",               # 32
    "papillary_area_ratio",         # 33
]

# ================== 输入变量定义（高风险默认值）==================
INPUT_VARIABLES = {
    "age": {
        "zh": "年龄", 
        "en": "Age", 
        "type": "select",
        "default": ">40",
        "options": {
            "<=40": {"zh": "≤40岁", "en": "≤40 years"}, 
            ">40": {"zh": ">40岁", "en": ">40 years"}
        }
    },
    "family_cancer_history": {
        "zh": "家族史", 
        "en": "Family Cancer History", 
        "type": "select",
        "default": "yes",
        "options": {
            "no": {"zh": "否", "en": "No"}, 
            "yes": {"zh": "是", "en": "Yes"}
        }
    },
    "sexual_history": {
        "zh": "性生活史", 
        "en": "Sexual History", 
        "type": "select",
        "default": "yes",
        "options": {
            "no": {"zh": "否", "en": "No"}, 
            "yes": {"zh": "是", "en": "Yes"}
        }
    },
    "parity": {
        "zh": "生育", 
        "en": "Parity", 
        "type": "select",
        "default": "yes",
        "options": {
            "no": {"zh": "否", "en": "No"}, 
            "yes": {"zh": "是", "en": "Yes"}
        }
    },
    "menopausal_status": {
        "zh": "绝经", 
        "en": "Menopausal Status", 
        "type": "select",
        "default": "yes",
        "options": {
            "no": {"zh": "否", "en": "No"}, 
            "yes": {"zh": "是", "en": "Yes"}
        }
    },
    "comorbidities": {
        "zh": "内科疾病", 
        "en": "Comorbidities", 
        "type": "select",
        "default": "yes",
        "options": {
            "no": {"zh": "否", "en": "No"}, 
            "yes": {"zh": "是", "en": "Yes"}
        }
    },
    "presenting_symptom": {
        "zh": "症状", 
        "en": "Presenting Symptom", 
        "type": "select",
        "default": "abnormal_bleeding",
        "options": {
            "abdominal_pain_bloating": {"zh": "腹痛、腹胀", "en": "Abdominal Pain/Bloating"}, 
            "physical_examination": {"zh": "体检发现", "en": "Physical Examination"},
            "abnormal_bleeding": {"zh": "异常流血、不规律流血", "en": "Abnormal/Irregular Bleeding"}
        }
    },
    "surgical_route": {
        "zh": "手术方式", 
        "en": "Surgical Route", 
        "type": "select",
        "default": "laparotomy",
        "options": {
            "laparotomy": {"zh": "开腹", "en": "Laparotomy"}, 
            "laparoscopy": {"zh": "腹腔镜", "en": "Laparoscopy"}
        }
    },
    "tumor_envelope_integrity": {
        "zh": "肿物破裂", 
        "en": "Tumor Envelope Integrity", 
        "type": "select",
        "default": "yes",
        "options": {
            "no": {"zh": "否", "en": "No"}, 
            "yes": {"zh": "是", "en": "Yes"}
        }
    },
    "fertility_sparing_surgery": {
        "zh": "保留生育功能", 
        "en": "Fertility-Sparing Surgery", 
        "type": "select",
        "default": "no",
        "options": {
            "no": {"zh": "否", "en": "No"}, 
            "yes": {"zh": "是", "en": "Yes"}
        }
    },
    "completeness_of_surgery": {
        "zh": "全面分期", 
        "en": "Completeness of Surgery", 
        "type": "select",
        "default": "no",
        "options": {
            "no": {"zh": "否", "en": "No"}, 
            "yes": {"zh": "是", "en": "Yes"}
        }
    },
    "omentectomy": {
        "zh": "清大网", 
        "en": "Omentectomy", 
        "type": "select",
        "default": "no",
        "options": {
            "no": {"zh": "否", "en": "No"}, 
            "yes": {"zh": "是", "en": "Yes"}
        }
    },
    "lymphadenectomy": {
        "zh": "清淋巴", 
        "en": "Lymphadenectomy", 
        "type": "select",
        "default": "no",
        "options": {
            "no": {"zh": "否", "en": "No"}, 
            "yes": {"zh": "是", "en": "Yes"}
        }
    },
    "histological_subtype": {
        "zh": "病理类型", 
        "en": "Histological Subtype", 
        "type": "select",
        "default": "serous",
        "options": {
            "serous": {"zh": "浆液性 (0)", "en": "Serous (0)"}, 
            "mucinous": {"zh": "粘液性 (1)", "en": "Mucinous (1)"}, 
            "seromucinous": {"zh": "浆粘液性 (2)", "en": "Seromucinous (2)"},
            "endometrioid": {"zh": "子宫内膜样 (3)", "en": "Endometrioid (3)"},
            "clear_cell": {"zh": "透明细胞 (4)", "en": "Clear Cell (4)"},
            "brenner": {"zh": "Brenner瘤 (5)", "en": "Brenner Tumor (5)"}
        }
    },
    "micropapillary": {
        "zh": "微乳头", 
        "en": "Micropapillary", 
        "type": "select",
        "default": "yes",
        "options": {
            "no": {"zh": "否", "en": "No"}, 
            "yes": {"zh": "是", "en": "Yes"}
        }
    },
    "microinfiltration": {
        "zh": "微浸润", 
        "en": "Microinfiltration", 
        "type": "select",
        "default": "yes",
        "options": {
            "no": {"zh": "否", "en": "No"}, 
            "yes": {"zh": "是", "en": "Yes"}
        }
    },
    "psammoma_bodies_calcification": {
        "zh": "钙化砂体", 
        "en": "Psammoma Bodies and Calcification", 
        "type": "select",
        "default": "yes",
        "options": {
            "no": {"zh": "否", "en": "No"}, 
            "yes": {"zh": "是", "en": "Yes"}
        }
    },
    "peritoneal_implantation": {
        "zh": "腹膜种植", 
        "en": "Peritoneal Implantation", 
        "type": "select",
        "default": "yes",
        "options": {
            "no": {"zh": "否", "en": "No"}, 
            "yes": {"zh": "是", "en": "Yes"}
        }
    },
    "ascites_cytology": {
        "zh": "腹水细胞学", 
        "en": "Ascites Cytology", 
        "type": "select",
        "default": "yes",
        "options": {
            "no": {"zh": "否", "en": "No"}, 
            "yes": {"zh": "是", "en": "Yes"}
        }
    },
    "figo_staging": {
        "zh": "分期", 
        "en": "FIGO Staging", 
        "type": "select",
        "default": "III",
        "options": {
            "I": {"zh": "I期", "en": "Stage I"}, 
            "II": {"zh": "II期", "en": "Stage II"}, 
            "III": {"zh": "III期", "en": "Stage III"}
        }
    },
    "unilateral_or_bilateral": {
        "zh": "单侧/双侧", 
        "en": "Unilateral or Bilateral", 
        "type": "select",
        "default": "bilateral",
        "options": {
            "unilateral": {"zh": "单侧", "en": "Unilateral"}, 
            "bilateral": {"zh": "双侧", "en": "Bilateral"}
        }
    },
    "tumor_size": {
        "zh": "肿瘤直径", 
        "en": "Tumor Size", 
        "type": "number", 
        "min": 0.1, 
        "max": 50, 
        "default": 15.0,
        "unit": {"zh": "cm", "en": "cm"}
    },
    "ca125": {
        "zh": "CA125", 
        "en": "CA125", 
        "type": "select",
        "default": "abnormal",
        "options": {
            "normal": {"zh": "正常 (0-35 U/mL)", "en": "Normal (0-35 U/mL)"}, 
            "abnormal": {"zh": "异常 (>35 U/mL)", "en": "Abnormal (>35 U/mL)"}
        }
    },
    "cea": {
        "zh": "CEA", 
        "en": "CEA", 
        "type": "select",
        "default": "abnormal",
        "options": {
            "normal": {"zh": "正常 (0-5 ng/mL)", "en": "Normal (0-5 ng/mL)"}, 
            "abnormal": {"zh": "异常 (>5 ng/mL)", "en": "Abnormal (>5 ng/mL)"}
        }
    },
    "ca199": {
        "zh": "CA199", 
        "en": "CA199", 
        "type": "select",
        "default": "abnormal",
        "options": {
            "normal": {"zh": "正常 (0-37 U/mL)", "en": "Normal (0-37 U/mL)"}, 
            "abnormal": {"zh": "异常 (>37 U/mL)", "en": "Abnormal (>37 U/mL)"}
        }
    },
    "afp": {
        "zh": "AFP", 
        "en": "AFP", 
        "type": "select",
        "default": "normal",
        "options": {
            "normal": {"zh": "正常 (0-9 ng/mL)", "en": "Normal (0-9 ng/mL)"}, 
            "abnormal": {"zh": "异常 (>9 ng/mL)", "en": "Abnormal (>9 ng/mL)"}
        }
    },
    "ca724": {
        "zh": "CA724", 
        "en": "CA724", 
        "type": "select",
        "default": "abnormal",
        "options": {
            "normal": {"zh": "正常 (0-6.9 U/mL)", "en": "Normal (0-6.9 U/mL)"}, 
            "abnormal": {"zh": "异常 (>6.9 U/mL)", "en": "Abnormal (>6.9 U/mL)"}
        }
    },
    "he4": {
        "zh": "HE4", 
        "en": "HE4", 
        "type": "select",
        "default": "abnormal",
        "options": {
            "normal": {"zh": "正常 (0-140 pmol/L)", "en": "Normal (0-140 pmol/L)"}, 
            "abnormal": {"zh": "异常 (>140 pmol/L)", "en": "Abnormal (>140 pmol/L)"}
        }
    },
    "smoking_drinking_history": {
        "zh": "吸烟史", 
        "en": "Smoking and Drinking History", 
        "type": "select",
        "default": "yes",
        "options": {
            "no": {"zh": "否", "en": "No"}, 
            "yes": {"zh": "是", "en": "Yes"}
        }
    },
    "receive_estrogens": {
        "zh": "雌激素暴露史", 
        "en": "Receive Estrogens", 
        "type": "select",
        "default": "yes",
        "options": {
            "no": {"zh": "否", "en": "No"}, 
            "yes": {"zh": "是", "en": "Yes"}
        }
    },
    "ovulation_induction": {
        "zh": "促排卵后", 
        "en": "Ovulation Induction", 
        "type": "select",
        "default": "yes",
        "options": {
            "no": {"zh": "否", "en": "No"}, 
            "yes": {"zh": "是", "en": "Yes"}
        }
    },
    "postoperative_adjuvant_therapy": {
        "zh": "术后化疗", 
        "en": "Postoperative Adjuvant Therapy", 
        "type": "select",
        "default": "no",
        "options": {
            "no": {"zh": "否", "en": "No"}, 
            "yes": {"zh": "是", "en": "Yes"}
        }
    },
    "type_of_lesion": {
        "zh": "病灶类型", 
        "en": "Type of Lesion", 
        "type": "select",
        "default": "endophytic",
        "options": {
            "exophytic": {"zh": "外生型", "en": "Exophytic"}, 
            "endophytic": {"zh": "内生型", "en": "Endophytic"}
        }
    },
    "papillary_area_ratio": {
        "zh": "乳头面积占比", 
        "en": "Papillary Area Ratio", 
        "type": "select",
        "default": ">50%",
        "options": {
            "<=50%": {"zh": "≤50%", "en": "≤50%"}, 
            ">50%": {"zh": ">50%", "en": ">50%"}
        }
    }
}

# ================== 变量分组（界面显示用）==================
VARIABLE_GROUPS = {
    "basic_info": [
        "age", "family_cancer_history", "sexual_history", "parity", 
        "menopausal_status", "comorbidities", "smoking_drinking_history", 
        "receive_estrogens", "ovulation_induction"
    ],
    "surgical_info": [
        "presenting_symptom", "surgical_route", "tumor_envelope_integrity", 
        "fertility_sparing_surgery", "completeness_of_surgery", "omentectomy", 
        "lymphadenectomy", "postoperative_adjuvant_therapy"
    ],
    "pathology_info": [
        "histological_subtype", "micropapillary", "microinfiltration", 
        "psammoma_bodies_calcification", "peritoneal_implantation", "ascites_cytology", 
        "figo_staging", "unilateral_or_bilateral", "tumor_size", "type_of_lesion", 
        "papillary_area_ratio"
    ],
    "tumor_markers": [
        "ca125", "cea", "ca199", "afp", "ca724", "he4"
    ]
}


# ================== 数据预处理类 ==================
class DataPreprocessor:
    def __init__(self, n_features_select=None, scaler_type='robust'):
        self.n_features_select = n_features_select
        self.scaler_type = scaler_type
        self.scaler = None
        self.selector = None
        self.selected_features = None
        
    def fit_transform(self, X, y=None, feature_names=None):
        if self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        if self.n_features_select and y is not None and self.n_features_select < X.shape[1]:
            self.selector = SelectKBest(mutual_info_classif, k=self.n_features_select)
            X_scaled = self.selector.fit_transform(X_scaled, y)
            if feature_names is not None:
                mask = self.selector.get_support()
                self.selected_features = [f for f, m in zip(feature_names, mask) if m]
        else:
            self.selected_features = feature_names
        return X_scaled
    
    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        if self.selector is not None:
            X_scaled = self.selector.transform(X_scaled)
        return X_scaled


# ================== 模型类 ==================
class SEBlock(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.fc(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3, use_se=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.se = SEBlock(dim) if use_se else nn.Identity()
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = self.block(x)
        out = self.se(out)
        return self.activation(x + self.dropout(out))


class EnhancedDeepSurv(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], drop_rate=0.3, n_res_blocks=2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dims[0], drop_rate) for _ in range(n_res_blocks)
        ])
        self.down_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.down_layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.GELU(),
                nn.Dropout(drop_rate)
            ))
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        x = self.input_proj(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        for down_layer in self.down_layers:
            x = down_layer(x)
        return self.output_layer(x).squeeze(1)


class EnhancedDeepHit(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], num_durations=10, drop_rate=0.3):
        super().__init__()
        layers = []
        in_d = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_d, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(drop_rate)
            ])
            in_d = h_dim
        layers.append(nn.Linear(in_d, num_durations))
        self.net = nn.Sequential(*layers)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)


class EnhancedDenoisingAE(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], latent_dim=64, dropout=0.2):
        super().__init__()
        encoder_layers = []
        in_d = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_d, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_d = h_dim
        encoder_layers.append(nn.Linear(in_d, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = []
        in_d = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_d, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_d = h_dim
        decoder_layers.append(nn.Linear(in_d, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x, noise_factor=0.1):
        if self.training and noise_factor > 0:
            x_noisy = x + torch.randn_like(x) * noise_factor
        else:
            x_noisy = x
        z = self.encoder(x_noisy)
        return self.decoder(z), z
    
    def encode(self, x):
        return self.encoder(x)


class EnhancedTransformer(nn.Module):
    def __init__(self, latent_dim, n_heads=4, ff_dim=256, n_layers=2, dropout=0.1):
        super().__init__()
        while latent_dim % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        self.input_norm = nn.LayerNorm(latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, z):
        if z.dim() == 2:
            z = z.unsqueeze(1)
        z = self.input_norm(z)
        z = self.transformer(z)
        z = z.squeeze(1)
        return self.output_proj(z)


class LearnableFusion(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x).squeeze(1)


# ================== 工具函数 ==================
def get_text(key, lang): 
    return TRANSLATIONS.get(key, {}).get(lang, key)


def encode_option(var, opt):
    """将选项值编码为数值"""
    special_encoding = {
        "histological_subtype": {
            "serous": 0, "mucinous": 1, "seromucinous": 2,
            "endometrioid": 3, "clear_cell": 4, "brenner": 5
        },
        "age": {"<=40": 0, ">40": 1},
        "papillary_area_ratio": {"<=50%": 0, ">50%": 1},
        "figo_staging": {"I": 0, "II": 1, "III": 2}
    }
    
    if var in special_encoding and opt in special_encoding[var]:
        return float(special_encoding[var][opt])
    
    opts = INPUT_VARIABLES.get(var, {}).get("options", {})
    try: 
        return float(list(opts.keys()).index(opt))
    except: 
        return 0.0


@st.cache_resource
def load_models(model_dir="results_clinical_enhanced_v3"):
    """加载模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    ok = False
    load_log = [f"📁 模型目录: {model_dir}", f"💻 设备: {device}"]
    
    try:
        req_files = [
            'model_ae.pt', 'model_trans.pt', 'model_deepsurv.pt', 
            'model_deephit.pt', 'model_fusion.pt', 'preprocessor.joblib', 
            'time_cuts.npy', 'ds_min_max.npy', 'best_parameters.json'
        ]
        
        missing = [f for f in req_files if not os.path.exists(os.path.join(model_dir, f))]
        
        if ok:
        # 打印关键信息
        print(f"[INFO] 预处理器输入特征数: {prep.scaler.n_features_in_}")
        print(f"[INFO] AE输入维度: {in_dim}")
        print(f"[INFO] 潜在空间维度: {lat}")
        print(f"[INFO] 融合特征维度: {fused}")
        print(f"[INFO] DeepSurv输出范围: [{ds_mm[0]:.4f}, {ds_mm[1]:.4f}]")
        print(f"[INFO] 时间切分点数: {len(time_cuts)}")
        
        # 检查ds_min_max范围是否合理
        if ds_mm[1] - ds_mm[0] < 0.1:
            print("[WARNING] DeepSurv输出范围过小，可能导致归一化问题！")

        
        if not missing:
            with open(os.path.join(model_dir, "best_parameters.json")) as f: 
                params = json.load(f)
            prep = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))
            time_cuts = np.load(os.path.join(model_dir, "time_cuts.npy"))
            ds_mm = np.load(os.path.join(model_dir, "ds_min_max.npy"))
            
            in_dim = prep.scaler.n_features_in_
            if hasattr(prep, 'selector') and prep.selector is not None:
                in_dim = prep.selector.k if hasattr(prep.selector, 'k') else in_dim
            
            ae_h1, ae_h2, lat = params.get('ae_h1', 256), params.get('ae_h2', 128), params.get('ae_latent', 64)
            fused = lat * 2
            
            ae = EnhancedDenoisingAE(in_dim, [ae_h1, ae_h2], lat)
            ae.load_state_dict(torch.load(os.path.join(model_dir, "model_ae.pt"), map_location=device))
            ae.eval()
            
            trans = EnhancedTransformer(lat, n_heads=4, ff_dim=256, n_layers=2)
            trans.load_state_dict(torch.load(os.path.join(model_dir, "model_trans.pt"), map_location=device))
            trans.eval()
            
            ds_h1, ds_h2, ds_h3 = params.get('ds_h1', 256), params.get('ds_h2', 128), params.get('ds_h3', 64)
            ds = EnhancedDeepSurv(fused, [ds_h1, ds_h2, ds_h3], drop_rate=params.get('ds_drop', 0.3))
            ds.load_state_dict(torch.load(os.path.join(model_dir, "model_deepsurv.pt"), map_location=device))
            ds.eval()
            
            dh_h1, dh_h2 = params.get('dh_h1', 256), params.get('dh_h2', 128)
            dh = EnhancedDeepHit(fused, [dh_h1, dh_h2], len(time_cuts) - 1)
            dh.load_state_dict(torch.load(os.path.join(model_dir, "model_deephit.pt"), map_location=device))
            dh.eval()
            
            fusion = LearnableFusion()
            fusion.load_state_dict(torch.load(os.path.join(model_dir, "model_fusion.pt"), map_location=device))
            fusion.eval()
            
            models = {
                'ae': ae.to(device), 'trans': trans.to(device), 'ds': ds.to(device), 
                'dh': dh.to(device), 'fusion': fusion.to(device), 'prep': prep, 
                'time_cuts': time_cuts, 'ds_mm': ds_mm, 'device': device, 'params': params
            }
            ok = True
            load_log.append("✅ 所有模型加载成功")
        else:
            load_log.append(f"❌ 缺少文件: {missing}")
    except Exception as e:
        load_log.append(f"❌ 加载错误: {e}")
    
    if not ok:
        in_dim, lat, fused, n_bins = len(VARIABLE_ORDER), 64, 128, 10
        models = {
            'ae': EnhancedDenoisingAE(in_dim, [256, 128], lat).to(device), 
            'trans': EnhancedTransformer(lat).to(device), 
            'ds': EnhancedDeepSurv(fused, [256, 128, 64]).to(device), 
            'dh': EnhancedDeepHit(fused, [256, 128], n_bins).to(device), 
            'fusion': LearnableFusion().to(device), 
            'prep': None, 'time_cuts': np.linspace(0, 120, 11), 
            'ds_mm': np.array([-5., 5.]), 'device': device, 'params': {}
        }
        for k in ['ae', 'trans', 'ds', 'dh', 'fusion']: 
            models[k].eval()
    
    models['ok'], models['log'] = ok, load_log
    return models


def preprocess(data, models):
    """按VARIABLE_ORDER顺序预处理数据 - 与训练流程一致"""
    feats = []
    
    # 按照训练时的特征顺序提取
    for v in VARIABLE_ORDER:
        info = INPUT_VARIABLES[v]
        if info['type'] == 'select':
            val = encode_option(v, data.get(v, info.get('default')))
        else:
            val = float(data.get(v, info.get('default', 0)))
        feats.append(val)
    
    X = np.array(feats, dtype=np.float32).reshape(1, -1)
    
    # 使用训练时保存的预处理器
    if models.get('prep') is not None:
        try: 
            X = models['prep'].transform(X)
        except Exception as e:
            print(f"[WARNING] Preprocessor failed: {e}")
            # 如果预处理器失败，使用简单标准化
            X = (X - np.mean(X)) / (np.std(X) + 1e-8)
    else: 
        # 没有预处理器时使用简单标准化
        X = (X - np.mean(X)) / (np.std(X) + 1e-8)
    
    return X


def predict(data, models):
    """执行预测 - 与训练代码流程完全一致"""
    dev = models['device']
    
    # Step 1: 预处理
    X_np = preprocess(data, models)
    X = torch.tensor(X_np, dtype=torch.float32, device=dev)
    
    with torch.no_grad():
        # Step 2: AE编码 (与训练代码一致)
        Z = models['ae'].encode(X)
        
        # Step 3: Transformer (与训练代码一致)
        T = models['trans'](Z)
        
        # Step 4: 融合特征 (与训练代码一致: Xf = concat([Z, T]))
        Xf = torch.cat([Z, T], dim=1)
        
        # Step 5: DeepSurv预测
        risk_ds_raw = models['ds'](Xf).cpu().numpy()
        risk_ds = risk_ds_raw.item() if risk_ds_raw.ndim == 0 else risk_ds_raw[0]
        
        # Step 6: DeepHit预测
        pmf = models['dh'](Xf).cpu().numpy()[0]
        
        # Step 7: 归一化DeepSurv输出 (与训练代码一致)
        # 训练代码: prob_ds_test = normalize_risk(risk_ds_test, min_ds, max_ds)
        min_ds, max_ds = models['ds_mm']
        
        # 使用与训练代码相同的normalize_risk函数
        range_val = max_ds - min_ds
        if range_val == 0:
            p_ds = 0.5
        else:
            p_ds = (risk_ds - min_ds) / range_val
            p_ds = np.clip(p_ds, 0, 1)
        
        # Step 8: DeepHit累积风险 (与训练代码一致)
        # 训练代码: target_bin = actual_n_bins // 2
        #           risk_dh_test = cif_test[:, target_bin]
        cif = np.cumsum(pmf)
        surv = 1 - cif
        target_bin = len(pmf) // 2
        r_dh = cif[target_bin]
        
        # Step 9: Fusion网络 (与训练代码一致)
        # 训练代码: test_in = torch.tensor(np.column_stack([prob_ds_test, risk_dh_test]), ...)
        #           p_final = fusion_net(test_in).cpu().numpy()
        fusion_input = torch.tensor([[p_ds, r_dh]], dtype=torch.float32, device=dev)
        final = models['fusion'](fusion_input).cpu().numpy()
        final = final.item() if final.ndim == 0 else final[0]
    
    # 计算各时间点风险
    tc = models['time_cuts']
    tp = (tc[:-1] + tc[1:]) / 2
    n = len(cif)
    
    def get_risk_at_time(target_time):
        # 找到对应的时间bin
        idx = np.searchsorted(tp, target_time)
        idx = min(max(idx, 0), n - 1)
        return float(cif[idx])
    
    return {
        'risk': float(final),
        'surv': surv,
        'cif': cif,
        'tp': tp,
        'r12': get_risk_at_time(12),
        'r36': get_risk_at_time(36),
        'r60': get_risk_at_time(60),
        'p_ds': float(p_ds),
        'r_dh': float(r_dh),
        'raw_ds': float(risk_ds)
    }


def batch_predict(df, models, lang):
    """批量预测"""
    results = []
    prog = st.progress(0)
    
    for i, row in df.iterrows():
        data = {}
        for v in VARIABLE_ORDER:
            for lg in ['zh', 'en']:
                col = INPUT_VARIABLES[v][lg]
                if col in row: 
                    data[v] = row[col]
                    break
            if v not in data and v in row: 
                data[v] = row[v]
        
        try:
            p = predict(data, models)
            lv = get_text("low_risk" if p['risk'] < 0.3 else ("medium_risk" if p['risk'] < 0.6 else "high_risk"), lang)
            m = get_text("months", lang)
            results.append({
                get_text("patient_id", lang): row.get('patient_id', row.get('患者编号', i + 1)), 
                get_text("overall_risk", lang): f"{p['risk']*100:.1f}%", 
                f"12{m}": f"{p['r12']*100:.1f}%", 
                f"36{m}": f"{p['r36']*100:.1f}%", 
                f"60{m}": f"{p['r60']*100:.1f}%", 
                get_text("risk_level", lang): lv, '_r': p['risk']
            })
        except Exception as e:
            st.warning(f"患者 {i+1} 预测失败: {e}")
        prog.progress((i + 1) / len(df))
    
    prog.empty()
    return pd.DataFrame(results)


def make_template(lang):
    """生成模板"""
    cols = [get_text("patient_id", lang)] + [INPUT_VARIABLES[v][lang] for v in VARIABLE_ORDER]
    data = {cols[0]: [1, 2, 3]}
    for i, v in enumerate(VARIABLE_ORDER):
        info = INPUT_VARIABLES[v]
        default = info.get('default', list(info.get('options', {}).keys())[0] if info['type'] == 'select' else 0)
        data[cols[i + 1]] = [default] * 3
    return pd.DataFrame(data)


# ================== 图表函数 ==================
def make_gauge(risk, lang):
    if risk < 0.3: 
        col, lv = "#27ae60", get_text("low_risk", lang)
    elif risk < 0.6: 
        col, lv = "#f39c12", get_text("medium_risk", lang)
    else: 
        col, lv = "#e74c3c", get_text("high_risk", lang)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=risk * 100,
        number={'suffix': '%', 'font': {'size': 64, 'color': col, 'family': 'Arial Black'}},
        title={'text': f"<b>{get_text('overall_risk', lang)}</b><br><span style='font-size:26px;color:{col}'>{lv}</span>", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': '#555', 'tickfont': {'size': 16}, 'dtick': 25},
            'bar': {'color': col, 'thickness': 0.7}, 'bgcolor': '#f0f0f0', 'borderwidth': 2, 'bordercolor': '#888',
            'steps': [{'range': [0, 30], 'color': 'rgba(39,174,96,0.2)'}, {'range': [30, 60], 'color': 'rgba(243,156,18,0.2)'}, {'range': [60, 100], 'color': 'rgba(231,76,60,0.2)'}]
        }
    ))
    fig.update_layout(height=350, margin=dict(l=30, r=30, t=100, b=30), paper_bgcolor='rgba(0,0,0,0)')
    return fig


def make_time_bar(r12, r36, r60, lang):
    labels = [get_text('month_12', lang), get_text('month_36', lang), get_text('month_60', lang)]
    vals = [r12 * 100, r36 * 100, r60 * 100]
    cols = ['#27ae60' if v < 30 else ('#f39c12' if v < 60 else '#e74c3c') for v in vals]
    
    fig = go.Figure(data=[go.Bar(x=labels, y=vals, marker_color=cols, text=[f'<b>{v:.1f}%</b>' for v in vals], textposition='outside', textfont=dict(size=20, color='#333'), width=0.5)])
    fig.update_layout(
        title=dict(text=f"<b>{get_text('time_risk', lang)}</b>", font=dict(size=18), x=0.5),
        xaxis=dict(tickfont=dict(size=16)), 
        yaxis=dict(title=f"<b>{get_text('risk_prob', lang)} (%)</b>", title_font=dict(size=16), tickfont=dict(size=14), range=[0, max(vals) * 1.35 if max(vals) > 0 else 100], gridcolor='#e8e8e8'),
        height=350, margin=dict(l=70, r=30, t=70, b=50), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white'
    )
    return fig


def make_survival_chart(surv, tp, lang):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tp, y=surv, mode='lines+markers', line=dict(color='#3498db', width=3), fill='tozeroy', fillcolor='rgba(52,152,219,0.15)', marker=dict(size=10, color='#3498db', line=dict(width=2, color='white'))))
    fig.update_layout(
        title=dict(text=f"<b>{get_text('survival_curve', lang)}</b>", font=dict(size=18), x=0.5),
        xaxis=dict(title=f"<b>{get_text('time_months', lang)}</b>", title_font=dict(size=16), tickfont=dict(size=14), gridcolor='#e8e8e8', dtick=12),
        yaxis=dict(title=f"<b>{get_text('survival_prob', lang)}</b>", title_font=dict(size=16), tickfont=dict(size=14), range=[0, 1.05], gridcolor='#e8e8e8', tickformat='.0%'),
        height=350, margin=dict(l=70, r=30, t=70, b=60), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white', showlegend=False
    )
    return fig


def make_cumulative_chart(cif, tp, lang):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tp, y=cif, mode='lines+markers', line=dict(color='#e74c3c', width=3), fill='tozeroy', fillcolor='rgba(231,76,60,0.15)', marker=dict(size=10, color='#e74c3c', symbol='square', line=dict(width=2, color='white'))))
    fig.update_layout(
        title=dict(text=f"<b>{get_text('cumulative_risk_curve', lang)}</b>", font=dict(size=18), x=0.5),
        xaxis=dict(title=f"<b>{get_text('time_months', lang)}</b>", title_font=dict(size=16), tickfont=dict(size=14), gridcolor='#e8e8e8', dtick=12),
        yaxis=dict(title=f"<b>{get_text('risk_prob', lang)}</b>", title_font=dict(size=16), tickfont=dict(size=14), range=[0, 1.05], gridcolor='#e8e8e8', tickformat='.0%'),
        height=350, margin=dict(l=70, r=30, t=70, b=60), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white', showlegend=False
    )
    return fig


def make_pie(df, lang):
    rc = get_text("risk_level", lang)
    h = len(df[df[rc].str.contains('High|高', case=False, na=False)]) if rc in df.columns else 0
    m = len(df[df[rc].str.contains('Intermediate|中', case=False, na=False)]) if rc in df.columns else 0
    l = len(df) - h - m
    
    fig = go.Figure(data=[go.Pie(
        labels=[get_text('low_risk', lang), get_text('medium_risk', lang), get_text('high_risk', lang)], 
        values=[l, m, h], marker_colors=['#27ae60', '#f39c12', '#e74c3c'], hole=0.45, 
        textinfo='label+percent+value', textfont=dict(size=15), pull=[0, 0, 0.05]
    )])
    fig.update_layout(
        title=dict(text=f"<b>{get_text('risk_distribution', lang)}</b>", font=dict(size=18), x=0.5), 
        height=380, margin=dict(l=20, r=20, t=70, b=20), paper_bgcolor='rgba(0,0,0,0)', 
        legend=dict(font=dict(size=14), orientation='h', yanchor='bottom', y=-0.12, xanchor='center', x=0.5)
    )
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
    
    story = [
        Paragraph("Cancer Recurrence Risk Report", ParagraphStyle('T', parent=styles['Heading1'], fontSize=18, spaceAfter=20, alignment=1)),
        Paragraph("Shengjing Hospital of China Medical University", styles['Normal']),
        Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']), Spacer(1, 20)
    ]
    
    data = [["Category", "Count", "%"], ["Total", str(total), "100%"], ["High", str(h), f"{h/total*100:.1f}%" if total else "0%"], ["Medium", str(m), f"{m/total*100:.1f}%" if total else "0%"], ["Low", str(l), f"{l/total*100:.1f}%" if total else "0%"]]
    tbl = Table(data, colWidths=[120, 80, 80])
    tbl.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white), ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
    
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
    
    story = [
        Paragraph("Patient Risk Assessment", ParagraphStyle('T', parent=styles['Heading1'], fontSize=18, spaceAfter=20, alignment=1)),
        Paragraph("Shengjing Hospital", styles['Normal']),
        Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']), Spacer(1, 20)
    ]
    
    data = [["Item", "Value"], ["Risk", f"{r*100:.1f}%"], ["Level", lv], ["12M", f"{res['r12']*100:.1f}%"], ["36M", f"{res['r36']*100:.1f}%"], ["60M", f"{res['r60']*100:.1f}%"]]
    tbl = Table(data, colWidths=[150, 150])
    tbl.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white), ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, -1), 11), ('GRID', (0, 0), (-1, -1), 1, colors.black), ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white])]))
    
    story.extend([tbl, Spacer(1, 20), Paragraph("For clinical reference only.", ParagraphStyle('D', fontSize=9, textColor=colors.grey))])
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# ================== 输入控件 ==================
def sel_widget(v, info, lang, pre=""):
    options_list = list(info['options'].keys())
    default_val = info.get('default', options_list[0])
    default_idx = options_list.index(default_val) if default_val in options_list else 0
    return st.selectbox(info[lang], options_list, index=default_idx, format_func=lambda x: info['options'][x][lang], key=f"{pre}{v}")


def num_widget(v, info, lang, pre=""):
    lbl = f"{info[lang]} ({info['unit'][lang]})" if 'unit' in info else info[lang]
    return st.number_input(lbl, float(info.get('min', 0)), float(info.get('max', 100)), float(info.get('default', 0)), key=f"{pre}{v}")


# ================== 主函数 ==================
def main():
    models = load_models()
    
    # 模型状态
    if models.get('ok', False):
        st.markdown('<div class="status-box status-success">✅ <b>模型状态</b>: 已成功加载训练好的模型</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-box status-error">❌ <b>模型状态</b>: 未找到训练好的模型，使用默认随机模型</div>', unsafe_allow_html=True)
        with st.expander("🔍 查看加载日志"):
            for log in models.get('log', []): 
                st.text(log)
    
    # 顶部栏
    if HAS_LOGO:
        st.markdown(f'<div class="top-bar"><div class="logo-section"><img src="data:image/png;base64,{LOGO_BASE64}" class="logo-img"><div class="logo-text"><h2>盛京医院</h2><p>中国医科大学附属盛京医院</p></div></div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="top-bar"><div class="logo-section"><div class="logo-text"><h2>🏥 盛京医院</h2><p>中国医科大学附属盛京医院</p></div></div></div>', unsafe_allow_html=True)
    
    # 语言选择
    col_space, col_lang = st.columns([10, 1])
    with col_lang:
        lang = LANGUAGES[st.selectbox("🌐", list(LANGUAGES.keys()), label_visibility="collapsed", key="lang")]
    
    # 头部
    if HAS_LOGO:
        st.markdown(f'<div class="hospital-header"><div class="header-logo"><img src="data:image/png;base64,{LOGO_BASE64}"></div><div class="header-text"><h1>🏥 {get_text("title", lang)}</h1><p class="subtitle">{get_text("subtitle", lang)}</p><p class="hospital-name">{get_text("hospital", lang)}</p></div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="hospital-header-nologo"><h1>🏥 {get_text("title", lang)}</h1><p class="subtitle">{get_text("subtitle", lang)}</p><p class="hospital-name">{get_text("hospital", lang)}</p></div>', unsafe_allow_html=True)
    
    # 标签页
    tab1, tab2 = st.tabs([f"📋 {get_text('single_patient', lang)}", f"📊 {get_text('batch_prediction', lang)}"])
    
    # ========== 单例预测 ==========
    with tab1:
        c1, c2, c3 = st.columns(3)
        data = {}
        
        with c1:
            st.markdown(f'<div class="module-card"><div class="module-title">📝 {get_text("basic_info", lang)}</div>', unsafe_allow_html=True)
            for v in VARIABLE_GROUPS["basic_info"]:
                info = INPUT_VARIABLES[v]
                data[v] = num_widget(v, info, lang, "s_") if info['type'] == 'number' else sel_widget(v, info, lang, "s_")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with c2:
            st.markdown(f'<div class="module-card"><div class="module-title surgery">🔪 {get_text("surgical_info", lang)}</div>', unsafe_allow_html=True)
            for v in VARIABLE_GROUPS["surgical_info"]:
                data[v] = sel_widget(v, INPUT_VARIABLES[v], lang, "s_")
            st.markdown('</div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="module-card"><div class="module-title pathology">🔬 {get_text("pathology_info", lang)}</div>', unsafe_allow_html=True)
            for v in VARIABLE_GROUPS["pathology_info"]:
                info = INPUT_VARIABLES[v]
                data[v] = num_widget(v, info, lang, "s_") if info['type'] == 'number' else sel_widget(v, info, lang, "s_")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="module-card"><div class="module-title markers">🧪 {get_text("tumor_markers", lang)}</div>', unsafe_allow_html=True)
        mc = st.columns(6)
        for i, v in enumerate(VARIABLE_GROUPS["tumor_markers"]):
            with mc[i]: 
                data[v] = sel_widget(v, INPUT_VARIABLES[v], lang, "s_")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        bc1, bc2, bc3 = st.columns([2, 1, 2])
        with bc2:
            predict_btn = st.button(f"🔮 {get_text('predict_button', lang)}", use_container_width=True, key="pred")
        
        if predict_btn:
            with st.spinner(get_text('processing', lang)):
                res = predict(data, models)
                
                # 调试信息
                with st.expander(f"🔧 {get_text('debug_info', lang)}"):
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        st.markdown(f"**{get_text('input_data', lang)}:**")
                        encoded = {v: f"{data[v]} → {encode_option(v, data[v])}" if INPUT_VARIABLES[v]['type'] == 'select' else data[v] for v in VARIABLE_ORDER}
                        st.json(encoded)
                    with col_d2:
                        st.markdown(f"**{get_text('processed_features', lang)}:**")
                        X_proc = preprocess(data, models)
                        st.write(f"Shape: {X_proc.shape}, Range: [{X_proc.min():.4f}, {X_proc.max():.4f}]")
                    
                    st.markdown("**模型输出:**")
                    dc = st.columns(4)
                    dc[0].metric("DeepSurv原始", f"{res.get('raw_ds', 0):.4f}")
                    dc[1].metric("DeepSurv归一化", f"{res.get('p_ds', 0):.4f}")
                    dc[2].metric("DeepHit中位", f"{res.get('r_dh', 0):.4f}")
                    dc[3].metric("融合风险", f"{res['risk']:.4f}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f'<div class="result-section"><div class="result-title">📊 {get_text("prediction_results", lang)}</div>', unsafe_allow_html=True)
                
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
                if r < 0.3: 
                    lv, adv, css = "low_risk", "advice_low", "low"
                elif r < 0.6: 
                    lv, adv, css = "medium_risk", "advice_medium", "medium"
                else: 
                    lv, adv, css = "high_risk", "advice_high", "high"
                
                st.markdown(f"""
                <div class="advice-box {css}">
                    <h4>💊 {get_text('clinical_advice', lang)} — {get_text(lv, lang)} ({r*100:.1f}%)</h4>
                    <pre style="white-space: pre-wrap; font-family: inherit; margin: 0; line-height: 1.8;">{get_text(adv, lang)}</pre>
                </div>
                """, unsafe_allow_html=True)
                
                # 导出
                st.markdown(f"#### 📥 {get_text('export_results', lang)}")
                ec1, ec2, ec3 = st.columns(3)
                with ec1:
                    df_exp = pd.DataFrame({
                        get_text('overall_risk', lang): [f"{res['risk']*100:.1f}%"], 
                        get_text('month_12', lang): [f"{res['r12']*100:.1f}%"], 
                        get_text('month_36', lang): [f"{res['r36']*100:.1f}%"], 
                        get_text('month_60', lang): [f"{res['r60']*100:.1f}%"]
                    })
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine='openpyxl') as w: 
                        df_exp.to_excel(w, index=False)
                    st.download_button(f"📊 {get_text('export_excel', lang)}", buf.getvalue(), f"result_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx", use_container_width=True)
                with ec2:
                    st.download_button(f"📄 {get_text('export_pdf', lang)}", make_single_pdf(res, lang), f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", "application/pdf", use_container_width=True)
    
    # ========== 批量预测 ==========
    with tab2:
        st.markdown(f"#### {get_text('step1', lang)}")
        tpl = make_template(lang)
        buf = io.StringIO()
        tpl.to_csv(buf, index=False, encoding='utf-8-sig')
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
                                fig = go.Figure(go.Histogram(x=res_df['_r'] * 100, nbinsx=20, marker_color='#3498db', opacity=0.8))
                                fig.add_vline(x=30, line_dash="dash", line_color="#27ae60", line_width=2)
                                fig.add_vline(x=60, line_dash="dash", line_color="#e74c3c", line_width=2)
                                fig.update_layout(
                                    title=dict(text=f"<b>{get_text('risk_distribution', lang)}</b>", font=dict(size=18), x=0.5), 
                                    xaxis=dict(title=f"<b>{get_text('risk_prob', lang)} (%)</b>"), 
                                    yaxis=dict(title=f"<b>{get_text('total_patients', lang)}</b>"), 
                                    height=380, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        disp = res_df.drop(columns=[c for c in res_df.columns if c.startswith('_')], errors='ignore')
                        
                        def hl(row):
                            v = str(row.get(rc, ''))
                            if 'High' in v or '高' in v: 
                                return ['background-color:#f8d7da'] * len(row)
                            if 'Intermediate' in v or '中' in v: 
                                return ['background-color:#fff3cd'] * len(row)
                            return ['background-color:#d4edda'] * len(row)
                        
                        st.dataframe(disp.style.apply(hl, axis=1), use_container_width=True, height=350)
                        
                        st.markdown(f"#### 📥 {get_text('export_results', lang)}")
                        e1, e2, e3 = st.columns(3)
                        with e1:
                            buf = io.StringIO()
                            disp.to_csv(buf, index=False, encoding='utf-8-sig')
                            st.download_button(f"📋 {get_text('export_csv', lang)}", buf.getvalue(), f"batch_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", use_container_width=True)
                        with e2:
                            buf = io.BytesIO()
                            with pd.ExcelWriter(buf, engine='openpyxl') as w: 
                                disp.to_excel(w, index=False)
                            st.download_button(f"📊 {get_text('export_excel', lang)}", buf.getvalue(), f"batch_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx", use_container_width=True)
                        with e3:
                            st.download_button(f"📄 {get_text('export_pdf', lang)}", make_pdf(res_df, lang), f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", "application/pdf", use_container_width=True)
                        
                        if h > 0:
                            st.markdown("---")
                            st.markdown(f"### ⚠️ {get_text('high_risk_attention', lang)}")
                            hdf = disp[disp[rc].str.contains('High|高', case=False, na=False)]
                            st.dataframe(hdf.style.apply(lambda x: ['background-color:#f8d7da'] * len(x), axis=1), use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # 页脚
    st.markdown("---")
    st.info(get_text('disclaimer', lang))
    
    if HAS_LOGO:
        st.markdown(f"""
        <div class="footer">
            <div class="footer-logo"><img src="data:image/png;base64,{LOGO_BASE64}"></div>
            <div class="footer-text">
                <p class="hospital-name">{get_text('hospital', lang)}</p>
                <p class="version">Cancer Recurrence Risk Prediction System v2.0</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="footer">
            <div class="footer-text">
                <p class="hospital-name">🏥 {get_text('hospital', lang)}</p>
                <p class="version">Cancer Recurrence Risk Prediction System v2.0</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
