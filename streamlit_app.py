"""
Streamlit Web Application for Cancer Recurrence Prediction
肿瘤复发预测网页应用
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
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
import io
import base64
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# ================== 页面配置 ==================
st.set_page_config(
    page_title="Cancer Recurrence Prediction | 肿瘤复发预测",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================== 自定义CSS样式 ==================
st.markdown("""
<style>
    /* 隐藏侧边栏 */
    [data-testid="collapsedControl"] {display: none}
    section[data-testid="stSidebar"] {display: none;}
    
    /* 主容器样式 */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* 标题样式 */
    .main-title {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-title h1 {
        color: white;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .main-title h3 {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        font-weight: normal;
    }
    .main-title .hospital {
        color: #FFD700;
        font-size: 1rem;
        font-weight: bold;
        margin-top: 0.5rem;
    }
    
    /* 卡片样式 */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    /* 分组标题样式 */
    .section-header {
        background: linear-gradient(90deg, #f8f9fa 0%, #ffffff 100%);
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
        font-weight: bold;
        color: #333;
    }
    
    /* 结果卡片样式 */
    .result-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* 风险等级颜色 */
    .risk-low {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 4px solid #28a745;
    }
    .risk-medium {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        border-left: 4px solid #ffc107;
    }
    .risk-high {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 4px solid #dc3545;
    }
    
    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 25px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* 选择框样式 */
    .stSelectbox > div > div {
        border-radius: 8px;
    }
    
    /* 标签页样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* 指标卡片样式 */
    [data-testid="metric-container"] {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    /* 页脚样式 */
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-top: 2rem;
        color: white;
    }
    .footer p {
        margin: 0.3rem 0;
    }
    .footer .hospital-name {
        font-size: 1.1rem;
        font-weight: bold;
        color: #FFD700;
    }
    
    /* 成功消息样式 */
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    /* 语言选择器样式 */
    .language-selector {
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ================== 语言配置 ==================
LANGUAGES = {"中文": "zh", "English": "en"}

# ================== 翻译字典 ==================
TRANSLATIONS = {
    "title": {"zh": "🏥 肿瘤复发风险预测系统", "en": "🏥 Cancer Recurrence Risk Prediction System"},
    "subtitle": {"zh": "基于深度学习的个体化预测模型", "en": "Deep Learning-Based Personalized Prediction Model"},
    "hospital": {"zh": "中国医科大学附属盛京医院", "en": "Shengjing Hospital of China Medical University"},
    "patient_info": {"zh": "📋 患者信息录入", "en": "📋 Patient Information Entry"},
    "single_patient": {"zh": "👤 单个患者预测", "en": "👤 Single Patient Prediction"},
    "batch_prediction": {"zh": "📊 批量患者预测", "en": "📊 Batch Patient Prediction"},
    "basic_info": {"zh": "📝 基本信息", "en": "📝 Basic Information"},
    "surgical_info": {"zh": "🔪 手术信息", "en": "🔪 Surgical Information"},
    "pathology_info": {"zh": "🔬 病理信息", "en": "🔬 Pathology Information"},
    "tumor_markers": {"zh": "🧪 肿瘤标志物", "en": "🧪 Tumor Markers"},
    "predict_button": {"zh": "🔮 开始预测", "en": "🔮 Start Prediction"},
    "prediction_results": {"zh": "📊 预测结果", "en": "📊 Prediction Results"},
    "overall_risk": {"zh": "总体复发风险", "en": "Overall Recurrence Risk"},
    "risk_level": {"zh": "风险等级", "en": "Risk Level"},
    "low_risk": {"zh": "低风险", "en": "Low Risk"},
    "medium_risk": {"zh": "中等风险", "en": "Medium Risk"},
    "high_risk": {"zh": "高风险", "en": "High Risk"},
    "survival_curve": {"zh": "📈 生存曲线预测", "en": "📈 Survival Curve Prediction"},
    "time_risk": {"zh": "时间点复发风险", "en": "Time-Point Recurrence Risk"},
    "clinical_advice": {"zh": "💊 临床建议", "en": "💊 Clinical Recommendations"},
    "disclaimer": {
        "zh": "⚠️ 免责声明：本系统仅供临床参考，不能替代专业医生的诊断。请结合临床实际情况综合判断。",
        "en": "⚠️ Disclaimer: This system is for clinical reference only and cannot replace professional medical diagnosis."
    },
    "months": {"zh": "个月", "en": " months"},
    "probability": {"zh": "概率", "en": "Probability"},
    "time": {"zh": "时间", "en": "Time"},
    "survival_probability": {"zh": "无复发生存概率", "en": "Recurrence-Free Survival"},
    "cumulative_risk": {"zh": "累积复发风险", "en": "Cumulative Recurrence Risk"},
    "upload_csv": {"zh": "上传CSV文件", "en": "Upload CSV File"},
    "download_template": {"zh": "📥 下载模板", "en": "📥 Download Template"},
    "batch_results": {"zh": "批量预测结果", "en": "Batch Prediction Results"},
    "export_excel": {"zh": "📊 导出Excel", "en": "📊 Export Excel"},
    "export_pdf": {"zh": "📄 导出PDF报告", "en": "📄 Export PDF Report"},
    "export_csv": {"zh": "📋 导出CSV", "en": "📋 Export CSV"},
    "patient_id": {"zh": "患者ID", "en": "Patient ID"},
    "total_patients": {"zh": "总患者数", "en": "Total Patients"},
    "high_risk_count": {"zh": "高风险", "en": "High Risk"},
    "medium_risk_count": {"zh": "中风险", "en": "Medium Risk"},
    "low_risk_count": {"zh": "低风险", "en": "Low Risk"},
    "risk_distribution": {"zh": "风险等级分布", "en": "Risk Level Distribution"},
    "risk_score_dist": {"zh": "风险分数分布", "en": "Risk Score Distribution"},
    "processing": {"zh": "正在分析中，请稍候...", "en": "Analyzing, please wait..."},
    "export_results": {"zh": "📥 导出结果", "en": "📥 Export Results"},
    "detailed_results": {"zh": "📋 详细结果", "en": "📋 Detailed Results"},
    "high_risk_attention": {"zh": "⚠️ 需重点关注的高风险患者", "en": "⚠️ High Risk Patients Requiring Attention"},
    "high_risk_warning": {"zh": "位患者被评估为高风险，建议密切随访！", "en": "patients classified as high risk, close follow-up recommended!"},
    "preview_template": {"zh": "预览模板", "en": "Preview Template"},
    "preview_data": {"zh": "预览数据", "en": "Preview Data"},
    "loaded_patients": {"zh": "成功加载患者数据", "en": "Successfully loaded patient data"},
    "file_error": {"zh": "文件处理错误", "en": "File processing error"},
    "file_format_hint": {"zh": "请确保您的文件格式与模板一致。", "en": "Please ensure your file format matches the template."},
    "step1": {"zh": "第一步：下载模板", "en": "Step 1: Download Template"},
    "step2": {"zh": "第二步：上传数据", "en": "Step 2: Upload Data"},
    "model_loaded": {"zh": "✅ 预测模型已就绪", "en": "✅ Prediction model ready"},
    "language_label": {"zh": "选择语言", "en": "Select Language"},
    "metric_label": {"zh": "指标", "en": "Metric"},
    "value_label": {"zh": "数值", "en": "Value"},
    "final_risk": {"zh": "综合风险评分", "en": "Overall Risk Score"},
    "month_risk_12": {"zh": "12个月复发风险", "en": "12-Month Recurrence Risk"},
    "month_risk_36": {"zh": "36个月复发风险", "en": "36-Month Recurrence Risk"},
    "month_risk_60": {"zh": "60个月复发风险", "en": "60-Month Recurrence Risk"},
    "advice_low": {
        "zh": """
**随访建议：**
- 常规随访，每6个月复查一次
- 保持健康生活方式
- 定期监测肿瘤标志物（CA125、HE4等）
- 每年进行盆腔超声检查
        """,
        "en": """
**Follow-up Recommendations:**
- Routine follow-up every 6 months
- Maintain healthy lifestyle
- Regular monitoring of tumor markers (CA125, HE4, etc.)
- Annual pelvic ultrasound examination
        """
    },
    "advice_medium": {
        "zh": """
**随访建议：**
- 加强随访，每3-4个月复查一次
- 考虑辅助化疗或其他辅助治疗
- 密切监测肿瘤标志物变化趋势
- 每3-6个月进行影像学检查
- 建议进行基因检测评估
        """,
        "en": """
**Follow-up Recommendations:**
- Enhanced follow-up every 3-4 months
- Consider adjuvant chemotherapy or other treatments
- Close monitoring of tumor marker trends
- Imaging examination every 3-6 months
- Recommend genetic testing evaluation
        """
    },
    "advice_high": {
        "zh": """
**随访建议：**
- 密切随访，每2-3个月复查一次
- 强烈建议进行辅助化疗
- 建议多学科会诊（MDT）
- 密切监测复发迹象
- 可考虑参加临床试验
- 加强心理支持和营养管理
        """,
        "en": """
**Follow-up Recommendations:**
- Close follow-up every 2-3 months
- Strongly recommend adjuvant chemotherapy
- Recommend multidisciplinary team (MDT) consultation
- Close monitoring for recurrence signs
- Consider clinical trial participation
- Enhanced psychological support and nutrition management
        """
    }
}

# ================== 输入变量配置 ==================
INPUT_VARIABLES = {
    "age": {
        "zh": "年龄", "en": "Age", "type": "number", 
        "min": 18, "max": 100, "default": 50,
        "unit": {"zh": "岁", "en": "years"}
    },
    "family_cancer_history": {
        "zh": "肿瘤家族史", "en": "Family Cancer History", "type": "select",
        "options": {"no": {"zh": "无", "en": "No"}, "yes": {"zh": "有", "en": "Yes"}}
    },
    "sexual_history": {
        "zh": "性生活史", "en": "Sexual History", "type": "select",
        "options": {"no": {"zh": "无", "en": "No"}, "yes": {"zh": "有", "en": "Yes"}}
    },
    "parity": {
        "zh": "生育次数", "en": "Parity", "type": "select",
        "options": {"0": {"zh": "0次", "en": "0"}, "1": {"zh": "1次", "en": "1"}, 
                   "2": {"zh": "2次", "en": "2"}, "3": {"zh": "≥3次", "en": "≥3"}}
    },
    "menopausal_status": {
        "zh": "绝经状态", "en": "Menopausal Status", "type": "select",
        "options": {"premenopausal": {"zh": "未绝经", "en": "Premenopausal"}, 
                   "postmenopausal": {"zh": "已绝经", "en": "Postmenopausal"}}
    },
    "comorbidities": {
        "zh": "合并内科疾病", "en": "Comorbidities", "type": "select",
        "options": {"no": {"zh": "无", "en": "None"}, "hypertension": {"zh": "高血压", "en": "Hypertension"},
                   "diabetes": {"zh": "糖尿病", "en": "Diabetes"}, "cardiovascular": {"zh": "心血管疾病", "en": "Cardiovascular"},
                   "multiple": {"zh": "多种疾病", "en": "Multiple"}}
    },
    "smoking_drinking_history": {
        "zh": "吸烟饮酒史", "en": "Smoking/Drinking History", "type": "select",
        "options": {"no": {"zh": "无", "en": "No"}, "smoking": {"zh": "吸烟", "en": "Smoking"},
                   "drinking": {"zh": "饮酒", "en": "Drinking"}, "both": {"zh": "均有", "en": "Both"}}
    },
    "receive_estrogens": {
        "zh": "雌激素暴露史", "en": "Estrogen Exposure", "type": "select",
        "options": {"no": {"zh": "无", "en": "No"}, "hrt": {"zh": "激素替代治疗", "en": "HRT"},
                   "contraceptive": {"zh": "口服避孕药", "en": "Oral Contraceptive"}, "other": {"zh": "其他", "en": "Other"}}
    },
    "ovulation_induction": {
        "zh": "促排卵治疗史", "en": "Ovulation Induction History", "type": "select",
        "options": {"no": {"zh": "无", "en": "No"}, "yes": {"zh": "有", "en": "Yes"}}
    },
    "presenting_symptom": {
        "zh": "主要症状", "en": "Presenting Symptom", "type": "select",
        "options": {"asymptomatic": {"zh": "无症状", "en": "Asymptomatic"}, "abdominal_pain": {"zh": "腹痛", "en": "Abdominal Pain"},
                   "bloating": {"zh": "腹胀", "en": "Bloating"}, "mass": {"zh": "腹部包块", "en": "Abdominal Mass"},
                   "bleeding": {"zh": "异常出血", "en": "Abnormal Bleeding"}, "other": {"zh": "其他", "en": "Other"}}
    },
    "surgical_route": {
        "zh": "手术方式", "en": "Surgical Approach", "type": "select",
        "options": {"laparoscopy": {"zh": "腹腔镜手术", "en": "Laparoscopy"}, "laparotomy": {"zh": "开腹手术", "en": "Laparotomy"},
                   "robotic": {"zh": "机器人辅助", "en": "Robotic"}, "conversion": {"zh": "中转开腹", "en": "Conversion"}}
    },
    "tumor_envelope_integrity": {
        "zh": "肿瘤包膜完整性", "en": "Tumor Capsule Integrity", "type": "select",
        "options": {"intact": {"zh": "完整", "en": "Intact"}, "ruptured_before": {"zh": "术前破裂", "en": "Ruptured Before Surgery"},
                   "ruptured_during": {"zh": "术中破裂", "en": "Ruptured During Surgery"}}
    },
    "fertility_sparing_surgery": {
        "zh": "保留生育功能", "en": "Fertility-Sparing Surgery", "type": "select",
        "options": {"no": {"zh": "否", "en": "No"}, "yes": {"zh": "是", "en": "Yes"}}
    },
    "completeness_of_surgery": {
        "zh": "手术完整性", "en": "Surgical Completeness", "type": "select",
        "options": {"incomplete": {"zh": "不完整分期", "en": "Incomplete Staging"}, "complete": {"zh": "完整分期", "en": "Complete Staging"}}
    },
    "omentectomy": {
        "zh": "大网膜切除", "en": "Omentectomy", "type": "select",
        "options": {"no": {"zh": "未切除", "en": "No"}, "partial": {"zh": "部分切除", "en": "Partial"}, "total": {"zh": "全切除", "en": "Total"}}
    },
    "lymphadenectomy": {
        "zh": "淋巴结清扫", "en": "Lymphadenectomy", "type": "select",
        "options": {"no": {"zh": "未清扫", "en": "No"}, "pelvic": {"zh": "盆腔淋巴结", "en": "Pelvic"},
                   "paraaortic": {"zh": "腹主动脉旁", "en": "Para-aortic"}, "both": {"zh": "盆腔+腹主动脉旁", "en": "Both"}}
    },
    "postoperative_adjuvant_therapy": {
        "zh": "术后辅助治疗", "en": "Adjuvant Therapy", "type": "select",
        "options": {"no": {"zh": "无", "en": "No"}, "chemotherapy": {"zh": "化疗", "en": "Chemotherapy"},
                   "targeted": {"zh": "靶向治疗", "en": "Targeted Therapy"}, "combined": {"zh": "联合治疗", "en": "Combined"}}
    },
    "histological_subtype": {
        "zh": "病理类型", "en": "Histological Subtype", "type": "select",
        "options": {"serous": {"zh": "浆液性", "en": "Serous"}, "mucinous": {"zh": "黏液性", "en": "Mucinous"},
                   "endometrioid": {"zh": "子宫内膜样", "en": "Endometrioid"}, "clear_cell": {"zh": "透明细胞", "en": "Clear Cell"},
                   "mixed": {"zh": "混合型", "en": "Mixed"}, "other": {"zh": "其他", "en": "Other"}}
    },
    "micropapillary": {
        "zh": "微乳头结构", "en": "Micropapillary Pattern", "type": "select",
        "options": {"no": {"zh": "无", "en": "No"}, "yes": {"zh": "有", "en": "Yes"}}
    },
    "microinfiltration": {
        "zh": "微浸润", "en": "Microinvasion", "type": "select",
        "options": {"no": {"zh": "无", "en": "No"}, "yes": {"zh": "有", "en": "Yes"}}
    },
    "psammoma_bodies_calcification": {
        "zh": "砂粒体/钙化", "en": "Psammoma Bodies", "type": "select",
        "options": {"no": {"zh": "无", "en": "No"}, "yes": {"zh": "有", "en": "Yes"}}
    },
    "peritoneal_implantation": {
        "zh": "腹膜种植", "en": "Peritoneal Implants", "type": "select",
        "options": {"no": {"zh": "无", "en": "No"}, "noninvasive": {"zh": "非浸润性", "en": "Non-invasive"}, "invasive": {"zh": "浸润性", "en": "Invasive"}}
    },
    "ascites_cytology": {
        "zh": "腹水细胞学", "en": "Ascites Cytology", "type": "select",
        "options": {"no_ascites": {"zh": "无腹水", "en": "No Ascites"}, "negative": {"zh": "阴性", "en": "Negative"}, "positive": {"zh": "阳性", "en": "Positive"}}
    },
    "figo_staging": {
        "zh": "FIGO分期", "en": "FIGO Stage", "type": "select",
        "options": {"IA": {"zh": "IA期", "en": "Stage IA"}, "IB": {"zh": "IB期", "en": "Stage IB"},
                   "IC1": {"zh": "IC1期", "en": "Stage IC1"}, "IC2": {"zh": "IC2期", "en": "Stage IC2"},
                   "IC3": {"zh": "IC3期", "en": "Stage IC3"}, "II": {"zh": "II期", "en": "Stage II"},
                   "IIIA": {"zh": "IIIA期", "en": "Stage IIIA"}, "IIIB": {"zh": "IIIB期", "en": "Stage IIIB"},
                   "IIIC": {"zh": "IIIC期", "en": "Stage IIIC"}}
    },
    "unilateral_or_bilateral": {
        "zh": "肿瘤侧别", "en": "Tumor Laterality", "type": "select",
        "options": {"left": {"zh": "左侧", "en": "Left"}, "right": {"zh": "右侧", "en": "Right"}, "bilateral": {"zh": "双侧", "en": "Bilateral"}}
    },
    "tumor_size": {
        "zh": "肿瘤最大径", "en": "Tumor Size", "type": "select",
        "options": {"<=5": {"zh": "≤5cm", "en": "≤5cm"}, "5-10": {"zh": "5-10cm", "en": "5-10cm"},
                   "10-15": {"zh": "10-15cm", "en": "10-15cm"}, ">15": {"zh": ">15cm", "en": ">15cm"}}
    },
    "type_of_lesion": {
        "zh": "病灶性质", "en": "Lesion Type", "type": "select",
        "options": {"cystic": {"zh": "囊性", "en": "Cystic"}, "solid": {"zh": "实性", "en": "Solid"}, "mixed": {"zh": "囊实混合", "en": "Mixed"}}
    },
    "papillary_area_ratio": {
        "zh": "乳头状区域占比", "en": "Papillary Area Ratio", "type": "select",
        "options": {"<10%": {"zh": "<10%", "en": "<10%"}, "10-30%": {"zh": "10-30%", "en": "10-30%"},
                   "30-50%": {"zh": "30-50%", "en": "30-50%"}, ">50%": {"zh": ">50%", "en": ">50%"}}
    },
    "ca125": {
        "zh": "CA125", "en": "CA125", "type": "select",
        "options": {"normal": {"zh": "正常(<35)", "en": "Normal(<35)"}, "mild": {"zh": "轻度升高(35-100)", "en": "Mild(35-100)"},
                   "moderate": {"zh": "中度升高(100-500)", "en": "Moderate(100-500)"}, "high": {"zh": "显著升高(>500)", "en": "High(>500)"}}
    },
    "cea": {
        "zh": "CEA", "en": "CEA", "type": "select",
        "options": {"normal": {"zh": "正常(<5)", "en": "Normal(<5)"}, "elevated": {"zh": "升高(≥5)", "en": "Elevated(≥5)"}}
    },
    "ca199": {
        "zh": "CA199", "en": "CA199", "type": "select",
        "options": {"normal": {"zh": "正常(<37)", "en": "Normal(<37)"}, "elevated": {"zh": "升高(≥37)", "en": "Elevated(≥37)"}}
    },
    "afp": {
        "zh": "AFP", "en": "AFP", "type": "select",
        "options": {"normal": {"zh": "正常(<10)", "en": "Normal(<10)"}, "elevated": {"zh": "升高(≥10)", "en": "Elevated(≥10)"}}
    },
    "ca724": {
        "zh": "CA724", "en": "CA724", "type": "select",
        "options": {"normal": {"zh": "正常(<6.9)", "en": "Normal(<6.9)"}, "elevated": {"zh": "升高(≥6.9)", "en": "Elevated(≥6.9)"}}
    },
    "he4": {
        "zh": "HE4", "en": "HE4", "type": "select",
        "options": {"normal": {"zh": "正常(<70)", "en": "Normal(<70)"}, "mild": {"zh": "轻度升高(70-140)", "en": "Mild(70-140)"},
                   "elevated": {"zh": "显著升高(>140)", "en": "High(>140)"}}
    }
}


# ================== 数据预处理器类 ==================
class DataPreprocessor:
    def __init__(self, select_k=None):
        self.scaler = StandardScaler()
        self.selector = None
        self.select_k = select_k
        self.feature_names = None
        
    def fit(self, X, y=None):
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        if self.select_k is not None and y is not None:
            self.selector = SelectKBest(f_classif, k=min(self.select_k, X.shape[1]))
            self.selector.fit(X_scaled, y)
        return self
    
    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        if self.selector is not None:
            X_scaled = self.selector.transform(X_scaled)
        return X_scaled
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


# ================== 模型定义 ==================
class SEBlock(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, max(dim // reduction, 1)), nn.ReLU(),
            nn.Linear(max(dim // reduction, 1), dim), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3, use_se=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim)
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
            nn.Linear(input_dim, hidden_dims[0]), nn.BatchNorm1d(hidden_dims[0]), nn.GELU(), nn.Dropout(drop_rate)
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dims[0], drop_rate) for _ in range(n_res_blocks)])
        self.down_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.down_layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.BatchNorm1d(hidden_dims[i+1]), nn.GELU(), nn.Dropout(drop_rate)
            ))
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
    def forward(self, x):
        x = self.input_proj(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        for down_layer in self.down_layers:
            x = down_layer(x)
        return self.output_layer(x).squeeze(-1)

class EnhancedDeepHit(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], num_durations=10, drop_rate=0.3):
        super().__init__()
        layers = []
        in_d = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_d, h_dim), nn.BatchNorm1d(h_dim), nn.GELU(), nn.Dropout(drop_rate)])
            in_d = h_dim
        layers.append(nn.Linear(in_d, num_durations))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)

class EnhancedDenoisingAE(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], latent_dim=64, dropout=0.2):
        super().__init__()
        encoder_layers = []
        in_d = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([nn.Linear(in_d, h_dim), nn.BatchNorm1d(h_dim), nn.GELU(), nn.Dropout(dropout)])
            in_d = h_dim
        encoder_layers.append(nn.Linear(in_d, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        decoder_layers = []
        in_d = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(in_d, h_dim), nn.BatchNorm1d(h_dim), nn.GELU(), nn.Dropout(dropout)])
            in_d = h_dim
        decoder_layers.append(nn.Linear(in_d, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    def encode(self, x):
        return self.encoder(x)
    def forward(self, x, noise_factor=0.1):
        z = self.encoder(x)
        return self.decoder(z), z

class EnhancedTransformer(nn.Module):
    def __init__(self, latent_dim, n_heads=4, ff_dim=256, n_layers=2, dropout=0.1):
        super().__init__()
        while latent_dim % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        self.input_norm = nn.LayerNorm(latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU(), nn.Dropout(dropout))
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
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


# ================== 工具函数 ==================
def get_text(key: str, lang: str) -> str:
    return TRANSLATIONS.get(key, {}).get(lang, key)

def get_var_label(var_name: str, lang: str) -> str:
    return INPUT_VARIABLES.get(var_name, {}).get(lang, var_name)

def get_option_label(var_name: str, option_key: str, lang: str) -> str:
    var_info = INPUT_VARIABLES.get(var_name, {})
    options = var_info.get("options", {})
    return options.get(option_key, {}).get(lang, option_key)

def encode_option(var_name: str, option_key: str) -> float:
    var_info = INPUT_VARIABLES.get(var_name, {})
    options = var_info.get("options", {})
    if options:
        keys = list(options.keys())
        try:
            return float(keys.index(option_key))
        except ValueError:
            return 0.0
    return 0.0

@st.cache_resource
def load_models(model_dir="results_clinical_enhanced_v3"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    use_pretrained = False
    
    try:
        required_files = ['model_ae.pt', 'model_trans.pt', 'model_deepsurv.pt', 'model_deephit.pt', 
                         'model_fusion.pt', 'preprocessor.joblib', 'time_cuts.npy', 'ds_min_max.npy', 'best_parameters.json']
        all_exist = all(os.path.exists(os.path.join(model_dir, f)) for f in required_files)
        
        if all_exist:
            with open(os.path.join(model_dir, "best_parameters.json"), "r") as f:
                params = json.load(f)
            preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))
            time_cuts = np.load(os.path.join(model_dir, "time_cuts.npy"))
            num_bins = len(time_cuts) - 1
            ds_min_max = np.load(os.path.join(model_dir, "ds_min_max.npy"))
            fcm_path = os.path.join(model_dir, "fcm_centers.npy")
            fcm_centers = np.load(fcm_path) if os.path.exists(fcm_path) else np.array([[0.3, 0.3], [0.7, 0.7]])
            
            input_dim = preprocessor.scaler.n_features_in_
            if hasattr(preprocessor, 'selector') and preprocessor.selector is not None:
                input_dim = getattr(preprocessor.selector, 'k', input_dim)
            
            latent_dim = params.get('ae_latent', 64)
            fused_dim = latent_dim * 2
            
            ae = EnhancedDenoisingAE(input_dim, [params.get('ae_h1', 256), params.get('ae_h2', 128)], latent_dim)
            ae.load_state_dict(torch.load(os.path.join(model_dir, "model_ae.pt"), map_location=device))
            ae.eval()
            
            trans = EnhancedTransformer(latent_dim)
            trans.load_state_dict(torch.load(os.path.join(model_dir, "model_trans.pt"), map_location=device))
            trans.eval()
            
            ds = EnhancedDeepSurv(fused_dim, [params.get('ds_h1', 256), params.get('ds_h2', 128), params.get('ds_h3', 64)], drop_rate=params.get('ds_drop', 0.3))
            ds.load_state_dict(torch.load(os.path.join(model_dir, "model_deepsurv.pt"), map_location=device))
            ds.eval()
            
            dh = EnhancedDeepHit(fused_dim, [params.get('dh_h1', 256), params.get('dh_h2', 128)], num_durations=num_bins)
            dh.load_state_dict(torch.load(os.path.join(model_dir, "model_deephit.pt"), map_location=device))
            dh.eval()
            
            fusion = LearnableFusion()
            fusion.load_state_dict(torch.load(os.path.join(model_dir, "model_fusion.pt"), map_location=device))
            fusion.eval()
            
            models = {'ae': ae.to(device), 'trans': trans.to(device), 'ds': ds.to(device), 'dh': dh.to(device),
                     'fusion': fusion.to(device), 'preprocessor': preprocessor, 'time_cuts': time_cuts,
                     'ds_min_max': ds_min_max, 'fcm_centers': fcm_centers, 'params': params, 'device': device, 'input_dim': input_dim}
            use_pretrained = True
    except Exception as e:
        use_pretrained = False
    
    if not use_pretrained:
        input_dim = len(INPUT_VARIABLES)
        latent_dim, fused_dim, num_bins = 64, 128, 10
        models = {'ae': EnhancedDenoisingAE(input_dim, [256, 128], latent_dim).to(device),
                 'trans': EnhancedTransformer(latent_dim).to(device),
                 'ds': EnhancedDeepSurv(fused_dim, [256, 128, 64]).to(device),
                 'dh': EnhancedDeepHit(fused_dim, [256, 128], num_bins).to(device),
                 'fusion': LearnableFusion().to(device), 'preprocessor': None,
                 'time_cuts': np.linspace(0, 120, num_bins + 1), 'ds_min_max': np.array([-5.0, 5.0]),
                 'fcm_centers': np.array([[0.3, 0.3], [0.7, 0.7]]), 'params': {}, 'device': device, 'input_dim': input_dim}
        for key in ['ae', 'trans', 'ds', 'dh', 'fusion']:
            models[key].eval()
    
    models['use_pretrained'] = use_pretrained
    return models

def preprocess_input(input_data: Dict, models: Dict) -> np.ndarray:
    feature_values = []
    for var_name in INPUT_VARIABLES.keys():
        value = input_data.get(var_name, None)
        var_info = INPUT_VARIABLES[var_name]
        if var_info['type'] == 'select':
            feature_values.append(encode_option(var_name, value) if value else 0.0)
        elif var_info['type'] == 'number':
            feature_values.append(float(value) if value is not None else float(var_info.get('default', 0)))
        else:
            feature_values.append(0.0)
    X = np.array(feature_values).reshape(1, -1)
    if models.get('preprocessor') is not None:
        try:
            X = models['preprocessor'].transform(X)
        except:
            X = (X - X.mean()) / (X.std() + 1e-8)
    else:
        X = (X - X.mean()) / (X.std() + 1e-8)
    return X

def predict_single(input_data: Dict, models: Dict) -> Dict:
    device = models['device']
    X = preprocess_input(input_data, models)
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        Z = models['ae'].encode(X_tensor)
        T = models['trans'](Z)
        Xf = torch.cat([Z, T], dim=1)
        risk_ds = models['ds'](Xf).cpu().numpy()
        risk_ds = risk_ds.item() if risk_ds.ndim == 0 else risk_ds[0]
        pmf = models['dh'](Xf).cpu().numpy()[0]
        min_ds, max_ds = models['ds_min_max']
        prob_ds = np.clip((risk_ds - min_ds) / (max_ds - min_ds + 1e-8), 0, 1)
        cif = np.cumsum(pmf)
        survival = 1 - cif
        target_bin = len(pmf) // 2
        risk_dh = cif[target_bin]
        fusion_input = torch.tensor([[prob_ds, risk_dh]], dtype=torch.float32, device=device)
        final_risk = models['fusion'](fusion_input).cpu().numpy()
        final_risk = final_risk.item() if final_risk.ndim == 0 else final_risk[0]
    
    time_cuts = models['time_cuts']
    time_points = (time_cuts[:-1] + time_cuts[1:]) / 2
    n_bins = len(cif)
    
    return {
        'final_risk': float(final_risk), 'risk_deepsurv': float(prob_ds), 'risk_deephit': float(risk_dh),
        'pmf': pmf, 'cif': cif, 'survival': survival, 'time_points': time_points,
        'risk_12m': float(cif[min(int(n_bins * 0.1), n_bins-1)]),
        'risk_36m': float(cif[min(int(n_bins * 0.3), n_bins-1)]),
        'risk_60m': float(cif[min(int(n_bins * 0.5), n_bins-1)])
    }

def predict_batch(df: pd.DataFrame, models: Dict, lang: str) -> pd.DataFrame:
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in df.iterrows():
        status_text.text(f"{'正在处理' if lang == 'zh' else 'Processing'} {idx + 1}/{len(df)}...")
        input_data = {}
        for var_name in INPUT_VARIABLES.keys():
            for col_lang in ['zh', 'en']:
                col_name = INPUT_VARIABLES[var_name][col_lang]
                if col_name in row:
                    input_data[var_name] = row[col_name]
                    break
            if var_name not in input_data and var_name in row:
                input_data[var_name] = row[var_name]
        
        try:
            pred = predict_single(input_data, models)
            risk_level = get_text("low_risk" if pred['final_risk'] < 0.3 else ("medium_risk" if pred['final_risk'] < 0.6 else "high_risk"), lang)
            results.append({
                get_text("patient_id", lang): row.get('patient_id', row.get('患者ID', idx + 1)),
                get_text("overall_risk", lang): f"{pred['final_risk']*100:.1f}%",
                f"12{get_text('months', lang)}": f"{pred['risk_12m']*100:.1f}%",
                f"36{get_text('months', lang)}": f"{pred['risk_36m']*100:.1f}%",
                f"60{get_text('months', lang)}": f"{pred['risk_60m']*100:.1f}%",
                get_text("risk_level", lang): risk_level,
                '_final_risk_value': pred['final_risk']
            })
        except:
            results.append({
                get_text("patient_id", lang): row.get('patient_id', row.get('患者ID', idx + 1)),
                get_text("overall_risk", lang): "Error", f"12{get_text('months', lang)}": "N/A",
                f"36{get_text('months', lang)}": "N/A", f"60{get_text('months', lang)}": "N/A",
                get_text("risk_level", lang): "Error", '_final_risk_value': 0
            })
        progress_bar.progress((idx + 1) / len(df))
    
    status_text.empty()
    progress_bar.empty()
    return pd.DataFrame(results)

def create_template_csv(lang: str) -> pd.DataFrame:
    columns = [get_text("patient_id", lang)]
    for var_name, var_info in INPUT_VARIABLES.items():
        columns.append(var_info[lang])
    sample_data = {columns[0]: [1, 2, 3]}
    for i, (var_name, var_info) in enumerate(INPUT_VARIABLES.items()):
        col_name = columns[i + 1]
        if var_info['type'] == 'select':
            options = list(var_info['options'].keys())
            sample_data[col_name] = [options[0]] * 3
        else:
            sample_data[col_name] = [var_info.get('default', 0)] * 3
    return pd.DataFrame(sample_data)


# ================== PDF生成 ==================
def generate_pdf_report(results_df: pd.DataFrame, lang: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=18, spaceAfter=30, alignment=1)
    story.append(Paragraph("Cancer Recurrence Risk Prediction Report", title_style))
    story.append(Paragraph("Shengjing Hospital of China Medical University", styles['Normal']))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    total = len(results_df)
    risk_col = get_text("risk_level", lang)
    high_risk = len(results_df[results_df[risk_col].str.contains('High|高', case=False, na=False)]) if risk_col in results_df.columns else 0
    medium_risk = len(results_df[results_df[risk_col].str.contains('Medium|中', case=False, na=False)]) if risk_col in results_df.columns else 0
    low_risk = len(results_df[results_df[risk_col].str.contains('Low|低', case=False, na=False)]) if risk_col in results_df.columns else 0
    
    summary_data = [["Metric", "Value"], ["Total Patients", str(total)],
                   ["High Risk", f"{high_risk} ({high_risk/total*100:.1f}%)" if total > 0 else "0"],
                   ["Medium Risk", f"{medium_risk} ({medium_risk/total*100:.1f}%)" if total > 0 else "0"],
                   ["Low Risk", f"{low_risk} ({low_risk/total*100:.1f}%)" if total > 0 else "0"]]
    
    summary_table = Table(summary_data, colWidths=[200, 200])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12), ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')), ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 30))
    story.append(Paragraph("Disclaimer: This report is for reference only.", ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=8, textColor=colors.grey)))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def generate_single_pdf_report(patient_data: Dict, results: Dict, lang: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=18, spaceAfter=30, alignment=1)
    story.append(Paragraph("Patient Risk Assessment Report", title_style))
    story.append(Paragraph("Shengjing Hospital of China Medical University", styles['Normal']))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    risk = results['final_risk']
    risk_level = "Low Risk" if risk < 0.3 else ("Medium Risk" if risk < 0.6 else "High Risk")
    
    results_data = [["Metric", "Value"], ["Overall Risk", f"{risk*100:.1f}%"], ["Risk Level", risk_level],
                   ["12-month Risk", f"{results['risk_12m']*100:.1f}%"],
                   ["36-month Risk", f"{results['risk_36m']*100:.1f}%"],
                   ["60-month Risk", f"{results['risk_60m']*100:.1f}%"]]
    
    results_table = Table(results_data, colWidths=[200, 200])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12), ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')])
    ]))
    story.append(results_table)
    story.append(Spacer(1, 30))
    story.append(Paragraph("Disclaimer: This report is for reference only.", ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=8, textColor=colors.grey)))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ================== 可视化函数 ==================
def create_gauge_chart(risk: float, lang: str) -> go.Figure:
    if risk < 0.3:
        color, risk_text = "#28a745", get_text("low_risk", lang)
    elif risk < 0.6:
        color, risk_text = "#ffc107", get_text("medium_risk", lang)
    else:
        color, risk_text = "#dc3545", get_text("high_risk", lang)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=risk * 100, domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{get_text('overall_risk', lang)}<br><span style='font-size:0.9em;color:{color}'>{risk_text}</span>", 'font': {'size': 16}},
        number={'suffix': '%', 'font': {'size': 50, 'color': color}},
        gauge={'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
               'bar': {'color': color, 'thickness': 0.75},
               'bgcolor': 'white', 'borderwidth': 2, 'bordercolor': "gray",
               'steps': [{'range': [0, 30], 'color': 'rgba(40, 167, 69, 0.3)'},
                        {'range': [30, 60], 'color': 'rgba(255, 193, 7, 0.3)'},
                        {'range': [60, 100], 'color': 'rgba(220, 53, 69, 0.3)'}],
               'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': risk * 100}}
    ))
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=60, b=30), paper_bgcolor='rgba(0,0,0,0)', font={'family': "Arial"})
    return fig

def create_survival_curve(survival: np.ndarray, time_points: np.ndarray, lang: str) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=(get_text("survival_probability", lang), get_text("cumulative_risk", lang)))
    
    fig.add_trace(go.Scatter(x=time_points, y=survival, mode='lines+markers', name=get_text("survival_probability", lang),
                            line=dict(color='#667eea', width=3), fill='tozeroy', fillcolor='rgba(102, 126, 234, 0.2)',
                            marker=dict(size=8, color='#667eea')), row=1, col=1)
    
    cif = 1 - survival
    fig.add_trace(go.Scatter(x=time_points, y=cif, mode='lines+markers', name=get_text("cumulative_risk", lang),
                            line=dict(color='#dc3545', width=3), fill='tozeroy', fillcolor='rgba(220, 53, 69, 0.2)',
                            marker=dict(size=8, color='#dc3545')), row=1, col=2)
    
    time_label = f"{get_text('time', lang)} ({get_text('months', lang).strip()})"
    fig.update_xaxes(title_text=time_label, row=1, col=1, gridcolor='#f0f0f0')
    fig.update_xaxes(title_text=time_label, row=1, col=2, gridcolor='#f0f0f0')
    fig.update_yaxes(title_text=get_text("probability", lang), range=[0, 1], row=1, col=1, gridcolor='#f0f0f0')
    fig.update_yaxes(title_text=get_text("probability", lang), range=[0, 1], row=1, col=2, gridcolor='#f0f0f0')
    fig.update_layout(height=380, showlegend=False, margin=dict(l=60, r=60, t=60, b=60), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def create_time_risk_bar(risk_12m: float, risk_36m: float, risk_60m: float, lang: str) -> go.Figure:
    months = get_text("months", lang)
    labels = [f"12{months}", f"36{months}", f"60{months}"]
    values = [risk_12m * 100, risk_36m * 100, risk_60m * 100]
    colors = ['#28a745' if v < 30 else ('#ffc107' if v < 60 else '#dc3545') for v in values]
    
    fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=colors, text=[f'{v:.1f}%' for v in values],
                                 textposition='outside', textfont=dict(size=14, color='#333'))])
    fig.update_layout(title=dict(text=get_text("time_risk", lang), font=dict(size=16)),
                     yaxis_title=f"{get_text('probability', lang)} (%)", yaxis_range=[0, 100],
                     height=320, margin=dict(l=60, r=40, t=60, b=40), paper_bgcolor='rgba(0,0,0,0)',
                     plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(gridcolor='#f0f0f0'))
    return fig

def create_risk_distribution_chart(results_df: pd.DataFrame, lang: str) -> go.Figure:
    risk_col = get_text("risk_level", lang)
    if risk_col in results_df.columns:
        high = len(results_df[results_df[risk_col].str.contains('High|高', case=False, na=False)])
        medium = len(results_df[results_df[risk_col].str.contains('Medium|中', case=False, na=False)])
        low = len(results_df[results_df[risk_col].str.contains('Low|低', case=False, na=False)])
    else:
        high = medium = low = 0
    
    fig = go.Figure(data=[go.Pie(labels=[get_text("low_risk", lang), get_text("medium_risk", lang), get_text("high_risk", lang)],
                                values=[low, medium, high], marker_colors=['#28a745', '#ffc107', '#dc3545'],
                                hole=0.5, textinfo='label+percent+value', textfont=dict(size=12),
                                pull=[0, 0, 0.05])])
    fig.update_layout(title=dict(text=get_text("risk_distribution", lang), font=dict(size=16)),
                     height=380, margin=dict(l=20, r=20, t=60, b=20), paper_bgcolor='rgba(0,0,0,0)',
                     legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5))
    return fig


# ================== 输入控件 ==================
def render_select_widget(var_name: str, var_info: Dict, lang: str, key_prefix: str = "") -> str:
    label = var_info[lang]
    options = var_info.get('options', {})
    option_keys = list(options.keys())
    format_func = lambda x: options[x][lang]
    return st.selectbox(label, options=option_keys, format_func=format_func, key=f"{key_prefix}{var_name}")

def render_number_widget(var_name: str, var_info: Dict, lang: str, key_prefix: str = "") -> float:
    label = var_info[lang]
    if 'unit' in var_info:
        label = f"{label} ({var_info['unit'][lang]})"
    return st.number_input(label, min_value=float(var_info.get('min', 0)), max_value=float(var_info.get('max', 100)),
                          value=float(var_info.get('default', 0)), key=f"{key_prefix}{var_name}")


# ================== 主应用 ==================
def main():
    models = load_models()
    
    # 语言选择
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        language = st.selectbox("🌐", options=list(LANGUAGES.keys()), index=0, key="lang_selector", label_visibility="collapsed")
    lang = LANGUAGES[language]
    
    # 主标题
    st.markdown(f"""
    <div class="main-title">
        <h1>{get_text('title', lang)}</h1>
        <h3>{get_text('subtitle', lang)}</h3>
        <p class="hospital">{get_text('hospital', lang)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 模型状态
    if models.get('use_pretrained', False):
        st.markdown(f'<div class="success-box">{get_text("model_loaded", lang)}</div>', unsafe_allow_html=True)
    
    # 选项卡
    tab1, tab2 = st.tabs([get_text("single_patient", lang), get_text("batch_prediction", lang)])
    
    # ==================== 单个患者预测 ====================
    with tab1:
        st.markdown(f'<div class="section-header">{get_text("patient_info", lang)}</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        input_data = {}
        
        with col1:
            st.markdown(f'<div class="section-header">{get_text("basic_info", lang)}</div>', unsafe_allow_html=True)
            for var_name in ['age', 'family_cancer_history', 'sexual_history', 'parity', 'menopausal_status', 
                           'comorbidities', 'smoking_drinking_history', 'receive_estrogens', 'ovulation_induction']:
                if var_name in INPUT_VARIABLES:
                    var_info = INPUT_VARIABLES[var_name]
                    input_data[var_name] = render_select_widget(var_name, var_info, lang, "s_") if var_info['type'] == 'select' else render_number_widget(var_name, var_info, lang, "s_")
        
        with col2:
            st.markdown(f'<div class="section-header">{get_text("surgical_info", lang)}</div>', unsafe_allow_html=True)
            for var_name in ['presenting_symptom', 'surgical_route', 'tumor_envelope_integrity', 'fertility_sparing_surgery',
                           'completeness_of_surgery', 'omentectomy', 'lymphadenectomy', 'postoperative_adjuvant_therapy']:
                if var_name in INPUT_VARIABLES:
                    var_info = INPUT_VARIABLES[var_name]
                    input_data[var_name] = render_select_widget(var_name, var_info, lang, "s_") if var_info['type'] == 'select' else render_number_widget(var_name, var_info, lang, "s_")
        
        with col3:
            st.markdown(f'<div class="section-header">{get_text("pathology_info", lang)}</div>', unsafe_allow_html=True)
            for var_name in ['histological_subtype', 'micropapillary', 'microinfiltration', 'psammoma_bodies_calcification',
                           'peritoneal_implantation', 'ascites_cytology', 'figo_staging', 'unilateral_or_bilateral',
                           'tumor_size', 'type_of_lesion', 'papillary_area_ratio']:
                if var_name in INPUT_VARIABLES:
                    var_info = INPUT_VARIABLES[var_name]
                    input_data[var_name] = render_select_widget(var_name, var_info, lang, "s_") if var_info['type'] == 'select' else render_number_widget(var_name, var_info, lang, "s_")
        
        st.markdown(f'<div class="section-header">{get_text("tumor_markers", lang)}</div>', unsafe_allow_html=True)
        marker_cols = st.columns(6)
        for i, var_name in enumerate(['ca125', 'cea', 'ca199', 'afp', 'ca724', 'he4']):
            with marker_cols[i]:
                input_data[var_name] = render_select_widget(var_name, INPUT_VARIABLES[var_name], lang, "s_")
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            predict_clicked = st.button(get_text("predict_button", lang), type="primary", use_container_width=True, key="single_predict")
        
        if predict_clicked:
            with st.spinner(get_text("processing", lang)):
                results = predict_single(input_data, models)
                
                st.markdown("---")
                st.markdown(f'<div class="section-header">{get_text("prediction_results", lang)}</div>', unsafe_allow_html=True)
                
                result_col1, result_col2 = st.columns([1, 2])
                with result_col1:
                    st.plotly_chart(create_gauge_chart(results['final_risk'], lang), use_container_width=True)
                    st.plotly_chart(create_time_risk_bar(results['risk_12m'], results['risk_36m'], results['risk_60m'], lang), use_container_width=True)
                
                with result_col2:
                    st.markdown(f'<div class="section-header">{get_text("survival_curve", lang)}</div>', unsafe_allow_html=True)
                    st.plotly_chart(create_survival_curve(results['survival'], results['time_points'], lang), use_container_width=True)
                
                # 临床建议
                st.markdown("---")
                st.markdown(f'<div class="section-header">{get_text("clinical_advice", lang)}</div>', unsafe_allow_html=True)
                
                risk = results['final_risk']
                if risk < 0.3:
                    risk_level, advice_key, css_class = "low_risk", "advice_low", "risk-low"
                elif risk < 0.6:
                    risk_level, advice_key, css_class = "medium_risk", "advice_medium", "risk-medium"
                else:
                    risk_level, advice_key, css_class = "high_risk", "advice_high", "risk-high"
                
                st.markdown(f"""
                <div class="info-card {css_class}">
                    <h3>{get_text('risk_level', lang)}: {get_text(risk_level, lang)} ({risk*100:.1f}%)</h3>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(get_text(advice_key, lang))
                
                # 导出
                st.markdown("---")
                st.markdown(f'<div class="section-header">{get_text("export_results", lang)}</div>', unsafe_allow_html=True)
                export_col1, export_col2 = st.columns(2)
                
                with export_col1:
                    if lang == 'zh':
                        detail_df = pd.DataFrame({'指标': ['综合风险', '12个月风险', '36个月风险', '60个月风险'],
                                                 '数值': [f"{results['final_risk']*100:.2f}%", f"{results['risk_12m']*100:.2f}%",
                                                         f"{results['risk_36m']*100:.2f}%", f"{results['risk_60m']*100:.2f}%"]})
                    else:
                        detail_df = pd.DataFrame({'Metric': ['Overall Risk', '12-Month Risk', '36-Month Risk', '60-Month Risk'],
                                                 'Value': [f"{results['final_risk']*100:.2f}%", f"{results['risk_12m']*100:.2f}%",
                                                          f"{results['risk_36m']*100:.2f}%", f"{results['risk_60m']*100:.2f}%"]})
                    
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        detail_df.to_excel(writer, sheet_name='Results', index=False)
                    st.download_button(get_text("export_excel", lang), data=excel_buffer.getvalue(),
                                      file_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                      mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                
                with export_col2:
                    st.download_button(get_text("export_pdf", lang), data=generate_single_pdf_report(input_data, results, lang),
                                      file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
    
    # ==================== 批量预测 ====================
    with tab2:
        st.markdown(f'<div class="section-header">{get_text("step1", lang)}</div>', unsafe_allow_html=True)
        template_df = create_template_csv(lang)
        csv_buffer = io.StringIO()
        template_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        st.download_button(get_text("download_template", lang), data=csv_buffer.getvalue(),
                          file_name=f"template_{lang}.csv", mime="text/csv")
        
        with st.expander(get_text("preview_template", lang)):
            st.dataframe(template_df, use_container_width=True)
        
        st.markdown("---")
        st.markdown(f'<div class="section-header">{get_text("step2", lang)}</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(get_text("upload_csv", lang), type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                st.success(f"✅ {get_text('loaded_patients', lang)}: {len(df)}")
                
                with st.expander(get_text("preview_data", lang)):
                    st.dataframe(df.head(10), use_container_width=True)
                
                if st.button(get_text("predict_button", lang), type="primary", key="batch_predict"):
                    with st.spinner(get_text("processing", lang)):
                        results_df = predict_batch(df, models, lang)
                        
                        st.markdown("---")
                        st.markdown(f'<div class="section-header">{get_text("batch_results", lang)}</div>', unsafe_allow_html=True)
                        
                        total = len(results_df)
                        risk_col = get_text("risk_level", lang)
                        high_count = len(results_df[results_df[risk_col].str.contains('High|高', case=False, na=False)]) if risk_col in results_df.columns else 0
                        medium_count = len(results_df[results_df[risk_col].str.contains('Medium|中', case=False, na=False)]) if risk_col in results_df.columns else 0
                        low_count = len(results_df[results_df[risk_col].str.contains('Low|低', case=False, na=False)]) if risk_col in results_df.columns else 0
                        
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric(get_text("total_patients", lang), total)
                        m2.metric(get_text("high_risk_count", lang), high_count)
                        m3.metric(get_text("medium_risk_count", lang), medium_count)
                        m4.metric(get_text("low_risk_count", lang), low_count)
                        
                        chart_col1, chart_col2 = st.columns(2)
                        with chart_col1:
                            st.plotly_chart(create_risk_distribution_chart(results_df, lang), use_container_width=True)
                        with chart_col2:
                            if '_final_risk_value' in results_df.columns:
                                risk_values = results_df['_final_risk_value'].values * 100
                                hist_fig = go.Figure(data=[go.Histogram(x=risk_values, nbinsx=20, marker_color='#667eea', opacity=0.75)])
                                hist_fig.add_vline(x=30, line_dash="dash", line_color="#28a745")
                                hist_fig.add_vline(x=60, line_dash="dash", line_color="#dc3545")
                                hist_fig.update_layout(title=get_text("risk_score_dist", lang), xaxis_title=f"{get_text('probability', lang)} (%)",
                                                      yaxis_title=get_text("total_patients", lang), height=380, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                                st.plotly_chart(hist_fig, use_container_width=True)
                        
                        st.markdown(f'<div class="section-header">{get_text("detailed_results", lang)}</div>', unsafe_allow_html=True)
                        display_df = results_df.drop(columns=[c for c in results_df.columns if c.startswith('_')], errors='ignore')
                        
                        def highlight_risk(row):
                            risk_col = get_text("risk_level", lang)
                            if risk_col in row:
                                val = str(row[risk_col])
                                if 'High' in val or '高' in val:
                                    return ['background-color: #f8d7da'] * len(row)
                                elif 'Medium' in val or '中' in val:
                                    return ['background-color: #fff3cd'] * len(row)
                                else:
                                    return ['background-color: #d4edda'] * len(row)
                            return [''] * len(row)
                        
                        st.dataframe(display_df.style.apply(highlight_risk, axis=1), use_container_width=True, height=400)
                        
                        st.markdown("---")
                        st.markdown(f'<div class="section-header">{get_text("export_results", lang)}</div>', unsafe_allow_html=True)
                        e1, e2, e3 = st.columns(3)
                        
                        with e1:
                            csv_export = io.StringIO()
                            display_df.to_csv(csv_export, index=False, encoding='utf-8-sig')
                            st.download_button(get_text("export_csv", lang), data=csv_export.getvalue(),
                                             file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
                        with e2:
                            excel_buf = io.BytesIO()
                            with pd.ExcelWriter(excel_buf, engine='openpyxl') as writer:
                                display_df.to_excel(writer, sheet_name='Results', index=False)
                            st.download_button(get_text("export_excel", lang), data=excel_buf.getvalue(),
                                             file_name=f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                        with e3:
                            st.download_button(get_text("export_pdf", lang), data=generate_pdf_report(results_df, lang),
                                             file_name=f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
                        
                        if high_count > 0:
                            st.markdown("---")
                            st.markdown(f'<div class="section-header">{get_text("high_risk_attention", lang)}</div>', unsafe_allow_html=True)
                            high_risk_df = display_df[display_df[risk_col].str.contains('High|高', case=False, na=False)]
                            st.dataframe(high_risk_df.style.apply(lambda x: ['background-color: #f8d7da'] * len(x), axis=1), use_container_width=True)
                            st.warning(f"⚠️ {high_count} {get_text('high_risk_warning', lang)}")
            
            except Exception as e:
                st.error(f"{get_text('file_error', lang)}: {str(e)}")
                st.info(get_text("file_format_hint", lang))
    
    # 页脚
    st.markdown("---")
    st.info(get_text("disclaimer", lang))
    st.markdown(f"""
    <div class="footer">
        <p class="hospital-name">{get_text('hospital', lang)}</p>
        <p>Cancer Recurrence Risk Prediction System v3.0</p>
        <p>© 2024 All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
