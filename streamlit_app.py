"""
肿瘤复发风险预测临床决策支持系统 - 修复版
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
    page_title="肿瘤复发预测系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================== CSS样式 ==================
st.markdown("""
<style>
    [data-testid="collapsedControl"] {display: none}
    section[data-testid="stSidebar"] {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .main .block-container {padding: 0.5rem 1.5rem 2rem 1.5rem; max-width: 100%;}
    .status-box {padding: 10px 15px; border-radius: 8px; margin-bottom: 10px; font-weight: 500;}
    .status-success {background: linear-gradient(90deg, #d4edda, #c3e6cb); border-left: 5px solid #28a745; color: #155724;}
    .status-error {background: linear-gradient(90deg, #f8d7da, #f5c6cb); border-left: 5px solid #dc3545; color: #721c24;}
    .module-card {background: #ffffff; border-radius: 8px; padding: 0.8rem; margin-bottom: 0.8rem; box-shadow: 0 2px 6px rgba(0,0,0,0.05); border: 1px solid #e8e8e8;}
    .module-title {background: linear-gradient(90deg, #3498db, #2980b9); color: white; padding: 0.4rem 0.6rem; border-radius: 5px; margin: -0.8rem -0.8rem 0.6rem -0.8rem; font-weight: 600; font-size: 0.85rem;}
    .module-title.pathology {background: linear-gradient(90deg, #9b59b6, #8e44ad);}
    .module-title.surgery {background: linear-gradient(90deg, #e67e22, #d35400);}
    .module-title.markers {background: linear-gradient(90deg, #1abc9c, #16a085);}
    .result-section {background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 12px; padding: 1.2rem; margin: 1rem 0; border: 2px solid #dee2e6;}
    .chart-container {background: white; border-radius: 10px; padding: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border: 1px solid #e8e8e8;}
    .advice-box {background: white; border-radius: 10px; padding: 1rem; margin: 1rem 0; border-left: 5px solid; box-shadow: 0 2px 8px rgba(0,0,0,0.06);}
    .advice-box.low {border-color: #28a745; background: linear-gradient(90deg, #f0fff0, white);}
    .advice-box.medium {border-color: #ffc107; background: linear-gradient(90deg, #fffef0, white);}
    .advice-box.high {border-color: #dc3545; background: linear-gradient(90deg, #fff0f0, white);}
    .stButton > button {background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); color: white; border: none; padding: 0.6rem 2rem; font-size: 1rem; font-weight: 600; border-radius: 25px;}
    .hospital-header {background: linear-gradient(135deg, #1a5276 0%, #2980b9 50%, #1a5276 100%); padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1rem; text-align: center; color: white;}
    .hospital-header h1 {font-size: 1.6rem; margin: 0 0 0.3rem 0;}
    .hospital-header .subtitle {font-size: 0.95rem; opacity: 0.9;}
    .hospital-header .hospital-name {color: #f1c40f; font-size: 0.9rem; font-weight: 600; margin-top: 0.4rem;}
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
                  "en": "⚠️ Note: Predictions are for clinical reference only."},
    "months": {"zh": "月", "en": "M"},
    "time_months": {"zh": "时间（月）", "en": "Time (Months)"},
    "survival_prob": {"zh": "生存概率", "en": "Survival Probability"},
    "risk_prob": {"zh": "复发概率", "en": "Recurrence Probability"},
    "month_12": {"zh": "12个月", "en": "12M"},
    "month_36": {"zh": "36个月", "en": "36M"},
    "month_60": {"zh": "60个月", "en": "60M"},
    "processing": {"zh": "正在评估中...", "en": "Assessing..."},
    "export_results": {"zh": "导出报告", "en": "Export Report"},
    "export_excel": {"zh": "导出Excel", "en": "Export Excel"},
    "export_pdf": {"zh": "导出PDF", "en": "Export PDF"},
    "debug_info": {"zh": "调试信息", "en": "Debug Info"},
    "advice_low": {
        "zh": "• 常规随访：每6个月复查\n• 影像检查：每年盆腔超声\n• 标志物：每6个月CA125、HE4",
        "en": "• Routine follow-up: Every 6 months\n• Imaging: Annual pelvic ultrasound\n• Markers: CA125, HE4 every 6 months"
    },
    "advice_medium": {
        "zh": "• 加强随访：每3-4个月复查\n• 影像检查：每6个月CT/MRI\n• 标志物：每3个月检测\n• 评估辅助治疗必要性",
        "en": "• Enhanced follow-up: Every 3-4 months\n• Imaging: CT/MRI every 6 months\n• Markers: Every 3 months"
    },
    "advice_high": {
        "zh": "• 密切随访：每2-3个月复查\n• 影像检查：每3个月CT/MRI\n• 标志物：每6-8周检测\n• 强烈建议辅助化疗\n• 建议MDT多学科会诊",
        "en": "• Close follow-up: Every 2-3 months\n• Imaging: CT/MRI every 3 months\n• Adjuvant chemo recommended\n• MDT consultation advised"
    }
}

# ================== 特征顺序（与训练数据完全一致）==================
FEATURE_ORDER = [
    'age',
    'family cancer history',
    'sexual history',
    'parity',
    'menopausal status',
    'comorbidities',
    'presenting symptom',
    'surgical route',
    'tumor envelope integrity',
    'fertility-sparing surgery',
    'completeness of surgery',
    'omentectomy',
    'lymphadenectomy',
    'histological subtype',
    'micropapillary',
    'microinfiltration',
    'psammoma bodies and calcification',
    'peritoneal implantation',
    'ascites cytology',
    'FIGO staging',
    'unilateral or bilateral',
    'tumor size',
    'CA125',
    'CEA',
    'CA199',
    'AFP',
    'CA724',
    'HE4',
    'smoking and drinking history',
    'receive estrogens',
    'ovulation induction',
    'postoperative adjuvant therapy',
    'type of lesion',
    'papillary area ratio',
]

# ================== 输入变量定义（编码与训练数据一致）==================
INPUT_VARIABLES = {
    # 0: age - 二分类 (0: ≤40, 1: >40)
    "age": {
        "zh": "年龄", "en": "Age", "type": "select", "default": 1,
        "options": {0: {"zh": "≤40岁", "en": "≤40 years"}, 1: {"zh": ">40岁", "en": ">40 years"}}
    },
    # 1: family cancer history - 二分类 (0: 否, 1: 是)
    "family cancer history": {
        "zh": "家族史", "en": "Family Cancer History", "type": "select", "default": 1,
        "options": {0: {"zh": "否", "en": "No"}, 1: {"zh": "是", "en": "Yes"}}
    },
    # 2: sexual history - 二分类 (0: 否, 1: 是)
    "sexual history": {
        "zh": "性生活史", "en": "Sexual History", "type": "select", "default": 1,
        "options": {0: {"zh": "否", "en": "No"}, 1: {"zh": "是", "en": "Yes"}}
    },
    # 3: parity - 二分类 (0: 否, 1: 是)
    "parity": {
        "zh": "生育", "en": "Parity", "type": "select", "default": 1,
        "options": {0: {"zh": "否", "en": "No"}, 1: {"zh": "是", "en": "Yes"}}
    },
    # 4: menopausal status - 二分类 (0: 否, 1: 是)
    "menopausal status": {
        "zh": "绝经", "en": "Menopausal Status", "type": "select", "default": 1,
        "options": {0: {"zh": "否", "en": "No"}, 1: {"zh": "是", "en": "Yes"}}
    },
    # 5: comorbidities - 二分类 (0: 否, 1: 是)
    "comorbidities": {
        "zh": "内科疾病", "en": "Comorbidities", "type": "select", "default": 1,
        "options": {0: {"zh": "否", "en": "No"}, 1: {"zh": "是", "en": "Yes"}}
    },
    # 6: presenting symptom - 三分类 (1, 2, 3) 注意从1开始！
    "presenting symptom": {
        "zh": "症状", "en": "Presenting Symptom", "type": "select", "default": 3,
        "options": {
            1: {"zh": "体检发现", "en": "Physical Examination"},
            2: {"zh": "腹痛、腹胀", "en": "Abdominal Pain/Bloating"}, 
            3: {"zh": "异常流血", "en": "Abnormal Bleeding"}
        }
    },
    # 7: surgical route - 二分类 (0: 开腹, 1: 腹腔镜)
    "surgical route": {
        "zh": "手术方式", "en": "Surgical Route", "type": "select", "default": 0,
        "options": {0: {"zh": "开腹", "en": "Laparotomy"}, 1: {"zh": "腹腔镜", "en": "Laparoscopy"}}
    },
    # 8: tumor envelope integrity - 二分类 (0: 完整, 1: 破裂)
    "tumor envelope integrity": {
        "zh": "肿物破裂", "en": "Tumor Envelope Integrity", "type": "select", "default": 1,
        "options": {0: {"zh": "否(完整)", "en": "No (Intact)"}, 1: {"zh": "是(破裂)", "en": "Yes (Ruptured)"}}
    },
    # 9: fertility-sparing surgery - 二分类 (0: 否, 1: 是)
    "fertility-sparing surgery": {
        "zh": "保留生育功能", "en": "Fertility-Sparing Surgery", "type": "select", "default": 0,
        "options": {0: {"zh": "否", "en": "No"}, 1: {"zh": "是", "en": "Yes"}}
    },
    # 10: completeness of surgery - 二分类 (0: 否, 1: 是)
    "completeness of surgery": {
        "zh": "全面分期", "en": "Completeness of Surgery", "type": "select", "default": 0,
        "options": {0: {"zh": "否", "en": "No"}, 1: {"zh": "是", "en": "Yes"}}
    },
    # 11: omentectomy - 二分类 (0: 否, 1: 是)
    "omentectomy": {
        "zh": "大网膜切除", "en": "Omentectomy", "type": "select", "default": 0,
        "options": {0: {"zh": "否", "en": "No"}, 1: {"zh": "是", "en": "Yes"}}
    },
    # 12: lymphadenectomy - 二分类 (0: 否, 1: 是)
    "lymphadenectomy": {
        "zh": "淋巴结清扫", "en": "Lymphadenectomy", "type": "select", "default": 0,
        "options": {0: {"zh": "否", "en": "No"}, 1: {"zh": "是", "en": "Yes"}}
    },
    # 13: histological subtype - 六分类 (0-5)
    "histological subtype": {
        "zh": "病理类型", "en": "Histological Subtype", "type": "select", "default": 0,
        "options": {
            0: {"zh": "浆液性", "en": "Serous"},
            1: {"zh": "粘液性", "en": "Mucinous"},
            2: {"zh": "浆粘液性", "en": "Seromucinous"},
            3: {"zh": "子宫内膜样", "en": "Endometrioid"},
            4: {"zh": "透明细胞", "en": "Clear Cell"},
            5: {"zh": "Brenner瘤", "en": "Brenner Tumor"}
        }
    },
    # 14: micropapillary - 二分类 (0: 否, 1: 是)
    "micropapillary": {
        "zh": "微乳头", "en": "Micropapillary", "type": "select", "default": 1,
        "options": {0: {"zh": "否", "en": "No"}, 1: {"zh": "是", "en": "Yes"}}
    },
    # 15: microinfiltration - 二分类 (0: 否, 1: 是)
    "microinfiltration": {
        "zh": "微浸润", "en": "Microinfiltration", "type": "select", "default": 1,
        "options": {0: {"zh": "否", "en": "No"}, 1: {"zh": "是", "en": "Yes"}}
    },
    # 16: psammoma bodies and calcification - 二分类 (0: 否, 1: 是)
    "psammoma bodies and calcification": {
        "zh": "砂粒体/钙化", "en": "Psammoma Bodies", "type": "select", "default": 1,
        "options": {0: {"zh": "否", "en": "No"}, 1: {"zh": "是", "en": "Yes"}}
    },
    # 17: peritoneal implantation - 二分类 (0: 否, 1: 是)
    "peritoneal implantation": {
        "zh": "腹膜种植", "en": "Peritoneal Implantation", "type": "select", "default": 1,
        "options": {0: {"zh": "否", "en": "No"}, 1: {"zh": "是", "en": "Yes"}}
    },
    # 18: ascites cytology - 二分类 (0: 阴性, 1: 阳性)
    "ascites cytology": {
        "zh": "腹水细胞学", "en": "Ascites Cytology", "type": "select", "default": 1,
        "options": {0: {"zh": "阴性", "en": "Negative"}, 1: {"zh": "阳性", "en": "Positive"}}
    },
    # 19: FIGO staging - 三分类 (1, 2, 3) 注意从1开始！
    "FIGO staging": {
        "zh": "FIGO分期", "en": "FIGO Staging", "type": "select", "default": 3,
        "options": {
            1: {"zh": "I期", "en": "Stage I"},
            2: {"zh": "II期", "en": "Stage II"},
            3: {"zh": "III期", "en": "Stage III"}
        }
    },
    # 20: unilateral or bilateral - 二分类 (0: 单侧, 1: 双侧)
    "unilateral or bilateral": {
        "zh": "单侧/双侧", "en": "Laterality", "type": "select", "default": 1,
        "options": {0: {"zh": "单侧", "en": "Unilateral"}, 1: {"zh": "双侧", "en": "Bilateral"}}
    },
    # 21: tumor size - 二分类！(0: ≤10cm, 1: >10cm)
    "tumor size": {
        "zh": "肿瘤大小", "en": "Tumor Size", "type": "select", "default": 1,
        "options": {0: {"zh": "≤10cm", "en": "≤10cm"}, 1: {"zh": ">10cm", "en": ">10cm"}}
    },
    # 22: CA125 - 二分类 (0: 正常, 1: 异常)
    "CA125": {
        "zh": "CA125", "en": "CA125", "type": "select", "default": 1,
        "options": {0: {"zh": "正常 (≤35 U/mL)", "en": "Normal (≤35)"}, 1: {"zh": "异常 (>35 U/mL)", "en": "Abnormal (>35)"}}
    },
    # 23: CEA - 二分类 (0: 正常, 1: 异常)
    "CEA": {
        "zh": "CEA", "en": "CEA", "type": "select", "default": 1,
        "options": {0: {"zh": "正常 (≤5 ng/mL)", "en": "Normal (≤5)"}, 1: {"zh": "异常 (>5 ng/mL)", "en": "Abnormal (>5)"}}
    },
    # 24: CA199 - 二分类 (0: 正常, 1: 异常)
    "CA199": {
        "zh": "CA199", "en": "CA199", "type": "select", "default": 1,
        "options": {0: {"zh": "正常 (≤37 U/mL)", "en": "Normal (≤37)"}, 1: {"zh": "异常 (>37 U/mL)", "en": "Abnormal (>37)"}}
    },
    # 25: AFP - 二分类 (0: 正常, 1: 异常)
    "AFP": {
        "zh": "AFP", "en": "AFP", "type": "select", "default": 0,
        "options": {0: {"zh": "正常 (≤9 ng/mL)", "en": "Normal (≤9)"}, 1: {"zh": "异常 (>9 ng/mL)", "en": "Abnormal (>9)"}}
    },
    # 26: CA724 - 二分类 (0: 正常, 1: 异常)
    "CA724": {
        "zh": "CA724", "en": "CA724", "type": "select", "default": 1,
        "options": {0: {"zh": "正常 (≤6.9 U/mL)", "en": "Normal (≤6.9)"}, 1: {"zh": "异常 (>6.9 U/mL)", "en": "Abnormal (>6.9)"}}
    },
    # 27: HE4 - 二分类 (0: 正常, 1: 异常)
    "HE4": {
        "zh": "HE4", "en": "HE4", "type": "select", "default": 1,
        "options": {0: {"zh": "正常 (≤140 pmol/L)", "en": "Normal (≤140)"}, 1: {"zh": "异常 (>140 pmol/L)", "en": "Abnormal (>140)"}}
    },
    # 28: smoking and drinking history - 二分类 (0: 否, 1: 是)
    "smoking and drinking history": {
        "zh": "吸烟饮酒史", "en": "Smoking/Drinking", "type": "select", "default": 1,
        "options": {0: {"zh": "否", "en": "No"}, 1: {"zh": "是", "en": "Yes"}}
    },
    # 29: receive estrogens - 二分类 (0: 否, 1: 是)
    "receive estrogens": {
        "zh": "雌激素暴露", "en": "Estrogen Exposure", "type": "select", "default": 1,
        "options": {0: {"zh": "否", "en": "No"}, 1: {"zh": "是", "en": "Yes"}}
    },
    # 30: ovulation induction - 二分类 (0: 否, 1: 是)
    "ovulation induction": {
        "zh": "促排卵史", "en": "Ovulation Induction", "type": "select", "default": 1,
        "options": {0: {"zh": "否", "en": "No"}, 1: {"zh": "是", "en": "Yes"}}
    },
    # 31: postoperative adjuvant therapy - 二分类 (0: 否, 1: 是)
    "postoperative adjuvant therapy": {
        "zh": "术后辅助治疗", "en": "Adjuvant Therapy", "type": "select", "default": 0,
        "options": {0: {"zh": "否", "en": "No"}, 1: {"zh": "是", "en": "Yes"}}
    },
    # 32: type of lesion - 二分类 (0: 外生型, 1: 内生型)
    "type of lesion": {
        "zh": "病灶类型", "en": "Lesion Type", "type": "select", "default": 1,
        "options": {0: {"zh": "外生型", "en": "Exophytic"}, 1: {"zh": "内生型", "en": "Endophytic"}}
    },
    # 33: papillary area ratio - 二分类 (0: ≤50%, 1: >50%)
    "papillary area ratio": {
        "zh": "乳头面积占比", "en": "Papillary Ratio", "type": "select", "default": 1,
        "options": {0: {"zh": "≤50%", "en": "≤50%"}, 1: {"zh": ">50%", "en": ">50%"}}
    }
}

# ================== 变量分组 ==================
VARIABLE_GROUPS = {
    "basic_info": [
        "age", "family cancer history", "sexual history", "parity",
        "menopausal status", "comorbidities", "smoking and drinking history",
        "receive estrogens", "ovulation induction"
    ],
    "surgical_info": [
        "presenting symptom", "surgical route", "tumor envelope integrity",
        "fertility-sparing surgery", "completeness of surgery", "omentectomy",
        "lymphadenectomy", "postoperative adjuvant therapy"
    ],
    "pathology_info": [
        "histological subtype", "micropapillary", "microinfiltration",
        "psammoma bodies and calcification", "peritoneal implantation", "ascites cytology",
        "FIGO staging", "unilateral or bilateral", "tumor size", "type of lesion",
        "papillary area ratio"
    ],
    "tumor_markers": ["CA125", "CEA", "CA199", "AFP", "CA724", "HE4"]
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
        self.fc = nn.Sequential(nn.Linear(dim, dim // reduction), nn.ReLU(), nn.Linear(dim // reduction, dim), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3, use_se=True):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim), nn.BatchNorm1d(dim))
        self.se = SEBlock(dim) if use_se else nn.Identity()
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.activation(x + self.dropout(self.se(self.block(x))))

class EnhancedDeepSurv(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], drop_rate=0.3, n_res_blocks=2):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dims[0]), nn.BatchNorm1d(hidden_dims[0]), nn.GELU(), nn.Dropout(drop_rate))
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dims[0], drop_rate) for _ in range(n_res_blocks)])
        self.down_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.down_layers.append(nn.Sequential(nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.BatchNorm1d(hidden_dims[i+1]), nn.GELU(), nn.Dropout(drop_rate)))
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
    def forward(self, x):
        x = self.input_proj(x)
        for rb in self.res_blocks:
            x = rb(x)
        for dl in self.down_layers:
            x = dl(x)
        return self.output_layer(x).squeeze(1)

class EnhancedDeepHit(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], num_durations=10, drop_rate=0.3):
        super().__init__()
        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop_rate)])
            in_d = h
        layers.append(nn.Linear(in_d, num_durations))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)

class EnhancedDenoisingAE(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], latent_dim=64, dropout=0.2):
        super().__init__()
        enc = []
        in_d = input_dim
        for h in hidden_dims:
            enc.extend([nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)])
            in_d = h
        enc.append(nn.Linear(in_d, latent_dim))
        self.encoder = nn.Sequential(*enc)
        dec = []
        in_d = latent_dim
        for h in reversed(hidden_dims):
            dec.extend([nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)])
            in_d = h
        dec.append(nn.Linear(in_d, input_dim))
        self.decoder = nn.Sequential(*dec)
    def encode(self, x):
        return self.encoder(x)
    def forward(self, x, noise_factor=0.1):
        if self.training and noise_factor > 0:
            x = x + torch.randn_like(x) * noise_factor
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
        z = self.transformer(z).squeeze(1)
        return self.output_proj(z)

class LearnableFusion(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())
    def forward(self, x):
        return self.net(x).squeeze(1)


# ================== 工具函数 ==================
def get_text(key, lang):
    return TRANSLATIONS.get(key, {}).get(lang, key)

@st.cache_resource
def load_models(model_dir="results_clinical_enhanced_v3"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {'ok': False, 'log': [], 'device': device}
    
    try:
        req = ['model_ae.pt', 'model_trans.pt', 'model_deepsurv.pt', 'model_deephit.pt', 'model_fusion.pt', 'preprocessor.joblib', 'time_cuts.npy', 'ds_min_max.npy', 'best_parameters.json']
        missing = [f for f in req if not os.path.exists(os.path.join(model_dir, f))]
        
        if not missing:
            with open(os.path.join(model_dir, "best_parameters.json")) as f:
                params = json.load(f)
            prep = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))
            time_cuts = np.load(os.path.join(model_dir, "time_cuts.npy"))
            ds_mm = np.load(os.path.join(model_dir, "ds_min_max.npy"))
            
            in_dim = prep.scaler.n_features_in_
            lat = params.get('ae_latent', 96)
            fused = lat * 2
            
            ae = EnhancedDenoisingAE(in_dim, [params.get('ae_h1', 512), params.get('ae_h2', 256)], lat)
            ae.load_state_dict(torch.load(os.path.join(model_dir, "model_ae.pt"), map_location=device))
            ae.eval().to(device)
            
            trans = EnhancedTransformer(lat)
            trans.load_state_dict(torch.load(os.path.join(model_dir, "model_trans.pt"), map_location=device))
            trans.eval().to(device)
            
            ds = EnhancedDeepSurv(fused, [params.get('ds_h1', 512), params.get('ds_h2', 64), params.get('ds_h3', 32)], params.get('ds_drop', 0.45))
            ds.load_state_dict(torch.load(os.path.join(model_dir, "model_deepsurv.pt"), map_location=device))
            ds.eval().to(device)
            
            n_dur = len(time_cuts) - 1
            dh = EnhancedDeepHit(fused, [params.get('dh_h1', 512), params.get('dh_h2', 64)], n_dur)
            dh.load_state_dict(torch.load(os.path.join(model_dir, "model_deephit.pt"), map_location=device))
            dh.eval().to(device)
            
            fusion = LearnableFusion()
            fusion.load_state_dict(torch.load(os.path.join(model_dir, "model_fusion.pt"), map_location=device))
            fusion.eval().to(device)
            
            models.update({'ae': ae, 'trans': trans, 'ds': ds, 'dh': dh, 'fusion': fusion, 'prep': prep, 'time_cuts': time_cuts, 'ds_mm': ds_mm, 'params': params, 'ok': True})
            models['log'].append("✅ 所有模型加载成功")
        else:
            models['log'].append(f"❌ 缺少文件: {missing}")
    except Exception as e:
        models['log'].append(f"❌ 加载错误: {e}")
    
    return models


def preprocess_input(data, models):
    """按训练时的特征顺序准备输入"""
    feats = []
    for feat_name in FEATURE_ORDER:
        val = data.get(feat_name, INPUT_VARIABLES[feat_name]['default'])
        feats.append(float(val))
    
    X = np.array(feats, dtype=np.float32).reshape(1, -1)
    
    if models.get('prep'):
        X = models['prep'].transform(X)
    
    return X


def predict(data, models):
    """执行预测"""
    dev = models['device']
    X_np = preprocess_input(data, models)
    X = torch.tensor(X_np, dtype=torch.float32, device=dev)
    
    with torch.no_grad():
        Z = models['ae'].encode(X)
        T = models['trans'](Z)
        Xf = torch.cat([Z, T], dim=1)
        
        risk_ds_raw = models['ds'](Xf).cpu().numpy().item()
        pmf = models['dh'](Xf).cpu().numpy()[0]
        
        # 归一化DeepSurv
        min_ds, max_ds = models['ds_mm']
        p_ds = np.clip((risk_ds_raw - min_ds) / (max_ds - min_ds + 1e-8), 0, 1)
        
        # DeepHit累积风险
        cif = np.cumsum(pmf)
        surv = 1 - cif
        target_bin = len(pmf) // 2
        r_dh = cif[target_bin]
        
        # Fusion
        fusion_in = torch.tensor([[p_ds, r_dh]], dtype=torch.float32, device=dev)
        final = models['fusion'](fusion_in).cpu().numpy().item()
    
    tc = models['time_cuts']
    tp = (tc[:-1] + tc[1:]) / 2
    
    def get_risk(t):
        idx = min(max(np.searchsorted(tp, t), 0), len(cif) - 1)
        return float(cif[idx])
    
    return {
        'risk': float(final), 'surv': surv, 'cif': cif, 'tp': tp,
        'r12': get_risk(12), 'r36': get_risk(36), 'r60': get_risk(60),
        'p_ds': float(p_ds), 'r_dh': float(r_dh), 'raw_ds': float(risk_ds_raw)
    }


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
        number={'suffix': '%', 'font': {'size': 56, 'color': col}},
        title={'text': f"<b>{get_text('overall_risk', lang)}</b><br><span style='font-size:24px;color:{col}'>{lv}</span>"},
        gauge={
            'axis': {'range': [0, 100], 'tickfont': {'size': 14}},
            'bar': {'color': col, 'thickness': 0.7},
            'steps': [{'range': [0, 30], 'color': 'rgba(39,174,96,0.2)'}, {'range': [30, 60], 'color': 'rgba(243,156,18,0.2)'}, {'range': [60, 100], 'color': 'rgba(231,76,60,0.2)'}]
        }
    ))
    fig.update_layout(height=320, margin=dict(l=30, r=30, t=80, b=30))
    return fig

def make_time_bar(r12, r36, r60, lang):
    labels = [get_text('month_12', lang), get_text('month_36', lang), get_text('month_60', lang)]
    vals = [r12 * 100, r36 * 100, r60 * 100]
    cols = ['#27ae60' if v < 30 else ('#f39c12' if v < 60 else '#e74c3c') for v in vals]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=cols, text=[f'{v:.1f}%' for v in vals], textposition='outside'))
    fig.update_layout(title=f"<b>{get_text('time_risk', lang)}</b>", yaxis=dict(range=[0, max(vals)*1.3 if max(vals) > 0 else 100]), height=320, margin=dict(l=50, r=30, t=60, b=40))
    return fig

def make_survival_chart(surv, tp, lang):
    fig = go.Figure(go.Scatter(x=tp, y=surv, mode='lines+markers', fill='tozeroy', line=dict(color='#3498db', width=3)))
    fig.update_layout(title=f"<b>{get_text('survival_curve', lang)}</b>", xaxis_title=get_text('time_months', lang), yaxis=dict(title=get_text('survival_prob', lang), range=[0, 1.05], tickformat='.0%'), height=320)
    return fig

def make_cumulative_chart(cif, tp, lang):
    fig = go.Figure(go.Scatter(x=tp, y=cif, mode='lines+markers', fill='tozeroy', line=dict(color='#e74c3c', width=3)))
    fig.update_layout(title=f"<b>{get_text('cumulative_risk_curve', lang)}</b>", xaxis_title=get_text('time_months', lang), yaxis=dict(title=get_text('risk_prob', lang), range=[0, 1.05], tickformat='.0%'), height=320)
    return fig


# ================== 输入控件 ==================
def render_select(feat_name, info, lang, prefix=""):
    options = list(info['options'].keys())
    default_idx = options.index(info['default']) if info['default'] in options else 0
    return st.selectbox(
        info[lang],
        options,
        index=default_idx,
        format_func=lambda x: info['options'][x][lang],
        key=f"{prefix}{feat_name}"
    )


# ================== 主函数 ==================
def main():
    models = load_models()
    
    # 状态显示
    if models.get('ok'):
        st.markdown('<div class="status-box status-success">✅ 模型已成功加载</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-box status-error">❌ 模型加载失败</div>', unsafe_allow_html=True)
        for log in models.get('log', []):
            st.text(log)
        return
    
    # 语言选择
    col1, col2 = st.columns([10, 1])
    with col2:
        lang = LANGUAGES[st.selectbox("🌐", list(LANGUAGES.keys()), label_visibility="collapsed")]
    
    # 头部
    st.markdown(f"""
    <div class="hospital-header">
        <h1>🏥 {get_text('title', lang)}</h1>
        <p class="subtitle">{get_text('subtitle', lang)}</p>
        <p class="hospital-name">{get_text('hospital', lang)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 输入表单
    st.markdown(f"### 📋 {get_text('single_patient', lang)}")
    
    data = {}
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(f'<div class="module-card"><div class="module-title">📝 {get_text("basic_info", lang)}</div>', unsafe_allow_html=True)
        for feat in VARIABLE_GROUPS["basic_info"]:
            data[feat] = render_select(feat, INPUT_VARIABLES[feat], lang, "s_")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with c2:
        st.markdown(f'<div class="module-card"><div class="module-title surgery">🔪 {get_text("surgical_info", lang)}</div>', unsafe_allow_html=True)
        for feat in VARIABLE_GROUPS["surgical_info"]:
            data[feat] = render_select(feat, INPUT_VARIABLES[feat], lang, "s_")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with c3:
        st.markdown(f'<div class="module-card"><div class="module-title pathology">🔬 {get_text("pathology_info", lang)}</div>', unsafe_allow_html=True)
        for feat in VARIABLE_GROUPS["pathology_info"]:
            data[feat] = render_select(feat, INPUT_VARIABLES[feat], lang, "s_")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 肿瘤标志物
    st.markdown(f'<div class="module-card"><div class="module-title markers">🧪 {get_text("tumor_markers", lang)}</div>', unsafe_allow_html=True)
    cols = st.columns(6)
    for i, feat in enumerate(VARIABLE_GROUPS["tumor_markers"]):
        with cols[i]:
            data[feat] = render_select(feat, INPUT_VARIABLES[feat], lang, "s_")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 预测按钮
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn = st.columns([2, 1, 2])
    with col_btn[1]:
        predict_btn = st.button(f"🔮 {get_text('predict_button', lang)}", use_container_width=True)
    
    if predict_btn:
        with st.spinner(get_text('processing', lang)):
            res = predict(data, models)
            
            # 调试信息
            with st.expander(f"🔧 {get_text('debug_info', lang)}"):
                st.write("**输入编码:**")
                st.json({k: int(v) for k, v in data.items()})
                st.write("**模型输出:**")
                cols = st.columns(4)
                cols[0].metric("DeepSurv原始", f"{res['raw_ds']:.4f}")
                cols[1].metric("DeepSurv归一化", f"{res['p_ds']:.4f}")
                cols[2].metric("DeepHit中位", f"{res['r_dh']:.4f}")
                cols[3].metric("融合风险", f"{res['risk']:.4f}")
            
            # 结果展示
            st.markdown(f'<div class="result-section">', unsafe_allow_html=True)
            
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                st.plotly_chart(make_gauge(res['risk'], lang), use_container_width=True)
            with r1c2:
                st.plotly_chart(make_time_bar(res['r12'], res['r36'], res['r60'], lang), use_container_width=True)
            
            r2c1, r2c2 = st.columns(2)
            with r2c1:
                st.plotly_chart(make_survival_chart(res['surv'], res['tp'], lang), use_container_width=True)
            with r2c2:
                st.plotly_chart(make_cumulative_chart(res['cif'], res['tp'], lang), use_container_width=True)
            
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
                <pre style="white-space: pre-wrap; font-family: inherit;">{get_text(adv, lang)}</pre>
            </div>
            """, unsafe_allow_html=True)
    
    # 免责声明
    st.markdown("---")
    st.info(get_text('disclaimer', lang))


if __name__ == "__main__":
    main()
