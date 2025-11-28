"""
Streamlit Web Application for Cancer Recurrence Prediction
肿瘤复发预测网页应用 - 增强版
===========================================================
Features:
- 中英文界面切换
- 个体患者风险预测 (下拉选择输入)
- 批量患者CSV导入预测
- 结果导出 PDF/Excel
- 生存曲线可视化
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
import os
import io
import base64
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from fpdf import FPDF
import tempfile

# ================== 页面配置 ==================
st.set_page_config(
    page_title="Cancer Recurrence Prediction | 肿瘤复发预测",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== 语言配置 ==================
LANGUAGES = {
    "中文": "zh",
    "English": "en"
}

# 翻译字典
TRANSLATIONS = {
    "title": {
        "zh": "🏥 肿瘤复发风险预测系统",
        "en": "🏥 Cancer Recurrence Risk Prediction System"
    },
    "subtitle": {
        "zh": "基于深度学习的个体化预测模型",
        "en": "Deep Learning-Based Personalized Prediction Model"
    },
    "patient_info": {
        "zh": "📋 患者信息录入",
        "en": "📋 Patient Information Entry"
    },
    "single_patient": {
        "zh": "单个患者预测",
        "en": "Single Patient Prediction"
    },
    "batch_prediction": {
        "zh": "批量患者预测",
        "en": "Batch Patient Prediction"
    },
    "basic_info": {
        "zh": "基本信息",
        "en": "Basic Information"
    },
    "medical_history": {
        "zh": "病史信息",
        "en": "Medical History"
    },
    "surgical_info": {
        "zh": "手术信息",
        "en": "Surgical Information"
    },
    "pathology_info": {
        "zh": "病理信息",
        "en": "Pathology Information"
    },
    "tumor_markers": {
        "zh": "肿瘤标志物",
        "en": "Tumor Markers"
    },
    "predict_button": {
        "zh": "🔮 开始预测",
        "en": "🔮 Start Prediction"
    },
    "prediction_results": {
        "zh": "📊 预测结果",
        "en": "📊 Prediction Results"
    },
    "overall_risk": {
        "zh": "总体复发风险",
        "en": "Overall Recurrence Risk"
    },
    "risk_level": {
        "zh": "风险等级",
        "en": "Risk Level"
    },
    "low_risk": {
        "zh": "低风险",
        "en": "Low Risk"
    },
    "medium_risk": {
        "zh": "中等风险",
        "en": "Medium Risk"
    },
    "high_risk": {
        "zh": "高风险",
        "en": "High Risk"
    },
    "survival_curve": {
        "zh": "生存曲线预测",
        "en": "Survival Curve Prediction"
    },
    "time_risk": {
        "zh": "时间点复发风险",
        "en": "Time-Point Recurrence Risk"
    },
    "risk_factors": {
        "zh": "主要风险因素",
        "en": "Major Risk Factors"
    },
    "clinical_advice": {
        "zh": "临床建议",
        "en": "Clinical Recommendations"
    },
    "disclaimer": {
        "zh": "⚠️ 免责声明：本系统仅供参考，不能替代专业医生的诊断。请结合临床实际情况综合判断。",
        "en": "⚠️ Disclaimer: This system is for reference only and cannot replace professional medical diagnosis."
    },
    "model_not_found": {
        "zh": "⚠️ 模型文件未找到，正在使用演示模式",
        "en": "⚠️ Model files not found, using demo mode"
    },
    "sidebar_title": {
        "zh": "⚙️ 设置",
        "en": "⚙️ Settings"
    },
    "language_select": {
        "zh": "选择语言",
        "en": "Select Language"
    },
    "months": {
        "zh": "个月",
        "en": " months"
    },
    "probability": {
        "zh": "概率",
        "en": "Probability"
    },
    "time": {
        "zh": "时间",
        "en": "Time"
    },
    "survival_probability": {
        "zh": "生存概率",
        "en": "Survival Probability"
    },
    "cumulative_risk": {
        "zh": "累积复发风险",
        "en": "Cumulative Recurrence Risk"
    },
    "upload_csv": {
        "zh": "上传CSV文件",
        "en": "Upload CSV File"
    },
    "download_template": {
        "zh": "下载模板",
        "en": "Download Template"
    },
    "batch_results": {
        "zh": "批量预测结果",
        "en": "Batch Prediction Results"
    },
    "export_excel": {
        "zh": "导出Excel",
        "en": "Export Excel"
    },
    "export_pdf": {
        "zh": "导出PDF报告",
        "en": "Export PDF Report"
    },
    "patient_id": {
        "zh": "患者ID",
        "en": "Patient ID"
    },
    "total_patients": {
        "zh": "总患者数",
        "en": "Total Patients"
    },
    "high_risk_count": {
        "zh": "高风险患者",
        "en": "High Risk Patients"
    },
    "medium_risk_count": {
        "zh": "中风险患者",
        "en": "Medium Risk Patients"
    },
    "low_risk_count": {
        "zh": "低风险患者",
        "en": "Low Risk Patients"
    },
    "risk_distribution": {
        "zh": "风险分布",
        "en": "Risk Distribution"
    },
    "advice_low": {
        "zh": """
        - 建议常规随访，每6个月复查一次
        - 保持健康生活方式
        - 定期监测肿瘤标志物
        """,
        "en": """
        - Recommend routine follow-up every 6 months
        - Maintain healthy lifestyle
        - Regular monitoring of tumor markers
        """
    },
    "advice_medium": {
        "zh": """
        - 建议加强随访，每3-4个月复查一次
        - 考虑辅助化疗或其他辅助治疗
        - 密切监测肿瘤标志物变化
        - 影像学检查频率增加
        """,
        "en": """
        - Recommend enhanced follow-up every 3-4 months
        - Consider adjuvant chemotherapy or other treatments
        - Close monitoring of tumor marker changes
        - Increased frequency of imaging examinations
        """
    },
    "advice_high": {
        "zh": """
        - 强烈建议密切随访，每2-3个月复查一次
        - 建议进行辅助化疗
        - 考虑多学科会诊(MDT)
        - 密切监测复发迹象
        - 可考虑临床试验
        """,
        "en": """
        - Strongly recommend close follow-up every 2-3 months
        - Recommend adjuvant chemotherapy
        - Consider multidisciplinary team (MDT) consultation
        - Close monitoring for recurrence signs
        - Consider clinical trials
        """
    },
    "select_option": {
        "zh": "请选择",
        "en": "Please select"
    },
    "input_value": {
        "zh": "请输入数值",
        "en": "Enter value"
    }
}

# 输入变量配置 - 增强版（带选项翻译）
INPUT_VARIABLES = {
    "age": {
        "zh": "年龄", 
        "en": "Age", 
        "type": "number", 
        "min": 18, 
        "max": 100, 
        "default": 50,
        "unit": {"zh": "岁", "en": "years"}
    },
    "family_cancer_history": {
        "zh": "家族史", 
        "en": "Family Cancer History", 
        "type": "select", 
        "options": {
            "no": {"zh": "无", "en": "No"},
            "yes": {"zh": "有", "en": "Yes"}
        }
    },
    "sexual_history": {
        "zh": "性生活史", 
        "en": "Sexual History", 
        "type": "select", 
        "options": {
            "no": {"zh": "无", "en": "No"},
            "yes": {"zh": "有", "en": "Yes"}
        }
    },
    "parity": {
        "zh": "生育次数", 
        "en": "Parity", 
        "type": "select",
        "options": {
            "0": {"zh": "0次", "en": "0"},
            "1": {"zh": "1次", "en": "1"},
            "2": {"zh": "2次", "en": "2"},
            "3": {"zh": "3次及以上", "en": "3 or more"}
        }
    },
    "menopausal_status": {
        "zh": "绝经状态", 
        "en": "Menopausal Status", 
        "type": "select", 
        "options": {
            "premenopausal": {"zh": "未绝经", "en": "Premenopausal"},
            "postmenopausal": {"zh": "已绝经", "en": "Postmenopausal"}
        }
    },
    "comorbidities": {
        "zh": "内科疾病", 
        "en": "Comorbidities", 
        "type": "select", 
        "options": {
            "no": {"zh": "无", "en": "None"},
            "hypertension": {"zh": "高血压", "en": "Hypertension"},
            "diabetes": {"zh": "糖尿病", "en": "Diabetes"},
            "cardiovascular": {"zh": "心血管疾病", "en": "Cardiovascular"},
            "multiple": {"zh": "多种疾病", "en": "Multiple"}
        }
    },
    "presenting_symptom": {
        "zh": "症状", 
        "en": "Presenting Symptom", 
        "type": "select", 
        "options": {
            "asymptomatic": {"zh": "无症状", "en": "Asymptomatic"},
            "abdominal_pain": {"zh": "腹痛", "en": "Abdominal Pain"},
            "bloating": {"zh": "腹胀", "en": "Bloating"},
            "mass": {"zh": "包块", "en": "Mass"},
            "bleeding": {"zh": "异常出血", "en": "Abnormal Bleeding"},
            "other": {"zh": "其他", "en": "Other"}
        }
    },
    "surgical_route": {
        "zh": "手术方式", 
        "en": "Surgical Route", 
        "type": "select", 
        "options": {
            "laparoscopy": {"zh": "腹腔镜", "en": "Laparoscopy"},
            "laparotomy": {"zh": "开腹手术", "en": "Laparotomy"},
            "robotic": {"zh": "机器人辅助", "en": "Robotic"},
            "conversion": {"zh": "中转开腹", "en": "Conversion"}
        }
    },
    "tumor_envelope_integrity": {
        "zh": "肿物破裂", 
        "en": "Tumor Envelope Integrity", 
        "type": "select", 
        "options": {
            "intact": {"zh": "完整", "en": "Intact"},
            "ruptured_before": {"zh": "术前破裂", "en": "Ruptured Before Surgery"},
            "ruptured_during": {"zh": "术中破裂", "en": "Ruptured During Surgery"}
        }
    },
    "fertility_sparing_surgery": {
        "zh": "保留生育功能", 
        "en": "Fertility-Sparing Surgery", 
        "type": "select", 
        "options": {
            "no": {"zh": "否", "en": "No"},
            "yes": {"zh": "是", "en": "Yes"}
        }
    },
    "completeness_of_surgery": {
        "zh": "全面分期", 
        "en": "Completeness of Surgery", 
        "type": "select", 
        "options": {
            "incomplete": {"zh": "不完全", "en": "Incomplete"},
            "complete": {"zh": "完全", "en": "Complete"}
        }
    },
    "omentectomy": {
        "zh": "大网膜切除", 
        "en": "Omentectomy", 
        "type": "select", 
        "options": {
            "no": {"zh": "未切除", "en": "No"},
            "partial": {"zh": "部分切除", "en": "Partial"},
            "total": {"zh": "全切除", "en": "Total"}
        }
    },
    "lymphadenectomy": {
        "zh": "淋巴结清扫", 
        "en": "Lymphadenectomy", 
        "type": "select", 
        "options": {
            "no": {"zh": "未清扫", "en": "No"},
            "pelvic": {"zh": "盆腔淋巴结", "en": "Pelvic"},
            "paraaortic": {"zh": "腹主动脉旁", "en": "Para-aortic"},
            "both": {"zh": "盆腔+腹主动脉旁", "en": "Both"}
        }
    },
    "histological_subtype": {
        "zh": "病理类型", 
        "en": "Histological Subtype", 
        "type": "select",
        "options": {
            "serous": {"zh": "浆液性", "en": "Serous"},
            "mucinous": {"zh": "粘液性", "en": "Mucinous"},
            "endometrioid": {"zh": "子宫内膜样", "en": "Endometrioid"},
            "clear_cell": {"zh": "透明细胞", "en": "Clear Cell"},
            "mixed": {"zh": "混合型", "en": "Mixed"},
            "other": {"zh": "其他", "en": "Other"}
        }
    },
    "micropapillary": {
        "zh": "微乳头结构", 
        "en": "Micropapillary", 
        "type": "select", 
        "options": {
            "no": {"zh": "无", "en": "No"},
            "yes": {"zh": "有", "en": "Yes"}
        }
    },
    "microinfiltration": {
        "zh": "微浸润", 
        "en": "Microinfiltration", 
        "type": "select", 
        "options": {
            "no": {"zh": "无", "en": "No"},
            "yes": {"zh": "有", "en": "Yes"}
        }
    },
    "psammoma_bodies_calcification": {
        "zh": "钙化砂体", 
        "en": "Psammoma Bodies and Calcification", 
        "type": "select", 
        "options": {
            "no": {"zh": "无", "en": "No"},
            "yes": {"zh": "有", "en": "Yes"}
        }
    },
    "peritoneal_implantation": {
        "zh": "腹膜种植", 
        "en": "Peritoneal Implantation", 
        "type": "select", 
        "options": {
            "no": {"zh": "无", "en": "No"},
            "noninvasive": {"zh": "非浸润性", "en": "Non-invasive"},
            "invasive": {"zh": "浸润性", "en": "Invasive"}
        }
    },
    "ascites_cytology": {
        "zh": "腹水细胞学", 
        "en": "Ascites Cytology", 
        "type": "select", 
        "options": {
            "no_ascites": {"zh": "无腹水", "en": "No Ascites"},
            "negative": {"zh": "阴性", "en": "Negative"},
            "positive": {"zh": "阳性", "en": "Positive"}
        }
    },
    "figo_staging": {
        "zh": "FIGO分期", 
        "en": "FIGO Staging", 
        "type": "select", 
        "options": {
            "IA": {"zh": "IA期", "en": "Stage IA"},
            "IB": {"zh": "IB期", "en": "Stage IB"},
            "IC1": {"zh": "IC1期", "en": "Stage IC1"},
            "IC2": {"zh": "IC2期", "en": "Stage IC2"},
            "IC3": {"zh": "IC3期", "en": "Stage IC3"},
            "II": {"zh": "II期", "en": "Stage II"},
            "IIIA": {"zh": "IIIA期", "en": "Stage IIIA"},
            "IIIB": {"zh": "IIIB期", "en": "Stage IIIB"},
            "IIIC": {"zh": "IIIC期", "en": "Stage IIIC"}
        }
    },
    "unilateral_or_bilateral": {
        "zh": "单侧/双侧", 
        "en": "Unilateral or Bilateral", 
        "type": "select", 
        "options": {
            "left": {"zh": "左侧", "en": "Left"},
            "right": {"zh": "右侧", "en": "Right"},
            "bilateral": {"zh": "双侧", "en": "Bilateral"}
        }
    },
    "tumor_size": {
        "zh": "肿瘤直径", 
        "en": "Tumor Size", 
        "type": "select",
        "options": {
            "<=5": {"zh": "≤5cm", "en": "≤5cm"},
            "5-10": {"zh": "5-10cm", "en": "5-10cm"},
            "10-15": {"zh": "10-15cm", "en": "10-15cm"},
            ">15": {"zh": ">15cm", "en": ">15cm"}
        }
    },
    "ca125": {
        "zh": "CA125", 
        "en": "CA125", 
        "type": "select",
        "options": {
            "normal": {"zh": "正常 (<35 U/mL)", "en": "Normal (<35 U/mL)"},
            "mild": {"zh": "轻度升高 (35-100 U/mL)", "en": "Mildly Elevated (35-100 U/mL)"},
            "moderate": {"zh": "中度升高 (100-500 U/mL)", "en": "Moderately Elevated (100-500 U/mL)"},
            "high": {"zh": "显著升高 (>500 U/mL)", "en": "Significantly Elevated (>500 U/mL)"}
        }
    },
    "cea": {
        "zh": "CEA", 
        "en": "CEA", 
        "type": "select",
        "options": {
            "normal": {"zh": "正常 (<5 ng/mL)", "en": "Normal (<5 ng/mL)"},
            "elevated": {"zh": "升高 (≥5 ng/mL)", "en": "Elevated (≥5 ng/mL)"}
        }
    },
    "ca199": {
        "zh": "CA199", 
        "en": "CA199", 
        "type": "select",
        "options": {
            "normal": {"zh": "正常 (<37 U/mL)", "en": "Normal (<37 U/mL)"},
            "elevated": {"zh": "升高 (≥37 U/mL)", "en": "Elevated (≥37 U/mL)"}
        }
    },
    "afp": {
        "zh": "AFP", 
        "en": "AFP", 
        "type": "select",
        "options": {
            "normal": {"zh": "正常 (<10 ng/mL)", "en": "Normal (<10 ng/mL)"},
            "elevated": {"zh": "升高 (≥10 ng/mL)", "en": "Elevated (≥10 ng/mL)"}
        }
    },
    "ca724": {
        "zh": "CA724", 
        "en": "CA724", 
        "type": "select",
        "options": {
            "normal": {"zh": "正常 (<6.9 U/mL)", "en": "Normal (<6.9 U/mL)"},
            "elevated": {"zh": "升高 (≥6.9 U/mL)", "en": "Elevated (≥6.9 U/mL)"}
        }
    },
    "he4": {
        "zh": "HE4", 
        "en": "HE4", 
        "type": "select",
        "options": {
            "normal": {"zh": "正常 (<70 pmol/L)", "en": "Normal (<70 pmol/L)"},
            "mild": {"zh": "轻度升高 (70-140 pmol/L)", "en": "Mildly Elevated (70-140 pmol/L)"},
            "elevated": {"zh": "显著升高 (>140 pmol/L)", "en": "Significantly Elevated (>140 pmol/L)"}
        }
    },
    "smoking_drinking_history": {
        "zh": "吸烟饮酒史", 
        "en": "Smoking and Drinking History", 
        "type": "select", 
        "options": {
            "no": {"zh": "无", "en": "No"},
            "smoking": {"zh": "吸烟", "en": "Smoking"},
            "drinking": {"zh": "饮酒", "en": "Drinking"},
            "both": {"zh": "吸烟+饮酒", "en": "Both"}
        }
    },
    "receive_estrogens": {
        "zh": "雌激素暴露史", 
        "en": "Receive Estrogens", 
        "type": "select", 
        "options": {
            "no": {"zh": "无", "en": "No"},
            "hrt": {"zh": "激素替代治疗", "en": "HRT"},
            "contraceptive": {"zh": "避孕药", "en": "Contraceptive"},
            "other": {"zh": "其他", "en": "Other"}
        }
    },
    "ovulation_induction": {
        "zh": "促排卵治疗史", 
        "en": "Ovulation Induction", 
        "type": "select", 
        "options": {
            "no": {"zh": "无", "en": "No"},
            "yes": {"zh": "有", "en": "Yes"}
        }
    },
    "postoperative_adjuvant_therapy": {
        "zh": "术后辅助治疗", 
        "en": "Postoperative Adjuvant Therapy", 
        "type": "select", 
        "options": {
            "no": {"zh": "未行辅助治疗", "en": "No"},
            "chemotherapy": {"zh": "化疗", "en": "Chemotherapy"},
            "targeted": {"zh": "靶向治疗", "en": "Targeted Therapy"},
            "combined": {"zh": "联合治疗", "en": "Combined"}
        }
    },
    "type_of_lesion": {
        "zh": "病灶类型", 
        "en": "Type of Lesion", 
        "type": "select", 
        "options": {
            "cystic": {"zh": "囊性", "en": "Cystic"},
            "solid": {"zh": "实性", "en": "Solid"},
            "mixed": {"zh": "囊实混合", "en": "Mixed"}
        }
    },
    "papillary_area_ratio": {
        "zh": "乳头面积占比", 
        "en": "Papillary Area Ratio", 
        "type": "select",
        "options": {
            "<10%": {"zh": "<10%", "en": "<10%"},
            "10-30%": {"zh": "10-30%", "en": "10-30%"},
            "30-50%": {"zh": "30-50%", "en": "30-50%"},
            ">50%": {"zh": ">50%", "en": ">50%"}
        }
    }
}


# ================== 模型定义 ==================

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
            layers.extend([
                nn.Linear(in_d, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(drop_rate)
            ])
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
        return self.net(x).squeeze(-1)


# ================== 工具函数 ==================

def get_text(key: str, lang: str) -> str:
    """获取翻译文本"""
    return TRANSLATIONS.get(key, {}).get(lang, key)


def get_option_label(var_name: str, option_key: str, lang: str) -> str:
    """获取选项的翻译标签"""
    var_info = INPUT_VARIABLES.get(var_name, {})
    options = var_info.get("options", {})
    option_info = options.get(option_key, {})
    return option_info.get(lang, option_key)


def encode_option(var_name: str, option_key: str) -> float:
    """将选项编码为数值"""
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
    """加载训练好的模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models = {}
    demo_mode = False
    
    try:
        # 加载参数
        with open(f"{model_dir}/best_parameters.json", "r") as f:
            params = json.load(f)
        
        # 加载预处理器
        preprocessor = joblib.load(f"{model_dir}/preprocessor.joblib")
        
        # 加载时间切分点
        time_cuts = np.load(f"{model_dir}/time_cuts.npy")
        num_bins = len(time_cuts) - 1
        
        # 加载DeepSurv min/max
        ds_min_max = np.load(f"{model_dir}/ds_min_max.npy")
        
        # 加载FCM中心
        fcm_centers = np.load(f"{model_dir}/fcm_centers.npy")
        
        # 确定输入维度
        input_dim = preprocessor.scaler.n_features_in_
        if preprocessor.selector is not None:
            input_dim = preprocessor.selector.k
        
        latent_dim = params.get('ae_latent', 64)
        fused_dim = latent_dim * 2
        
        # 加载模型
        ae = EnhancedDenoisingAE(input_dim, [params.get('ae_h1', 256), params.get('ae_h2', 128)], latent_dim)
        ae.load_state_dict(torch.load(f"{model_dir}/model_ae.pt", map_location=device))
        ae.eval()
        
        trans = EnhancedTransformer(latent_dim)
        trans.load_state_dict(torch.load(f"{model_dir}/model_trans.pt", map_location=device))
        trans.eval()
        
        ds = EnhancedDeepSurv(fused_dim, [params.get('ds_h1', 256), params.get('ds_h2', 128), params.get('ds_h3', 64)], drop_rate=params.get('ds_drop', 0.3))
        ds.load_state_dict(torch.load(f"{model_dir}/model_deepsurv.pt", map_location=device))
        ds.eval()
        
        dh = EnhancedDeepHit(fused_dim, [params.get('dh_h1', 256), params.get('dh_h2', 128)], num_durations=num_bins)
        dh.load_state_dict(torch.load(f"{model_dir}/model_deephit.pt", map_location=device))
        dh.eval()
        
        fusion = LearnableFusion()
        fusion.load_state_dict(torch.load(f"{model_dir}/model_fusion.pt", map_location=device))
        fusion.eval()
        
        models = {
            'ae': ae.to(device),
            'trans': trans.to(device),
            'ds': ds.to(device),
            'dh': dh.to(device),
            'fusion': fusion.to(device),
            'preprocessor': preprocessor,
            'time_cuts': time_cuts,
            'ds_min_max': ds_min_max,
            'fcm_centers': fcm_centers,
            'params': params,
            'device': device
        }
        
    except Exception as e:
        st.warning(f"模型加载失败: {e}")
        demo_mode = True
        
        # 演示模式
        device = torch.device("cpu")
        input_dim = len(INPUT_VARIABLES)
        latent_dim = 64
        fused_dim = latent_dim * 2
        num_bins = 10
        
        models = {
            'ae': EnhancedDenoisingAE(input_dim, [256, 128], latent_dim).to(device),
            'trans': EnhancedTransformer(latent_dim).to(device),
            'ds': EnhancedDeepSurv(fused_dim, [256, 128, 64]).to(device),
            'dh': EnhancedDeepHit(fused_dim, [256, 128], num_bins).to(device),
            'fusion': LearnableFusion().to(device),
            'preprocessor': None,
            'time_cuts': np.linspace(0, 120, num_bins + 1),
            'ds_min_max': np.array([-5.0, 5.0]),
            'fcm_centers': np.array([[0.3, 0.3], [0.7, 0.7]]),
            'params': {},
            'device': device
        }
        
        for key in ['ae', 'trans', 'ds', 'dh', 'fusion']:
            models[key].eval()
    
    return models, demo_mode


def preprocess_input(input_data: Dict, models: Dict, demo_mode: bool) -> np.ndarray:
    """预处理输入数据"""
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
    
    if models['preprocessor'] is not None and not demo_mode:
        try:
            X = models['preprocessor'].transform(X)
        except:
            pass
    
    return X


def predict_single(input_data: Dict, models: Dict, demo_mode: bool) -> Dict:
    """单个患者预测"""
    device = models['device']
    
    X = preprocess_input(input_data, models, demo_mode)
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
    risk_12m = cif[min(int(n_bins * 0.1), n_bins-1)]
    risk_36m = cif[min(int(n_bins * 0.3), n_bins-1)]
    risk_60m = cif[min(int(n_bins * 0.5), n_bins-1)]
    
    return {
        'final_risk': float(final_risk),
        'risk_deepsurv': float(prob_ds),
        'risk_deephit': float(risk_dh),
        'pmf': pmf,
        'cif': cif,
        'survival': survival,
        'time_points': time_points,
        'risk_12m': float(risk_12m),
        'risk_36m': float(risk_36m),
        'risk_60m': float(risk_60m)
    }


def predict_batch(df: pd.DataFrame, models: Dict, demo_mode: bool, lang: str) -> pd.DataFrame:
    """批量患者预测"""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in df.iterrows():
        status_text.text(f"Processing patient {idx + 1}/{len(df)}...")
        
        input_data = {}
        for var_name in INPUT_VARIABLES.keys():
            col_name_zh = INPUT_VARIABLES[var_name]['zh']
            col_name_en = INPUT_VARIABLES[var_name]['en']
            
            if col_name_zh in row:
                input_data[var_name] = row[col_name_zh]
            elif col_name_en in row:
                input_data[var_name] = row[col_name_en]
            elif var_name in row:
                input_data[var_name] = row[var_name]
        
        try:
            pred = predict_single(input_data, models, demo_mode)
            
            risk_level = "low_risk" if pred['final_risk'] < 0.3 else ("medium_risk" if pred['final_risk'] < 0.6 else "high_risk")
            
            results.append({
                get_text("patient_id", lang): row.get('patient_id', row.get('患者ID', idx + 1)),
                get_text("overall_risk", lang): f"{pred['final_risk']*100:.1f}%",
                f"12{get_text('months', lang)}": f"{pred['risk_12m']*100:.1f}%",
                f"36{get_text('months', lang)}": f"{pred['risk_36m']*100:.1f}%",
                f"60{get_text('months', lang)}": f"{pred['risk_60m']*100:.1f}%",
                get_text("risk_level", lang): get_text(risk_level, lang),
                '_final_risk_value': pred['final_risk']
            })
        except Exception as e:
            results.append({
                get_text("patient_id", lang): row.get('patient_id', row.get('患者ID', idx + 1)),
                get_text("overall_risk", lang): "Error",
                f"12{get_text('months', lang)}": "N/A",
                f"36{get_text('months', lang)}": "N/A",
                f"60{get_text('months', lang)}": "N/A",
                get_text("risk_level", lang): "Error",
                '_final_risk_value': 0
            })
        
        progress_bar.progress((idx + 1) / len(df))
    
    status_text.empty()
    progress_bar.empty()
    
    return pd.DataFrame(results)


def create_template_csv(lang: str) -> pd.DataFrame:
    """创建CSV模板"""
    columns = ['patient_id' if lang == 'en' else '患者ID']
    
    for var_name, var_info in INPUT_VARIABLES.items():
        col_name = var_info['en'] if lang == 'en' else var_info['zh']
        columns.append(col_name)
    
    # 创建示例数据
    sample_data = {
        columns[0]: [1, 2, 3]
    }
    
    for i, (var_name, var_info) in enumerate(INPUT_VARIABLES.items()):
        col_name = columns[i + 1]
        if var_info['type'] == 'select':
            options = list(var_info['options'].keys())
            sample_data[col_name] = [options[0]] * 3
        else:
            sample_data[col_name] = [var_info.get('default', 0)] * 3
    
    return pd.DataFrame(sample_data)


# ================== PDF生成 ==================

class PDFReport(FPDF):
    def __init__(self, lang='zh'):
        super().__init__()
        self.lang = lang
        # 使用内置字体，避免中文字体问题
        self.add_page()
        
    def header(self):
        self.set_font('Helvetica', 'B', 16)
        title = "Cancer Recurrence Risk Prediction Report" if self.lang == 'en' else "Cancer Recurrence Risk Report"
        self.cell(0, 10, title, 0, 1, 'C')
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


def generate_pdf_report(results_df: pd.DataFrame, lang: str) -> bytes:
    """生成PDF报告"""
    pdf = PDFReport(lang)
    pdf.set_font('Helvetica', '', 10)
    
    # 标题
    pdf.set_font('Helvetica', 'B', 14)
    title = "Batch Prediction Results" if lang == 'en' else "Batch Prediction Results"
    pdf.cell(0, 10, title, 0, 1, 'L')
    pdf.ln(5)
    
    # 摘要统计
    pdf.set_font('Helvetica', 'B', 12)
    summary_title = "Summary Statistics" if lang == 'en' else "Summary Statistics"
    pdf.cell(0, 10, summary_title, 0, 1, 'L')
    
    pdf.set_font('Helvetica', '', 10)
    
    total = len(results_df)
    risk_col = get_text("risk_level", lang)
    
    if risk_col in results_df.columns:
        high_risk = len(results_df[results_df[risk_col].str.contains('High|高', case=False, na=False)])
        medium_risk = len(results_df[results_df[risk_col].str.contains('Medium|中', case=False, na=False)])
        low_risk = len(results_df[results_df[risk_col].str.contains('Low|低', case=False, na=False)])
    else:
        high_risk = medium_risk = low_risk = 0
    
    pdf.cell(0, 8, f"Total Patients: {total}", 0, 1)
    pdf.cell(0, 8, f"High Risk: {high_risk} ({high_risk/total*100:.1f}%)" if total > 0 else "High Risk: 0", 0, 1)
    pdf.cell(0, 8, f"Medium Risk: {medium_risk} ({medium_risk/total*100:.1f}%)" if total > 0 else "Medium Risk: 0", 0, 1)
    pdf.cell(0, 8, f"Low Risk: {low_risk} ({low_risk/total*100:.1f}%)" if total > 0 else "Low Risk: 0", 0, 1)
    pdf.ln(10)
    
    # 详细结果表格
    pdf.set_font('Helvetica', 'B', 12)
    detail_title = "Detailed Results" if lang == 'en' else "Detailed Results"
    pdf.cell(0, 10, detail_title, 0, 1, 'L')
    
    # 表格头
    pdf.set_font('Helvetica', 'B', 8)
    display_cols = [col for col in results_df.columns if not col.startswith('_')]
    
    col_width = 190 / len(display_cols)
    for col in display_cols:
        pdf.cell(col_width, 8, str(col)[:15], 1, 0, 'C')
    pdf.ln()
    
    # 表格数据
    pdf.set_font('Helvetica', '', 8)
    for _, row in results_df.head(50).iterrows():  # 限制50行
        for col in display_cols:
            value = str(row[col])[:15] if col in row else ""
            pdf.cell(col_width, 6, value, 1, 0, 'C')
        pdf.ln()
    
    # 免责声明
    pdf.ln(10)
    pdf.set_font('Helvetica', 'I', 8)
    disclaimer = "Disclaimer: This report is for reference only and cannot replace professional medical diagnosis."
    pdf.multi_cell(0, 5, disclaimer)
    
    # 生成时间
    pdf.ln(5)
    pdf.cell(0, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    
    return pdf.output(dest='S').encode('latin-1')


def generate_single_pdf_report(patient_data: Dict, results: Dict, lang: str) -> bytes:
    """生成单个患者PDF报告"""
    pdf = PDFReport(lang)
    pdf.set_font('Helvetica', '', 10)
    
    # 患者信息
    pdf.set_font('Helvetica', 'B', 14)
    info_title = "Patient Information" if lang == 'en' else "Patient Information"
    pdf.cell(0, 10, info_title, 0, 1, 'L')
    
    pdf.set_font('Helvetica', '', 10)
    for var_name, value in patient_data.items():
        if var_name in INPUT_VARIABLES:
            var_info = INPUT_VARIABLES[var_name]
            label = var_info['en'] if lang == 'en' else var_info['zh']
            
            if var_info['type'] == 'select' and value:
                display_value = get_option_label(var_name, value, lang)
            else:
                display_value = str(value)
            
            pdf.cell(0, 6, f"{label}: {display_value}", 0, 1)
    
    pdf.ln(10)
    
    # 预测结果
    pdf.set_font('Helvetica', 'B', 14)
    result_title = "Prediction Results" if lang == 'en' else "Prediction Results"
    pdf.cell(0, 10, result_title, 0, 1, 'L')
    
    pdf.set_font('Helvetica', '', 12)
    
    risk = results['final_risk']
    if risk < 0.3:
        risk_level = "Low Risk" if lang == 'en' else "Low Risk"
    elif risk < 0.6:
        risk_level = "Medium Risk" if lang == 'en' else "Medium Risk"
    else:
        risk_level = "High Risk" if lang == 'en' else "High Risk"
    
    pdf.cell(0, 8, f"Overall Risk: {risk*100:.1f}%", 0, 1)
    pdf.cell(0, 8, f"Risk Level: {risk_level}", 0, 1)
    pdf.cell(0, 8, f"12-month Risk: {results['risk_12m']*100:.1f}%", 0, 1)
    pdf.cell(0, 8, f"36-month Risk: {results['risk_36m']*100:.1f}%", 0, 1)
    pdf.cell(0, 8, f"60-month Risk: {results['risk_60m']*100:.1f}%", 0, 1)
    
    # 免责声明
    pdf.ln(10)
    pdf.set_font('Helvetica', 'I', 8)
    disclaimer = "Disclaimer: This report is for reference only and cannot replace professional medical diagnosis."
    pdf.multi_cell(0, 5, disclaimer)
    
    pdf.ln(5)
    pdf.cell(0, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    
    return pdf.output(dest='S').encode('latin-1')


# ================== 可视化函数 ==================

def create_gauge_chart(risk: float, lang: str) -> go.Figure:
    """创建仪表盘图"""
    if risk < 0.3:
        color = "green"
        risk_text = get_text("low_risk", lang)
    elif risk < 0.6:
        color = "orange"
        risk_text = get_text("medium_risk", lang)
    else:
        color = "red"
        risk_text = get_text("high_risk", lang)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{get_text('overall_risk', lang)}<br><span style='font-size:0.8em'>{risk_text}</span>"},
        number={'suffix': '%', 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': 'white',
            'borderwidth': 2,
            'steps': [
                {'range': [0, 30], 'color': 'rgba(0, 255, 0, 0.3)'},
                {'range': [30, 60], 'color': 'rgba(255, 165, 0, 0.3)'},
                {'range': [60, 100], 'color': 'rgba(255, 0, 0, 0.3)'}
            ],
            'threshold': {
                'line': {'color': 'black', 'width': 4},
                'thickness': 0.75,
                'value': risk * 100
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_survival_curve(survival: np.ndarray, time_points: np.ndarray, lang: str) -> go.Figure:
    """创建生存曲线图"""
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        get_text("survival_probability", lang),
        get_text("cumulative_risk", lang)
    ))
    
    fig.add_trace(
        go.Scatter(
            x=time_points, y=survival,
            mode='lines+markers',
            name=get_text("survival_probability", lang),
            line=dict(color='blue', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 100, 255, 0.2)'
        ),
        row=1, col=1
    )
    
    cif = 1 - survival
    fig.add_trace(
        go.Scatter(
            x=time_points, y=cif,
            mode='lines+markers',
            name=get_text("cumulative_risk", lang),
            line=dict(color='red', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.2)'
        ),
        row=1, col=2
    )
    
    time_label = get_text("time", lang) + " (" + get_text("months", lang).strip() + ")"
    
    fig.update_xaxes(title_text=time_label, row=1, col=1)
    fig.update_xaxes(title_text=time_label, row=1, col=2)
    fig.update_yaxes(title_text=get_text("probability", lang), range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text=get_text("probability", lang), range=[0, 1], row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False, margin=dict(l=50, r=50, t=50, b=50))
    
    return fig


def create_time_risk_bar(risk_12m: float, risk_36m: float, risk_60m: float, lang: str) -> go.Figure:
    """创建时间点风险柱状图"""
    months_text = get_text("months", lang)
    
    fig = go.Figure(data=[
        go.Bar(
            x=[f"12{months_text}", f"36{months_text}", f"60{months_text}"],
            y=[risk_12m * 100, risk_36m * 100, risk_60m * 100],
            marker_color=['#2ecc71', '#f39c12', '#e74c3c'],
            text=[f'{risk_12m*100:.1f}%', f'{risk_36m*100:.1f}%', f'{risk_60m*100:.1f}%'],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=get_text("time_risk", lang),
        yaxis_title=get_text("probability", lang) + " (%)",
        yaxis_range=[0, 100],
        height=350,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def create_risk_distribution_chart(results_df: pd.DataFrame, lang: str) -> go.Figure:
    """创建风险分布图"""
    risk_col = get_text("risk_level", lang)
    
    if risk_col in results_df.columns:
        high_risk = len(results_df[results_df[risk_col].str.contains('High|高', case=False, na=False)])
        medium_risk = len(results_df[results_df[risk_col].str.contains('Medium|中', case=False, na=False)])
        low_risk = len(results_df[results_df[risk_col].str.contains('Low|低', case=False, na=False)])
    else:
        high_risk = medium_risk = low_risk = 0
    
    fig = go.Figure(data=[
        go.Pie(
            labels=[get_text("low_risk", lang), get_text("medium_risk", lang), get_text("high_risk", lang)],
            values=[low_risk, medium_risk, high_risk],
            marker_colors=['#2ecc71', '#f39c12', '#e74c3c'],
            hole=0.4,
            textinfo='label+percent+value'
        )
    ])
    
    fig.update_layout(
        title=get_text("risk_distribution", lang),
        height=400
    )
    
    return fig


# ================== 输入控件 ==================

def render_select_widget(var_name: str, var_info: Dict, lang: str, key_prefix: str = "") -> str:
    """渲染下拉选择控件"""
    label = f"{var_info['zh']} / {var_info['en']}" if lang == "zh" else f"{var_info['en']} / {var_info['zh']}"
    
    options = var_info.get('options', {})
    option_keys = list(options.keys())
    
    # 创建显示标签
    format_func = lambda x: f"{options[x]['zh']} / {options[x]['en']}" if lang == "zh" else f"{options[x]['en']} / {options[x]['zh']}"
    
    selected = st.selectbox(
        label,
        options=option_keys,
        format_func=format_func,
        key=f"{key_prefix}{var_name}"
    )
    
    return selected


def render_number_widget(var_name: str, var_info: Dict, lang: str, key_prefix: str = "") -> float:
    """渲染数值输入控件"""
    label = f"{var_info['zh']} / {var_info['en']}" if lang == "zh" else f"{var_info['en']} / {var_info['zh']}"
    
    if 'unit' in var_info:
        unit = var_info['unit'].get(lang, '')
        label = f"{label} ({unit})"
    
    value = st.number_input(
        label,
        min_value=float(var_info.get('min', 0)),
        max_value=float(var_info.get('max', 100)),
        value=float(var_info.get('default', 0)),
        key=f"{key_prefix}{var_name}"
    )
    
    return value


# ================== 主应用 ==================

def main():
    # 侧边栏
    with st.sidebar:
        st.title("⚙️ Settings / 设置")
        language = st.selectbox(
            "Language / 语言",
            options=list(LANGUAGES.keys()),
            index=0
        )
        lang = LANGUAGES[language]
        
        st.markdown("---")
        st.markdown("""
        ### About / 关于
        
        **Models / 模型:**
        - DeepSurv
        - DeepHit
        - Autoencoder + Transformer
        
        **Version / 版本:** 3.0
        """)
    
    # 加载模型
    models, demo_mode = load_models()
    
    # 主标题
    st.title(get_text("title", lang))
    st.markdown(f"### {get_text('subtitle', lang)}")
    
    if demo_mode:
        st.warning(get_text("model_not_found", lang))
    
    st.markdown("---")
    
    # 选项卡
    tab1, tab2 = st.tabs([
        f"👤 {get_text('single_patient', lang)}", 
        f"📊 {get_text('batch_prediction', lang)}"
    ])
    
    # ==================== 单个患者预测 ====================
    with tab1:
        st.header(get_text("patient_info", lang))
        
        col1, col2, col3 = st.columns(3)
        input_data = {}
        
        # 基本信息
        with col1:
            st.subheader(f"📝 {get_text('basic_info', lang)}")
            basic_vars = ['age', 'family_cancer_history', 'sexual_history', 'parity', 
                         'menopausal_status', 'comorbidities', 'smoking_drinking_history',
                         'receive_estrogens', 'ovulation_induction']
            for var_name in basic_vars:
                if var_name in INPUT_VARIABLES:
                    var_info = INPUT_VARIABLES[var_name]
                    if var_info['type'] == 'select':
                        input_data[var_name] = render_select_widget(var_name, var_info, lang, "single_")
                    else:
                        input_data[var_name] = render_number_widget(var_name, var_info, lang, "single_")
        
        # 手术信息
        with col2:
            st.subheader(f"🔪 {get_text('surgical_info', lang)}")
            surgical_vars = ['presenting_symptom', 'surgical_route', 'tumor_envelope_integrity',
                           'fertility_sparing_surgery', 'completeness_of_surgery', 'omentectomy',
                           'lymphadenectomy', 'postoperative_adjuvant_therapy']
            for var_name in surgical_vars:
                if var_name in INPUT_VARIABLES:
                    var_info = INPUT_VARIABLES[var_name]
                    if var_info['type'] == 'select':
                        input_data[var_name] = render_select_widget(var_name, var_info, lang, "single_")
                    else:
                        input_data[var_name] = render_number_widget(var_name, var_info, lang, "single_")
        
        # 病理信息
        with col3:
            st.subheader(f"🔬 {get_text('pathology_info', lang)}")
            pathology_vars = ['histological_subtype', 'micropapillary', 'microinfiltration',
                            'psammoma_bodies_calcification', 'peritoneal_implantation', 
                            'ascites_cytology', 'figo_staging', 'unilateral_or_bilateral',
                            'tumor_size', 'type_of_lesion', 'papillary_area_ratio']
            for var_name in pathology_vars:
                if var_name in INPUT_VARIABLES:
                    var_info = INPUT_VARIABLES[var_name]
                    if var_info['type'] == 'select':
                        input_data[var_name] = render_select_widget(var_name, var_info, lang, "single_")
                    else:
                        input_data[var_name] = render_number_widget(var_name, var_info, lang, "single_")
        
        # 肿瘤标志物
        st.subheader(f"🧪 {get_text('tumor_markers', lang)}")
        marker_cols = st.columns(6)
        marker_vars = ['ca125', 'cea', 'ca199', 'afp', 'ca724', 'he4']
        for i, var_name in enumerate(marker_vars):
            with marker_cols[i]:
                var_info = INPUT_VARIABLES[var_name]
                input_data[var_name] = render_select_widget(var_name, var_info, lang, "single_")
        
        st.markdown("---")
        
        # 预测按钮
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            predict_clicked = st.button(
                get_text("predict_button", lang),
                type="primary",
                use_container_width=True,
                key="single_predict"
            )
        
        if predict_clicked:
            with st.spinner("Predicting... / 预测中..."):
                results = predict_single(input_data, models, demo_mode)
                
                st.markdown("---")
                st.header(get_text("prediction_results", lang))
                
                result_col1, result_col2 = st.columns([1, 2])
                
                with result_col1:
                    gauge_fig = create_gauge_chart(results['final_risk'], lang)
                    st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    bar_fig = create_time_risk_bar(
                        results['risk_12m'], 
                        results['risk_36m'], 
                        results['risk_60m'],
                        lang
                    )
                    st.plotly_chart(bar_fig, use_container_width=True)
                
                with result_col2:
                    st.subheader(get_text("survival_curve", lang))
                    survival_fig = create_survival_curve(
                        results['survival'],
                        results['time_points'],
                        lang
                    )
                    st.plotly_chart(survival_fig, use_container_width=True)
                
                # 临床建议
                st.markdown("---")
                st.subheader(get_text("clinical_advice", lang))
                
                risk = results['final_risk']
                if risk < 0.3:
                    risk_level = "low_risk"
                    advice_key = "advice_low"
                    st.success(f"**{get_text('risk_level', lang)}: {get_text(risk_level, lang)}** ({risk*100:.1f}%)")
                elif risk < 0.6:
                    risk_level = "medium_risk"
                    advice_key = "advice_medium"
                    st.warning(f"**{get_text('risk_level', lang)}: {get_text(risk_level, lang)}** ({risk*100:.1f}%)")
                else:
                    risk_level = "high_risk"
                    advice_key = "advice_high"
                    st.error(f"**{get_text('risk_level', lang)}: {get_text(risk_level, lang)}** ({risk*100:.1f}%)")
                
                st.markdown(get_text(advice_key, lang))
                
                # 导出按钮
                st.markdown("---")
                export_col1, export_col2 = st.columns(2)
                
                with export_col1:
                    # 导出Excel
                    detail_df = pd.DataFrame({
                        'Metric': ['Final Risk', 'DeepSurv Risk', 'DeepHit Risk', 
                                  '12-month Risk', '36-month Risk', '60-month Risk'],
                        'Value': [f"{results['final_risk']*100:.2f}%",
                                 f"{results['risk_deepsurv']*100:.2f}%",
                                 f"{results['risk_deephit']*100:.2f}%",
                                 f"{results['risk_12m']*100:.2f}%",
                                 f"{results['risk_36m']*100:.2f}%",
                                 f"{results['risk_60m']*100:.2f}%"]
                    })
                    
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        detail_df.to_excel(writer, sheet_name='Results', index=False)
                    excel_data = excel_buffer.getvalue()
                    
                    st.download_button(
                        label=f"📥 {get_text('export_excel', lang)}",
                        data=excel_data,
                        file_name=f"prediction_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with export_col2:
                    # 导出PDF
                    pdf_data = generate_single_pdf_report(input_data, results, lang)
                    st.download_button(
                        label=f"📄 {get_text('export_pdf', lang)}",
                        data=pdf_data,
                        file_name=f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
    
    # ==================== 批量预测 ====================
    with tab2:
        st.header(get_text("batch_prediction", lang))
        
        # 下载模板
        st.subheader(f"1️⃣ {get_text('download_template', lang)}")
        template_df = create_template_csv(lang)
        
        csv_buffer = io.StringIO()
        template_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label=f"📥 {get_text('download_template', lang)} (CSV)",
            data=csv_data,
            file_name=f"prediction_template_{lang}.csv",
            mime="text/csv"
        )
        
        # 预览模板
        with st.expander("Preview Template / 预览模板"):
            st.dataframe(template_df, use_container_width=True)
        
        st.markdown("---")
        
        # 上传文件
        st.subheader(f"2️⃣ {get_text('upload_csv', lang)}")
        uploaded_file = st.file_uploader(
            get_text("upload_csv", lang),
            type=['csv', 'xlsx'],
            help="Upload a CSV or Excel file with patient data / 上传包含患者数据的CSV或Excel文件"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Loaded {len(df)} patients / 已加载 {len(df)} 位患者")
                
                with st.expander("Preview Data / 预览数据"):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # 批量预测按钮
                if st.button(get_text("predict_button", lang), type="primary", key="batch_predict"):
                    with st.spinner("Processing... / 处理中..."):
