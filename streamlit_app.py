"""
Streamlit Web Application for Cancer Recurrence Prediction
肿瘤复发预测网页应用
===========================================================
中国医科大学附属盛京医院
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import tempfile
import os

# ================== 页面配置 ==================
st.set_page_config(
    page_title="肿瘤复发预测系统 - 盛京医院",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 隐藏侧边栏
st.markdown(
    """
    <style>
        [data-testid="collapsedControl"] {
            display: none
        }
        section[data-testid="stSidebar"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ================== 文本配置 ==================
TEXTS = {
    "title": "🏥 肿瘤复发风险预测系统",
    "subtitle": "基于深度学习的个体化预测模型",
    "hospital": "中国医科大学附属盛京医院",
    "patient_info": "📋 患者信息录入",
    "single_patient": "单个患者预测",
    "batch_prediction": "批量患者预测",
    "basic_info": "基本信息",
    "surgical_info": "手术信息",
    "pathology_info": "病理信息",
    "tumor_markers": "肿瘤标志物",
    "predict_button": "🔮 开始预测",
    "prediction_results": "📊 预测结果",
    "overall_risk": "总体复发风险",
    "risk_level": "风险等级",
    "low_risk": "低风险",
    "medium_risk": "中等风险",
    "high_risk": "高风险",
    "survival_curve": "生存曲线预测",
    "time_risk": "时间点复发风险",
    "clinical_advice": "临床建议",
    "disclaimer": "⚠️ 免责声明：本系统仅供临床参考，不能替代专业医生的诊断。请结合临床实际情况综合判断。",
    "months": "个月",
    "probability": "概率",
    "time": "时间",
    "survival_probability": "生存概率",
    "cumulative_risk": "累积复发风险",
    "upload_csv": "上传CSV文件",
    "download_template": "下载模板",
    "batch_results": "批量预测结果",
    "export_excel": "导出Excel",
    "export_pdf": "导出PDF报告",
    "patient_id": "患者ID",
    "total_patients": "总患者数",
    "high_risk_count": "高风险患者",
    "medium_risk_count": "中风险患者",
    "low_risk_count": "低风险患者",
    "risk_distribution": "风险分布",
    "processing": "处理中...",
    "advice_low": """
- 建议常规随访，每6个月复查一次
- 保持健康生活方式
- 定期监测肿瘤标志物
    """,
    "advice_medium": """
- 建议加强随访，每3-4个月复查一次
- 考虑辅助化疗或其他辅助治疗
- 密切监测肿瘤标志物变化
- 影像学检查频率增加
    """,
    "advice_high": """
- 强烈建议密切随访，每2-3个月复查一次
- 建议进行辅助化疗
- 考虑多学科会诊(MDT)
- 密切监测复发迹象
- 可考虑临床试验
    """
}

# 输入变量配置
INPUT_VARIABLES = {
    "age": {
        "label": "年龄",
        "type": "number", 
        "min": 18, 
        "max": 100, 
        "default": 50,
        "unit": "岁"
    },
    "family_cancer_history": {
        "label": "家族史",
        "type": "select", 
        "options": {"no": "无", "yes": "有"}
    },
    "sexual_history": {
        "label": "性生活史",
        "type": "select", 
        "options": {"no": "无", "yes": "有"}
    },
    "parity": {
        "label": "生育次数",
        "type": "select",
        "options": {"0": "0次", "1": "1次", "2": "2次", "3": "3次及以上"}
    },
    "menopausal_status": {
        "label": "绝经状态",
        "type": "select", 
        "options": {"premenopausal": "未绝经", "postmenopausal": "已绝经"}
    },
    "comorbidities": {
        "label": "内科疾病",
        "type": "select", 
        "options": {
            "no": "无",
            "hypertension": "高血压",
            "diabetes": "糖尿病",
            "cardiovascular": "心血管疾病",
            "multiple": "多种疾病"
        }
    },
    "presenting_symptom": {
        "label": "症状",
        "type": "select", 
        "options": {
            "asymptomatic": "无症状",
            "abdominal_pain": "腹痛",
            "bloating": "腹胀",
            "mass": "包块",
            "bleeding": "异常出血",
            "other": "其他"
        }
    },
    "surgical_route": {
        "label": "手术方式",
        "type": "select", 
        "options": {
            "laparoscopy": "腹腔镜",
            "laparotomy": "开腹手术",
            "robotic": "机器人辅助",
            "conversion": "中转开腹"
        }
    },
    "tumor_envelope_integrity": {
        "label": "肿物破裂",
        "type": "select", 
        "options": {
            "intact": "完整",
            "ruptured_before": "术前破裂",
            "ruptured_during": "术中破裂"
        }
    },
    "fertility_sparing_surgery": {
        "label": "保留生育功能",
        "type": "select", 
        "options": {"no": "否", "yes": "是"}
    },
    "completeness_of_surgery": {
        "label": "全面分期",
        "type": "select", 
        "options": {"incomplete": "不完全", "complete": "完全"}
    },
    "omentectomy": {
        "label": "大网膜切除",
        "type": "select", 
        "options": {"no": "未切除", "partial": "部分切除", "total": "全切除"}
    },
    "lymphadenectomy": {
        "label": "淋巴结清扫",
        "type": "select", 
        "options": {
            "no": "未清扫",
            "pelvic": "盆腔淋巴结",
            "paraaortic": "腹主动脉旁",
            "both": "盆腔+腹主动脉旁"
        }
    },
    "histological_subtype": {
        "label": "病理类型",
        "type": "select",
        "options": {
            "serous": "浆液性",
            "mucinous": "粘液性",
            "endometrioid": "子宫内膜样",
            "clear_cell": "透明细胞",
            "mixed": "混合型",
            "other": "其他"
        }
    },
    "micropapillary": {
        "label": "微乳头结构",
        "type": "select", 
        "options": {"no": "无", "yes": "有"}
    },
    "microinfiltration": {
        "label": "微浸润",
        "type": "select", 
        "options": {"no": "无", "yes": "有"}
    },
    "psammoma_bodies_calcification": {
        "label": "钙化砂体",
        "type": "select", 
        "options": {"no": "无", "yes": "有"}
    },
    "peritoneal_implantation": {
        "label": "腹膜种植",
        "type": "select", 
        "options": {
            "no": "无",
            "noninvasive": "非浸润性",
            "invasive": "浸润性"
        }
    },
    "ascites_cytology": {
        "label": "腹水细胞学",
        "type": "select", 
        "options": {
            "no_ascites": "无腹水",
            "negative": "阴性",
            "positive": "阳性"
        }
    },
    "figo_staging": {
        "label": "FIGO分期",
        "type": "select", 
        "options": {
            "IA": "IA期",
            "IB": "IB期",
            "IC1": "IC1期",
            "IC2": "IC2期",
            "IC3": "IC3期",
            "II": "II期",
            "IIIA": "IIIA期",
            "IIIB": "IIIB期",
            "IIIC": "IIIC期"
        }
    },
    "unilateral_or_bilateral": {
        "label": "单侧/双侧",
        "type": "select", 
        "options": {
            "left": "左侧",
            "right": "右侧",
            "bilateral": "双侧"
        }
    },
    "tumor_size": {
        "label": "肿瘤直径",
        "type": "select",
        "options": {
            "<=5": "≤5cm",
            "5-10": "5-10cm",
            "10-15": "10-15cm",
            ">15": ">15cm"
        }
    },
    "ca125": {
        "label": "CA125",
        "type": "select",
        "options": {
            "normal": "正常 (<35 U/mL)",
            "mild": "轻度升高 (35-100 U/mL)",
            "moderate": "中度升高 (100-500 U/mL)",
            "high": "显著升高 (>500 U/mL)"
        }
    },
    "cea": {
        "label": "CEA",
        "type": "select",
        "options": {
            "normal": "正常 (<5 ng/mL)",
            "elevated": "升高 (>=5 ng/mL)"
        }
    },
    "ca199": {
        "label": "CA199",
        "type": "select",
        "options": {
            "normal": "正常 (<37 U/mL)",
            "elevated": "升高 (>=37 U/mL)"
        }
    },
    "afp": {
        "label": "AFP",
        "type": "select",
        "options": {
            "normal": "正常 (<10 ng/mL)",
            "elevated": "升高 (>=10 ng/mL)"
        }
    },
    "ca724": {
        "label": "CA724",
        "type": "select",
        "options": {
            "normal": "正常 (<6.9 U/mL)",
            "elevated": "升高 (>=6.9 U/mL)"
        }
    },
    "he4": {
        "label": "HE4",
        "type": "select",
        "options": {
            "normal": "正常 (<70 pmol/L)",
            "mild": "轻度升高 (70-140 pmol/L)",
            "elevated": "显著升高 (>140 pmol/L)"
        }
    },
    "smoking_drinking_history": {
        "label": "吸烟饮酒史",
        "type": "select", 
        "options": {
            "no": "无",
            "smoking": "吸烟",
            "drinking": "饮酒",
            "both": "吸烟+饮酒"
        }
    },
    "receive_estrogens": {
        "label": "雌激素暴露史",
        "type": "select", 
        "options": {
            "no": "无",
            "hrt": "激素替代治疗",
            "contraceptive": "避孕药",
            "other": "其他"
        }
    },
    "ovulation_induction": {
        "label": "促排卵治疗史",
        "type": "select", 
        "options": {"no": "无", "yes": "有"}
    },
    "postoperative_adjuvant_therapy": {
        "label": "术后辅助治疗",
        "type": "select", 
        "options": {
            "no": "未行辅助治疗",
            "chemotherapy": "化疗",
            "targeted": "靶向治疗",
            "combined": "联合治疗"
        }
    },
    "type_of_lesion": {
        "label": "病灶类型",
        "type": "select", 
        "options": {
            "cystic": "囊性",
            "solid": "实性",
            "mixed": "囊实混合"
        }
    },
    "papillary_area_ratio": {
        "label": "乳头面积占比",
        "type": "select",
        "options": {
            "<10%": "<10%",
            "10-30%": "10-30%",
            "30-50%": "30-50%",
            ">50%": ">50%"
        }
    }
}


# ================== 模型定义 ==================

class SEBlock(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, max(dim // reduction, 1)),
            nn.ReLU(),
            nn.Linear(max(dim // reduction, 1), dim),
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
def load_models():
    """加载模型（演示模式）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        'time_cuts': np.linspace(0, 120, num_bins + 1),
        'ds_min_max': np.array([-5.0, 5.0]),
        'device': device
    }
    
    for key in ['ae', 'trans', 'ds', 'dh', 'fusion']:
        models[key].eval()
    
    return models


def preprocess_input(input_data: Dict, models: Dict) -> np.ndarray:
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
    X = (X - X.mean()) / (X.std() + 1e-8)
    
    return X


def predict_single(input_data: Dict, models: Dict) -> Dict:
    """单个患者预测"""
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


def predict_batch(df: pd.DataFrame, models: Dict) -> pd.DataFrame:
    """批量患者预测"""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in df.iterrows():
        status_text.text(f"正在处理第 {idx + 1}/{len(df)} 位患者...")
        
        input_data = {}
        for var_name in INPUT_VARIABLES.keys():
            col_name = INPUT_VARIABLES[var_name]['label']
            
            if col_name in row:
                input_data[var_name] = row[col_name]
            elif var_name in row:
                input_data[var_name] = row[var_name]
        
        try:
            pred = predict_single(input_data, models)
            
            if pred['final_risk'] < 0.3:
                risk_level = "低风险"
            elif pred['final_risk'] < 0.6:
                risk_level = "中等风险"
            else:
                risk_level = "高风险"
            
            results.append({
                "患者ID": row.get('patient_id', row.get('患者ID', idx + 1)),
                "总体复发风险": f"{pred['final_risk']*100:.1f}%",
                "12个月风险": f"{pred['risk_12m']*100:.1f}%",
                "36个月风险": f"{pred['risk_36m']*100:.1f}%",
                "60个月风险": f"{pred['risk_60m']*100:.1f}%",
                "风险等级": risk_level,
                '_final_risk_value': pred['final_risk']
            })
        except Exception as e:
            results.append({
                "患者ID": row.get('patient_id', row.get('患者ID', idx + 1)),
                "总体复发风险": "错误",
                "12个月风险": "N/A",
                "36个月风险": "N/A",
                "60个月风险": "N/A",
                "风险等级": "错误",
                '_final_risk_value': 0
            })
        
        progress_bar.progress((idx + 1) / len(df))
    
    status_text.empty()
    progress_bar.empty()
    
    return pd.DataFrame(results)


def create_template_csv() -> pd.DataFrame:
    """创建CSV模板"""
    columns = ['患者ID']
    
    for var_name, var_info in INPUT_VARIABLES.items():
        columns.append(var_info['label'])
    
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

def generate_pdf_report(results_df: pd.DataFrame) -> bytes:
    """生成PDF报告"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=1
    )
    story.append(Paragraph("Cancer Recurrence Risk Prediction Report", title_style))
    story.append(Paragraph("Shengjing Hospital of China Medical University", styles['Normal']))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Summary Statistics", styles['Heading2']))
    story.append(Spacer(1, 10))
    
    total = len(results_df)
    high_risk = len(results_df[results_df['风险等级'].str.contains('高', na=False)])
    medium_risk = len(results_df[results_df['风险等级'].str.contains('中', na=False)])
    low_risk = len(results_df[results_df['风险等级'].str.contains('低', na=False)])
    
    summary_data = [
        ["Metric", "Value"],
        ["Total Patients", str(total)],
        ["High Risk", f"{high_risk} ({high_risk/total*100:.1f}%)" if total > 0 else "0"],
        ["Medium Risk", f"{medium_risk} ({medium_risk/total*100:.1f}%)" if total > 0 else "0"],
        ["Low Risk", f"{low_risk} ({low_risk/total*100:.1f}%)" if total > 0 else "0"]
    ]
    
    summary_table = Table(summary_data, colWidths=[200, 200])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 30))
    
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey
    )
    story.append(Paragraph("Disclaimer: This report is for reference only.", disclaimer_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def generate_single_pdf_report(patient_data: Dict, results: Dict) -> bytes:
    """生成单个患者PDF报告"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=1
    )
    story.append(Paragraph("Patient Risk Assessment Report", title_style))
    story.append(Paragraph("Shengjing Hospital of China Medical University", styles['Normal']))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Prediction Results", styles['Heading2']))
    story.append(Spacer(1, 10))
    
    risk = results['final_risk']
    risk_level = "Low Risk" if risk < 0.3 else ("Medium Risk" if risk < 0.6 else "High Risk")
    
    results_data = [
        ["Metric", "Value"],
        ["Overall Risk", f"{risk*100:.1f}%"],
        ["Risk Level", risk_level],
        ["12-month Risk", f"{results['risk_12m']*100:.1f}%"],
        ["36-month Risk", f"{results['risk_36m']*100:.1f}%"],
        ["60-month Risk", f"{results['risk_60m']*100:.1f}%"]
    ]
    
    results_table = Table(results_data, colWidths=[200, 200])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    story.append(results_table)
    story.append(Spacer(1, 30))
    
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey
    )
    story.append(Paragraph("Disclaimer: This report is for reference only.", disclaimer_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ================== 可视化函数 ==================

def create_gauge_chart(risk: float) -> go.Figure:
    """创建仪表盘图"""
    if risk < 0.3:
        color = "green"
        risk_text = TEXTS["low_risk"]
    elif risk < 0.6:
        color = "orange"
        risk_text = TEXTS["medium_risk"]
    else:
        color = "red"
        risk_text = TEXTS["high_risk"]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{TEXTS['overall_risk']}<br><span style='font-size:0.8em'>{risk_text}</span>"},
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


def create_survival_curve(survival: np.ndarray, time_points: np.ndarray) -> go.Figure:
    """创建生存曲线图"""
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        TEXTS["survival_probability"],
        TEXTS["cumulative_risk"]
    ))
    
    fig.add_trace(
        go.Scatter(
            x=time_points, y=survival,
            mode='lines+markers',
            name=TEXTS["survival_probability"],
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
            name=TEXTS["cumulative_risk"],
            line=dict(color='red', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.2)'
        ),
        row=1, col=2
    )
    
    time_label = f"{TEXTS['time']} ({TEXTS['months']})"
    
    fig.update_xaxes(title_text=time_label, row=1, col=1)
    fig.update_xaxes(title_text=time_label, row=1, col=2)
    fig.update_yaxes(title_text=TEXTS["probability"], range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text=TEXTS["probability"], range=[0, 1], row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False, margin=dict(l=50, r=50, t=50, b=50))
    
    return fig


def create_time_risk_bar(risk_12m: float, risk_36m: float, risk_60m: float) -> go.Figure:
    """创建时间点风险柱状图"""
    months_text = TEXTS["months"]
    
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
        title=TEXTS["time_risk"],
        yaxis_title=f"{TEXTS['probability']} (%)",
        yaxis_range=[0, 100],
        height=350,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


def create_risk_distribution_chart(results_df: pd.DataFrame) -> go.Figure:
    """创建风险分布图"""
    high_risk = len(results_df[results_df['风险等级'].str.contains('高', na=False)])
    medium_risk = len(results_df[results_df['风险等级'].str.contains('中', na=False)])
    low_risk = len(results_df[results_df['风险等级'].str.contains('低', na=False)])
    
    fig = go.Figure(data=[
        go.Pie(
            labels=[TEXTS["low_risk"], TEXTS["medium_risk"], TEXTS["high_risk"]],
            values=[low_risk, medium_risk, high_risk],
            marker_colors=['#2ecc71', '#f39c12', '#e74c3c'],
            hole=0.4,
            textinfo='label+percent+value'
        )
    ])
    
    fig.update_layout(
        title=TEXTS["risk_distribution"],
        height=400
    )
    
    return fig


# ================== 输入控件 ==================

def render_select_widget(var_name: str, var_info: Dict, key_prefix: str = "") -> str:
    """渲染下拉选择控件"""
    label = var_info['label']
    options = var_info.get('options', {})
    option_keys = list(options.keys())
    
    format_func = lambda x: options[x]
    
    selected = st.selectbox(
        label,
        options=option_keys,
        format_func=format_func,
        key=f"{key_prefix}{var_name}"
    )
    
    return selected


def render_number_widget(var_name: str, var_info: Dict, key_prefix: str = "") -> float:
    """渲染数值输入控件"""
    label = var_info['label']
    
    if 'unit' in var_info:
        label = f"{label} ({var_info['unit']})"
    
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
    # 加载模型
    models = load_models()
    
    # 主标题和医院信息
    st.markdown(
        f"""
        <div style='text-align: center; padding: 10px;'>
            <h1>{TEXTS['title']}</h1>
            <h3>{TEXTS['subtitle']}</h3>
            <p style='color: #1E88E5; font-size: 18px; font-weight: bold;'>{TEXTS['hospital']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # 选项卡
    tab1, tab2 = st.tabs([
        f"👤 {TEXTS['single_patient']}", 
        f"📊 {TEXTS['batch_prediction']}"
    ])
    
    # ==================== 单个患者预测 ====================
    with tab1:
        st.header(TEXTS["patient_info"])
        
        col1, col2, col3 = st.columns(3)
        input_data = {}
        
        # 基本信息
        with col1:
            st.subheader(f"📝 {TEXTS['basic_info']}")
            basic_vars = ['age', 'family_cancer_history', 'sexual_history', 'parity', 
                         'menopausal_status', 'comorbidities', 'smoking_drinking_history',
                         'receive_estrogens', 'ovulation_induction']
            for var_name in basic_vars:
                if var_name in INPUT_VARIABLES:
                    var_info = INPUT_VARIABLES[var_name]
                    if var_info['type'] == 'select':
                        input_data[var_name] = render_select_widget(var_name, var_info, "single_")
                    else:
                        input_data[var_name] = render_number_widget(var_name, var_info, "single_")
        
        # 手术信息
        with col2:
            st.subheader(f"🔪 {TEXTS['surgical_info']}")
            surgical_vars = ['presenting_symptom', 'surgical_route', 'tumor_envelope_integrity',
                           'fertility_sparing_surgery', 'completeness_of_surgery', 'omentectomy',
                           'lymphadenectomy', 'postoperative_adjuvant_therapy']
            for var_name in surgical_vars:
                if var_name in INPUT_VARIABLES:
                    var_info = INPUT_VARIABLES[var_name]
                    if var_info['type'] == 'select':
                        input_data[var_name] = render_select_widget(var_name, var_info, "single_")
                    else:
                        input_data[var_name] = render_number_widget(var_name, var_info, "single_")
        
        # 病理信息
        with col3:
            st.subheader(f"🔬 {TEXTS['pathology_info']}")
            pathology_vars = ['histological_subtype', 'micropapillary', 'microinfiltration',
                            'psammoma_bodies_calcification', 'peritoneal_implantation', 
                            'ascites_cytology', 'figo_staging', 'unilateral_or_bilateral',
                            'tumor_size', 'type_of_lesion', 'papillary_area_ratio']
            for var_name in pathology_vars:
                if var_name in INPUT_VARIABLES:
                    var_info = INPUT_VARIABLES[var_name]
                    if var_info['type'] == 'select':
                        input_data[var_name] = render_select_widget(var_name, var_info, "single_")
                    else:
                        input_data[var_name] = render_number_widget(var_name, var_info, "single_")
        
        # 肿瘤标志物
        st.subheader(f"🧪 {TEXTS['tumor_markers']}")
        marker_cols = st.columns(6)
        marker_vars = ['ca125', 'cea', 'ca199', 'afp', 'ca724', 'he4']
        for i, var_name in enumerate(marker_vars):
            with marker_cols[i]:
                var_info = INPUT_VARIABLES[var_name]
                input_data[var_name] = render_select_widget(var_name, var_info, "single_")
        
        st.markdown("---")
        
        # 预测按钮
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            predict_clicked = st.button(
                TEXTS["predict_button"],
                type="primary",
                use_container_width=True,
                key="single_predict"
            )
        
        if predict_clicked:
            with st.spinner(TEXTS["processing"]):
                results = predict_single(input_data, models)
                
                st.markdown("---")
                st.header(TEXTS["prediction_results"])
                
                result_col1, result_col2 = st.columns([1, 2])
                
                with result_col1:
                    gauge_fig = create_gauge_chart(results['final_risk'])
                    st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    bar_fig = create_time_risk_bar(
                        results['risk_12m'], 
                        results['risk_36m'], 
                        results['risk_60m']
                    )
                    st.plotly_chart(bar_fig, use_container_width=True)
                
                with result_col2:
                    st.subheader(TEXTS["survival_curve"])
                    survival_fig = create_survival_curve(
                        results['survival'],
                        results['time_points']
                    )
                    st.plotly_chart(survival_fig, use_container_width=True)
                
                # 临床建议
                st.markdown("---")
                st.subheader(TEXTS["clinical_advice"])
                
                risk = results['final_risk']
                if risk < 0.3:
                    risk_level = "low_risk"
                    advice_key = "advice_low"
                    st.success(f"**{TEXTS['risk_level']}: {TEXTS[risk_level]}** ({risk*100:.1f}%)")
                elif risk < 0.6:
                    risk_level = "medium_risk"
                    advice_key = "advice_medium"
                    st.warning(f"**{TEXTS['risk_level']}: {TEXTS[risk_level]}** ({risk*100:.1f}%)")
                else:
                    risk_level = "high_risk"
                    advice_key = "advice_high"
                    st.error(f"**{TEXTS['risk_level']}: {TEXTS[risk_level]}** ({risk*100:.1f}%)")
                
                st.markdown(TEXTS[advice_key])
                
                # 导出按钮
                st.markdown("---")
                st.subheader("📥 导出结果")
                export_col1, export_col2 = st.columns(2)
                
                with export_col1:
                    detail_df = pd.DataFrame({
                        '指标': ['总体风险', 'DeepSurv风险', 'DeepHit风险', 
                                '12个月风险', '36个月风险', '60个月风险'],
                        '数值': [f"{results['final_risk']*100:.2f}%",
                                f"{results['risk_deepsurv']*100:.2f}%",
                                f"{results['risk_deephit']*100:.2f}%",
                                f"{results['risk_12m']*100:.2f}%",
                                f"{results['risk_36m']*100:.2f}%",
                                f"{results['risk_60m']*100:.2f}%"]
                    })
                    
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        detail_df.to_excel(writer, sheet_name='预测结果', index=False)
                    excel_data = excel_buffer.getvalue()
                    
                    st.download_button(
                        label=f"📥 {TEXTS['export_excel']}",
                        data=excel_data,
                        file_name=f"预测结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with export_col2:
                    pdf_data = generate_single_pdf_report(input_data, results)
                    st.download_button(
                        label=f"📄 {TEXTS['export_pdf']}",
                        data=pdf_data,
                        file_name=f"预测报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
    
    # ==================== 批量预测 ====================
    with tab2:
        st.header(TEXTS["batch_prediction"])
        
        # 下载模板
        st.subheader(f"1️⃣ {TEXTS['download_template']}")
        template_df = create_template_csv()
        
        csv_buffer = io.StringIO()
        template_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label=f"📥 {TEXTS['download_template']} (CSV)",
            data=csv_data,
            file_name="预测模板.csv",
            mime="text/csv"
        )
        
        with st.expander("预览模板"):
            st.dataframe(template_df, use_container_width=True)
        
        st.markdown("---")
        
        # 上传文件
        st.subheader(f"2️⃣ {TEXTS['upload_csv']}")
        uploaded_file = st.file_uploader(
            TEXTS["upload_csv"],
            type=['csv', 'xlsx'],
            help="上传包含患者数据的CSV或Excel文件"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ 已加载 {len(df)} 位患者数据")
                
                with st.expander("预览数据"):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # 批量预测按钮
                if st.button(TEXTS["predict_button"], type="primary", key="batch_predict"):
                    with st.spinner(TEXTS["processing"]):
                        results_df = predict_batch(df, models)
                        
                        st.markdown("---")
                        st.header(TEXTS["batch_results"])
                        
                        # 统计摘要
                        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                        
                        total = len(results_df)
                        high_count = len(results_df[results_df['风险等级'].str.contains('高', na=False)])
                        medium_count = len(results_df[results_df['风险等级'].str.contains('中', na=False)])
                        low_count = len(results_df[results_df['风险等级'].str.contains('低', na=False)])
                        
                        with summary_col1:
                            st.metric(TEXTS["total_patients"], total)
                        with summary_col2:
                            st.metric(TEXTS["high_risk_count"], high_count)
                        with summary_col3:
                            st.metric(TEXTS["medium_risk_count"], medium_count)
                        with summary_col4:
                            st.metric(TEXTS["low_risk_count"], low_count)
                        
                        # 风险分布图
                        chart_col1, chart_col2 = st.columns(2)
                        
                        with chart_col1:
                            pie_fig = create_risk_distribution_chart(results_df)
                            st.plotly_chart(pie_fig, use_container_width=True)
                        
                        with chart_col2:
                            if '_final_risk_value' in results_df.columns:
                                risk_values = results_df['_final_risk_value'].values * 100
                                
                                hist_fig = go.Figure(data=[
                                    go.Histogram(
                                        x=risk_values,
                                        nbinsx=20,
                                        marker_color='steelblue',
                                        opacity=0.75
                                    )
                                ])
                                
                                hist_fig.add_vline(x=30, line_dash="dash", line_color="green", 
                                                   annotation_text="低/中")
                                hist_fig.add_vline(x=60, line_dash="dash", line_color="red",
                                                   annotation_text="中/高")
                                
                                hist_fig.update_layout(
                                    title="风险分数分布",
                                    xaxis_title="风险分数 (%)",
                                    yaxis_title="患者数量",
                                    height=400
                                )
                                
                                st.plotly_chart(hist_fig, use_container_width=True)
                        
                        # 显示结果表格
                        st.subheader("📋 详细结果")
                        
                        display_df = results_df.drop(columns=[col for col in results_df.columns if col.startswith('_')], errors='ignore')
                        
                        def highlight_risk(row):
                            if '高' in str(row.get('风险等级', '')):
                                return ['background-color: #ffcccc'] * len(row)
                            elif '中' in str(row.get('风险等级', '')):
                                return ['background-color: #fff3cd'] * len(row)
                            else:
                                return ['background-color: #d4edda'] * len(row)
                        
                        styled_df = display_df.style.apply(highlight_risk, axis=1)
                        st.dataframe(styled_df, use_container_width=True, height=400)
                        
                        # 导出选项
                        st.markdown("---")
                        st.subheader("📥 导出结果")
                        
                        export_col1, export_col2, export_col3 = st.columns(3)
                        
                        with export_col1:
                            csv_export = io.StringIO()
                            display_df.to_csv(csv_export, index=False, encoding='utf-8-sig')
                            csv_export_data = csv_export.getvalue()
                            
                            st.download_button(
                                label="📥 导出CSV",
                                data=csv_export_data,
                                file_name=f"批量预测结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        with export_col2:
                            excel_buffer = io.BytesIO()
                            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                display_df.to_excel(writer, sheet_name='预测结果', index=False)
                                
                                summary_data = {
                                    '指标': [TEXTS["total_patients"], TEXTS["high_risk_count"], 
                                            TEXTS["medium_risk_count"], TEXTS["low_risk_count"]],
                                    '数值': [total, high_count, medium_count, low_count]
                                }
                                summary_df = pd.DataFrame(summary_data)
                                summary_df.to_excel(writer, sheet_name='统计摘要', index=False)
                            
                            excel_export_data = excel_buffer.getvalue()
                            
                            st.download_button(
                                label=f"📥 {TEXTS['export_excel']}",
                                data=excel_export_data,
                                file_name=f"批量预测结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        
                        with export_col3:
                            pdf_data = generate_pdf_report(results_df)
                            st.download_button(
                                label=f"📄 {TEXTS['export_pdf']}",
                                data=pdf_data,
                                file_name=f"批量预测报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf"
                            )
                        
                        # 高风险患者列表
                        if high_count > 0:
                            st.markdown("---")
                            st.subheader("⚠️ 需关注的高风险患者")
                            
                            high_risk_df = display_df[display_df['风险等级'].str.contains('高', na=False)]
                            
                            st.dataframe(
                                high_risk_df.style.apply(lambda x: ['background-color: #ffcccc'] * len(x), axis=1),
                                use_container_width=True
                            )
                            
                            st.warning(f"⚠️ {high_count} 位患者被评估为高风险，需要密切随访！")
                
            except Exception as e:
                st.error(f"文件处理错误: {str(e)}")
                st.info("请确保您的文件格式与模板一致。")
    
    # 页脚免责声明
    st.markdown("---")
    st.info(TEXTS["disclaimer"])
    
    # 页脚信息
    st.markdown(
        f"""
        <div style='text-align: center; color: gray; padding: 20px;'>
            <p style='font-size: 16px; font-weight: bold;'>{TEXTS['hospital']}</p>
            <p>肿瘤复发风险预测系统 v3.0</p>
            <p>© 2024 All Rights Reserved</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
