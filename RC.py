import streamlit as st
import pandas as pd
import datetime
import altair as alt
import numpy as np
import io
import re
import textwrap, os, re, io, json, math, datetime as dt
from typing import List, Dict, Optional, Tuple
import json
import re
from pathlib import Path

st.markdown("""
    <style>
    .refresh-button {
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 100;
    }
    </style>
    <div class="refresh-button">
        <form action="" method="get">
            <button type="submit"> Refresh</button>
        </form>
    </div>
""", unsafe_allow_html=True)

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .css-1dp5vir {display: none;} 
    </style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.set_page_config(layout="wide")
st.title("📑 Rate Card Generator")

# 1) 放在文件最上方：初始化开关
if "hide_intro" not in st.session_state:
    st.session_state["hide_intro"] = False  # 初始显示横幅

def intro_banner():
    # st.markdown("""
    # <div style='padding: 1rem; border-radius: 0.5rem; background-color: #F0F7F6;'>
    #   <h3 style='color: #225560;'>👋 欢迎使用Rate Card生成器</h3>
    #   <p style='font-size:16px; color:#444;'>
    #     👉 上传前注意事项:

    #     1. 每个Excel的表头都在第一行，即没有隐藏行（特别注意TrainCost_Xi'an）  
    #     2. 一个Excel中只有一行表头（特别注意TrainCost_Chengdu）  
    #     3. 拖车费从第三张sheet开始读取  
    #   </p>
    #   

    #   </p>
    # </div>
    # """, unsafe_allow_html=True)
    st.markdown("""
        <div style='padding: 1rem; border-radius: 0.5rem; background-color: #F0F7F6;'>
        <h3 style='color: #225560;'>⚠️ 上传前注意事项</h3>
        <div style='font-size:16px; color:#333; line-height:1.8; margin-left:0.2rem;'>
            1. 每个Excel的表头都在第一行，即没有隐藏行（特别注意 TrainCost_Xi'an）<br>
            2. 一个Excel中只有一行表头（特别注意 TrainCost_Chengdu）<br>
            3. 拖车费从第三张 sheet 开始读取<br>
            4. 请确认 Route 内容为传统<b>线路<b>
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)

# 2) 页面顶部：按开关显示/隐藏横幅
if not st.session_state["hide_intro"]:
    intro_banner()
# -------- Sidebar Uploads --------
with st.sidebar:
    st.header("Upload files")
    train_files = st.file_uploader("Train Cost (support multiple files)", type=["xls", "xlsx"], accept_multiple_files=True)
    truck_file = st.file_uploader("Truck｜拖车费", type=["xls", "xlsx"])
    buffer_file = st.file_uploader("Buffer", type=["xls", "xlsx"])
    # st.markdown("---")
    st.markdown("> 说明：本应用**不保存**任何数据；所有处理都在内存完成。")
    # 👉 按钮控制显示/隐藏逻辑说明
    if "show_logic" not in st.session_state:
        st.session_state["show_logic"] = False

    if st.button("📖 查看系统处理逻辑", use_container_width=True):
        st.session_state["show_logic"] = not st.session_state["show_logic"]

    if st.session_state["show_logic"]:
        st.markdown("""
        ### 系统处理逻辑
        1. **上传文件**：需要上传 Train Cost（可多文件）、Truck（拖车费）、Buffer 三类表格。  
        2. **Train Cost**：只保留月初价格，`Valid To` 自动改成月底，合并完成后的`Cost`已包含leasing/additional cost。
        3. **Truck**：从第 3 个 sheet 开始读，换算时效为天，Cost 自动除以汇率。  
        4. **Buffer**：支持在网页端编辑。
        5. **生成总表**：以Buffer表中的线路为准，与Train Cost价格合并，最后与拖车费合并，得到 T-T、D-T、T-D 三种服务范围价格。  
        6. **特殊规则**：出口 W 减 Handling Fee 200 RMB，路线名和站点做特殊调整（如马拉-罗兹）。  
        7. **输出结果**：可下载 CSV，也可导出离线 HTML。  
        """)

# ---------- Column picking logic ----------
STD_COLS = {
    "Flow": ["Flow"],
    "Origin Terminal": ["Origin Terminal", "origin terminal", "origin", "pol", "始发站", "起运站"],
    "Route": ["Route", "Routing","route", "线路", "去程"],
    "Dest Terminal": ["dest terminal", "destination terminal", "pod", "目的站", "到达站"],
    "Service Scope": ["service scope", "scope", "服务范围"],
    "Lead-Time(Day)": ["lead-time(day)", "lead time", "transit", "tt", "时效", "天"],
    "Container Type": ["Container", "container type", "Container Type"],
    "Valid From": ["Valid From", "valid from", "起始", "有效期自"],
    "Valid To": ["Valid To", "截止", "有效期至"],
    "remark": ["remark", "remarks", "备注", "说明"]
}

NEED_ORDER = ["Flow", "Origin Terminal","Route","Dest Terminal","Service Scope","Lead-Time(Day)","Container Type", "cost", "Valid From","Valid To","remark"]

def pick_first_match(colnames, patterns):
    # 确保列名都是字符串
    colnames = [str(c) for c in colnames]
    col_lower_map = {c.lower().strip(): c for c in colnames}
    for p in patterns:
        if p.lower() in col_lower_map:
            return col_lower_map[p.lower()]
        for c in colnames:
            if p.lower() in c.lower():
                return c
    return None

# def detect_cost_col(colnames):
#     colnames = [str(c) for c in colnames]
#     for c in colnames:
#         lc = c.lower()
#         if "cost" in lc and "forecast" not in lc:
#             return c
#     return None

def detect_leasing_col(colnames):
    colnames = [str(c) for c in colnames]
    for c in colnames:
        if "leasing" in c.lower():
            return c
    return None

# # 用括号把整个数字模式包起来（有一个捕获组）
RE_NUM_ALL = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")

# 统一：把列名里的 \n / 不间断空格 / 多空格 去掉
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def norm(c):
        c = str(c).replace("\u00A0", " ")   # nbsp
        c = c.replace("\n", " ")
        c = re.sub(r"\s+", " ", c)
        return c.strip()
    df = df.copy()
    df.columns = [norm(c) for c in df.columns]
    return df

# 清理：把“表头文字跑到数据里”的行去掉（如某格 = 该列列名）
def _drop_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        mask = df[c].astype(str).str.strip().str.lower() == str(c).strip().lower()
        df = df.loc[~mask]
    return df

def normalize_train_cost(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    # ★ 新增：标准化列名 + 去掉误入的数据表头行
    df = _normalize_columns(df)
    df = _drop_header_rows(df)

    cols = [str(c) for c in df.columns.tolist()]
    out = pd.DataFrame()

    # 1) 固定字段映射（原来的 STD_COLS 不用改；列名已被标准化，\n 已变空格）
    for name in ["Flow","Origin Terminal","Route","Dest Terminal","Service Scope","Lead-Time(Day)","Container Type","Valid From","Valid To","remark"]:
        pats = STD_COLS.get(name, [name])
        col = pick_first_match(cols, pats)
        out[name] = df[col] if col else np.nan

    # 2) Lead-Time(Day) 数值化
    if "Lead-Time(Day)" in out.columns:
        out["Lead-Time(Day)"] = pd.to_numeric(out["Lead-Time(Day)"], errors="coerce")

    # 3) 提取数字的小工具（保留你的实现）
    def _num(s: pd.Series) -> pd.Series:
        s = s.astype(str).str.replace("\u00A0", " ", regex=False)
        s = s.str.replace(r"\(([^)]+)\)", r"-\1", regex=True).str.strip()
        lists = s.str.findall(RE_NUM_ALL)
        def _avg(lst):
            if not lst:
                return 0.0
            vals = [float(x.replace(",", "")) for x in lst]
            return float(np.mean(vals))
        return lists.apply(_avg).astype(float)

    # 4) leasing
    leasing_col = detect_leasing_col(cols)
    leasing_num = _num(df[leasing_col]) if leasing_col else pd.Series(0.0, index=df.index)
    out["leasing"] = leasing_num

    # 5) cost：把所有含 cost（排除 forecast）的列相加
    cost_cols = [c for c in cols if ("cost" in c.lower()) and ("forecast" not in c.lower())]
    if cost_cols:
        parts = [_num(df[c]).rename(c) for c in cost_cols]
        cost_sum = pd.concat(parts, axis=1).sum(axis=1)
    else:
        cost_sum = pd.Series(0.0, index=df.index)
    out["cost"] = cost_sum + leasing_num

    # 6) Container Type 兜底（列名已标准化，这步仍然保留更稳）
    container_cols = [c for c in cols if "container" in c.lower()]
    if container_cols:
        tmp = df[container_cols].astype(str).replace({"nan": np.nan, "None": np.nan}).bfill(axis=1)
        out["Container Type"] = tmp.iloc[:, 0].astype(str).str.strip()
        out.loc[out["Container Type"].isin(["", "nan", "None"]), "Container Type"] = np.nan

    # 7) 文本列去空格 & 统一大小写（可选：把 Flow 统一成大写）
    for c in ["Flow","Origin Terminal","Route","Dest Terminal","Service Scope","Valid From","Valid To","remark","Container Type"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()
    if "Flow" in out.columns:
        out["Flow"] = out["Flow"].astype(str).str.upper().str.strip()

    # 8) 列顺序
    ordered = [c for c in NEED_ORDER if c in out.columns]
    if "Container Type" in out.columns and "Container Type" not in ordered:
        ordered.append("Container Type")
    out = out.reindex(columns=ordered)

    # 如需隐藏中间列，保留你原注释：
    out.drop(columns=["leasing"], inplace=True, errors="ignore")

    out["__source_file__"] = source_file
    return out


# ---------- Main ----------
def build_total_cost_table(df_buffer, df_train, df_truck,
                           remark_text: str = "",
                           payload40: float = None,
                           overload_text: str = "",
                           note_text: str = "") -> pd.DataFrame:
    # ---------- 小工具 ----------
    def _norm_cols(df):
        d = df.copy()
        d.columns = [c.strip() for c in d.columns]
        return d

    def _as_num(s):
        return pd.to_numeric(s, errors="coerce")
    # ---------- 统一列名空格/大小写 ----------
    df_buffer = _norm_cols(df_buffer)
    df_train  = _norm_cols(df_train)
    df_truck  = _norm_cols(df_truck)

    # 关键字段统一为 string 去空格，避免合并时匹配失败
    for d in (df_buffer, df_train, df_truck):
        for k in ["Origin Terminal", "Route", "Dest Terminal", "Container Type"]:
            if k in d.columns:
                d[k] = d[k].astype("string").str.strip()

    # ---------- 1) Buffer + Train 四键合并，得到 T-T 基础 ----------
    merge_keys = ["Origin Terminal", "Route", "Dest Terminal", "Container Type"]
    missing_in_buffer = [k for k in merge_keys if k not in df_buffer.columns]
    missing_in_train  = [k for k in merge_keys if k not in df_train.columns]
    if missing_in_buffer:
        raise ValueError(f"Buffer 缺少字段: {missing_in_buffer}")
    if missing_in_train:
        raise ValueError(f"Train 缺少字段: {missing_in_train}")

    tt_base = df_buffer.merge(
        df_train,
        on=merge_keys,
        how="inner",
        suffixes=("", "_train")
    )

    # 数值化
    if "Buffer" not in tt_base.columns:
        raise ValueError("Buffer 表中缺少 'Buffer' 列")
    for c in ["Buffer", "cost", "leasing", "Lead-Time(Day)"]:
        if c in tt_base.columns:
            tt_base[c] = _as_num(tt_base[c]).fillna(0)

    # T-T 成本（按你的口径：Buffer + cost(包含leasing)）
    tt_base["TT_total"] = tt_base["Buffer"].fillna(0) + tt_base["cost"].fillna(0)
    # T-T 时效 = 纯火车段时效
    tt_base["TT_leadtime"] = tt_base["Lead-Time(Day)"].fillna(0)

    # 组装 T-T 记录（City/Province 置空）
    tt_final = pd.DataFrame({
        "Flow": tt_base["Flow"],
        "Pickup/Delivery City": "",
        "Province": "",
        "Origin Terminal": tt_base["Origin Terminal"],
        "Dest Terminal": tt_base["Dest Terminal"],
        "Route": tt_base["Route"],
        "Service Scope": "T-T",
        "Lead-Time(Day)": tt_base["TT_leadtime"],
        "Valid From": tt_base["Valid From"] if "Valid From" in tt_base.columns else "",
        "Valid To": tt_base["Valid To"] if "Valid To" in tt_base.columns else "",
        "Total Cost": tt_base["TT_total"]
    })

    def pick_col(df, candidates, default_val=""):
        for c in candidates:
            if c in df.columns:
                return df[c]
        return pd.Series([default_val] * len(df), index=df.index)

    # ---------- 2) D-T（出口，W）：城市→始发站 的拖车 + T-T ----------
    truck_w = df_truck[df_truck["Route"].astype(str).str.upper().str.strip() == "W"].copy()
    tt_w    = tt_base[tt_base["Flow"].astype(str).str.upper().str.strip() == "W"].copy()

    dt_merge = tt_w.merge(
        truck_w,
        on="Origin Terminal",
        how="left",
        suffixes=("", "_truck")
    )

    # 兼容不同列名（是否带 _truck 后缀）
    dt_merge["truck_cost"] = pd.to_numeric(pick_col(dt_merge, ["Cost_truck", "Cost"], 0), errors="coerce").fillna(0)
    dt_merge["truck_lt"]   = pd.to_numeric(pick_col(dt_merge, ["Lead-Time (Day)_truck", "Lead-Time (Day)"], 0), errors="coerce").fillna(0)
    dt_merge["DT_total"]    = dt_merge["TT_total"] + dt_merge["truck_cost"]
    dt_merge["DT_leadtime"] = dt_merge["TT_leadtime"] + dt_merge["truck_lt"]

    dt_final = pd.DataFrame({
        "Flow": dt_merge["Flow"],
        "Pickup/Delivery City": pick_col(dt_merge, ["Pickup/Delivery City", "Pickup/Delivery City_truck"]),
        "Province": pick_col(dt_merge, ["Province", "Province_truck"]),
        "Origin Terminal": dt_merge["Origin Terminal"],
        "Dest Terminal": dt_merge["Dest Terminal"],
        "Route": dt_merge["Route"],
        "Service Scope": "D-T",
        "Lead-Time(Day)": dt_merge["DT_leadtime"],
        "Valid From": pick_col(dt_merge, ["Valid From"], ""),
        "Valid To": pick_col(dt_merge, ["Valid To"], ""),
        "Total Cost": dt_merge["DT_total"],
    })

    # ---------- 3) T-D（进口，E）：到达站→城市 的拖车 + T-T ----------
    truck_e = df_truck[df_truck["Route"].astype(str).str.upper().str.strip() == "E"].copy()
    tt_e    = tt_base[tt_base["Flow"].astype(str).str.upper().str.strip() == "E"].copy()

    td_merge = tt_e.merge(
        truck_e,
        left_on="Dest Terminal",
        right_on="Origin Terminal",
        how="left",
        suffixes=("", "_truck")
    )

    td_merge["truck_cost"] = pd.to_numeric(pick_col(td_merge, ["Cost_truck", "Cost"], 0), errors="coerce").fillna(0)
    td_merge["truck_lt"]   = pd.to_numeric(pick_col(td_merge, ["Lead-Time (Day)_truck", "Lead-Time (Day)"], 0), errors="coerce").fillna(0)

    # 注意：合并后左表的火车“始发站”在 'Origin Terminal'，
    # 右表卡车的“起点站”（等于火车的到达站）在 'Origin Terminal_truck'
    td_merge["TD_total"]    = td_merge["TT_total"] + td_merge["truck_cost"]
    td_merge["TD_leadtime"] = td_merge["TT_leadtime"] + td_merge["truck_lt"]

    td_final = pd.DataFrame({
        "Flow": td_merge["Flow"],
        "Pickup/Delivery City": pick_col(td_merge, ["Pickup/Delivery City", "Pickup/Delivery City_truck"]),
        "Province": pick_col(td_merge, ["Province", "Province_truck"]),
        # 取火车“始发站”（左表）
        "Origin Terminal": pick_col(td_merge, ["Origin Terminal"]),
        "Dest Terminal": td_merge["Dest Terminal"],
        "Route": td_merge["Route"],
        "Service Scope": "T-D",
        "Lead-Time(Day)": td_merge["TD_leadtime"],
        "Valid From": pick_col(td_merge, ["Valid From"], ""),
        "Valid To": pick_col(td_merge, ["Valid To"], ""),
        "Total Cost": td_merge["TD_total"],
    })


    # ---------- 4) 合并三类记录 ----------
    final_df = pd.concat([tt_final, dt_final, td_final], ignore_index=True)

    # ---------- 5) Handling Fee ----------
    mask_w   = final_df["Flow"].astype(str).str.upper().str.strip() == "W"
    # 只对 W 的行减 200
    final_df.loc[final_df["Flow"] == "W", "Total Cost"] -= 200
    final_df["Handling Fee"] = np.where(final_df["Flow"].astype(str).str.upper().str.strip() == "W", 200, 0)

    # ---------- 6) 列顺序与数值类型 ----------
    final_df["Lead-Time(Day)"] = _as_num(final_df["Lead-Time(Day)"])
    final_df["Total Cost"]     = _as_num(final_df["Total Cost"])
    # final_df = final_df[[c for c in final_cols if c in final_df.columns]]
    final_df["Route"] = final_df["Route"].replace({
        "传统线路": "Public",
        "传统路线": "Public",
        "全程时刻表": "Super Express"
    })
    def _norm(s: pd.Series) -> pd.Series:
        return (s.astype(str)
                .str.strip()
                .str.replace(r"[\u2018\u2019\u2032]", "'", regex=True)  
                .str.replace(r"[\u2013\u2014\u2212]", "-", regex=True) 
                .str.upper())

    # 1. 成都 → Lodz 的 Super Express 改成 成都 → Malaszewicze 的 Super Express
    mask_out = (
        _norm(final_df["Origin Terminal"]).eq("CHENGDU") &
        _norm(final_df["Dest Terminal"]).eq("LODZ") &
        _norm(final_df["Route"]).eq("SUPER EXPRESS")
    )
    final_df.loc[mask_out, "Dest Terminal"] = "Malaszewicze"

    # 2. Lodz → 成都 的 Super Express 改成 Malaszewicze → 成都 的 Super Express
    mask_in = (
        _norm(final_df["Origin Terminal"]).eq("LODZ") &
        _norm(final_df["Dest Terminal"]).eq("CHENGDU") &
        _norm(final_df["Route"]).eq("SUPER EXPRESS")
    )
    final_df.loc[mask_in, "Origin Terminal"] = "Malaszewicze"


    # ---------- Remark & 全局附加列 ----------
    final_df["Remark"] = remark_text or ""
    final_df["Extra Cost of Overload"] = overload_text or ""
    mask_cd_ml = (
    ((final_df["Origin Terminal"] == "Chengdu") & (final_df["Dest Terminal"] == "Malaszewicze")) |
    ((final_df["Origin Terminal"] == "Malaszewicze") & (final_df["Dest Terminal"] == "Chengdu")))
    final_df["Note"] = np.where(mask_cd_ml, note_text, "")
    final_df["40' Payload Limited (ton)"] = payload40 if payload40 is not None else ""

    # ---------- 列顺序 ----------
    ordered_cols = [
        "Flow","Pickup/Delivery City","Province","Origin Terminal","Dest Terminal","Route",
        "Service Scope","Lead-Time(Day)","Valid From","Valid To",
        "Total Cost","Handling Fee",
        "40' Payload Limited (ton)","Extra Cost of Overload",
        "Note","Remark"
    ]
    final_df = final_df[[c for c in ordered_cols if c in final_df.columns]]
    return final_df

tab1, tab2 = st.tabs(["Data", "Text"])
# ---------- Train Data Processing ----------
with tab1:
    st.number_input("USD:CNY Rate", min_value=0.0, step=0.01, value=7.0, key="rate")
    if train_files:
        dfs = []
        for f in train_files:
            try:
                df_raw = pd.read_excel(f)
            except Exception:
                f.seek(0)
                df_raw = pd.read_excel(f, sheet_name=0, engine="openpyxl")
            norm = normalize_train_cost(df_raw, f.name)
            dfs.append(norm)
        all_train = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=NEED_ORDER)
        df_train = all_train.dropna(how="all") 
        df_train = df_train.dropna(subset=["cost"])
        # 1) 规范日期列
        df_train["Valid From"] = pd.to_datetime(df_train["Valid From"], errors="coerce")
        df_train["Valid To"]   = pd.to_datetime(df_train["Valid To"],   errors="coerce")

        # 2) 仅保留 Valid From 在每月 1 号的价格
        mask_first = df_train["Valid From"].dt.day == 1
        df_train = df_train.loc[mask_first].copy()

        # 3) 把 Valid To 设为当月最后一天
        df_train["Valid To"] = df_train["Valid From"] + pd.offsets.MonthEnd(0)

        df_train["Valid From"] = df_train["Valid From"].dt.date
        df_train["Valid To"]   = df_train["Valid To"].dt.date
        with st.expander("查看合并后的Train Cost"):
            st.success(f"Train Cost 已加载，合并后 {len(df_train)} 行。")
            # df_train = st.data_editor(df_train, use_container_width=True)
            st.dataframe(df_train[NEED_ORDER], use_container_width=True)
            
    # ---------- Truck Data Processing ----------
    if truck_file:
        dfs_truck = []
        xls = pd.ExcelFile(truck_file)  # 读取 Truck 文件
        sheets = xls.sheet_names[2:]  # 从第三个 sheet 开始读取
        for sheet in sheets:
            df_truck_raw = pd.read_excel(xls, sheet_name=sheet)
            dfs_truck.append(df_truck_raw)
        all_truck = pd.concat(dfs_truck, ignore_index=True) if dfs_truck else pd.DataFrame()
        # 去掉空白列
        all_truck = all_truck.dropna(axis=1, how='all')
        # 去掉空白行
        all_truck = all_truck.dropna(axis=0, how='all')
        if 'Lead-Time（hour)' in all_truck.columns:
            all_truck['Lead-Time (Day)'] = np.ceil(all_truck['Lead-Time（hour)'] / 24).astype(int)
            all_truck.drop('Lead-Time（hour)', axis=1, inplace=True)
        # 修改列名
        origin_col = [col for col in all_truck.columns if 'origin terminal' in col.lower()]
        if origin_col:
            all_truck['Origin Terminal'] = all_truck[origin_col]
            all_truck.drop(origin_col, axis=1, inplace=True)  
        route_col = [col for col in all_truck.columns if 'route' in col.lower()]
        if route_col:
            all_truck['Route'] = all_truck[route_col]
            all_truck.drop(route_col, axis=1, inplace=True)  
        pro_col = [col for col in all_truck.columns if 'province' in col.lower()]
        if pro_col:
            all_truck['Province'] = all_truck[pro_col]
            all_truck.drop(pro_col, axis=1, inplace=True)  

        # 处理 cost 列，找到包含 cost 的列并除以 7
        usd_cny = float(st.session_state.get("rate", 7.0) or 7.0) 
        cost_col = [col for col in all_truck.columns if 'cost' in col.lower()]
        if cost_col:
            all_truck['Cost'] = np.round(all_truck[cost_col[0]] / usd_cny).astype(int)
            all_truck.drop(cost_col[0], axis=1, inplace=True)
        all_truck['Valid From'] = pd.to_datetime(all_truck['Valid From'], errors='coerce').dt.date
        all_truck['Valid To'] = pd.to_datetime(all_truck['Valid To'], errors='coerce').dt.date
        # Reorder columns
        column_order = [
            "Route",
            "Origin Terminal",
            "Pickup/Delivery City",
            "Province",
            "Lead-Time (Day)",
            "Valid From",
            "Valid To",
            "Cost"
        ]
        # Reorder the columns as per the new column_order list
        #df_truck = all_truck[column_order] if all(col in all_truck.columns for col in column_order) else all_truck
        df_truck = all_truck[column_order]
        # 显示合并后的 Truck 数据
        with st.expander("查看拖车费数据"):
            st.success(f"拖车费数据已加载，共{len(df_truck)} 行。")
            st.dataframe(df_truck, use_container_width=True)

    # ---------- Buffer Data Processing ----------
    if buffer_file:
        all_buffer = pd.read_excel(buffer_file)
        columns = ["Origin Terminal", "Route", "Dest Terminal", "Container Type", "Buffer"]
        df_buffer = all_buffer[columns]
        def _norm(s: pd.Series) -> pd.Series:
            return (s.astype(str)
                .str.strip()
                .str.replace(r"[\u2018\u2019\u2032]", "'", regex=True)  
                .str.replace(r"[\u2013\u2014\u2212]", "-", regex=True) 
                .str.upper())  # 如需大小写不敏感匹配
        df_buffer = df_buffer.dropna(how="all")
        mask_1 = (
            _norm(df_buffer["Origin Terminal"]).eq("MALASZEWICZE") &
            _norm(df_buffer["Dest Terminal"]).eq("CHENGDU")
            )
        df_buffer.loc[mask_1, ["Origin Terminal"]] = ["Lodz"]
        mask_2 = (
            _norm(df_buffer["Origin Terminal"]).eq("CHENGDU") &
            _norm(df_buffer["Dest Terminal"]).eq("MALASZEWICZE")
            )
        df_buffer.loc[mask_2, ["Route","Dest Terminal"]] = ["全程时刻表","Lodz"]
        st.session_state.setdefault("buffer_expanded", False)  # 默认不展开
        with st.expander("查看Buffer数据", expanded=st.session_state["buffer_expanded"]):
            st.success(f"Buffer数据已加载，共{len(df_buffer)} 行。")
            # 首次进入时，把原数据放到会话里，避免编辑时闪回
            if "buffer_edit_df" not in st.session_state:
                st.session_state["buffer_edit_df"] = df_buffer.copy()
            with st.form("buffer_edit_form"):
                edited = st.data_editor(
                    st.session_state["buffer_edit_df"],
                    use_container_width=True,
                    hide_index=True,
                    num_rows="dynamic",
                    column_config={
                        "Add_Fixed": st.column_config.NumberColumn("Add_Fixed", step=1.0, format="%.2f"),
                        "Add_Percent": st.column_config.NumberColumn("Add_Percent", step=0.01, format="%.4f",
                                            help="可输入0.05或5%；后台可再统一为小数"),
                        "remark": st.column_config.TextColumn("remark", max_chars=200),
                    },
                )
                saved = st.form_submit_button("保存 Buffer 编辑")
            if saved:
                st.session_state["buffer_edit_df"] = edited.copy()
                st.success("已保存 Buffer 编辑内容。")
        df_buffer = st.session_state["buffer_edit_df"]

with tab2:
    st.header("Global Settings")
    defaults = {
        "payload40": 20,
        "payload20": 20,
        "overload_text": "250USD/40'/20'(20-23 ton)",
        "note_text": "Actual arrival/departure station is Lodz i/o Malaszewicze.",
        "remark_text": (
            "1. WB leadtime include CN terminal customs landing time\n"
            "2. EB leadtime exclude CN terminal customs lead time of estimated 1-2 days\n"
            "3. Overload cost in case cargo weight more than 23 ton, please check with us case by case.\n"
            "4. Due to impact of Chinese New Year, if your pre/on carriage happen in week 4 to week 7, trucking fee will increase 30%."
        )
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    st.number_input("40' Payload Limited (ton)", min_value=0, step=1, key="payload40")
    st.text_input("Extra Cost of Overload", key="overload_text")
    st.text_area("Remark", height=160, key="remark_text")

# ---------- Export HTML -------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------
COLUMN_MAP = {
    # 你已有的原始列名 → HTML 期望的列名（右边是最终输出给 HTML 的键）
    "From": "From",
    "Province": "Province",
    "To": "To",
    "Origin Terminal": "Origin Terminal",
    "Dest Terminal": "Dest Terminal",
    "Service Scope": "Service Scope",
    "Route": "Route",
    "Lead-Time(Day)": "Lead-Time(Day)",
    "Total Cost": "Total Cost",
    "Handling Fee": "Handling Fee",
    "Valid From": "Valid From",
    "Valid To": "Valid To",
    "40' Payload Limited (ton)": "40' Payload Limited (ton)", 
    "Extra Cost of Overload": "Extra Cost of Overload",
    "Note": "Note",
    "Remark": "Remark"
}

REQUIRED_COLS = [
    "From", "Province", "To",
    "Origin Terminal", "Dest Terminal",
    "Service Scope", "Route",
    "Lead-Time(Day)",
    "Total Cost", "Handling Fee",
    "Valid From", "Valid To",
    "40' Payload Limited (ton)",
    "Extra Cost of Overload",
    "Note","Remark"
]

def _prepare_df_for_html(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # 统一生成/复制输出列
    for src, dst in COLUMN_MAP.items():
        if src in d.columns:
            if dst != src:
                d[dst] = d[src]
        else:
            d[dst] = np.nan

    # 只保留前端需要的列（并按顺序）
    d = d[REQUIRED_COLS]

    # 数值列：保持为 float
    for col in ["Total Cost", "Handling Fee"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0).astype(float)

    # 日期列：转成字符串（避免 JSON 序列化成时间戳或 NaT）
    for col in ["Valid From", "Valid To"]:
        if col in d.columns:
            d[col] = d[col].astype(str).replace({"NaT": "", "nat": ""})

    # 其他空值 → None（让 JSON 为 null）
    d = d.where(pd.notna(d), None)

    return d
import json, re, base64
def inject_df_into_html(df: pd.DataFrame, template_path: str) -> str:
    html = Path(template_path).read_text(encoding="utf-8")

    # DataFrame -> JSON（确保数字是 number、日期是字符串）
    payload = df.to_dict(orient="records")
    json_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    b64 = base64.b64encode(json_bytes).decode("ascii")

    # 用 Base64 注入，前端用 atob 再 JSON.parse，还原成数组
    pattern = r"const\s+csvData\s*=\s*\[[\s\S]*?\];"
    replacement = f"const csvData = JSON.parse(atob('{b64}'));"
    new_html = re.sub(pattern, replacement, html)

    return new_html


if buffer_file and train_files and truck_file:
    # —— 生成 final df —— #
    final_df = build_total_cost_table(
        df_buffer=df_buffer,
        df_train=df_train,
        df_truck=df_truck,
        remark_text=st.session_state["remark_text"],
        payload40=st.session_state["payload40"],
        overload_text=st.session_state["overload_text"],
        note_text=st.session_state["note_text"],
    )
    def _norm(s):
        if pd.isna(s):
            return ""
        return str(s).strip()

    def _norm_scope(s):
        # 统一成大写、去空格、统一短横线
        s = _norm(s).upper().replace(" ", "")
        s = s.replace("—", "-").replace("–", "-")
        return s

    def _norm_flow(s):
        return _norm(s).upper()

    def _compute_from(row):
        flow = _norm_flow(row["Flow"])
        scope = _norm_scope(row["Service Scope"])
        # T-T：两端都是站点
        if scope == "T-T":
            return row["Origin Terminal"]
        # 其它 scope：按 W/E 规则
        if flow == "W":
            return row["Pickup/Delivery City"]
        elif flow == "E":
            return row["Origin Terminal"]
        # 兜底（未知 flow）：优先起点站，否则城市
        return row["Origin Terminal"] if _norm(row["Origin Terminal"]) else row["Pickup/Delivery City"]

    def _compute_to(row):
        flow = _norm_flow(row["Flow"])
        scope = _norm_scope(row["Service Scope"])
        if scope == "T-T":
            return row["Dest Terminal"]
        if flow == "W":
            return row["Dest Terminal"]
        elif flow == "E":
            return row["Pickup/Delivery City"]
        # 兜底（未知 flow）：优先目的站，否则城市
        return row["Dest Terminal"] if _norm(row["Dest Terminal"]) else row["Pickup/Delivery City"]

    final_df["From"] = final_df.apply(_compute_from, axis=1)
    final_df["To"]   = final_df.apply(_compute_to,   axis=1)

    # 如需把 From/To 靠前显示，可重排列顺序（可选）
    prefer_order = ["Flow", "Service Scope", "From", "To"]
    rest = [c for c in final_df.columns if c not in prefer_order]
    
    # 生成df_final
    final_df = final_df[prefer_order + rest]

    st.success(f"总表生成完成：{len(final_df)} 行")

    with st.expander("预览"):
        st.dataframe(final_df, use_container_width=True)
        st.download_button(
            "📥 下载合并结果 CSV",
            data=final_df.to_csv(index=False).encode("utf-8"),
            file_name="RC_Raw_DB.csv",
            mime="text/csv"
        )

    st.header("Export Rate Card HTML")    
    if "final_df" in globals():
        df_ready = _prepare_df_for_html(final_df)
        html_template_path = "RC.html"
        if Path(html_template_path).exists():
            if st.button("📤 导出 Rate Card HTML", use_container_width=True):
                out_html = inject_df_into_html(df_ready, html_template_path)
                st.download_button(
                    label="下载 Rate Card.html",
                    data=out_html.encode("utf-8"),
                    file_name="RateCard.html",
                    mime="text/html",
                    use_container_width=True
                )
                st.success("已生成并可下载离线可用的 HTML。")
        else:
            st.error("找不到模板文件 Rate Card.html，请把模板放到程序同目录。")
    else:
        st.warning("final_df 尚未生成或未在当前作用域。请先生成 final_df。")
else:
    st.info("请上传完整表格。")


if "final_df" in locals() and isinstance(final_df, pd.DataFrame) and not final_df.empty:
    if not st.session_state["hide_intro"]:
        st.session_state["hide_intro"] = True
        st.rerun()   # 立刻重绘，这样本次就不再显示横幅



