import streamlit as st
import pandas as pd
import datetime
import altair as alt
import numpy as np
import io
import re
import textwrap, os, re, io, json, math, datetime as dt
from typing import List, Dict, Optional, Tuple

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
st.title("Rate Card Generator")

st.markdown("""
<div style='padding: 1rem; border-radius: 0.5rem; background-color: #F0F7F6;'>
  <h3 style='color: #225560;'>👋 欢迎使用Rate Card生成器</h3>
  <p style='font-size:16px; color:#444;'>
    本工具帮助您通过 <b>Train Cost</b>，<b>Buffer Table</b> 与 <b>FCL Net Cost</b>，
    快速生成Rate Card，体现各条路线的价格。
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)  # ⬅︎ 留白高度自定义

# -------- Sidebar Uploads --------
with st.sidebar:
    st.header("Upload files")
    train_files = st.file_uploader("Train Cost · 班列站到站 (support multiple files)", type=["xls", "xlsx"], accept_multiple_files=True)
    buffer_file = st.file_uploader("Buffer", type=["xls", "xlsx"])
    truck_file = st.file_uploader("Truck · 门到站拖车费", type=["xls", "xlsx"])
    st.markdown("---")
    st.markdown("> 说明：本应用**不保存**任何数据；所有处理都在内存完成。")

# ---------- Column picking logic ----------
STD_COLS = {
    "Flow": ["Flow"],
    "Origin Terminal": ["origin terminal", "origin", "pol", "始发站", "起运站"],
    "Route": ["Route", "Routing","route", "线路", "去程"],
    "Dest Terminal": ["dest terminal", "destination terminal", "pod", "目的站", "到达站"],
    "Service Scope": ["service scope", "scope", "服务范围"],
    "Lead-Time(Day)": ["lead-time(day)", "lead time", "transit", "tt", "时效", "天"],
    "valid from": ["valid from", "起始", "有效期自"],
    "valid to": ["valid to", "截止", "有效期至"],
    "remark": ["remark", "remarks", "备注", "说明"],
}

NEED_ORDER = ["Flow", "Origin Terminal","Route","Dest Terminal","Service Scope","Lead-Time(Day)","container type", "cost","leasing", "valid from","valid to","remark"]

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

def detect_cost_col(colnames):
    colnames = [str(c) for c in colnames]
    for c in colnames:
        lc = c.lower()
        if "cost" in lc and "forecast" not in lc:
            return c
    return None

def detect_leasing_col(colnames):
    colnames = [str(c) for c in colnames]
    for c in colnames:
        if "leasing" in c.lower():
            return c
    return None

# def normalize_train_cost(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
#     cols = df.columns.tolist()
#     out = pd.DataFrame()

#     # map fixed columns
#     for std_name, pats in STD_COLS.items():
#         src = pick_first_match(cols, pats)
#         if src is not None:
#             out[std_name] = df[src]
#         else:
#             out[std_name] = np.nan

#     # numbers
#     lt_col = pick_first_match(cols, STD_COLS["Lead-Time(Day)"])
#     if lt_col: out["Lead-Time(Day)"] = pd.to_numeric(df[lt_col], errors="coerce")

#     cost_col = detect_cost_col(cols)
#     leasing_col = detect_leasing_col(cols)
#     out["cost"] = pd.to_numeric(df[cost_col], errors="coerce") if cost_col else np.nan
#     out["leasing"] = pd.to_numeric(df[leasing_col], errors="coerce") if leasing_col else np.nan

#     # trim spaces
#     for c in ["Origin Terminal","Route","Dest Terminal","Service Scope","valid from","valid to","remark"]:
#         out[c] = out[c].astype(str).str.strip()

#     # Keep only the specified order
#     out = out[[c for c in NEED_ORDER if c in out.columns]]

#     # Attach source file for debugging (not exported)
#     out["__source_file__"] = source_file
#     return out
# 用括号把整个数字模式包起来（有一个捕获组）
RE_NUM = r'(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)'


def normalize_train_cost(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    cols = [str(c) for c in df.columns.tolist()]
    out = pd.DataFrame()

    # 1) 固定字段映射
    for name in ["Flow","Origin Terminal","Route","Dest Terminal","Service Scope","Lead-Time(Day)","valid from","valid to","remark"]:
        pats = STD_COLS.get(name, [name])
        col = pick_first_match(cols, pats)
        out[name] = df[col] if col else np.nan

    # Lead-Time(Day) 数值化
    if "Lead-Time(Day)" in out.columns:
        out["Lead-Time(Day)"] = pd.to_numeric(out["Lead-Time(Day)"], errors="coerce")

    # 小工具：提取第一个数字
    def _num(s: pd.Series) -> pd.Series:
        s = s.astype(str).str.replace("\u00A0", " ", regex=False).str.strip()
        x = s.str.extract(RE_NUM, expand=False)
        x = x.str.replace(",", "", regex=False)
        return pd.to_numeric(x, errors="coerce").fillna(0.0)

    # 2) leasing：数值化
    leasing_col = detect_leasing_col(cols)
    leasing_num = _num(df[leasing_col]) if leasing_col else pd.Series(0.0, index=df.index)
    out["leasing"] = leasing_num

    # 3) cost：把所有含 cost（排除 forecast）的列相加
    cost_cols = [c for c in cols if ("cost" in c.lower()) and ("forecast" not in c.lower())]
    if cost_cols:
        parts = [_num(df[c]).rename(c) for c in cost_cols]
        cost_sum = pd.concat(parts, axis=1).sum(axis=1)
    else:
        cost_sum = pd.Series(0.0, index=df.index)

    # 最终 cost = 所有 cost 之和 + leasing
    out["cost"] = cost_sum + leasing_num

    # 4) container type：取含 container 的列的第一个非空
    container_cols = [c for c in cols if "container" in c.lower()]
    if container_cols:
        tmp = df[container_cols].astype(str).replace({"nan": np.nan, "None": np.nan}).bfill(axis=1)
        out["container type"] = tmp.iloc[:, 0].astype(str).str.strip()
        out.loc[out["container type"].isin(["", "nan", "None"]), "container type"] = np.nan

    # 5) 文本列去空格 
    for c in ["Flow","Origin Terminal","Route","Dest Terminal","Service Scope","valid from","valid to","remark","container type"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()

    # 6) 列顺序
    ordered = [c for c in NEED_ORDER if c in out.columns]  
    if "container type" in out.columns and "container type" not in ordered:
        ordered.append("container type")
    out = out.reindex(columns=ordered)

    out["__source_file__"] = source_file
    return out

def fix_leasing(series):
    s = pd.Series(series).astype(str).str.replace("\u00A0"," ",regex=False).str.strip()
    x = s.str.extract(RE_NUM, expand=False).str.replace(",", "", regex=False)
    return pd.to_numeric(x, errors="coerce").fillna(0.0)

# ---------- Train Data Processing ----------
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
    #df_train["leasing"] = fix_leasing(df_train["leasing"])

    # show combined table
    with st.expander("查看合并后的Train Cost"):
        st.success(f"Train Cost 已加载，合并后 {len(df_train)} 行。")
        st.dataframe(df_train[NEED_ORDER], use_container_width=True)
else:
    st.info("请在左侧上传 Train Cost 文件（可多选）。")

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
    cost_col = [col for col in all_truck.columns if 'cost' in col.lower()]
    if cost_col:
        all_truck['Cost'] = np.round(all_truck[cost_col[0]] / 7).astype(int)
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
    with st.expander("查看合并后的拖车费数据"):
        st.success(f"拖车费数据已加载，合并后 {len(df_truck)} 行。")
        st.dataframe(df_truck, use_container_width=True)
else:
    st.info("请在左侧上传 FCL Net Cost 文件。")

# ---------- Buffer Data Processing ----------
if buffer_file:
    all_buffer = pd.read_excel(buffer_file)
    columns = ["Origin Terminal", "Route", "Dest Terminal", "Container Type", "Buffer", "RC"]
    df_buffer = all_buffer[columns]
    df_buffer = df_buffer.dropna(how="all")
    with st.expander("查看Buffer数据"):
        st.success(f"Buffer数据已加载，合并后 {len(df_buffer)} 行。")
        st.dataframe(df_buffer, use_container_width=True)
else:
    st.info("请在左侧上传 Buffer 文件。")

# ---------- Main ----------
def _pick_first(colnames, keys):
    colnames = [str(c) for c in colnames]
    for k in keys:
        k = k.lower()
        for c in colnames:
            if k in c.lower():
                return c
    return None

def normalize_truck(df_truck: pd.DataFrame) -> pd.DataFrame:
    """
    规范化拖车表 -> 长表：
    返回列：City, Terminal, Direction(Export/Import), Truck_Cost, Truck_Buffer
    支持两种形态：
      1) 只有一列 Cost（则视为进出口同价）
      2) 分别有 Export/Import 成本列；可选 Buffer 列
    """
    df = df_truck.copy()
    cols = df.columns.tolist()

    city_col = _pick_first(cols, ["pickup/delivery city", "city", "国内城市"])
    # 站点列可能叫 Origin Terminal / Dest Terminal（你的整理里保留了 Origin Terminal）
    term_col = _pick_first(cols, ["origin terminal", "terminal", "站点", "ramp", "station"])
    if city_col is None or term_col is None:
        st.error("拖车表缺少城市或站点列（如 Pickup/Delivery City, Origin Terminal）。")
        return pd.DataFrame(columns=["City","Terminal","Direction","Truck_Cost","Truck_Buffer"])

    # 价格列
    exp_cost_col = _pick_first(cols, ["export cost", "export_rate", "export", "exp cost"])
    imp_cost_col = _pick_first(cols, ["import cost", "import_rate", "import", "imp cost"])
    one_cost_col = _pick_first(cols, ["cost", "price", "rate"])  # 只有单列时使用
    # 拖车 buffer（可选）
    exp_buf_col = _pick_first(cols, ["export buffer", "exp buffer", "export surcharge"])
    imp_buf_col = _pick_first(cols, ["import buffer", "imp buffer", "import surcharge"])
    one_buf_col = _pick_first(cols, ["buffer", "surcharge", "add"])  # 单列 buffer（少见）

    out = []

    def to_num(x):
        return pd.to_numeric(x, errors="coerce")

    if exp_cost_col or imp_cost_col:
        # 分列版：逐行拉平
        for _, r in df.iterrows():
            # Export 行
            if exp_cost_col:
                out.append({
                    "City": r[city_col],
                    "Terminal": r[term_col],
                    "Direction": "Export",
                    "Truck_Cost": to_num(r.get(exp_cost_col, 0)) if exp_cost_col else 0,
                    "Truck_Buffer": to_num(r.get(exp_buf_col, 0)) if exp_buf_col else 0,
                })
            # Import 行
            if imp_cost_col:
                out.append({
                    "City": r[city_col],
                    "Terminal": r[term_col],
                    "Direction": "Import",
                    "Truck_Cost": to_num(r.get(imp_cost_col, 0)) if imp_cost_col else 0,
                    "Truck_Buffer": to_num(r.get(imp_buf_col, 0)) if imp_buf_col else 0,
                })
    else:
        # 单列版：复制两行（Export/Import同价）
        if one_cost_col is None:
            st.error("拖车表未找到成本列（如 Cost/Rate/Price）。")
            return pd.DataFrame(columns=["City","Terminal","Direction","Truck_Cost","Truck_Buffer"])

        for _, r in df.iterrows():
            for d in ["Export", "Import"]:
                out.append({
                    "City": r[city_col],
                    "Terminal": r[term_col],
                    "Direction": d,
                    "Truck_Cost": to_num(r.get(one_cost_col, 0)),
                    "Truck_Buffer": to_num(r.get(one_buf_col, 0)) if one_buf_col else 0,
                })

    out = pd.DataFrame(out)
    if out.empty:
        return out

    # 清洗
    out["City"] = out["City"].astype(str).str.strip()
    out["Terminal"] = out["Terminal"].astype(str).str.strip()
    out["Direction"] = out["Direction"].astype(str)
    out["Truck_Cost"] = pd.to_numeric(out["Truck_Cost"], errors="coerce").fillna(0.0)
    out["Truck_Buffer"] = pd.to_numeric(out["Truck_Buffer"], errors="coerce").fillna(0.0)

    # 去掉无效
    out = out[(out["City"] != "") & (out["Terminal"] != "")]
    return out

def normalize_rail(df_train: pd.DataFrame, use_leasing=True) -> pd.DataFrame:
    """
    规范化铁路费（含线路）：
    输入：df_train 至少含 Origin Terminal, Dest Terminal, cost；若含 Route/Route 也会带上
    输出列：Origin Terminal, Dest Terminal, Route(可选), Rail_Cost
    """
    need_cols_base = ["Origin Terminal", "Dest Terminal", "cost"]
    for c in need_cols_base:
        if c not in df_train.columns:
            st.error(f"Train Cost 缺少列：{c}")
            return pd.DataFrame(columns=["Origin Terminal","Dest Terminal","Route","Rail_Cost"])

    rail = df_train.copy()
    rail["Origin Terminal"] = rail["Origin Terminal"].astype(str).str.strip()
    rail["Dest Terminal"]   = rail["Dest Terminal"].astype(str).str.strip()

    # 找线路列（Route/Route/线路）
    Route_col = None
    Route_col = Route_col or ("Route" if "Route" in rail.columns else None)
    Route_col = Route_col or _pick_first(rail.columns, ["route", "线路"])
    if Route_col and Route_col != "Route":
        rail["Route"] = rail[Route_col]
    elif "Route" not in rail.columns:
        rail["Route"] = np.nan  # 统一列存在，可能全空

    rail_cost = pd.to_numeric(rail["cost"], errors="coerce").fillna(0.0)
    if use_leasing and "leasing" in rail.columns:
        rail_cost = rail_cost + pd.to_numeric(rail["leasing"], errors="coerce").fillna(0.0)
    rail["Rail_Cost"] = rail_cost

    key_cols = ["Origin Terminal", "Dest Terminal", "Route"]
    rail_out = rail.groupby(key_cols, as_index=False)["Rail_Cost"].sum()
    return rail_out

def normalize_buffer(df_buffer: pd.DataFrame) -> pd.DataFrame:
    """
    规范化铁路段 Buffer：
    优先使用含线路键 (Origin, Dest, Route)，否则使用 (Origin, Dest) 并对各 Route 广播。
    输出：Origin Terminal, Dest Terminal, Route(可能为空), Rail_Buffer
    """
    base_need = ["Origin Terminal", "Dest Terminal", "Buffer"]
    for c in base_need:
        if c not in df_buffer.columns:
            st.error(f"Buffer 缺少列：{c}")
            return pd.DataFrame(columns=["Origin Terminal","Dest Terminal","Route","Rail_Buffer"])

    buf = df_buffer.copy()
    buf["Origin Terminal"] = buf["Origin Terminal"].astype(str).str.strip()
    buf["Dest Terminal"]   = buf["Dest Terminal"].astype(str).str.strip()
    buf["Buffer"] = pd.to_numeric(buf["Buffer"], errors="coerce").fillna(0.0)

    # 识别线路列
    Route_col = None
    Route_col = Route_col or ("Route" if "Route" in buf.columns else None)
    Route_col = Route_col or _pick_first(buf.columns, ["route", "线路"])
    if Route_col and Route_col != "Route":
        buf["Route"] = buf[Route_col]
    elif "Route" not in buf.columns:
        buf["Route"] = np.nan

    # 如果 Buffer 没有线路维度，就只按 (O,D) 聚合
    if buf["Route"].isna().all():
        buf_out = buf.groupby(["Origin Terminal","Dest Terminal"], as_index=False)["Buffer"].sum()
        buf_out["Route"] = np.nan
    else:
        buf_out = buf.groupby(["Origin Terminal","Dest Terminal","Route"], as_index=False)["Buffer"].sum()

    buf_out = buf_out.rename(columns={"Buffer": "Rail_Buffer"})
    return buf_out

def build_total(df_truck_long: pd.DataFrame,
                df_rail: pd.DataFrame,
                df_buf: pd.DataFrame) -> pd.DataFrame:
    """
    生成总表（含 Route 维度）：
      Export:  City→Origin(拖车 Export) + Origin→Dest(铁路, 路线)
      Import:  Origin→Dest(铁路, 路线) + Dest→City(拖车 Import)
    输出列：City, Origin Terminal, Dest Terminal, Route, Direction, Cost, Buffer, Total
    """
    # ---- 将铁路 Buffer 合到铁路，优先按 (O,D,Route) 匹配，匹配不到时退化为 (O,D) ----
    rail = df_rail.copy()
    # 先尝试按三键 merge
    merged = rail.merge(
        df_buf, on=["Origin Terminal","Dest Terminal","Route"], how="left"
    )
    # 对于三键没匹配到且 buf 只有 (O,D) 的情况，再用 (O,D) 回填
    if "Rail_Buffer" not in merged.columns:
        merged["Rail_Buffer"] = 0.0
    needs_fill = merged["Rail_Buffer"].isna()
    if needs_fill.any():
        od_buf = df_buf[df_buf["Route"].isna()][["Origin Terminal","Dest Terminal","Rail_Buffer"]].drop_duplicates()
        merged.loc[needs_fill, "Rail_Buffer"] = merged[needs_fill].merge(
            od_buf, on=["Origin Terminal","Dest Terminal"], how="left"
        )["Rail_Buffer_y"].values
    merged["Rail_Buffer"] = pd.to_numeric(merged["Rail_Buffer"], errors="coerce").fillna(0.0)

    rail = merged  # 现在 rail 里有 Rail_Cost 与 Rail_Buffer，与 Route 对齐

    # ---- Export：Truck(Direction=Export) 连接 Origin Terminal ----
    truck_exp = df_truck_long[df_truck_long["Direction"] == "Export"].copy()
    exp = rail.merge(truck_exp, left_on="Origin Terminal", right_on="Terminal", how="left")
    exp_cost = pd.to_numeric(exp["Rail_Cost"], errors="coerce").fillna(0.0) + \
               pd.to_numeric(exp["Truck_Cost"], errors="coerce").fillna(0.0)
    exp_buf  = pd.to_numeric(exp["Rail_Buffer"], errors="coerce").fillna(0.0) + \
               pd.to_numeric(exp["Truck_Buffer"], errors="coerce").fillna(0.0)
    exp_out = pd.DataFrame({
        "City": exp["City"],
        "Origin Terminal": exp["Origin Terminal"],
        "Dest Terminal": exp["Dest Terminal"],
        "Route": exp["Route"],
        "Direction": "Export",
        "Cost": exp_cost.round(2),
        "Buffer": exp_buf.round(2),
    })
    exp_out["Total"] = (exp_out["Cost"] + exp_out["Buffer"]).round(2)

    # ---- Import：Truck(Direction=Import) 连接 Dest Terminal ----
    truck_imp = df_truck_long[df_truck_long["Direction"] == "Import"].copy()
    imp = rail.merge(truck_imp, left_on="Dest Terminal", right_on="Terminal", how="left")
    imp_cost = pd.to_numeric(imp["Rail_Cost"], errors="coerce").fillna(0.0) + \
               pd.to_numeric(imp["Truck_Cost"], errors="coerce").fillna(0.0)
    imp_buf  = pd.to_numeric(imp["Rail_Buffer"], errors="coerce").fillna(0.0) + \
               pd.to_numeric(imp["Truck_Buffer"], errors="coerce").fillna(0.0)
    imp_out = pd.DataFrame({
        "City": imp["City"],
        "Origin Terminal": imp["Origin Terminal"],
        "Dest Terminal": imp["Dest Terminal"],
        "Route": imp["Route"],
        "Direction": "Import",
        "Cost": imp_cost.round(2),
        "Buffer": imp_buf.round(2),
    })
    imp_out["Total"] = (imp_out["Cost"] + imp_out["Buffer"]).round(2)

    total = pd.concat([exp_out, imp_out], ignore_index=True)

    # 清洗及排序
    total = total.dropna(subset=["City","Origin Terminal","Dest Terminal"])
    total = total.drop_duplicates()
    total = total[["City","Origin Terminal","Dest Terminal","Route","Direction","Cost","Buffer","Total"]]
    return total


# ===== 实际调用（在你已有的 df_train / df_truck / df_buffer 准备好之后执行） =====
if ('df_train' in locals()) and ('df_truck' in locals()) and ('df_buffer' in locals()):
    truck_long = normalize_truck(df_truck)
    rail_norm  = normalize_rail(df_train, use_leasing=True)  # 若不想把 leasing 算进 Cost，改为 False
    buf_norm   = normalize_buffer(df_buffer)

    total_df = build_total(truck_long, rail_norm, buf_norm)

    st.markdown("### ✅ 合并完成 · Total Cost")
    st.dataframe(total_df, use_container_width=True)

    # 可选：导出
    @st.cache_data
    def _to_csv(df):
        return df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "下载 Total Cost CSV",
        data=_to_csv(total_df),
        file_name="Total_Cost.csv",
        mime="text/csv",
    )
else:
    st.info("请先确保 Train Cost、Truck、Buffer 三类数据都已加载。")
