import copy
from datetime import date
from random import randint
import pandas as pd
from bokeh.io import show
from bokeh.models import ColumnDataSource, DataTable, DateFormatter, TableColumn, NumberFormatter, CustomJS
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.palettes import Oranges256, Blues256
from bokeh.models import Select, Spinner, Checkbox
from bokeh.layouts import column, row
from bokeh.core.enums import Dimensions
from bokeh.plotting import output_file, save
import numpy as np
import os

np.set_printoptions(suppress=True, linewidth=600, precision=4)




def get_cmap(field_name):
    colors = list(Oranges256[76:]) + list(Blues256[76:][::-1])
    cmap = linear_cmap(field_name=field_name, palette=colors, low=-1, high=1)
    return cmap


def get_df(path, is_row_normalized=False):
    x0_coeff, eps_coeff, node_coeff = np.load(path).values()

    if node_coeff[:, 0].mean() > 1:
        # discrete integer time step, [0, 999]
        time_idxs = ["%03d"%x for x in node_coeff[:, 0].tolist()]
    else:
        # continuous float time step, [0, 1]
        time_idxs = ["%0.3f"%x for x in node_coeff[:, 0].tolist()]
    
    dfx0mg = pd.DataFrame()
    dfx0mg.insert(0, "equiv", np.sum(x0_coeff, axis=1))
    dfx0mg.insert(1, "ideal", node_coeff[1:, 1])
    
    dfepsmg = pd.DataFrame()
    dfepsmg.insert(0, "equiv", np.linalg.norm(eps_coeff, axis=1))
    dfepsmg.insert(1, "ideal", node_coeff[1:, 2])

    if is_row_normalized:
        x0_coeff = x0_coeff/np.diag(x0_coeff)[:, None]
        eps_coeff = eps_coeff/(np.diag(eps_coeff, 1)[:, None]+1E-8)
     
    dfx0 = pd.DataFrame(data=x0_coeff, columns=time_idxs[:-1])
    dfx0.insert(0, "time", time_idxs[1:])
    
    dfeps = pd.DataFrame(data=eps_coeff, columns=time_idxs)
    dfeps.insert(0, "time", time_idxs[1:])

    return dfx0, dfx0mg, dfeps, dfepsmg


def create_table_columns(columns):
    table_columns = []
    for col in columns:
        if col in ["time"]:
            formatter = NumberFormatter(format="0.[0]", background_color="pink", text_align="center")
            column = TableColumn(field=col, title=col, formatter=formatter, width=40, sortable=False)
        elif col in ["equiv"]:
            formatter = NumberFormatter(format="0.[000]", background_color="orange", text_align="center")
            column = TableColumn(field=col, title=col, formatter=formatter, width=80, sortable=False)
        elif col in ["ideal"]:
            formatter = NumberFormatter(format="0.[000]", background_color="greenyellow", text_align="center")
            column = TableColumn(field=col, title=col, formatter=formatter, width=80, sortable=False)
        else:
            formatter = NumberFormatter(format="0.[000]", background_color=get_cmap(col), text_align="center")
            column = TableColumn(field=col, title=col, formatter=formatter, width=40, sortable=False)
            
        table_columns.append(column)
    return table_columns


def create_coeff_pool(coeff_list_path, coeff_pool):
    df = pd.read_csv(coeff_list_path, index_col=0)
    df = df[df["alg"] != "ode heun"]
    
    for row in df.itertuples():
        if row.alg not in ["ddpm"]:
            continue
        create_one_coeff(row.alg, row.step, row.path, False, coeff_pool)
        create_one_coeff(row.alg, row.step, row.path, True, coeff_pool)
    return


def create_one_coeff(alg, step, path, is_row_normalized, coeff_pool):
    dfx0, dfx0mg, dfeps, dfepsmg = get_df(path, is_row_normalized)
    
    src_x0, src_x0_mg = ColumnDataSource(dfx0), ColumnDataSource(dfx0mg)
    tc_x0, tc_x0_mg = create_table_columns(dfx0.columns), create_table_columns(dfx0mg.columns)

    src_eps, src_eps_mg = ColumnDataSource(dfeps), ColumnDataSource(dfepsmg)
    tc_eps, tc_eps_mg = create_table_columns(dfeps.columns), create_table_columns(dfepsmg.columns)
    
    rn_flag = "normalized" if is_row_normalized else "original"
    
    x0_name = "%s_%d_pred_x0_%s"%(alg, step, rn_flag)
    coeff_pool[x0_name] = (src_x0, tc_x0, src_x0_mg, tc_x0_mg)
    
    eps_name = "%s_%d_noise_%s"%(alg, step, rn_flag)
    coeff_pool[eps_name] = (src_eps, tc_eps, src_eps_mg, tc_eps_mg)
    return


def create_step_options():
    options_1 = ["5", "10", "15", "18", "20", "24", "25", "30", "40", "50"]
    options_2 = ["6", "10", "12", "18", "20", "24", "28", "30", "40", "50"]
    options_3 = ["6", "9", "12", "15", "18", "24", "30", "36", "42", "51"]
    return options_1, options_2, options_3


def datatable_tx():
    arr_step_opts = create_step_options()
    
    dfs = get_df("D:\\codes\\WeSee\\NaturalDiffusion\\results\\ddpm\\ddpm_simpy_010.npz")
    dfx0, dfx0mg, dfeps, dfepsmg = dfs
    src_x0, src_x0_mg = ColumnDataSource(dfx0), ColumnDataSource(dfx0mg)
    src_eps, src_eps_mg = ColumnDataSource(dfeps), ColumnDataSource(dfepsmg)

    tc_x0 = create_table_columns(dfx0.columns)
    tc_x0_mg = create_table_columns(dfx0mg.columns)

    tc_eps = create_table_columns(dfeps.columns)
    tc_eps_mg = create_table_columns(dfepsmg.columns)

    coeff_pool = {}
    coeff_pool["ddpm_10_pred_x0_original"] = (src_x0, tc_x0, src_x0_mg, tc_x0_mg)
    coeff_pool["ddpm_10_noise_original"] = (src_eps, tc_eps, src_eps_mg, tc_eps_mg)
    coeff_pool["ddpm_10_pred_x0_normalized"] = (src_x0, tc_x0, src_x0_mg, tc_x0_mg)
    coeff_pool["ddpm_10_noise_normalized"] = (src_eps, tc_eps, src_eps_mg, tc_eps_mg)
    # for ii in range(200):
    #     coeff_pool["a_%d"%ii] = (src_eps, tc_eps, src_eps_mg, tc_eps_mg)
    
    # output_file(filename="VisualizeCoeffMatrix.html", title="Visualize Coefficient Matrix")

    # coeff_pool = {}
    # create_coeff_pool("./all_coeff_matrix.csv", coeff_pool)
    # src_x0, tc_x0, src_x0_mg, tc_x0_mg = coeff_pool["ddpm_10_pred_x0_original"]
    
    alg_opts = ["ddpm", "ddim", "sde euler", "ode euler", "flow match euler",
                "deis tab3", "dpmsolver2s", "dpmsolver3s", "dpmsolver++2s", "dpmsolver++3s"]
    
    alg_sel = Select(title="select algorithm", value="ddpm", options=alg_opts)
    step_sel = Select(title="select step", value="10", options=arr_step_opts[0])
    x0_or_eps = Select(title="pred_x0 or noise", value="pred_x0", options=["pred_x0", "noise"])
    row_normalized = Select(title="row normalized", value="original", options=["original", "normalized"])

    col_width_spin = Spinner(title="table column width", low=20, high=120, step=2, value=40, width=100)
    table_mat = DataTable(source=src_x0, columns=tc_x0, index_position=None, autosize_mode="none",
                          height=290, height_policy="auto", width_policy="min", min_width=440, resizable="both")
    table_margin = DataTable(source=src_x0_mg, columns=tc_x0_mg, index_position=None, autosize_mode="none",
                             height=290, height_policy="auto", width_policy="min", min_width=160)
    
    callback = CustomJS(args=dict(alg_sel=alg_sel, step_sel=step_sel, x0_or_eps=x0_or_eps,
                                  row_normalized=row_normalized, col_width_spin=col_width_spin,
                                  arr_step_opts=arr_step_opts, coeff_pool=coeff_pool,
                                  table_mat=table_mat, table_margin=table_margin),
                        code="""
        var width = window.innerWidth;
        var step = step_sel.value;
        var x0_or_eps = x0_or_eps.value;
        var col_width = col_width_spin.value;
        var alg = alg_sel.value;
        var rn_flag = row_normalized.value;
        var col_width = col_width_spin.value;
        
        if (alg == "ode heun" || alg == "dpmsolver2s" || alg == "dpmsolver++2s") {
            var options = arr_step_opts[1];
        }
        else if (alg == "dpmsolver3s" || alg == "dpmsolver++3s") {
            var options = arr_step_opts[2];
        }
        else {
            var options = arr_step_opts[0];
        }
        step_sel.options = options;
        step_sel.value = options.includes(step) ? step : options[1];
        step = step_sel.value;
        
        name = alg + "_" + step + "_" + x0_or_eps + "_" + rn_flag;
        
        var src_mat = coeff_pool[name][0];
        console.log(src_mat);
        var tc_mat = coeff_pool[name][1];
        var src_margin = coeff_pool[name][2];
        var tc_margin = coeff_pool[name][3];
        
        for (let i=0; i < tc_mat.length; i++) {
            tc_mat[i].width = col_width;
        }
        for (let i=0; i < tc_margin.length; i++) {
            tc_margin[i].width = col_width*2;
        }
        
        table_mat.source = src_mat;
        table_mat.columns = tc_mat;
        table_mat.min_width = Math.min(width-4*col_width, tc_mat.length*col_width);
        table_mat.height = 26*(src_mat.length+1);
        
        table_margin.source = src_margin;
        table_margin.columns = tc_margin;
        table_margin.min_width = 4*col_width
        table_margin.height = 26*(src_margin.length+1);
        """)
    
    callback_alg = CustomJS(args=dict(arr_step_opts=arr_step_opts,
                                      alg_sel=alg_sel, step_sel=step_sel, table_mat=table_mat),
                            code="""
        var alg = alg_sel.value;
        var step = step_sel.value;
        
        if (alg == "ode heun" || alg == "dpmsolver2s" || alg == "dpmsolver++2s") {
            var options = arr_step_opts[1];
        }
        else if (alg == "dpmsolver3s" || alg == "dpmsolver++3s") {
            var options = arr_step_opts[2];
        }
        else {
            var options = arr_step_opts[0];
        }
        step_sel.options = options;
        step_sel.value = options.includes(step) ? step : options[1];
        """)

    alg_sel.js_on_change("value", callback)
    x0_or_eps.js_on_change("value", callback)
    step_sel.js_on_change("value", callback)
    row_normalized.js_on_change("value", callback)
    
    show(column(row(alg_sel, x0_or_eps, step_sel,  row_normalized, col_width_spin, width=500), row(table_mat, table_margin)))
    return


def generate_coeff_matrix_tx():
    from AnalyzeDDPMDDIM import ddpm_simpy_analyze_coeff, ddim_simpy_analyze_coeff
    from AnalyzeFlowMatching import flow_simpy_analyze_coeff
    from AnalyzeEulerHeun import analyze_ode, analyze_sde, analyze_heun
    from AnalyzeDPMSolver import analyze_dpmsolver_2s, analyze_dpmsolver_3s
    from AnalyzeDPMSolver import analyze_dpmsolver_pp_2s, analyze_dpmsolver_pp_3s
    from AnalyzeDEIS import analyze_tab

    opts1, opts2, opts3 = create_step_options()
    
    infos = []
    for step in opts1:
        print("order 1", step)
        step = int(step)
        # ddpm_simpy_analyze_coeff(step)
        # ddim_simpy_analyze_coeff(step)
        # flow_simpy_analyze_coeff(step)
        # analyze_tab(step)
        # analyze_sde(step)
        # analyze_ode(step)
        infos.append(["ddpm", "ddpm\\ddpm_simpy", step])
        infos.append(["ddim", "ddim\\ddim_simpy", step])
        infos.append(["flow match euler", "flow_euler\\flow_euler_simpy", step])
        infos.append(["deis tab3", "deis\\deis_tab", step])
        infos.append(["sde euler", "euler_heun\\sde_euler", step])
        infos.append(["ode euler", "euler_heun\\ode_euler", step])

    for step in opts2:
        print("order 2", step)
        step = int(step)//2
        # analyze_heun(step)
        # analyze_dpmsolver_2s(step)
        # analyze_dpmsolver_pp_2s(step)
        infos.append(["ode heun", "euler_heun\\ode_heun", step*2])
        infos.append(["dpmsolver2s", "dpmsolver\\dpmsolver2s", step*2])
        infos.append(["dpmsolverpp2s", "dpmsolverpp\\dpmsolverpp2s", step*2])

    for step in opts3:
        print("order 3", step)
        step = int(step)//3
        # analyze_dpmsolver_3s(step)
        # analyze_dpmsolver_pp_3s(step)
        infos.append(["dpmsolver3s", "dpmsolver\\dpmsolver3s", step*3])
        infos.append(["dpmsolverpp3s", "dpmsolverpp\\dpmsolverpp3s", step*3])

    df = pd.DataFrame(infos, columns=["alg", "prefix", "step"])
    paths = []
    for row in df.itertuples():
        path = os.path.join("./results", "%s_%03d.npz"%(row.prefix, row.step))
        path = os.path.abspath(path)
        if not os.path.exists(path):
            print(path)
            break
        paths.append(path)
    df["path"] = paths
    df.to_csv("all_coeff_matrix.csv")
    return


if __name__ == "__main__":
    datatable_tx()
    # generate_coeff_matrix_tx()