import json
import textwrap
import math

# =============================================================================
# =============================================================================

def hex_to_rgb_tuple(hex_color):
    """Converts a HEX color code to an RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def generate_latex_stages_plot(input_json_path, output_tex_path):
    """
    Generates a .tex file with a PGFPlots groupplot that matches the user's
    NEW target format (grouping by stage, 1 row x N columns).
    *** UPDATED TO FIX SPACING AND CLIPPING ISSUES ***
    """
    
    try:
        with open(input_json_path, 'r') as f:
            scenario_data = json.load(f)
        figure_title = scenario_data['figure_title']
        figure_data = scenario_data['figure_data'] # figure_data is a list of stages
        aesthetics = scenario_data['aesthetics']
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: Could not read or parse '{input_json_path}'. Reason: {e}")
        return
        
    if not figure_data:
        print(f"No data to generate LaTeX for '{figure_title}'.")
        return

    # --- 1. Color Definitions ---
    color_defs_list = []
    for system, hex_code in aesthetics['color_map'].items():
        rgb = hex_to_rgb_tuple(hex_code)
        color_name = system.replace('-', '').lower() + 'color'
        color_defs_list.append(f"\\definecolor{{{color_name}}}{{RGB}}{{{rgb[0]},{rgb[1]},{rgb[2]}}}")
    color_defs = "\n".join(color_defs_list)

    # =====================================================================
    # =====================================================================

    # --- 2. Build each \nextgroupplot block for each Stage ---
    subplot_blocks = []
    
    symbolic_x_coords = aesthetics['system_order']
    
    for i, stage_info in enumerate(figure_data):
        plot_title = stage_info['subplot_title']
        
        addplot_cmds = []
        bar_heights = stage_info['bar_heights']
        
        for system in aesthetics['system_order']:
            height = bar_heights.get(system, 0)
            color_name = system.replace('-', '').lower() + 'color'
            pattern = aesthetics.get('hatch_map', {}).get(system)
            
            if pattern:
                style = f"fill={color_name}, pattern={pattern}, pattern color=black"
            else:
                style = f"fill={color_name}"
            
            if i > 0:
                style += ", forget plot"
                
            addplot_cmds.append(f"\\addplot[{style}] coordinates {{({system.replace('-', ' ')},{height:.3f})}};")
        
        addplot_str = "\n        ".join(addplot_cmds)

        ann_data = stage_info['arachne_annotations']
        ann_text_vs_m = f"{ann_data.get('vs_megatron', 0):.2f}x"
        ann_text_vs_d = f"{ann_data.get('vs_deepspeed', 0):.2f}x"
        full_ann_text = f"{{{ann_text_vs_m}, {ann_text_vs_d}}}"
        
        arachne_height = bar_heights.get('Arachne', 0)
        
        annotation_node = (f"\\node[font=\\scriptsize, rotate=45, anchor=south west, inner sep=1pt, yshift=3pt] "
                           f"at (axis cs:Arachne,{arachne_height}) {full_ann_text};")
        
        block = textwrap.dedent(f"""
        % --- Subplot: {plot_title} ---
        \\nextgroupplot[title={{{plot_title}}}]
        {addplot_str}
        
        % Annotation for Arachne
        {annotation_node}""")
        subplot_blocks.append(block)

    # --- 3. Assemble the final groupplot environment ---
    num_plots = len(figure_data)
    legend_labels = [l.replace('_', '\\_') for l in aesthetics['system_order']]
    symbolic_x_coords_latex = [s.replace('-', ' ') for s in symbolic_x_coords]
    
    latex_code = textwrap.dedent(f"""
        % =================================================
        % PGFPlot for: {figure_title}
        % Auto-generated from: {input_json_path}
        % Target format: 1 row of stages, with systems on the x-axis.
        % IMPORTANT: Make sure your main .tex file includes \\usepgfplotslibrary{{patterns}}
        % =================================================
        {color_defs}
        
        \\begin{{figure*}}[t!]
        \\centering
        \\begin{{tikzpicture}}
        \\begin{{groupplot}}[
            group style={{
                group size={num_plots} by 1,
                horizontal sep=1.8cm
            }},
            ybar,
            bar width=20pt,
            width=5.5cm,
            height=5cm,
            ymin=0,
            clip=false, % keep annotations fully visible
            enlarge x limits=0.35,
            ylabel={{Speedup vs. {aesthetics['primary_baseline']} (x)}},
            symbolic x coords={{{','.join(symbolic_x_coords_latex)}}},
            xmin={symbolic_x_coords_latex[0]}, % pin x-axis start
            xmax={symbolic_x_coords_latex[-1]}, % pin x-axis end
            xtick=data,
            xticklabel style={{rotate=45, anchor=east}},
            ymajorgrids=true,
            title style={{yshift=-1.5ex}},
            legend style={{
                at={{(0.5, 1.25)}},
                anchor=north,
                legend columns=-1,
                draw=none,
                fill=none
            }}
        ]
        
        {''.join(subplot_blocks)}

        \\end{{groupplot}}
        
        \\end{{tikzpicture}}
        \\caption{{{figure_title}}}
        \\label{{fig:{output_tex_path.replace('.tex', '').replace('_', '-')}}}
        \\end{{figure*}}
    """)

    with open(output_tex_path, 'w') as f:
        f.write(latex_code)
    print(f"[LATEX FIGURE] Groupplot code for stages has been saved to '{output_tex_path}'")


# =============================================================================
# =============================================================================
if __name__ == "__main__":
    input_json_file = "plot_data_stages.json" 
    output_tex_file = "figure_stages_speedup.tex"
    print(f"\n--- Generating LaTeX for {input_json_file} ---")
    generate_latex_stages_plot(
        input_json_path=input_json_file,
        output_tex_path=output_tex_file
    )