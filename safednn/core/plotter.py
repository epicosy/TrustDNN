import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt


def generate_colors_and_shades(n_colors, n_shades):
    # Define a palette
    palette = sns.color_palette("Set2", n_colors)

    # Generate lighter shades for each color in the palette
    colors_and_shades = []
    for color in palette:
        shades = sns.light_palette(color, n_colors=n_shades, reverse=True)
        colors_and_shades.append([color] + shades)

    # Convert colors and shades to hex format
    colors_and_shades_hex = [[sns.color_palette([color]).as_hex()[0] for color in group] for group in colors_and_shades]

    return colors_and_shades_hex


class Plotter:
    def __init__(self, figures_path, style: str = "darkgrid", context: str = "paper", font_scale: float = 1.8,
                 palette: str = 'Set2'):
        self.fig_size = (16, 9)
        self.font_size = 20
        self.labels_size = 24
        self.font_scale = font_scale
        self.palette = palette
        self.figures_path = figures_path
        sns.set_theme(style=style, context=context, rc={"grid.linewidth": 3}, font_scale=font_scale)

    def box_plot(self, df: pd.DataFrame, x: str, y: str, tag: str, hue: str = None, x_label: str = None,
                 y_label: str = None, invert: bool = False, rotate: bool = False):

        if not x_label:
            x_label = x.capitalize()

        if not y_label:
            y_label = y.capitalize()

        if invert:
            x, y = y, x
            x_label, y_label = y_label, x_label

        output_path = self.figures_path / f'box_plot_{tag}.png'
        plt.figure(figsize=self.fig_size)

        # Box plot
        sns.boxplot(data=df, x=x, y=y, hue=hue, notch=True, flierprops={"marker": "x"}, palette=self.palette,
                    medianprops={"color": "white"})

        plt.xlabel(x_label, fontweight='bold')
        plt.ylabel(y_label, fontweight='bold')

        if rotate:
            plt.yticks(rotation=45) if invert else plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.show()

    def bar_plot(self, data: pd.DataFrame, x: str, y: str, tag: str, hue: str = None, x_label: str = None,
                 y_label: str = None, transparent: bool = False, error_bars: bool = False, orient: str = 'v'):
        if not x_label:
            x_label = x.capitalize()

        if not y_label:
            y_label = y.capitalize()

        output_path = self.figures_path / f'bar_plot_{tag}.png'
        plt.figure(figsize=self.fig_size)

        # Bar plot
        sns.barplot(data=data, x=x, y=y, hue=hue, palette=self.palette, errorbar="sd" if error_bars else None,
                    errcolor=".4", linewidth=2.5, capsize=0.1, orient=orient)

        plt.xlabel(x_label, fontweight='bold', fontsize=self.font_size)
        plt.ylabel(y_label, fontweight='bold', fontsize=self.font_size)

        plt.xticks(fontsize=self.font_size-2)
        plt.yticks(fontsize=self.font_size-2)

        plt.legend(loc='best', fontsize=self.labels_size)
        plt.tight_layout()
        plt.savefig(str(output_path), transparent=transparent)
        plt.show()

    def stacked_bar_plot(self, data: pd.DataFrame, x: str, stack: str, hue: str, y: str, y_label: str, tag: str,
                         transparent: bool = False, x_label: str = None):
        output_path = self.figures_path / f'stacked_bar_plot_{tag}.png'
        fix, ax = plt.subplots(figsize=self.fig_size)

        if not x_label:
            x_label = x.capitalize()

        sns.set(style="whitegrid")  # Setting seaborn style

        pivot_data = data.pivot_table(index=[x, hue], columns=stack, values=y)

        # Plot the stacked bar plot
        pivot_data.plot.bar(stacked=True, ax=ax)

        plt.xlabel(x_label, fontsize=self.labels_size, fontweight='bold')
        plt.ylabel(y_label, fontsize=self.labels_size, fontweight='bold')

        plt.xticks(fontsize=self.font_size)

        plt.legend(loc='best', fontsize=self.labels_size)
        plt.tight_layout()
        plt.savefig(str(output_path), transparent=transparent)
        plt.show()
