import seaborn as sns
import pandas as pd

from matplotlib import pyplot as plt


class Plotter:
    def __init__(self, figures_path, style: str = "darkgrid", context: str = "paper", font_scale: float = 1.8,
                 palette: str = 'Set2'):
        self.fig_size = (16, 9)
        self.font_size = 20
        self.labels_size = 24
        self.font_scale = font_scale
        self.palette = palette
        self.figures_path = figures_path
        sns.set_theme(style=style, context=context, rc={"grid.linewidth": 3},
                      font_scale=font_scale)

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
