import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path


def mpl_style_params(new_rcparams=None):
    rcParams = mpl.rcParams
    # rcParams['lines.linestyle'] = '--'
    # rcParams['legend.fontsize'] = 'large' #'x-large'
    rcParams['legend.shadow'] = False
    rcParams['lines.marker'] = 'o'
    rcParams['lines.markersize'] = 2
    # rcParams['lines.linewidth'] = 3.5  # 2
    rcParams['ytick.major.width'] = 2.5
    rcParams['xtick.major.width'] = 2.5
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['axes.grid'] = True
    rcParams['figure.autolayout'] = False
    rcParams['savefig.format'] = 'png'
    rcParams["scatter.marker"] = "o"  # "x"
    rcParams.update({'font.size': 16})

    # Say, "the default sans-serif font is COMIC SANS"
    # rcParams['font.sans-serif'] = 'Liberation Sans'
    # Then, "ALWAYS use sans-serif fonts"
    # rcParams['font.family'] = "sans-serif"
    rcParams.update({
        "text.usetex": True,  # Use LaTeX to write all text
        "font.family": "serif",  # Use serif fonts
        "font.serif": ["Computer Modern"],
    })

    if new_rcparams:
        rcParams.update(new_rcparams)

    Path(rcParams["savefig.directory"]).mkdir(parents=True, exist_ok=True)

    return rcParams


if __name__ == '__main__':
    rcParams = mpl_style_params()
    mpl.rcParams.update(rcParams)

    from matplotlib.pyplot import subplots, xlabel, ylabel, grid, show
    s = fr"Mean (-): (74.06$\pm$2.56) $\mu$m\\Nominal (--): 73 $\mu$m"

    fig, ay = subplots()
    plt.text(0.5, 0.5, s)
    # Using the specialized math font elsewhere, plus a different font
    xlabel(r"The quick brown fox jumps over the lazy dog", fontsize=18)
    # No math formatting, for comparison
    ylabel(r'Italic and just Arial and not-math-font', fontsize=18)
    grid()
    # plt.savefig("abe.png")
    show()
