import sys
import warnings
import logging

from PIL import Image, ImageFont, ImageDraw
from PIL.Image import Image as ImageType
from sklearn.decomposition import PCA
import cv2
import numpy as np
from matplotlib import pyplot as plt, colors as mcolors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from .cli import BaseMLCLI, BaseMLArgs


def yes_no_prompt(question):
    print(f"{question} [Y/n]: ", end="")
    response = input().lower()
    return response == "" or response.startswith("y")


def find_optimal_components(features, threshold=0.95):
    pca = PCA()
    pca.fit(features)
    explained_variance = pca.explained_variance_ratio_
    # 累積寄与率が95%を超える次元数を選択する例
    cumulative_variance = np.cumsum(explained_variance)
    optimal_n = np.argmax(cumulative_variance >= threshold) + 1
    return min(optimal_n, len(features) - 1)


def get_platform_font():
    if sys.platform == 'win32':
        # Windows
        font_path = 'C:\\Windows\\Fonts\\msgothic.ttc'  # MSゴシック
    elif sys.platform == 'darwin':
        # macOS
        font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'
    else:
        # Linux
        # font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf' # TODO: propagation
        font_path = '/usr/share/fonts/TTF/DejaVuSans.ttf'
    return font_path


def create_frame(size, color, text, font):
    frame = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, size, size), outline=color, width=4)
    text_color = 'white' if mcolors.rgb_to_hsv(mcolors.hex2color(color))[2]<0.9 else 'black'
    bbox = np.array(draw.textbbox((0, 0), text, font=font))
    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    draw.rectangle((4, 4, bbox[2]+4, bbox[3]+4), fill=color)
    draw.text((1, 1), text, font=font, fill=text_color)
    return frame



def hover_images_on_scatters(scatters, imagess, ax=None, offset=(150, 30)):
    if ax is None:
        ax = plt.gca()
    fig = ax.figure

    def as_image(image_or_path):
        if isinstance(image_or_path, np.ndarray):
            return image_or_path
        if isinstance(image_or_path, ImageType):
            return image_or_path
        if isinstance(image_or_path, str):
            return Image.open(image_or_path)
        raise RuntimeError('Invalid param', image_or_path)

    imagebox = OffsetImage(as_image(imagess[0][0]), zoom=.5)
    imagebox.image.axes = ax
    annot = AnnotationBbox(
            imagebox,
            xy=(0, 0),
            # xybox=(256, 256),
            # xycoords='data',
            boxcoords='offset points',
            # boxcoords=('axes fraction', 'data'),
            pad=0.1,
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3'),
            zorder=100)
    annot.set_visible(False)
    ax.add_artist(annot)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes != ax:
            return
        for n, (sc, ii) in enumerate(zip(scatters, imagess)):
            cont, index = sc.contains(event)
            if cont:
                i = index['ind'][0]
                pos = sc.get_offsets()[i]
                annot.xy = pos
                annot.xybox = pos + np.array(offset)
                image = as_image(ii[i])
                # text = unique_code[n]
                # annot.set_text(text)
                # annot.get_bbox_patch().set_facecolor(cmap(int(text)/10))
                imagebox.set_data(image)
                annot.set_visible(True)
                fig.canvas.draw_idle()
                return

        if vis:
            annot.set_visible(False)
            fig.canvas.draw_idle()
            return

    fig.canvas.mpl_connect('motion_notify_event', hover)

def is_in_streamlit_context():
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", module="streamlit.*")
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        return ctx is not None
    except ImportError:
        return False
