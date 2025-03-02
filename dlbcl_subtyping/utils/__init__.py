from PIL import Image
from PIL.Image import Image as ImageType
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


from .cli import BaseMLCLI, BaseMLArgs

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
