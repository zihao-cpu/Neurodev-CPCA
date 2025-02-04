import nibabel as nib
import numpy as np
import PIL
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from io import BytesIO
from IPython.display import display
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from templateflow import api as tflow
nib.imageglobals.logger.level = 40

def adjust_ax(ax, dx=0, dy=0, dw=0, dh=0):
    pos = ax.get_position()
    new_pos = [pos.x0 + dx, pos.y0 + dy, pos.width + dw, pos.height + dh]
    ax.set_position(new_pos)


def get_ratio(x, y):
    return (np.max(x) - np.min(x)) / (np.max(y) - np.min(y))


class plot_surface:
    def __init__(self):
        # Load surfaces
        
        self.height=170
        self.width=235
        
        self.load_surf()
        self.load_atlas()
        
        # Setup plotting layout
        self.LAYOUT = {
            "scene": {f"{dim}axis": self._axis_config() for dim in ("x", "y", "z")},
            "paper_bgcolor": "#fff",
            "hovermode": False,
            "margin": {"l": 0, "r": 0, "b": 0, "t": 0, "pad": 0},
        }
    
    
    def load_surf(self):
        self.surf_lh = nib.load(tflow.get('fsLR', hemi='L', extension='surf.gii', suffix='veryinflated'))
        self.surf_rh = nib.load(tflow.get('fsLR', hemi='L', extension='surf.gii', suffix='veryinflated'))
        self.coords_lh, self.faces_lh = self.surf_lh.agg_data('pointset'), self.surf_lh.agg_data('triangle')
        self.coords_rh, self.faces_rh = self.surf_rh.agg_data('pointset'), self.surf_rh.agg_data('triangle')
    
    def load_atlas(self, name='schaefer', scale=200, network=17):
        self.atlas_file = nib.load(f'../data/Parcellation/Schaefer2018_{scale}Parcels_{network}Networks_order.dlabel.nii')
        self.atlas_data = self.atlas_file.get_fdata().squeeze()                   
        
        n_vx = len(self.atlas_data)
        
        self.mid_point = n_vx // 2
        self.mid_roi = scale // 2

        self.atlas_lh = self.atlas_data[:self.mid_point]
        self.atlas_rh = self.atlas_data[self.mid_point:]
        self.atlas_rh[self.atlas_rh > 0] -= self.mid_roi
    
    def _axis_config(self):
        return {
            "showgrid": False,
            "showline": False,
            "ticks": "",
            "title": "",
            "showticklabels": False,
            "zeroline": False,
            "showspikes": False,
            "spikesides": False,
            "showbackground": False,
        }

    def get_camerasetting(self, hemi):
        zoom_scale = 1.4
        if hemi == 'lh':
            zoom_left = 1.35
            zoom_right = 1.45
        elif hemi == 'rh':
            zoom_left = 1.45
            zoom_right = 1.35

        return {
            "left": {"eye": {"x": -zoom_left, "y": 0, "z": 0}, "up": {"x": 0, "y": 0, "z": 1}, "center": {"x": 0, "y": 0, "z": 0}},
            "right": {"eye": {"x": zoom_right, "y": 0, "z": 0}, "up": {"x": 0, "y": 0, "z": 1}, "center": {"x": 0, "y": 0, "z": 0}},
            "top": {"eye": {"x": 0, "y": 0, "z": zoom_scale}, "up": {"x": 0, "y": 1, "z": 0}, "center": {"x": 0, "y": 0, "z": 0}},
            "bottom": {"eye": {"x": 0, "y": 0, "z": -zoom_scale}, "up": {"x": 0, "y": 1, "z": 0}, "center": {"x": 0, "y": 0, "z": 0}},
            "front": {"eye": {"x": 0, "y": zoom_scale, "z": 0}, "up": {"x": 0, "y": 0, "z": 1}, "center": {"x": 0, "y": 0, "z": 0}},
            "back": {"eye": {"x": 0, "y": -zoom_scale, "z": 0}, "up": {"x": 0, "y": 0, "z": 1}, "center": {"x": 0, "y": 0, "z": 0}},
        }

    def minmax(self, x, vmin, vmax):
        return (x - vmin) / (vmax - vmin)

    def get_vertexcolor(self, data, hemi='lh', cmap='jet', vmax=None, vmin=None):
        if hemi == 'lh':
            atlas = self.atlas_lh
        else:
            atlas = self.atlas_rh
            
        colormap = plt.get_cmap(cmap)
        data_vtx = np.zeros(len(atlas), dtype=float)
        
        for i, value in enumerate(data):
            data_vtx[atlas == i+1] = value

        if vmax is None or vmin is None:
            mask = atlas != 0
            values = data_vtx[mask]
            vmin = values.min() if vmin is None else vmin
            vmax = values.max() if vmax is None else vmax

        normalized_data = self.minmax(data_vtx, vmin, vmax)
        rgba = colormap(normalized_data)[:, :3]
        rgb = np.round(rgba * 255).astype(np.uint8)
        
        hex_format = np.vectorize(lambda r, g, b: f'#{r:02x}{g:02x}{b:02x}')
        vertex_color = hex_format(rgb[:, 0], rgb[:, 1], rgb[:, 2])
        vertex_color[atlas == 0] = '#000000'
        
        return vertex_color

    def plot_mesh(self, hemi):
        if hemi == 'lh':
            coords = self.coords_lh.copy()
            faces = self.faces_lh.copy()
            atlas = self.atlas_lh.copy()
        elif hemi == 'rh':
            coords = self.coords_rh.copy()
            faces = self.faces_rh.copy()
            atlas = self.atlas_rh.copy()

        x, y, z = coords.T
        i, j, k = faces.T
        
        line_3d = self.get_border(atlas, coords, faces)
        mesh_3d = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k)
        
        fig = go.Figure(data=[mesh_3d, line_3d],  layout=go.Layout(height=self.height, width=self.width))
        fig.update_layout(**self.LAYOUT)
        return fig

    def generate_image_bytes(self, fig, vertexcolor, hemi='lh', view='left'):
        scene_camera = self.get_camerasetting(hemi)
        fig.update_layout(scene_camera=scene_camera[view], **self.LAYOUT)
        fig.data[0].vertexcolor = vertexcolor
        img = fig.to_image(format='jpg', scale=1)
        return PIL.Image.open(BytesIO(img))


    def plot_hemisphere(self, vertexcolor, hemi, view):
        fig = self.plot_mesh(hemi)
        return self.generate_image_bytes(fig, vertexcolor, hemi, view)

    def plot_hemispheres(self, data, cmap='jet', n_jobs=4, vmin=None, vmax=None, show=True):

        vertex_color_lh = self.get_vertexcolor(data[:self.mid_roi], hemi='lh', cmap=cmap, vmin=vmin, vmax=vmax)
        vertex_color_rh = self.get_vertexcolor(data[self.mid_roi:], hemi='rh', cmap=cmap, vmin=vmin, vmax=vmax)

        list_params = [
            dict(hemi='lh', view='left', vertexcolor=vertex_color_lh),
            dict(hemi='lh', view='right', vertexcolor=vertex_color_lh),
            dict(hemi='rh', view='left', vertexcolor=vertex_color_rh),
            dict(hemi='rh', view='right', vertexcolor=vertex_color_rh),
        ]

        list_img = Parallel(n_jobs=n_jobs)(delayed(self.plot_hemisphere)(**params) for params in list_params)

        total_width = 4 * self.width
        total_height = self.height
        concatenated_img = PIL.Image.new('RGB', (total_width, total_height))

        for i, img in enumerate(list_img):
            concatenated_img.paste(img, (i * self.width, 0))

        if show:
            display(concatenated_img)
        else:
            return concatenated_img
        
    def get_border(self, atlas, coords, faces):
        face_labels = atlas[faces]  # Shape: (n_faces, 3)

        # Identify edges where the ROI labels are different
        edge_masks = [
            face_labels[:, 0] != face_labels[:, 1],
            face_labels[:, 1] != face_labels[:, 2],
            face_labels[:, 2] != face_labels[:, 0]
        ]

        # Create an array of all possible edges (3 edges per face)
        edges = np.vstack([
            faces[:, [0, 1]][edge_masks[0]],
            faces[:, [1, 2]][edge_masks[1]],
            faces[:, [2, 0]][edge_masks[2]]
        ])

        # Sort and remove duplicate edges directly
        edges = np.sort(edges, axis=1)
        border_edges = np.unique(edges, axis=0)

        # Extract coordinates for line endpoints
        line_coords = coords[border_edges].reshape(-1, 3)

        # Split into x, y, z for scatter plot and insert NaNs for breaks
        x_lines = np.insert(line_coords[:, 0], np.arange(2, line_coords.shape[0], 2), np.nan)
        y_lines = np.insert(line_coords[:, 1], np.arange(2, line_coords.shape[0], 2), np.nan)
        z_lines = np.insert(line_coords[:, 2], np.arange(2, line_coords.shape[0], 2), np.nan)

        line_3d = go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode='lines',
            line=dict(color='#3f3f3f', width=2)
        )
        return line_3d
    
    
    def plot_dynamics(self, list_data, hemi='lh', cmap='jet', n_jobs=8, vmin=None, vmax=None, show=True):
        """
        list_data: (n_samples, n_rois)
        """
        assert list_data.shape[1] == self.mid_roi * 2, f"Number of ROIs must be {self.mid_roi * 2}"
        
        if hemi == 'lh':
            list_data = list_data[:, :self.mid_roi]
            atlas = self.atlas_lh.copy()
            top_view = 'left'
            
        elif hemi == 'rh':
            list_data = list_data[:, self.mid_roi:]
            atlas = self.atlas_rh.copy()
            top_view = 'right'


        fig = self.plot_mesh(hemi=hemi)
        

        list_params = []
        for view in ['left', 'right']:
            for data in list_data:
                list_params.append(dict(view=view, hemi=hemi, vertexcolor=self.get_vertexcolor(data, hemi=hemi, cmap=cmap, vmin=vmin, vmax=vmax)))

        # Use Parallel to generate images concurrently
        list_img = Parallel(n_jobs=n_jobs)(delayed(self.generate_image_bytes)(fig, **params) for params in list_params)


        total_width = len(list_data) * self.width
        total_height = 2 * self.height

        concatenated_img = PIL.Image.new('RGB', (total_width, total_height))
        for i, params in enumerate(list_params):
            if hemi == 'lh':
                row_left = 0
                row_right = self.height
            elif hemi == 'rh':
                row_left = self.height
                row_right = 0
            
            if params['view'] == 'left':
                row = row_left
                col = i * self.width
            else:
                row = row_right
                col = (i % len(list_data)) * self.width
            concatenated_img.paste(list_img[i], (col, row))
        if show:
            display(concatenated_img)
        else:
            return concatenated_img
    
