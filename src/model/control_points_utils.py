import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import SimpleITK as sitk
import json 
from pathlib import Path


def test_control_points_2d_3d(control_points, mask: sitk.Image):
    
    mask_arr = sitk.GetArrayFromImage(mask)   # [z, y, x]

    
    label_colors = {
        1: "red",
        2: "lime",
        3: "deepskyblue"
    }

    # PLOT 2D CONTROL POINTS
    
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.set_title("Control points - projection XY")
    ax1.set_facecolor("black")

    for label, pts_with_dirs in control_points.points.items():
        pts_idx = []

        for p, d in pts_with_dirs:
            idx = mask.TransformPhysicalPointToIndex(tuple(map(float, p)))  # (x,y,z)
            pts_idx.append(idx)

        if len(pts_idx) > 0:
            pts_idx = np.array(pts_idx)
            x = pts_idx[:, 0]
            y = pts_idx[:, 1]

            ax1.scatter(
                x, y,
                s=50,
                c=label_colors.get(label, "white"),
                label=f"label {label}"
            )

    ax1.invert_yaxis()
    ax1.set_aspect("equal")
    ax1.legend()
    plt.tight_layout()

    
    # PLOT 2D mask 
    mask_xy = np.max(mask_arr, axis=0)  # [y, x]

    cmap = mcolors.ListedColormap(["black", "red", "lime", "deepskyblue"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.set_title("Mask labels - projection XY")
    ax2.imshow(mask_xy, cmap=cmap, norm=norm, origin="upper")
    ax2.set_aspect("equal")
    plt.tight_layout()

   
    def plot_3d_scene(ax, elev, azim, title):
        ax.set_title(title)

        
        for label in sorted(control_points.points.keys()):
            region = np.argwhere(mask_arr == label)   # [N, 3] in ordine (z,y,x)

            if len(region) > 0:
                
                step = max(1, len(region) // 4000)
                region_sub = region[::step]

                z = region_sub[:, 0]
                y = region_sub[:, 1]
                x = region_sub[:, 2]

                ax.scatter(
                    x, y, z,
                    s=1,
                    c=label_colors.get(label, "white"),
                    alpha=0.15
                )

        # plot dei control points
        for label, pts_with_dirs in control_points.points.items():
            pts_idx = []

            for p, d in pts_with_dirs:
                idx = mask.TransformPhysicalPointToIndex(tuple(map(float, p)))  # (x,y,z)
                pts_idx.append(idx)

            if len(pts_idx) > 0:
                pts_idx = np.array(pts_idx)
                x = pts_idx[:, 0]
                y = pts_idx[:, 1]
                z = pts_idx[:, 2]

                ax.scatter(
                    x, y, z,
                    s=60,
                    c=label_colors.get(label, "white"),
                    edgecolors="black",
                    depthshade=True,
                    label=f"CP label {label}"
                )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=elev, azim=azim)
        ax.legend()

    
    # PLOT 3D: view 1
    
    fig3 = plt.figure(figsize=(9, 8))
    ax3 = fig3.add_subplot(111, projection="3d")
    plot_3d_scene(
        ax=ax3,
        elev=20,
        azim=45,
        title="Validation 3D: mask + control points (view 1)"
    )
    plt.tight_layout()

    # PLOT 3D: view 2
    
    fig4 = plt.figure(figsize=(9, 8))
    ax4 = fig4.add_subplot(111, projection="3d")
    plot_3d_scene(
        ax=ax4,
        elev=35,
        azim=135,
        title="Validation 3D: mask + control points (view 2)"
    )
    plt.tight_layout()

    plt.show()

    fig1.savefig("control_points_2d.png", dpi=200, bbox_inches="tight")
    fig2.savefig("mask_2d.png", dpi=200, bbox_inches="tight")
    fig3.savefig("mask_control_points_3d_view1.png", dpi=200, bbox_inches="tight")
    fig4.savefig("mask_control_points_3d_view2.png", dpi=200, bbox_inches="tight")

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)

    print(
        "Saved plots: "
        "control_points_2d.png, "
        "mask_2d.png, "
        "mask_control_points_3d_view1.png, "
        "mask_control_points_3d_view2.png"
    )


def _frame_name(frame_path: Path) -> str:
        """Return a clean frame name without .nii.gz / .nii suffix."""
        name = frame_path.name
        if name.endswith(".nii.gz"):
            return name[:-7]
        if name.endswith(".nii"):
            return name[:-4]
        return frame_path.stem


def _serialize_points(points_dict: dict, labels: list[int]) -> dict:
    """
    Convert:
        points[label] = [(point, direction), ...]
    into:
        {
          "1": [
            {"point": [x,y,z], "direction": [dx,dy,dz]},
            ...
          ],
          ...
        }
    """
    out = {}
    for label in labels:
        pts = points_dict.get(label, [])
        out[str(label)] = [
            {
                "point": [float(c) for c in point],
                "direction": [float(c) for c in direction],
            }
            for point, direction in pts
        ]
    return out


def plot_anchor_first_alignment(anchor_mask_path, aligned_first_mask, patient_name, out_dir):
    """
    Plot anchor mask and aligned first patient mask.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # read anchor mask
    anchor_mask = sitk.ReadImage(str(anchor_mask_path), sitk.sitkUInt8)

    # convert to numpy arrays
    anchor_arr = sitk.GetArrayFromImage(anchor_mask) > 0          # [z, y, x]
    first_arr = sitk.GetArrayFromImage(aligned_first_mask) > 0    # [z, y, x]

    # choose a representative slice
    union_arr = anchor_arr | first_arr

    if np.any(union_arr):
        z_slice = int(np.round(np.mean(np.argwhere(union_arr)[:, 0])))
    else:
        z_slice = anchor_arr.shape[0] // 2

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(anchor_arr[z_slice], cmap="gray")
    axes[0].set_title("Anchor mask")
    axes[0].axis("off")

    axes[1].imshow(first_arr[z_slice], cmap="gray")
    axes[1].set_title("First frame aligned")
    axes[1].axis("off")

    axes[2].imshow(anchor_arr[z_slice], cmap="Reds", alpha=0.45)
    axes[2].imshow(first_arr[z_slice], cmap="Blues", alpha=0.45)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(f"{patient_name} - anchor vs first frame, z={z_slice}")
    fig.tight_layout()

    out_path = out_dir / f"{patient_name}_anchor_vs_first_frame.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

