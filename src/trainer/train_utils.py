import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import SimpleITK as sitk


def test_control_points_2d_3d(control_points, mask: sitk.Image):
    """
    Produce:
    1) plot 2D dei control points
    2) plot 2D della mask
    3) plot 3D mask + control points

    Assunzioni:
    - mask è 3D
    - control_points.points[label] contiene physical points (x,y,z)
    """

    mask_arr = sitk.GetArrayFromImage(mask)   # [z, y, x]

    # stessi colori per labels e punti
    label_colors = {
        1: "red",
        2: "lime",
        3: "deepskyblue"
    }

    # -----------------------------
    # 1) PLOT 2D SOLO CONTROL POINTS
    # proiezione XY
    # -----------------------------
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.set_title("Control points - proiezione XY")
    ax1.set_facecolor("black")

    for label, pts_phys in control_points.points.items():
        pts_idx = []

        for p in pts_phys:
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

    # -----------------------------
    # 2) PLOT 2D SOLO MASK
    # proiezione XY con max lungo z
    # -----------------------------
    mask_xy = np.max(mask_arr, axis=0)  # [y, x]

    cmap = mcolors.ListedColormap(["black", "red", "lime", "deepskyblue"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.set_title("Mask labels - proiezione XY")
    ax2.imshow(mask_xy, cmap=cmap, norm=norm, origin="upper")
    ax2.set_aspect("equal")
    plt.tight_layout()

    # -----------------------------
    # 3) PLOT 3D: MASK + CONTROL POINTS
    # per non appesantire troppo il plot,
    # mostro i voxel della mask con uno stride
    # -----------------------------
    fig3 = plt.figure(figsize=(9, 8))
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.set_title("Verifica 3D: mask + control points")

    # plot dei voxel della mask per ogni label
    for label in sorted(control_points.points.keys()):
        region = np.argwhere(mask_arr == label)   # [N, 3] in ordine (z,y,x)

        if len(region) > 0:
            # sottocampionamento per alleggerire il plot
            step = max(1, len(region) // 4000)
            region_sub = region[::step]

            z = region_sub[:, 0]
            y = region_sub[:, 1]
            x = region_sub[:, 2]

            ax3.scatter(
                x, y, z,
                s=1,
                c=label_colors.get(label, "white"),
                alpha=0.15
            )

    # plot dei control points
    for label, pts_phys in control_points.points.items():
        pts_idx = []

        for p in pts_phys:
            idx = mask.TransformPhysicalPointToIndex(tuple(map(float, p)))  # (x,y,z)
            pts_idx.append(idx)

        if len(pts_idx) > 0:
            pts_idx = np.array(pts_idx)
            x = pts_idx[:, 0]
            y = pts_idx[:, 1]
            z = pts_idx[:, 2]

            ax3.scatter(
                x, y, z,
                s=60,
                c=label_colors.get(label, "white"),
                edgecolors="black",
                depthshade=True,
                label=f"CP label {label}"
            )

    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.legend()

    plt.tight_layout()
    plt.show()
    fig1.savefig("control_points_2d.png", dpi=200, bbox_inches="tight")
    fig2.savefig("mask_2d.png", dpi=200, bbox_inches="tight")
    fig3.savefig("mask_control_points_3d.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    print("Saved plots: control_points_2d.png, mask_2d.png, mask_control_points_3d.png")