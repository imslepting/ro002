import os
import open3d as o3d

def preview_arm():
    mesh_dir = "assets/ra605_710/visual"
    
    if not os.path.exists(mesh_dir):
        print(f"Error: Directory {mesh_dir} not found.")
        return

    mesh_files = [f for f in os.listdir(mesh_dir) if f.lower().endswith(".stl")]
    mesh_files.sort()
    
    if not mesh_files:
        print(f"Error: No STL files found in {mesh_dir}")
        return

    print(f"Loading {len(mesh_files)} mesh files from {mesh_dir}...")
    
    geometries = []
    # Distinct colors for different links
    colors = [
        [0.7, 0.7, 0.7],  # Base
        [0.8, 0.3, 0.3],  # Link 1
        [0.3, 0.8, 0.3],  # Link 2
        [0.3, 0.3, 0.8],  # Link 3
        [0.8, 0.8, 0.3],  # Link 4
        [0.8, 0.3, 0.8],  # Link 5
        [0.3, 0.8, 0.8],  # Link 6
    ]

    for i, filename in enumerate(mesh_files):
        filepath = os.path.join(mesh_dir, filename)
        mesh = o3d.io.read_triangle_mesh(filepath)
        if mesh.is_empty():
            print(f"Warning: {filename} is empty.")
            continue
            
        mesh.compute_vertex_normals()
        color = colors[i % len(colors)]
        mesh.paint_uniform_color(color)
        geometries.append(mesh)
        print(f"Loaded {filename}")

    # Add a coordinate frame at the origin (base)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    geometries.append(frame)

    print("Opening 3D viewer. Please close the window to continue.")
    o3d.visualization.draw_geometries(geometries, window_name="HIWIN RA/RT605-710-G Model Preview")

if __name__ == "__main__":
    preview_arm()
