import math
import numpy as np

def build_scene(stress_test_field=True, n_field_max=160, seed=1234):
    hero = np.array([
        [0.0,   -1000.0, 0.0,  1000.0, 0.92, 0.92, 0.85, 2.0],
        [0.0,    0.5,   -1.5,  0.5,    0.85, 0.15, 0.15, 0.0],
        [-1.1,   0.5,   -1.2,  0.5,    0.92, 0.92, 0.95, 1.0],
        [1.1,    0.5,   -1.8,  0.5,    0.15, 0.30, 0.85, 0.0],
        [0.0,    1.3,   -2.6,  0.35,   0.95, 0.75, 0.15, 1.0],
        [-2.1,   0.35,  -0.6,  0.35,   0.15, 0.75, 0.35, 0.0],
        [1.9,    0.35,  -0.4,  0.35,   0.95, 0.45, 0.10, 1.0],
        [-0.6,   0.25,   0.5,  0.25,   0.85, 0.30, 0.75, 0.0],
    ], dtype=np.float64)

    if not stress_test_field:
        return hero

    rng = np.random.default_rng(seed)
    hero_x = hero[1:, 0]
    hero_z = hero[1:, 2]
    hero_r = hero[1:, 3]

    field_rows = []
    for a in range(-8, 9):
        for b in range(-8, 3):
            cx = a * 1.0 + rng.uniform(-0.4, 0.4)
            cz = b * 1.0 + rng.uniform(-0.4, 0.4) - 1.0
            if cz > 2.2:
                continue
            radius = rng.uniform(0.14, 0.22)
            d = np.sqrt((hero_x - cx) ** 2 + (hero_z - cz) ** 2)
            if np.any(d < (hero_r + radius + 0.25)):
                continue
            if rng.random() < 0.78:
                material = 0.0
                color = rng.uniform(0.15, 0.9, size=3)
            else:
                material = 1.0
                color = rng.uniform(0.6, 0.95, size=3)
            field_rows.append([cx, radius, cz, radius, color[0], color[1], color[2], material])
            if len(field_rows) >= n_field_max:
                break
        if len(field_rows) >= n_field_max:
            break

    field = np.array(field_rows, dtype=np.float64)
    return np.vstack([hero, field])

spheres = build_scene()
n = spheres.shape[0]
print("Total spheres:", n)

def fmt_rows(rows):
    return ",\n".join("    {" + ", ".join(f"{v:.6f}f" for v in row) + "}" for row in rows)

with open("src/scene_data.cuh", "w") as f:
    f.write("// Auto-generated from the validated Python scene (seed=1234).\n")
    f.write("// Identical 168-sphere layout as the original numba.cuda version.\n")
    f.write("// Include ONLY from kernel.cu - __constant__ must be defined once.\n")
    f.write("#pragma once\n\n")
    f.write(f"#define NUM_SPHERES {n}\n\n")
    f.write("// cx, cy, cz, radius, r, g, b, material (0=diffuse, 1=metal, 2=checker)\n")
    f.write("__device__ __constant__ float d_spheres[NUM_SPHERES][8] = {\n")
    f.write(fmt_rows(spheres))
    f.write("\n};\n")

with open("src/scene_host.h", "w") as f:
    f.write("// Host-accessible copy of the exact same scene data, for the CPU\n")
    f.write("// reference renderer used in the CPU-vs-GPU benchmark.\n")
    f.write("#pragma once\n\n")
    f.write(f"#define NUM_SPHERES_HOST {n}\n")
    f.write("extern const float h_spheres[NUM_SPHERES_HOST][8];\n")

with open("src/scene_host.cpp", "w") as f:
    f.write('#include "scene_host.h"\n\n')
    f.write("const float h_spheres[NUM_SPHERES_HOST][8] = {\n")
    f.write(fmt_rows(spheres))
    f.write("\n};\n")

print("Wrote src/scene_data.cuh, src/scene_host.h, src/scene_host.cpp")
