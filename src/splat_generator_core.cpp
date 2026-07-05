#include <torch/extension.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

struct Wall {
    float start_x, start_z;
    float end_x, end_z;
    float height;
    float normal_x, normal_z;
    int wall_type; // 0 - wall, 1 - door

    Wall(float sx, float sz, float ex, float ez, float h, float nx, float nz, int wt)
        : start_x(sx), start_z(sz), end_x(ex), end_z(ez), height(h), normal_x(nx), normal_z(nz), wall_type(wt) {}
};

struct Floor {
    float min_x, max_x;
    float min_z, max_z;
    float y;

    Floor(float minx, float maxx, float minz, float maxz, float yy)
        : min_x(minx), max_x(maxx), min_z(minz), max_z(maxz), y(yy) {}
};

inline void calculate_lighting(
    float x, float y, float z, float angle, float height, bool is_floor,
    const std::vector<float>& base_color,
    const std::vector<float>& corners_flat,
    float out_rgb[3]
) {
    float ao_factor = 1.0f;
    float shadow_zone = 0.3f;

    if (y < shadow_zone) {
        ao_factor *= 0.45f + 0.55f * (y / shadow_zone);
    } else if ((height - y) < shadow_zone) {
        ao_factor *= 0.45f + 0.55f * ((height - y) / shadow_zone);
    }

    if (!is_floor) {
        for (size_t i = 0; i + 1 < corners_flat.size(); i += 2) {
            float cx = corners_flat[i], cz = corners_flat[i + 1];
            float dx = x - cx, dz = z - cz;
            float dist = std::sqrt(dx * dx + dz * dz);
            if (dist < 0.3f) {
                ao_factor *= 0.5f + 0.5f * (dist / 0.3f);
            }
        }
    }

    ao_factor = std::max(0.2f, ao_factor);
    float sun_angle = static_cast<float>(M_PI) / 4.0f;
    float sun_factor = is_floor ? 1.05f : 1.0f + 0.12f * std::cos(angle - sun_angle);

    out_rgb[0] = std::max(0.0f, std::min(1.0f, base_color[0] * ao_factor * sun_factor));
    out_rgb[1] = std::max(0.0f, std::min(1.0f, base_color[1] * ao_factor * sun_factor));
    out_rgb[2] = std::max(0.0f, std::min(1.0f, base_color[2] * ao_factor * sun_factor));
}

py::tuple generate_scene_cxx(
    std::vector<Wall> walls, std::vector<Floor> floors, float density,
    std::vector<float> wall_color, std::vector<float> floor_color,
    std::vector<float> corners_flat
) {
    size_t num_walls = walls.size();
    size_t num_floors = floors.size();

    std::vector<int> wall_steps_h(num_walls, 0);
    std::vector<int> wall_steps_v(num_walls, 0);
    std::vector<long> wall_offsets(num_walls, 0);
    long total_wall_points = 0;

    for (size_t i = 0; i < num_walls; ++i) {
        float dx = walls[i].end_x - walls[i].start_x;
        float dz = walls[i].end_z - walls[i].start_z;
        float length = std::sqrt(dx * dx + dz * dz);
        if (length < 1e-4f) continue;

        int num_steps_h = std::max(2, static_cast<int>(length * density));
        float y_min = (walls[i].wall_type == 1) ? 2.1f : 0.0f;
        float y_max = walls[i].height;
        
        int num_steps_v = (y_min >= y_max) ? 0 : std::max(2, static_cast<int>((y_max - y_min) * density));
        
        wall_steps_h[i] = num_steps_h;
        wall_steps_v[i] = num_steps_v;
        wall_offsets[i] = total_wall_points;
        total_wall_points += static_cast<long>(num_steps_h) * num_steps_v;
    }

    std::vector<int> floor_steps_x(num_floors, 0);
    std::vector<int> floor_steps_z(num_floors, 0);
    std::vector<long> floor_offsets(num_floors, 0);
    long total_floor_points = 0;

    for (size_t i = 0; i < num_floors; ++i) {
        float len_x = floors[i].max_x - floors[i].min_x;
        float len_z = floors[i].max_z - floors[i].min_z;
        int num_x = std::max(2, static_cast<int>(len_x * density));
        int num_z = std::max(2, static_cast<int>(len_z * density));

        floor_steps_x[i] = num_x;
        floor_steps_z[i] = num_z;
        floor_offsets[i] = total_floor_points;
        total_floor_points += static_cast<long>(num_x) * num_z;
    }

    long total_points = total_wall_points + total_floor_points;

    std::vector<float> out_xyz(total_points * 3);
    std::vector<float> out_scale(total_points * 3);
    std::vector<float> out_rotation(total_points * 4);
    std::vector<float> out_opacity(total_points);
    std::vector<float> out_rgb(total_points * 3);

    float step_h = 1.0f / density;
    float step_v = 1.0f / density;
    float offset_dist = 0.06f;

    // Parallel processing for Walls
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_walls; ++i) {
        int num_steps_h = wall_steps_h[i];
        int num_steps_v = wall_steps_v[i];
        if (num_steps_h == 0 || num_steps_v == 0) continue;

        const auto& wall = walls[i];
        float dx = wall.end_x - wall.start_x;
        float dz = wall.end_z - wall.start_z;
        float length = std::sqrt(dx * dx + dz * dz);
        float ux = dx / length, uz = dz / length;
        float angle = std::atan2(-dz, dx);
        float y_min = (wall.wall_type == 1) ? 2.1f : 0.0f;
        float y_max = wall.height;

        long global_point_idx = wall_offsets[i];
        float jitter_max_tangent = step_h * 0.15f;
        float jitter_max_v = step_v * 0.15f;

        for (int h = 0; h < num_steps_h; ++h) {
            float alpha = static_cast<float>(h) / (num_steps_h - 1);
            float curr_x = wall.start_x + alpha * dx;
            float curr_z = wall.start_z + alpha * dz;

            for (int v = 0; v < num_steps_v; ++v) {
                float curr_y = y_min + (static_cast<float>(v) / (num_steps_v - 1)) * (y_max - y_min);

                float tangent_jitter = std::sin(h * 12.9898f + v * 78.233f) * jitter_max_tangent;
                float y_jitter = std::cos(h * 4.1414f + v * 23.131f) * jitter_max_v;

                float jittered_x = curr_x + ux * tangent_jitter;
                float jittered_z = curr_z + uz * tangent_jitter;
                float jittered_y = std::max(y_min, std::min(y_max, curr_y + y_jitter));

                float shifted_x = jittered_x + wall.normal_x * offset_dist;
                float shifted_z = jittered_z + wall.normal_z * offset_dist;

                long write_idx = global_point_idx + (h * num_steps_v + v);

                out_xyz[write_idx * 3 + 0] = shifted_x;
                out_xyz[write_idx * 3 + 1] = jittered_y;
                out_xyz[write_idx * 3 + 2] = shifted_z;

                out_scale[write_idx * 3 + 0] = step_h * 1.9f;
                out_scale[write_idx * 3 + 1] = step_v * 1.9f;
                out_scale[write_idx * 3 + 2] = 0.001f;

                out_rotation[write_idx * 4 + 0] = static_cast<float>(std::cos(angle / 2.0));
                out_rotation[write_idx * 4 + 1] = 0.0f;
                out_rotation[write_idx * 4 + 2] = static_cast<float>(std::sin(angle / 2.0));
                out_rotation[write_idx * 4 + 3] = 0.0f;

                out_opacity[write_idx] = 1.0f;

                float rgb[3];
                calculate_lighting(shifted_x, jittered_y, shifted_z, angle, wall.height, false, wall_color, corners_flat, rgb);
                out_rgb[write_idx * 3 + 0] = rgb[0];
                out_rgb[write_idx * 3 + 1] = rgb[1];
                out_rgb[write_idx * 3 + 2] = rgb[2];
            }
        }
    }

    // Parallel processing for Floors
    #pragma omp parallel for schedule(dynamic)
    for (size_t f = 0; f < num_floors; ++f) {
        int num_x = floor_steps_x[f];
        int num_z = floor_steps_z[f];
        const auto& floor = floors[f];
        
        float len_x = floor.max_x - floor.min_x;
        float len_z = floor.max_z - floor.min_z;

        long global_point_idx = total_wall_points + floor_offsets[f];
        float jitter_max_x = step_h * 0.15f;
        float jitter_max_z = step_v * 0.15f;

        for (int i = 0; i < num_x; ++i) {
            float base_x = floor.min_x + (static_cast<float>(i) / (num_x - 1)) * len_x;
            for (int j = 0; j < num_z; ++j) {
                float base_z = floor.min_z + (static_cast<float>(j) / (num_z - 1)) * len_z;

                float noise_x = std::sin(i * 12.9898f + j * 78.233f) * jitter_max_x;
                float noise_z = std::cos(i * 4.1414f + j * 23.131f) * jitter_max_z;

                float final_x = base_x + noise_x;
                float final_z = base_z + noise_z;

                long write_idx = global_point_idx + (i * num_z + j);

                out_xyz[write_idx * 3 + 0] = final_x;
                out_xyz[write_idx * 3 + 1] = floor.y;
                out_xyz[write_idx * 3 + 2] = final_z;

                out_scale[write_idx * 3 + 0] = step_h * 1.9f;
                out_scale[write_idx * 3 + 1] = 0.0f;
                out_scale[write_idx * 3 + 2] = step_v * 1.9f;

                out_rotation[write_idx * 4 + 0] = 1.0f;
                out_rotation[write_idx * 4 + 1] = 0.0f;
                out_rotation[write_idx * 4 + 2] = 0.0f;
                out_rotation[write_idx * 4 + 3] = 0.0f;

                out_opacity[write_idx] = 1.0f;

                float rgb[3];
                calculate_lighting(final_x, floor.y, final_z, 0.0f, 3.0f, true, floor_color, corners_flat, rgb);
                out_rgb[write_idx * 3 + 0] = rgb[0];
                out_rgb[write_idx * 3 + 1] = rgb[1];
                out_rgb[write_idx * 3 + 2] = rgb[2];
            }
        }
    }

    return py::make_tuple(out_xyz, out_scale, out_rotation, out_opacity, out_rgb);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<Wall>(m, "Wall")
        .def(py::init<float, float, float, float, float, float, float, int>());

    py::class_<Floor>(m, "Floor")
        .def(py::init<float, float, float, float, float>());

    m.def("generate_scene", &generate_scene_cxx, "Fast analytical scene generation core");
}