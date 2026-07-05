#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

struct Wall {
    float start_x, start_z;
    float end_x, end_z;
    float height;
    float normal_x, normal_z;
    int wall_type; // 0 - wall, 1 - door
};

struct Floor {
    float min_x, max_x;
    float min_z, max_z;
    float y;
};

std::vector<float> calculate_lighting(float x, float y, float z, float angle, float height, bool is_floor, const std::vector<float>& base_color) {
    float ao_factor = 1.0f;
    float shadow_zone = 0.3f;
    
    if (y < shadow_zone) {
        ao_factor *= 0.45f + 0.55f * (y / shadow_zone);
    } else if ((height - y) < shadow_zone) {
        ao_factor *= 0.45f + 0.55f * ((height - y) / shadow_zone);
    }
    
    ao_factor = std::max(0.2f, ao_factor);
    float sun_angle = M_PI / 4.0f;
    float sun_factor = is_floor ? 1.05f : 1.0f + 0.12f * std::cos(angle - sun_angle);
    
    return {
        std::max(0.0f, std::min(1.0f, base_color[0] * ao_factor * sun_factor)),
        std::max(0.0f, std::min(1.0f, base_color[1] * ao_factor * sun_factor)),
        std::max(0.0f, std::min(1.0f, base_color[2] * ao_factor * sun_factor))
    };
}

// Функція-моноліт, яка збирає всю сцену відразу
py::tuple generate_scene_cxx(std::vector<Wall> walls, std::vector<Floor> floors, float density, std::vector<float> wall_color, std::vector<float> floor_color) {
    std::vector<std::vector<float>> out_xyz;
    std::vector<std::vector<float>> out_scale;
    std::vector<std::vector<float>> out_rotation;
    std::vector<float> out_opacity;
    std::vector<std::vector<float>> out_rgb;

    float step_h = 1.0f / density;
    float step_v = 1.0f / density;
    float offset_dist = 0.04f;

    // 1. СТІНИ
    for (const auto& wall : walls) {
        float dx = wall.end_x - wall.start_x;
        float dz = wall.end_z - wall.start_z;
        float length = std::sqrt(dx*dx + dz*dz);
        if (length < 1e-4) continue;

        float angle = std::atan2(-dz, dx);
        int num_steps_h = std::max(2, static_cast<int>(length * density));
        
        float y_min = (wall.wall_type == 1) ? 2.1f : 0.0f;
        float y_max = wall.height;
        if (y_min >= y_max) continue;
        
        int num_steps_v = std::max(2, static_cast<int>((y_max - y_min) * density));

        for (int h = 0; h < num_steps_h; ++h) {
            float alpha = static_cast<float>(h) / (num_steps_h - 1);
            float curr_x = wall.start_x + alpha * dx;
            float curr_z = wall.start_z + alpha * dz;

            float shifted_x = curr_x + wall.normal_x * offset_dist;
            float shifted_z = curr_z + wall.normal_z * offset_dist;

            for (int v = 0; v < num_steps_v; ++v) {
                float curr_y = y_min + (static_cast<float>(v) / (num_steps_v - 1)) * (y_max - y_min);

                out_xyz.push_back({shifted_x, curr_y, shifted_z});
                // Було: out_scale.push_back({step_h * 1.3f, step_v * 1.3f, 0.001f});
                out_scale.push_back({step_h * 1.9f, step_v * 1.9f, 0.001f});
                out_rotation.push_back({static_cast<float>(std::cos(angle / 2.0)), 0.0f, static_cast<float>(std::sin(angle / 2.0)), 0.0f});
                out_opacity.push_back(1.0f);
                out_rgb.push_back(calculate_lighting(shifted_x, curr_y, shifted_z, angle, wall.height, false, wall_color));
            }
        }
    }

    // 2. ПІДЛОГА
    // 2. ПІДЛОГА
for (const auto& floor : floors) {
    float len_x = floor.max_x - floor.min_x;
    float len_z = floor.max_z - floor.min_z;

    int num_x = std::max(2, static_cast<int>(len_x * density));
    int num_z = std::max(2, static_cast<int>(len_z * density));

    // Напів-випадковий шум для руйнування періодичності сітки
    float jitter_max_x = step_h * 0.15f;
    float jitter_max_z = step_v * 0.15f;

    for (int i = 0; i < num_x; ++i) {
        float base_x = floor.min_x + (static_cast<float>(i) / (num_x - 1)) * len_x;
        for (int j = 0; j < num_z; ++j) {
            float base_z = floor.min_z + (static_cast<float>(j) / (num_z - 1)) * len_z;

            // Генеруємо легкий зсув на основі індексів (детерміновано, без rand())
            float noise_x = std::sin(i * 12.9898f + j * 78.233f) * jitter_max_x;
            float noise_z = std::cos(i * 4.1414f + j * 23.131f) * jitter_max_z;

            float final_x = base_x + noise_x;
            float final_z = base_z + noise_z;

            out_xyz.push_back({final_x, floor.y, final_z});
            out_scale.push_back({step_h * 1.9f, 0.0f, step_v * 1.9f}); // Зберігаємо нове перекриття
            out_rotation.push_back({1.0f, 0.0f, 0.0f, 0.0f});
            out_opacity.push_back(1.0f);
            out_rgb.push_back(calculate_lighting(final_x, floor.y, final_z, 0.0f, 3.0f, true, floor_color));
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