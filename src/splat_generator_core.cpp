#include <torch/extension.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Структура параметричного отвору (вікно, двері, прохід)
struct Opening {
    float horiz_start; // Відстань від початку стіни (в метрах) до початку отвору
    float horiz_end;   // Відстань від початку стіни до кінця отвору
    float vert_start;  // Нижня межа отвору від підлоги (0.0 для дверей, ~0.9 для вікна)
    float vert_end;    // Верхня межа отвору (~2.1 для дверей, ~2.0 для вікна)

    Opening(float hs, float he, float vs, float ve)
        : horiz_start(hs), horiz_end(he), vert_start(vs), vert_end(ve) {}
};

// Структура сферичного руйнування (дірка від прильоту/вибуху)
struct DestructionHole {
    float x, y, z;     // Світові 3D-координати центру пролому
    float radius;      // Радіус руйнування (умовні 5см - 50см)

    DestructionHole(float cx, float cy, float cz, float r)
        : x(cx), y(cy), z(cz), radius(r) {}
};

// Модернізована структура стіни
struct Wall {
    float start_x, start_z;
    float end_x, end_z;
    float height;
    float normal_x, normal_z;
    std::vector<Opening> openings; // Динамічний список отворів у цій стіні

    Wall(float sx, float sz, float ex, float ez, float h, float nx, float nz, std::vector<Opening> ops)
        : start_x(sx), start_z(sz), end_x(ex), end_z(ez), height(h), normal_x(nx), normal_z(nz), openings(ops) {}
};

struct Floor {
    float min_x, max_x;
    float min_z, max_z;
    float y;

    Floor(float minx, float maxx, float minz, float maxz, float yy)
        : min_x(minx), max_x(maxx), min_z(minz), max_z(maxz), y(yy) {}
};

// Функція розрахунку запікання світла та Ambient Occlusion (AO)
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

// Головна функція генерації сцени
py::tuple generate_scene_cxx(
    std::vector<Wall> walls, 
    std::vector<Floor> floors, 
    std::vector<DestructionHole> holes,
    float density,
    std::vector<float> wall_color, 
    std::vector<float> floor_color,
    std::vector<float> corners_flat
) {
    size_t num_walls = walls.size();
    size_t num_floors = floors.size();

    std::vector<int> wall_steps_h(num_walls, 0);
    std::vector<int> wall_steps_v(num_walls, 0);
    std::vector<long> wall_offsets(num_walls, 0);
    std::vector<long> wall_actual_counts(num_walls, 0); 
    
    long total_wall_points = 0;

    // Константи кроку та зміщення (визначені ОДИН раз на початку)
    float step_h = 1.0f / density;
    float step_v = 1.0f / density;
    float offset_dist = 0.06f;

    // --- 1. COUNTING PASS ДЛЯ СТІН (Аналіз вікон, дверей та руйнувань) ---
    for (size_t i = 0; i < num_walls; ++i) {
        float dx = walls[i].end_x - walls[i].start_x;
        float dz = walls[i].end_z - walls[i].start_z;
        float length = std::sqrt(dx * dx + dz * dz);
        if (length < 1e-4f) continue;

        int num_steps_h = std::max(2, static_cast<int>(length * density));
        int num_steps_v = std::max(2, static_cast<int>(walls[i].height * density));
        
        wall_steps_h[i] = num_steps_h;
        wall_steps_v[i] = num_steps_v;
        wall_offsets[i] = total_wall_points;

        long valid_points_in_wall = 0;

        for (int h = 0; h < num_steps_h; ++h) {
            float alpha = static_cast<float>(h) / (num_steps_h - 1);
            float t_dist = alpha * length; 
            float curr_x = walls[i].start_x + alpha * dx;
            float curr_z = walls[i].start_z + alpha * dz;

            for (int v = 0; v < num_steps_v; ++v) {
                float curr_y = (static_cast<float>(v) / (num_steps_v - 1)) * walls[i].height;

                // Перевірка отворів (вікна / двері)
                bool inside_opening = false;
                for (const auto& op : walls[i].openings) {
                    if (t_dist >= op.horiz_start && t_dist <= op.horiz_end &&
                        curr_y >= op.vert_start && curr_y <= op.vert_end) {
                        inside_opening = true;
                        break;
                    }
                }
                if (inside_opening) continue;

                // Перевірка сфер деструкції (дірки)
                float shifted_x = curr_x + walls[i].normal_x * offset_dist;
                float shifted_z = curr_z + walls[i].normal_z * offset_dist;
                
                bool inside_hole = false;
                for (const auto& hole : holes) {
                    float hdx = shifted_x - hole.x;
                    float hdy = curr_y - hole.y;
                    float hdz = shifted_z - hole.z;
                    if ((hdx*hdx + hdy*hdy + hdz*hdz) <= (hole.radius * hole.radius)) {
                        inside_hole = true;
                        break;
                    }
                }
                if (inside_hole) continue;

                valid_points_in_wall++;
            }
        }
        wall_actual_counts[i] = valid_points_in_wall;
        total_wall_points += valid_points_in_wall;
    }

    // --- 2. РОЗРАХУНОК КРОКІВ ДЛЯ ПІДЛОГИ ---
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

    // --- 3.ВИДІЛЕННЯ ПАМ'ЯТІ ПІД ОСТАТОЧНІ МАСИВИ ---
    long total_points = total_wall_points + total_floor_points;

    std::vector<float> out_xyz(total_points * 3);
    std::vector<float> out_scale(total_points * 3);
    std::vector<float> out_rotation(total_points * 4);
    std::vector<float> out_opacity(total_points);
    std::vector<float> out_rgb(total_points * 3);

    // --- 4. ПАРАЛЕЛЬНИЙ ЦИКЛ ГЕНЕРАЦІЇ СТІН ---
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_walls; ++i) {
        int num_steps_h = wall_steps_h[i];
        int num_steps_v = wall_steps_v[i];
        if (num_steps_h == 0 || num_steps_v == 0 || wall_actual_counts[i] == 0) continue;

        const auto& wall = walls[i];
        float dx = wall.end_x - wall.start_x;
        float dz = wall.end_z - wall.start_z;
        float length = std::sqrt(dx * dx + dz * dz);
        float ux = dx / length, uz = dz / length;
        float angle = std::atan2(-dz, dx);

        long global_write_idx = wall_offsets[i]; 
        long local_counter = 0;                  

        float jitter_max_tangent = step_h * 0.15f;
        float jitter_max_v = step_v * 0.15f;

        for (int h = 0; h < num_steps_h; ++h) {
            float alpha = static_cast<float>(h) / (num_steps_h - 1);
            float t_dist = alpha * length;
            float curr_x = wall.start_x + alpha * dx;
            float curr_z = wall.start_z + alpha * dz;

            for (int v = 0; v < num_steps_v; ++v) {
                float curr_y = (static_cast<float>(v) / (num_steps_v - 1)) * wall.height;

                bool skip = false;
                for (const auto& op : wall.openings) {
                    if (t_dist >= op.horiz_start && t_dist <= op.horiz_end &&
                        curr_y >= op.vert_start && curr_y <= op.vert_end) { skip = true; break; }
                }
                if (skip) continue;

                float tangent_jitter = std::sin(h * 12.9898f + v * 78.233f) * jitter_max_tangent;
                float y_jitter = std::cos(h * 4.1414f + v * 23.131f) * jitter_max_v;

                float jittered_x = curr_x + ux * tangent_jitter;
                float jittered_z = curr_z + uz * tangent_jitter;
                float jittered_y = std::max(0.0f, std::min(wall.height, curr_y + y_jitter));

                float shifted_x = jittered_x + wall.normal_x * offset_dist;
                float shifted_z = jittered_z + wall.normal_z * offset_dist;

                for (const auto& hole : holes) {
                    float hdx = shifted_x - hole.x;
                    float hdy = jittered_y - hole.y;
                    float hdz = shifted_z - hole.z;
                    if ((hdx*hdx + hdy*hdy + hdz*hdz) <= (hole.radius * hole.radius)) { skip = true; break; }
                }
                if (skip) continue;

                long write_idx = global_write_idx + local_counter;
                local_counter++;

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

    // --- 5. ПАРАЛЕЛЬНИЙ ЦИКЛ ГЕНЕРАЦІЇ ПІДЛОГИ ---
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

// ═══════════════════════════════════════════════════════════════════
//  LAYER 2 — КОЛІЗІЙНА / СИМУЛЯЦІЙНА ГЕОМЕТРІЯ (для LOS/фізики)
// ═══════════════════════════════════════════════════════════════════
// На відміну від generate_scene_cxx (яка генерує ХМАРУ ТОЧОК для
// візуалізації), ця функція генерує РЕАЛЬНУ ТРИКУТНУ СІТКУ (вершини +
// індекси) — точні box'и для стін і плит для підлог. Проти цієї геометрії
// THREE.Raycaster + three-mesh-bvh на клієнті робитимуть точні перевірки
// LOS/зіткнень замість крихкого shadow-cubemap трюку проти хмари точок.
//
// СВІДОМИЙ КОМПРОМІС (задокументовано в ARCHITECTURE.md): отвори (Opening)
// вирізаються ТОЧНОЮ інтервальною математикою (як і для точок), а дірки
// від руйнувань (DestructionHole) — через рівномірну сітку комірок
// (~cell_size, типово 0.2м): комірка або повністю є, або повністю відсутня.
// Простіше й надійніше за справжній 3D CSG-буль; точності в кілька
// сантиметрів цілком достатньо для LOS/пробиття кулею в CQB-симуляції.
//
// Матеріал меша на клієнті варто рендерити з side:THREE.DoubleSide —
// порядок вершин граней тут не гарантовано консистентний для backface
// culling, але для raycasting-only геометрії це не має значення.

inline void add_box(
    std::vector<float>& verts, std::vector<uint32_t>& indices,
    const float corners[8][3]
) {
    uint32_t base = static_cast<uint32_t>(verts.size() / 3);
    for (int i = 0; i < 8; ++i) {
        verts.push_back(corners[i][0]);
        verts.push_back(corners[i][1]);
        verts.push_back(corners[i][2]);
    }
    // corners: 0..3 = нижня грань, 4..7 = відповідні кути верхньої грані
    static const int faces[6][4] = {
        {0,1,2,3}, {4,7,6,5}, {0,4,5,1}, {3,2,6,7}, {0,3,7,4}, {1,5,6,2}
    };
    for (auto& f : faces) {
        indices.push_back(base + f[0]); indices.push_back(base + f[1]); indices.push_back(base + f[2]);
        indices.push_back(base + f[0]); indices.push_back(base + f[2]); indices.push_back(base + f[3]);
    }
}

inline bool point_in_any_opening(float t, float y, const std::vector<Opening>& openings) {
    for (const auto& op : openings) {
        if (t >= op.horiz_start && t <= op.horiz_end && y >= op.vert_start && y <= op.vert_end) return true;
    }
    return false;
}

inline bool point_in_any_hole(float x, float y, float z, const std::vector<DestructionHole>& holes) {
    for (const auto& h : holes) {
        float dx = x - h.x, dy = y - h.y, dz = z - h.z;
        if (dx * dx + dy * dy + dz * dz <= h.radius * h.radius) return true;
    }
    return false;
}

py::tuple generate_collision_mesh_cxx(
    std::vector<Wall> walls,
    std::vector<Floor> floors,
    std::vector<DestructionHole> holes,
    float cell_size
) {
    std::vector<float> vertices;
    std::vector<uint32_t> indices;

    float wall_thickness = 0.12f;
    float floor_thickness = 0.12f;

    // ── СТІНИ ──
    for (const auto& wall : walls) {
        float dx = wall.end_x - wall.start_x;
        float dz = wall.end_z - wall.start_z;
        float length = std::sqrt(dx * dx + dz * dz);
        if (length < 1e-4f) continue;

        float ux = dx / length, uz = dz / length;
        float nx = wall.normal_x, nz = wall.normal_z;

        int num_cells_h = std::max(1, static_cast<int>(std::ceil(length / cell_size)));
        int num_cells_v = std::max(1, static_cast<int>(std::ceil(wall.height / cell_size)));
        float cell_h = length / num_cells_h;
        float cell_v = wall.height / num_cells_v;

        for (int h = 0; h < num_cells_h; ++h) {
            float t0 = h * cell_h, t1 = (h + 1) * cell_h;
            float t_mid = (t0 + t1) * 0.5f;

            for (int v = 0; v < num_cells_v; ++v) {
                float y0 = v * cell_v, y1 = (v + 1) * cell_v;
                float y_mid = (y0 + y1) * 0.5f;

                if (point_in_any_opening(t_mid, y_mid, wall.openings)) continue;

                float cx = wall.start_x + ux * t_mid;
                float cz = wall.start_z + uz * t_mid;
                if (point_in_any_hole(cx, y_mid, cz, holes)) continue;

                float p0x = wall.start_x + ux * t0, p0z = wall.start_z + uz * t0;
                float p1x = wall.start_x + ux * t1, p1z = wall.start_z + uz * t1;
                float halfT = wall_thickness * 0.5f;

                float corners[8][3] = {
                    { p0x - nx * halfT, y0, p0z - nz * halfT },
                    { p1x - nx * halfT, y0, p1z - nz * halfT },
                    { p1x + nx * halfT, y0, p1z + nz * halfT },
                    { p0x + nx * halfT, y0, p0z + nz * halfT },
                    { p0x - nx * halfT, y1, p0z - nz * halfT },
                    { p1x - nx * halfT, y1, p1z - nz * halfT },
                    { p1x + nx * halfT, y1, p1z + nz * halfT },
                    { p0x + nx * halfT, y1, p0z + nz * halfT }
                };
                add_box(vertices, indices, corners);
            }
        }
    }

    // ── ПІДЛОГИ ──
    for (const auto& floor : floors) {
        float len_x = floor.max_x - floor.min_x;
        float len_z = floor.max_z - floor.min_z;
        if (len_x < 1e-4f || len_z < 1e-4f) continue;

        int num_cells_x = std::max(1, static_cast<int>(std::ceil(len_x / cell_size)));
        int num_cells_z = std::max(1, static_cast<int>(std::ceil(len_z / cell_size)));
        float cell_x = len_x / num_cells_x;
        float cell_z = len_z / num_cells_z;
        float halfT = floor_thickness * 0.5f;

        for (int i = 0; i < num_cells_x; ++i) {
            float x0 = floor.min_x + i * cell_x, x1 = floor.min_x + (i + 1) * cell_x;
            float x_mid = (x0 + x1) * 0.5f;

            for (int j = 0; j < num_cells_z; ++j) {
                float z0 = floor.min_z + j * cell_z, z1 = floor.min_z + (j + 1) * cell_z;
                float z_mid = (z0 + z1) * 0.5f;

                if (point_in_any_hole(x_mid, floor.y, z_mid, holes)) continue;

                float corners[8][3] = {
                    { x0, floor.y - halfT, z0 }, { x1, floor.y - halfT, z0 },
                    { x1, floor.y - halfT, z1 }, { x0, floor.y - halfT, z1 },
                    { x0, floor.y + halfT, z0 }, { x1, floor.y + halfT, z0 },
                    { x1, floor.y + halfT, z1 }, { x0, floor.y + halfT, z1 }
                };
                add_box(vertices, indices, corners);
            }
        }
    }

    return py::make_tuple(vertices, indices);
}

// --- 6. РЕЄСТРАЦІЯ МОДУЛЯ ДЛЯ PYBIND11 ---
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<Opening>(m, "Opening")
        .def(py::init<float, float, float, float>());

    py::class_<DestructionHole>(m, "DestructionHole")
        .def(py::init<float, float, float, float>());

    py::class_<Wall>(m, "Wall")
        .def(py::init<float, float, float, float, float, float, float, std::vector<Opening>>());

    py::class_<Floor>(m, "Floor")
        .def(py::init<float, float, float, float, float>());

    m.def("generate_scene", &generate_scene_cxx, "Parametric scene generation core with destruction support");
    m.def("generate_collision_mesh", &generate_collision_mesh_cxx, "Generate exact box/plane collision mesh (Layer 2) for LOS/physics");
}