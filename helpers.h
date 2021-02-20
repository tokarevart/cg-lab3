#pragma once

#include "sptalgs.h"
#include "mesh.h"
#include "view.h"

struct Inter {
    spt::vec3d at;
    spt::vec3d normal;
    std::size_t face_id;
};

std::optional<Inter> ray_mesh_nearest_intersection(
    spt::vec3d origin, spt::vec3d dir, const Mesh& mesh) {

    spt::vec3d normal;
    spt::vec3d at;
    std::size_t face_id;
    double max_z = std::numeric_limits<double>::lowest();
    bool no_inters = true;
    for (std::size_t i = 0; i < mesh.surface.size(); ++i) {
        auto face = mesh.surface[i];
        auto ointer = spt::ray_intersect_triangle(
            origin, dir,
            mesh.verts[face.verts[0]],
            mesh.verts[face.verts[1]],
            mesh.verts[face.verts[2]]
            );
        if (!ointer.has_value()) {
            continue;
        }

        auto inter = ointer.value();
        if (inter[2] > max_z) {
            max_z = inter[2];
            at = inter;
            normal = mesh.face_normal(face);
            no_inters = false;
            face_id = i;
        }
    }

    if (no_inters) {
        return std::nullopt;
    } else {
        return Inter{at, normal, face_id};
    }
}

std::optional<Inter> vertray_mesh_nearest_intersection(
    std::size_t origin_vert, spt::vec3d dir, const Mesh& mesh) {

    spt::vec3d normal;
    spt::vec3d at;
    std::size_t face_id;
    double max_z = std::numeric_limits<double>::lowest();
    bool no_inters = true;
    for (std::size_t i = 0; i < mesh.surface.size(); ++i) {
        auto face = mesh.surface[i];
        if (face.contains_vert(origin_vert)) {
            continue;
        }

        auto origin = mesh.verts[origin_vert];
        auto ointer = spt::ray_intersect_triangle(
            origin, dir,
            mesh.verts[face.verts[0]],
            mesh.verts[face.verts[1]],
            mesh.verts[face.verts[2]]
            );
        if (!ointer.has_value()) {
            continue;
        }

        auto inter = ointer.value();
        if (inter[2] > max_z) {
            max_z = inter[2];
            at = inter;
            normal = mesh.face_normal(face);
            no_inters = false;
            face_id = i;
        }
    }

    if (no_inters) {
        return std::nullopt;
    } else {
        return Inter{at, normal, face_id};
    }
}

bool vertray_intersect_mesh(
    std::size_t origin_vert, spt::vec3d dir, const Mesh& mesh) {

    for (std::size_t i = 0; i < mesh.surface.size(); ++i) {
        auto face = mesh.surface[i];
        if (face.contains_vert(origin_vert)) {
            continue;
        }

        auto origin = mesh.verts[origin_vert];
        auto ointer = spt::ray_intersect_triangle(
            origin, dir,
            mesh.verts[face.verts[0]],
            mesh.verts[face.verts[1]],
            mesh.verts[face.verts[2]]
            );
        if (ointer.has_value()) {
            return true;
        }
    }
    return false;
}

std::array<int, 2> min_max_image_y(
    const Viewport& viewport, ViewTransformer vtran, const Mesh& mesh) {

    int min_y = std::max(
        0, static_cast<int>(
               vtran.to_viewport(
                   std::array{0.0, mesh.max_coor(1)})[1]));
    int max_y = std::min(
        static_cast<int>(
            vtran.to_viewport(
                std::array{0.0, mesh.min_coor(1)})[1] + 1),
        viewport.height());
    return { min_y, max_y };
}

std::optional<spt::vec2d> segm_horizline_intersection(
    const std::array<spt::vec2d, 2> pts, double y, double& t
    ) {
    double p0y = pts[0][1];
    double p1y = pts[1][1];
    double maxabsy = std::max(std::abs(p0y), std::abs(p1y));
    double scaled_eps = std::numeric_limits<double>::epsilon() * maxabsy;
    if (std::abs(p0y - p1y) <= scaled_eps) {
        return std::nullopt;
    }
    t = static_cast<double>(y - p0y) / (p1y - p0y);
    if (t < 0.0 || t > 1.0) {
        return std::nullopt;
    } else {
        return pts[0] + t * (pts[1] - pts[0]);
    }
}

std::optional<spt::vec2d> segm_horizline_intersection(
    const std::array<spt::vec2d, 2> pts, double y
    ) {
    double t;
    return segm_horizline_intersection(pts, y, t);
}

struct EdgeInter {
    spt::vec2d p;
    Edge edge;
};

std::optional<std::array<EdgeInter, 2>> triangle_horizline_intersections(
    const Mesh& mesh, const Face& face, double y, std::array<double, 2>& ts
    ) {
    std::array<spt::vec2d, 2> pres;
    std::array<Edge, 2> eres;
    std::size_t res_i = 0;
    std::size_t prev = face.verts[2];
    auto prevpos = spt::vec2d(mesh.verts[prev]);
    for (int i = 0; i < 3; ++i) {
        std::size_t cur = face.verts[i];
        auto curpos = spt::vec2d(mesh.verts[cur]);
        double t;
        auto ointer = segm_horizline_intersection({prevpos, curpos}, y, t);
        if (ointer.has_value()) {
            if (curpos[1] == y) {
                std::size_t next = face.verts[0];
                if (i < 2) {
                    next = face.verts[i + 1];
                }
                auto nextpos = spt::vec2d(mesh.verts[next]);
                if ((curpos[1] - prevpos[1]) * (nextpos[1] - curpos[1]) < 0) {
                    pres[res_i] = ointer.value();
                    eres[res_i] = { prev, cur };
                    ts[res_i] = t;
                    if (++res_i == 2) {
                        break;
                    }
                }
            } else {
                pres[res_i] = ointer.value();
                eres[res_i] = { prev, cur };
                ts[res_i] = t;
                if (++res_i == 2) {
                    break;
                }
            }
        }
        prev = cur;
        prevpos = curpos;
    }
    if (res_i < 2) {
        return std::nullopt;
    }

    if (pres[0][0] > pres[1][0]) {
        std::swap(pres[0], pres[1]);
        std::swap(eres[0], eres[1]);
        std::swap(ts[0], ts[1]);
    }
    return std::array{
        EdgeInter{ pres[0], eres[0] },
        EdgeInter{ pres[1], eres[1] }
    };
}

std::optional<std::array<EdgeInter, 2>> triangle_horizline_intersections(
    const Mesh& mesh, const Face& face, double y
    ) {
    std::array<double, 2> ts;
    return triangle_horizline_intersections(mesh, face, y, ts);
}

template <typename T>
T interpolate(const T& v0, const T& v1, double t) {
    return v0 + t * (v1 - v0);
}

template <typename T>
T interpolate_along_edge(const std::vector<T>& vals, Edge edge, double t) {
    T v0 = vals[edge[0]];
    T v1 = vals[edge[1]];
    return interpolate(v0, v1, t);
}

std::array<int, 2> image_x_draw_range(
    const Viewport& viewport, ViewTransformer vtran,
    const std::array<spt::vec2d, 2>& pts) {

    return {
        std::max(static_cast<int>(vtran.to_viewport(pts[0])[0]), 0),
        std::min(static_cast<int>(vtran.to_viewport(pts[1])[0]) + 1, viewport.width())
    };
}
