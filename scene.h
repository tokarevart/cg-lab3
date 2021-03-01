#pragma once

#include "point-light.h"
#include "mesh.h"
#include "view.h"
#include "camera.h"
#include "helpers.h"

struct LocalScene {
    PointLight light;
    Mesh mesh;

    double local_intensity(spt::vec3d at, spt::vec3d normal) const {
        return light.local_intensity(at, normal);
    }

    double local_intensity_normalized(spt::vec3d at, spt::vec3d normal) const {
        return light.local_intensity_normalized(at, normal);
    }

    LocalScene& translate(spt::vec3d move) {
        light.pos += move;
        for (auto& vert : mesh.verts) {
            vert += move;
        }
        return *this;
    }

    LocalScene& rotate(spt::mat3d rot) {
        light.pos = spt::dot(rot, light.pos);
        for (auto& vert : mesh.verts) {
            vert = spt::dot(rot, vert);
        }
        for (auto& normal : mesh.normals) {
            normal = spt::dot(rot, normal);
        }
        return *this;
    }

    LocalScene& linear_transform(spt::mat3d a, spt::vec3d b) {
        light.pos = spt::dot(a, light.pos) + b;
        for (auto& vert : mesh.verts) {
            vert = spt::dot(a, vert) + b;
        }
        for (auto& normal : mesh.normals) {
            normal = spt::dot(a, normal);
        }
        return *this;
    }

    LocalScene& backface_cull() {
        std::vector<Face> filtered;
        for (auto& face : mesh.surface) {
            if (mesh.face_normal(face)[2] > 0) {
                filtered.push_back(face);
            }
        }
        mesh.surface = std::move(filtered);
        return *this;
    }

    LocalScene& occlusion_cull(ViewTransformer vtran) {
        std::vector<std::size_t> vis_verts;

        spt::vec3d backward({0.0, 0.0, 1.0});
        double upper_x = vtran.to_world(spt::vec2i({vtran.viewport->width(), 0}))[0];
        double lower_x = vtran.to_world(spt::vec2i({0, 0}))[0];
        double upper_y = vtran.to_world(spt::vec2i({0, -1}))[1];
        double lower_y = vtran.to_world(spt::vec2i({0, vtran.viewport->height() - 1}))[1];
        for (std::size_t i = 0; i < mesh.verts.size(); ++i) {
            double x = mesh.verts[i][0];
            double y = mesh.verts[i][1];
            double z = mesh.verts[i][2];
            auto inside = [](double l, double c, double u) {
                return c > l && c < u;
            };
            if (z <= 0
                && inside(lower_x, x, upper_x)
                && inside(lower_y, y, upper_y)
                && !vertray_intersect_mesh(i, backward, mesh)) {
                vis_verts.push_back(i);
            }
        }

        std::vector<Face> vis_faces;
        auto bl = spt::vec2d(vtran.bottom_left());
        auto br = spt::vec2d(vtran.bottom_right());
        auto tl = spt::vec2d(vtran.top_left());
        auto tr = spt::vec2d(vtran.top_right());
        for (auto& face : mesh.surface) {
            auto prev = spt::vec2d(mesh.verts[face.verts.back()]);
            for (std::size_t i = 0; i < 3; ++i) {
                auto cur = spt::vec2d(mesh.verts[face.verts[i]]);
                auto bounds = std::array{
                    std::pair(bl, br),
                    std::pair(br, tr),
                    std::pair(tl, tr),
                    std::pair(bl, tl)
                };
                if (std::any_of(bounds.begin(), bounds.end(), [&prev, &cur](auto& p) {
                        return spt::segment_intersect_segment(p.first, p.second, prev, cur).has_value();
                    })) {
                    vis_faces.push_back(face);
                    break;
                }
                prev = cur;
            }
        }

        auto face_visible = [](auto vis_verts, const Face& face) {
            for (std::size_t vert_id : face.verts) {
                if (std::find(
                        vis_verts.begin(),
                        vis_verts.end(),
                        vert_id) == vis_verts.end()) {
                    return false;
                }
            }
            return true;
        };

        for (auto& face : mesh.surface) {
            if (face_visible(vis_verts, face)) {
                vis_faces.push_back(face);
            }
        }

        for (auto& face : vis_faces) {
            for (std::size_t v : face.verts) {
                if (std::find(vis_verts.begin(), vis_verts.end(), v) == vis_verts.end()) {
                    vis_verts.push_back(v);
                }
            }
        }

        auto find_pos = [](const std::vector<std::size_t>& cont, std::size_t val) {
            std::size_t i = 0;
            while (i < cont.size()) {
                if (cont[i] == val) {
                    break;
                }
                ++i;
            }
            return i;
        };

        for (auto& face : vis_faces) {
            for (std::size_t& v : face.verts) {
                v = find_pos(vis_verts, v);
            }
        }

        std::vector<Vert> vis_verts_poses;
        std::vector<spt::vec3d> vis_verts_normals;
        vis_verts_poses.reserve(vis_verts.size());
        vis_verts_normals.reserve(vis_verts.size());
        for (std::size_t v : vis_verts) {
            vis_verts_poses.push_back(mesh.verts[v]);
            vis_verts_normals.push_back(mesh.normals[v]);
        }

        mesh.verts = std::move(vis_verts_poses);
        mesh.surface = std::move(vis_faces);
        mesh.normals = std::move(vis_verts_normals);
        return *this;
    }

    LocalScene& transform_camera(Camera cam) {
        auto rot = spt::mat3d(cam.orient).transpose().inversed();
        translate(-cam.pos);
        rotate(rot);
        return *this;
    }
};

struct GhostScene : public LocalScene {
    Camera cam;

    double local_intensity(spt::vec3d at, spt::vec3d normal) const {
        return light.intensity(at, normal, cam.orient[2]);
    }

    double local_intensity_normalized(spt::vec3d at, spt::vec3d normal) const {
        return light.intensity_normalized(at, normal, -cam.orient[2]);
    }

    GhostScene(LocalScene loc, Camera actual_cam, Camera ghost_cam) {
        light = loc.light;
        mesh = std::move(loc.mesh);

        auto cl = actual_cam.pos;
        auto cg = ghost_cam.pos;
        auto l = actual_cam.orient;
        auto g = ghost_cam.orient;
        auto inv_g = g.inversed();
        auto a = spt::dot(l, inv_g).transpose();
        auto inv_g_tr = inv_g.transpose();
        auto b = spt::dot(inv_g_tr, cl - cg);

        cam = actual_cam;
        cam.orient = spt::dot(inv_g_tr, actual_cam.orient);
        cam.pos += cl - cg;
        linear_transform(a, b);
    }
};
