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

    void remove_invisible_faces(const std::vector<std::size_t>& invis_verts) {
        if (invis_verts.empty()) {
            return;
        }

        std::vector<std::size_t> vis_verts;
        vis_verts.reserve(mesh.verts.size() - invis_verts.size());
        std::size_t cur_ivert_i = 0;
        for (std::size_t i = 0; i < mesh.verts.size(); ++i) {
            if (cur_ivert_i < invis_verts.size()
                && i == invis_verts[cur_ivert_i]) {
                ++cur_ivert_i;
            } else {
                vis_verts.push_back(i);
            }
        }

        auto face_visible = [&vis_verts](const Face& face) {
            for (std::size_t vert_id : face.verts) {
                if (std::find(
                        vis_verts.begin(),
                        vis_verts.end(),
                        vert_id) != vis_verts.end()) {
                    return true;
                }
            }
            return false;
        };

        std::vector<Face> vis_faces;
        for (auto& face : mesh.surface) {
            if (face_visible(face)) {
                vis_faces.push_back(face);
            }
        }

        mesh.surface = std::move(vis_faces);
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
        std::vector<std::size_t> invis_verts;

        spt::vec3d backward({0.0, 0.0, 1.0});
        for (std::size_t i = 0; i < mesh.verts.size(); ++i) {
            if (vertray_intersect_mesh(i, backward, mesh)) {
                invis_verts.push_back(i);
            }
        }

        double upper_x = vtran.to_localworld(spt::vec2i({vtran.viewport->width(), 0}))[0];
        double lower_x = vtran.to_localworld(spt::vec2i({0, 0}))[0];
        double upper_y = vtran.to_localworld(spt::vec2i({0, -1}))[1];
        double lower_y = vtran.to_localworld(spt::vec2i({0, vtran.viewport->height() - 1}))[1];
        for (std::size_t i = 0; i < mesh.verts.size(); ++i) {
            double x = mesh.verts[i][0];
            double y = mesh.verts[i][1];
            double z = mesh.verts[i][2];
            auto inside = [](double l, double c, double u) {
                return c > l && c < u;
            };
            if ( z > 0 ||
                !inside(lower_x, x, upper_x) ||
                !inside(lower_y, y, upper_y)) {
                invis_verts.push_back(i);
            }
        }

        std::sort(invis_verts.begin(), invis_verts.end());
        remove_invisible_faces(invis_verts);
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
