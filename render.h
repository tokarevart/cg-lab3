#pragma once

#include "view.h"
#include "helpers.h"

template <typename TScene>
void render_full(Viewport& viewport, ViewTransformer vtran, const TScene& scene) {
    auto& mesh = scene.mesh;
    auto minmax_y = min_max_image_y(viewport, vtran, mesh);
    for (int ay = minmax_y[0]; ay < minmax_y[1]; ++ay) {
        for (int ax = 0; ax < viewport.width(); ++ax) {
            spt::vec2i absolute({ax, ay});
            auto origin = vtran.to_localworld(absolute);
            auto dir = spt::vec3d({0.0, 0.0, -1.0});
            auto ointer = ray_mesh_nearest_intersection(origin, dir, mesh);
            if (ointer.has_value()) {
                auto inter = ointer.value();
                double intensity = scene.local_intensity_normalized(inter.at, inter.normal);
                viewport.draw_point_grayscale({ax, ay}, intensity);
            }
        }
    }
}

template <typename TScene>
void render_simple(Viewport& viewport, ViewTransformer vtran, const TScene& scene) {
    auto& mesh = scene.mesh;
    std::vector<double> intesities;
    intesities.reserve(mesh.surface.size());
    for (auto& face : mesh.surface) {
        auto center = mesh.face_center(face);
        auto normal = mesh.face_normal(face);
        double intensity = scene.local_intensity_normalized(center, normal);
        intesities.push_back(intensity);
    }

    auto minmax_y = min_max_image_y(viewport, vtran, mesh);
    for (int ay = minmax_y[0]; ay < minmax_y[1]; ++ay) {
        for (int ax = 0; ax < viewport.width(); ++ax) {
            spt::vec2i absolute({ax, ay});
            auto origin = vtran.to_localworld(absolute);
            auto dir = spt::vec3d({0.0, 0.0, -1.0});
            auto ointer = ray_mesh_nearest_intersection(origin, dir, mesh);
            if (ointer.has_value()) {
                auto face_id = ointer.value().face_id;
                viewport.draw_point_grayscale({ax, ay}, intesities[face_id]);
            }
        }
    }
}

template <typename TScene>
void render_gouraud(Viewport& viewport, ViewTransformer vtran, const TScene& scene) {
    auto& mesh = scene.mesh;
    std::vector<double> verts_intens;
    verts_intens.reserve(mesh.verts.size());
    for (std::size_t i = 0; i < mesh.verts.size(); ++i) {
        double inten = scene.local_intensity_normalized(mesh.verts[i], mesh.normals[i]);
        verts_intens.push_back(inten);
    }

    auto minmax_y = min_max_image_y(viewport, vtran, mesh);
    for (int ay = minmax_y[0]; ay < minmax_y[1]; ++ay) {
        for (auto& face : mesh.surface) {
            double world_y = vtran.to_localworld(spt::vec2d({0.0, static_cast<double>(ay)}))[1];
            std::array<double, 2> ts;
            auto oxinters = triangle_horizline_intersections(mesh, face, world_y, ts);
            if (!oxinters.has_value()) {
                continue;
            }
            auto xinters = oxinters.value();

            std::array<double, 2> intens;
            for (std::size_t i = 0; i < 2; ++i) {
                intens[i] = interpolate_along_edge(
                    verts_intens, xinters[i].edge, ts[i]);
            }

            auto xrange = image_x_draw_range(viewport, vtran, {xinters[0].p, xinters[1].p});
            int axlen = xrange[1] - xrange[0];
            double delta_inten = 0.0;
            if (axlen > 1) {
                delta_inten = (intens[1] - intens[0]) / (axlen - 1);
            }
            for (int rel_ax = 0; rel_ax < axlen; ++rel_ax) {
                viewport.draw_point_grayscale(
                    {rel_ax + xrange[0], ay},
                    intens[0] + rel_ax * delta_inten);
            }
        }
    }
}

template <typename TScene>
void render_phong(Viewport& viewport, ViewTransformer vtran, const TScene& scene) {
    auto& mesh = scene.mesh;
    auto minmax_y = min_max_image_y(viewport, vtran, mesh);
    for (int ay = minmax_y[0]; ay < minmax_y[1]; ++ay) {
        for (auto& face : mesh.surface) {
            double world_y = vtran.to_localworld(spt::vec2d({0.0, static_cast<double>(ay)}))[1];
            std::array<double, 2> ts;
            auto oxinters = triangle_horizline_intersections(mesh, face, world_y, ts);
            if (!oxinters.has_value()) {
                continue;
            }
            auto xinters = oxinters.value();

            std::array<spt::vec3d, 2> bnd_pts;
            std::array<spt::vec3d, 2> bnd_normals;
            for (std::size_t i = 0; i < 2; ++i) {
                auto& edge = xinters[i].edge;
                auto v0 = mesh.verts[edge[0]];
                auto v1 = mesh.verts[edge[1]];
                bnd_pts[i] = v0 + ts[i] * (v1 - v0);
                bnd_normals[i] = interpolate_along_edge(
                    mesh.normals, edge, ts[i]);
            }

            auto xrange = image_x_draw_range(viewport, vtran, {xinters[0].p, xinters[1].p});
            int axlen = xrange[1] - xrange[0];
            spt::vec3d delta_p;
            spt::vec3d delta_normal;
            if (axlen > 1) {
                delta_p = (bnd_pts[1] - bnd_pts[0]) / (axlen - 1);
                delta_normal = (bnd_normals[1] - bnd_normals[0]) / (axlen - 1);
            }

            for (int rel_ax = 0; rel_ax < axlen; ++rel_ax) {
                auto p = bnd_pts[0] + static_cast<double>(rel_ax) * delta_p;
                auto normal = (bnd_normals[0] + static_cast<double>(rel_ax) * delta_normal).normalize();
                double intensity = scene.local_intensity_normalized(p, normal);
                viewport.draw_point_grayscale({rel_ax + xrange[0], ay}, intensity);
            }
        }
    }
}
