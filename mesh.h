#pragma once

#include <vector>
#include "sptops.h"

using Vert = spt::vec3d;
using Edge = std::array<std::size_t, 2>;

struct Face {
    std::array<std::size_t, 3> verts;

    Edge edge(std::size_t i) const {
        if (i < 2) {
            return {verts[i], verts[i + 1]};
        } else {
            return {verts[i], verts[0]};
        }
    }

    bool contains_vert(std::size_t vert_id) const {
        if (verts[0] == vert_id ||
            verts[1] == vert_id ||
            verts[2] == vert_id) {
            return true;
        } else {
            return false;
        }
    }
};

using Surface = std::vector<Face>;

struct Mesh {
    std::vector<Vert> verts;
    Surface surface;
    std::vector<spt::vec3d> normals; // vertex normals

    double min_coor(std::size_t c) const {
        double minc = std::numeric_limits<double>::max();
        for (auto& v : verts) {
            if (v[c] < minc) {
                minc = v[c];
            }
        }
        return minc;
    }

    double max_coor(std::size_t c) const {
        double maxc = std::numeric_limits<double>::lowest();
        for (auto& v : verts) {
            if (v[c] > maxc) {
                maxc = v[c];
            }
        }
        return maxc;
    }

    spt::vec3d face_normal(const Face& face) const {
        auto edgevec0 = verts[face.verts[1]] - verts[face.verts[0]];
        auto edgevec1 = verts[face.verts[2]] - verts[face.verts[1]];
        return spt::cross(edgevec0, edgevec1).normalize();
    }

    spt::vec3d face_center(const Face& face) const {
        return (verts[face.verts[0]]
                + verts[face.verts[1]]
                + verts[face.verts[2]]) / 3.0;
    }

    void update_normals() {
        normals = std::vector(verts.size(), spt::vec3d());
        for (auto& face : surface) {
            auto normal = face_normal(face);
            for (std::size_t vid : face.verts) {
                normals[vid] += normal;
            }
        }
        for (auto& normal : normals) {
            normal.normalize();
        }
    }

    Mesh() {}
    Mesh(const std::vector<Vert>& verts, const Surface& surface)
        : verts(verts), surface(surface), normals(verts.size(), spt::vec3d()) {
        update_normals();
    }
};
