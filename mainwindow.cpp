#include "mainwindow.h"

#include "sptalgs.h"
#include <QVector2D>
#include <QGraphicsPixmapItem>
#include <algorithm>
#include <memory>

#include "ui_mainwindow.h"

template <typename T>
QList<typename T::value_type> qlist_from_vec(T vec) {
    return QList(vec.x.begin(), vec.x.end());
}

struct Viewport {
    QImage* image;

    int width() const {
        return image->width();
    }

    int height() const {
        return image->height();
    }

    // 0 <= intensity <= 1
    void draw_point_grayscale(QPoint p, double intensity) {
        int c = std::min(static_cast<int>(intensity * 255.0), 255);
        image->setPixel(p, qRgb(c, c, c));
    }

    Viewport() {}
    Viewport(QImage& im) : image(&im) {}
};

struct ViewTransformer {
    const Viewport* viewport;
    double pixelsize;
    spt::vec2d image_center;

    spt::vec3d to_localworld(spt::vec2i pos) const {
        spt::vec2d fpos(pos);
        spt::vec2d res2d = (fpos - image_center) / pixelsize;
        return spt::vec3d({res2d[0], -res2d[1], 0.0});
    }

    spt::vec2i to_viewport(spt::vec2d pos) const {
        pos[1] = -pos[1];
        auto scaled = pos * pixelsize;
        return spt::vec2i(scaled + image_center);
    }

    spt::vec2i proj_on_viewport(spt::vec3d pos) const {
        return to_viewport(spt::vec2d({pos[0], pos[1]}));
    }

    ViewTransformer(const Viewport& view, double pixelsize)
        : viewport(&view), pixelsize(pixelsize) {
        image_center = spt::vec2d({viewport->width() * 0.5, viewport->height() * 0.5});
    }
};

struct AmbientLight {
    double ia, ka;

    double intensity() const {
        return ia * ka;
    }

    double max_intensity() const {
        return intensity();
    }
};

struct DiffuseLight {
    double il, kd;

    double intensity(spt::vec3d light, spt::vec3d normal) const {
        return il * kd * std::max(0.0, -spt::dot(normal, light));
    }

    double max_intensity() const {
        return il * kd;
    }
};

struct PhongLight {
    double il, ks, n;

    double intensity(spt::vec3d light, spt::vec3d normal, spt::vec3d camdir) const {
        auto proj = spt::dot(light, normal) * normal;
        auto refl = light - proj - proj;
        return il * ks * std::max(0.0, std::pow(-spt::dot(refl, camdir), n));
    }

    double local_intensity(spt::vec3d light, spt::vec3d normal) const {
        auto projz = spt::dot(light, normal) * normal[2];
        auto reflz = light[2] - projz - projz;
        return il * ks * std::max(0.0, std::pow(reflz, n));
    }

    double max_intensity() const {
        return il * ks;
    }
};

struct SimpleIllum {
    AmbientLight ambient;
    DiffuseLight diffuse;
    PhongLight phong;

    double intensity(spt::vec3d light, spt::vec3d normal, spt::vec3d camdir) const {
        return ambient.intensity()
               + diffuse.intensity(light, normal)
               + phong.intensity(light, normal, camdir);
    }

    double intensity_normalized(spt::vec3d light, spt::vec3d normal, spt::vec3d camdir) const {
        return intensity(light, normal, camdir) / max_intensity();
    }

    double local_intensity(spt::vec3d light, spt::vec3d normal) const {
        return ambient.intensity()
               + diffuse.intensity(light, normal)
               + phong.local_intensity(light, normal);
    }

    double local_intensity_normalized(spt::vec3d light, spt::vec3d normal) const {
        return local_intensity(light, normal) / max_intensity();
    }

    double max_intensity() const {
        return ambient.max_intensity()
               + diffuse.max_intensity()
               + phong.max_intensity();
    }
};

struct PointLight {
    spt::vec3d pos;
    SimpleIllum illum;

    double intensity(spt::vec3d at, spt::vec3d normal, spt::vec3d camdir) const {
        return illum.intensity((at - this->pos).normalize(), normal, camdir);
    }

    double intensity_normalized(spt::vec3d at, spt::vec3d normal, spt::vec3d camdir) const {
        return illum.intensity_normalized((at - this->pos).normalize(), normal, camdir);
    }

    double local_intensity(spt::vec3d at, spt::vec3d normal) const {
        return illum.local_intensity((at - this->pos).normalize(), normal);
    }

    double local_intensity_normalized(spt::vec3d at, spt::vec3d normal) const {
        return illum.local_intensity_normalized((at - this->pos).normalize(), normal);
    }

    PointLight() {}

    PointLight(spt::vec3d pos, SimpleIllum illum)
        : pos(pos), illum(illum) {}
};

struct Camera {
    spt::vec3d pos;
    spt::mat3d orient;
};

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

struct LocalScene {
    Camera cam;
    PointLight light;
    Mesh mesh;

    virtual double local_intensity(spt::vec3d at, spt::vec3d normal) const {
        return light.local_intensity(at, normal);
    }

    virtual double local_intensity_normalized(spt::vec3d at, spt::vec3d normal) const {
        return light.local_intensity_normalized(at, normal);
    }

    void translate(spt::vec3d move) {
        light.pos += move;
        for (auto& vert : mesh.verts) {
            vert += move;
        }
    }

    void rotate(spt::mat3d rot) {
        light.pos = spt::dot(rot, light.pos);
        for (auto& vert : mesh.verts) {
            vert = spt::dot(rot, vert);
        }
        for (auto& normal : mesh.normals) {
            normal = spt::dot(rot, normal);
        }
    }

    void linear_transform(spt::mat3d a, spt::vec3d b) {
        light.pos = spt::dot(a, light.pos) + b;
        for (auto& vert : mesh.verts) {
            vert = spt::dot(a, vert) + b;
        }
        for (auto& normal : mesh.normals) {
            normal = spt::dot(a, normal);
        }
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

    void backface_cull() {
        std::vector<Face> filtered;
        for (auto& face : mesh.surface) {
            if (mesh.face_normal(face)[2] > 0) {
                filtered.push_back(face);
            }
        }
        mesh.surface = std::move(filtered);
    }

    void occlusion_cull(ViewTransformer vtran) {
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
    }
};

struct GhostScene : public LocalScene {
    double local_intensity(spt::vec3d at, spt::vec3d normal) const override {
        return light.intensity(at, normal, cam.orient[2]);
    }

    double local_intensity_normalized(spt::vec3d at, spt::vec3d normal) const override {
        return light.intensity_normalized(at, normal, -cam.orient[2]);
    }
};

struct GhostSceneBuilder {
    std::unique_ptr<GhostScene> locscene;
    spt::mat3d a;
    spt::vec3d b;

    std::unique_ptr<GhostScene> build() {
        locscene->linear_transform(a, b);
        return std::move(locscene);
    }
};

struct Scene {
    Camera cam;
    PointLight light;
    Mesh mesh;

    template <typename TScene = LocalScene>
    std::unique_ptr<TScene> local(const Camera* loccam = nullptr) const {
        if (!loccam) {
            loccam = &cam;
        }
        auto loc = std::make_unique<TScene>();
        loc->mesh = mesh;
        loc->light = light;
        loc->cam = *loccam;
        auto rot = loccam->orient.transposed().inversed();
        loc->translate(-loc->cam.pos);
        loc->rotate(rot);
        return std::move(loc);
    }

    GhostSceneBuilder ghost_scene_builder(Camera debcam) const {
        auto loc = local<GhostScene>();
        auto finalrot_tr = spt::dot(loc->cam.orient, debcam.orient.inversed());
        auto finalrot = finalrot_tr.transpose();

        auto cl = loc->cam.pos;
        auto cg = debcam.pos;
        auto l = loc->cam.orient;
        auto g = debcam.orient;
        auto inv_g = g.inversed();
        auto a = spt::dot(l, inv_g).transpose();
        auto int_g_tr = inv_g.transpose();
        auto b = spt::dot(int_g_tr, cl - cg);

        GhostSceneBuilder builder;
        builder.a = a;
        builder.b = b;
        builder.locscene = std::move(loc);
        auto camori = builder.locscene->cam.orient;
        builder.locscene->cam.orient = spt::dot(int_g_tr, camori);
        return builder;
    }
};

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

void render_full_local(Viewport& viewport, ViewTransformer vtran, const LocalScene& loc) {
    auto& mesh = loc.mesh;
    auto minmax_y = min_max_image_y(viewport, vtran, mesh);
    for (int ay = minmax_y[0]; ay < minmax_y[1]; ++ay) {
        for (int ax = 0; ax < viewport.width(); ++ax) {
            spt::vec2i absolute({ax, ay});
            auto origin = vtran.to_localworld(absolute);
            auto dir = spt::vec3d({0.0, 0.0, -1.0});
            auto ointer = ray_mesh_nearest_intersection(origin, dir, mesh);
            if (ointer.has_value()) {
                auto inter = ointer.value();
                double intensity = loc.local_intensity_normalized(inter.at, inter.normal);
                viewport.draw_point_grayscale({ax, ay}, intensity);
            }
        }
    }
}

void render_full(Viewport& viewport, ViewTransformer vtran, const Scene& scene) {
    auto loc = scene.local();
    loc->backface_cull();
    loc->occlusion_cull(vtran);
    render_full_local(viewport, vtran, *loc);
}

void render_simple_local(Viewport& viewport, ViewTransformer vtran, const LocalScene& loc) {
    auto& mesh = loc.mesh;
    std::vector<double> intesities;
    intesities.reserve(mesh.surface.size());
    for (auto& face : mesh.surface) {
        auto center = mesh.face_center(face);
        auto normal = mesh.face_normal(face);
        double intensity = loc.local_intensity_normalized(center, normal);
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

void render_simple(Viewport& viewport, ViewTransformer vtran, const Scene& scene) {
    auto loc = scene.local();
    loc->backface_cull();
    loc->occlusion_cull(vtran);
    render_simple_local(viewport, vtran, *loc);
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

void render_gouraud_local(Viewport& viewport, ViewTransformer vtran, const LocalScene& loc) {
    auto& mesh = loc.mesh;
    std::vector<double> verts_intens;
    verts_intens.reserve(mesh.verts.size());
    for (std::size_t i = 0; i < mesh.verts.size(); ++i) {
        double inten = loc.local_intensity_normalized(mesh.verts[i], mesh.normals[i]);
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

void render_gouraud(Viewport& viewport, ViewTransformer vtran, const Scene& scene) {
    auto loc = scene.local();
    loc->backface_cull();
    loc->occlusion_cull(vtran);
    render_gouraud_local(viewport, vtran, *loc);
}

void render_phong_local(Viewport& viewport, ViewTransformer vtran, const LocalScene& loc) {
    auto& mesh = loc.mesh;

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
                double intensity = loc.local_intensity_normalized(p, normal);
                viewport.draw_point_grayscale({rel_ax + xrange[0], ay}, intensity);
            }
        }
    }
}

void render_phong(Viewport& viewport, ViewTransformer vtran, const Scene& scene) {
    auto loc = scene.local();
    loc->backface_cull();
    loc->occlusion_cull(vtran);
    render_phong_local(viewport, vtran, *loc);
}

spt::mat3d direct_z_along_vec(spt::mat3d orient, spt::vec3d v) {
    auto y = spt::cross(v, orient[0]).normalize();
    auto x = spt::cross(y, v).normalize();
    v.normalize();
    return spt::mat3d(x, y, v);
}

Scene prepare_scene(Ui::MainWindow* ui) {
    Camera cam;
    cam.pos = spt::vec3d({
        ui->campos_x_sbox->value(),
        ui->campos_y_sbox->value(),
        ui->campos_z_sbox->value()
    });
    cam.orient = spt::mat3d(direct_z_along_vec(
        spt::mat3d::identity(),
        -spt::vec3d({
            ui->camdir_x_sbox->value(),
            ui->camdir_y_sbox->value(),
            ui->camdir_z_sbox->value()
        })));

    spt::vec3d lightpos({
        ui->light_x_sbox->value(),
        ui->light_y_sbox->value(),
        ui->light_z_sbox->value()
    });
    double ia = 1.0;
    double ka = 0.15;
    double il = 10.0;
    double kd = ka;
    double ks = 0.8;
    double n = 5.0;
    AmbientLight ambient;
    ambient.ia = ia;
    ambient.ka = ka;
    DiffuseLight diffuse;
    diffuse.il = il;
    diffuse.kd = kd;
    PhongLight phong;
    phong.il = il;
    phong.ks = ks;
    phong.n = n;
    PointLight light(lightpos, SimpleIllum{ambient, diffuse, phong});

    std::vector<Vert> verts{
        std::array{-1.0, -1.0, 1.0},
        std::array{1.0, -1.0, 1.0},
        std::array{1.0, 1.0, 1.0},
        std::array{-1.0, 1.0, 1.0},
        std::array{0.0, 0.0, 1.0}
    };
    Surface surface{
        {0, 1, 4},
        {1, 2, 4},
        {2, 3, 4},
        {3, 0, 4}
    };
    Mesh mesh(verts, surface);

    Scene scene;
    scene.cam = cam;
    scene.light = light;
    scene.mesh = mesh;
    return scene;
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {
    ui->setupUi(this);
    for (int i = 0; i < 4; ++i) {
        gscenes[i] = new QGraphicsScene();
    }

    auto set_scene = [](QGraphicsView *gview, QGraphicsScene *scene) {
        gview->setScene(scene);
        gview->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        gview->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    };

    set_scene(ui->gview0, gscenes[0]);
    set_scene(ui->gview1, gscenes[1]);
    set_scene(ui->gview2, gscenes[2]);
    set_scene(ui->gview3, gscenes[3]);
}

MainWindow::~MainWindow() { delete ui; }

void MainWindow::on_render_button_clicked() {
    Scene scene = prepare_scene(ui);

    double pixelsize = 100.0;
    auto render_and_show = [&scene, pixelsize]
        (std::function<void(Viewport&, ViewTransformer, const LocalScene&)> render_fn,
         QGraphicsView* gview, QGraphicsScene *gscene) {

        QImage image(gview->size(), QImage::Format_RGB32);
        image.fill(Qt::white);
        Viewport viewport(image);
        ViewTransformer vtran(viewport, pixelsize);
        auto loc = scene.local<LocalScene>();
//        Camera ghcam;
//        ghcam.pos = spt::vec3d({0.0, 0.0, 2.0});
//        ghcam.orient = spt::mat3d::identity();
//        auto builder = scene.ghost_scene_builder(ghcam);
//        auto loc = builder.build();
        render_fn(viewport, vtran, *loc);

        gscene->clear();
        gscene->setSceneRect(gview->rect());
        gscene->addPixmap(QPixmap::fromImage(image))->setPos(0, 0);
    };

    render_and_show(render_full_local, ui->gview0, gscenes[0]);
    render_and_show(render_simple_local, ui->gview1, gscenes[1]);
    render_and_show(render_gouraud_local, ui->gview2, gscenes[2]);
    render_and_show(render_phong_local, ui->gview3, gscenes[3]);
}
