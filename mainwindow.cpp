#include "mainwindow.h"

#include "sptalgs.h"
#include <QVector2D>
#include <QGraphicsPixmapItem>
#include <algorithm>

#include "ui_mainwindow.h"

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
};

struct DiffuseLight {
    double il, kd;

    double intensity(spt::vec3d light, spt::vec3d normal) const {
        return il * kd * std::max(0.0, -spt::dot(normal, light));
    }
};

struct SimpleIllum {
    AmbientLight ambient;
    DiffuseLight diffuse;

    double intensity(spt::vec3d light, spt::vec3d normal) const {
        return ambient.intensity() + diffuse.intensity(normal, light);
    }
};

struct PointLight {
    spt::vec3d pos;
    SimpleIllum illum;

    double compute_intensity(spt::vec3d at, spt::vec3d normal) const {
        return illum.intensity((at - this->pos).normalize(), normal);
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

struct Scene {
    Camera cam;
    PointLight light;
    Mesh mesh;
};

struct LocalScene {
    PointLight light;
    Mesh mesh;

    void remove_invisible_verts(const std::vector<std::size_t>& invis_verts) {
        if (invis_verts.empty()) {
            return;
        }

        std::vector<Vert> vis_verts;
        std::vector<std::size_t> vis_verts_ids;
        vis_verts.reserve(mesh.verts.size() - invis_verts.size());
        std::size_t cur_ivert_i = 0;
        for (std::size_t i = 0; i < mesh.verts.size(); ++i) {
            if (i == invis_verts[cur_ivert_i]) {
                ++cur_ivert_i;
            } else {
                vis_verts.push_back(mesh.verts[i]);
                vis_verts_ids.push_back(i);
            }
        }

        auto face_visible = [&vis_verts_ids](const Face& face) {
            for (std::size_t vert_id : face.verts) {
                if (std::find(
                        vis_verts_ids.begin(),
                        vis_verts_ids.end(),
                        vert_id) != vis_verts_ids.end()) {
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

        mesh.verts = std::move(vis_verts);
        mesh.surface = std::move(vis_faces);
    }

    // does not remove dangling vertices
    void retain_faces(const std::vector<Face>& faces) {
        mesh.surface = faces;
        // ...
    }

    void backface_cull() {
        std::vector<Face> filtered;
        for (auto& face : mesh.surface) {
            if (mesh.face_normal(face)[2] > 0) {
                filtered.push_back(face);
            }
        }
        retain_faces(filtered);
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
        double upper_y = vtran.to_localworld(spt::vec2i({0, 0}))[1];
        double lower_y = vtran.to_localworld(spt::vec2i({0, vtran.viewport->height()}))[1];
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

        remove_invisible_verts(invis_verts);
    }

    LocalScene(Scene scene) : light(scene.light), mesh(scene.mesh) {
        auto campos = scene.cam.pos;
        spt::mat3d rot = scene.cam.orient.transpose().inversed();
        light.pos = spt::dot(rot, light.pos - campos);
        for (auto& vert : mesh.verts) {
            vert = spt::dot(rot, vert - campos);
        }
        for (auto& normal : mesh.normals) {
            normal = spt::dot(rot, normal);
        }
    }
};

void render_full(Viewport& viewport, ViewTransformer vtran, const Scene& scene) {
    LocalScene locscene(scene);
    locscene.backface_cull();
    locscene.occlusion_cull(vtran);
    auto& mesh = locscene.mesh;
    auto& light = locscene.light;
    int min_y = vtran.to_viewport(std::array{0.0, mesh.max_coor(1)})[1];
    int max_y = vtran.to_viewport(std::array{0.0, mesh.min_coor(1)})[1] + 1;
    for (int ay = min_y; ay < max_y; ++ay) {
        for (int ax = 0; ax < viewport.width(); ++ax) {
            spt::vec2i absolute({ax, ay});
            auto origin = vtran.to_localworld(absolute);
            auto dir = spt::vec3d({0.0, 0.0, -1.0});
            auto ointer = ray_mesh_nearest_intersection(origin, dir, mesh);
            if (ointer.has_value()) {
                auto inter = ointer.value();
                double intensity = light.compute_intensity(inter.at, inter.normal);
                viewport.draw_point_grayscale({ax, ay}, intensity);
            }
        }
    }
}

void render_simple(Viewport& viewport, ViewTransformer vtran, const Scene& scene) {
    LocalScene locscene(scene);
    locscene.backface_cull();
    locscene.occlusion_cull(vtran);
    auto& mesh = locscene.mesh;
    auto& light = locscene.light;
    std::vector<double> intesities;
    intesities.reserve(mesh.surface.size());
    for (auto& face : mesh.surface) {
        auto center = mesh.face_center(face);
        auto normal = mesh.face_normal(face);
        double intensity = light.compute_intensity(center, normal);
        intesities.push_back(intensity);
    }

    int min_y = vtran.to_viewport(std::array{0.0, mesh.max_coor(1)})[1];
    int max_y = vtran.to_viewport(std::array{0.0, mesh.min_coor(1)})[1] + 1;
    for (int ay = min_y; ay < max_y; ++ay) {
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

std::optional<spt::vec2d> segm_horizline_intersection(
    const std::array<spt::vec2d, 2> pts, double y
) {
    double p0y = pts[0][1];
    double p1y = pts[1][1];
    double maxabsy = std::max(std::abs(p0y), std::abs(p1y));
    double scaled_eps = std::numeric_limits<double>::epsilon() * maxabsy;
    if (std::abs(p0y - p1y) <= scaled_eps) {
        return std::nullopt;
    }
    double t = static_cast<double>(y - p0y) / (p1y - p0y);
    if (t < 0.0 || t > 1.0) {
        return std::nullopt;
    } else {
        return pts[0] + t * (pts[1] - pts[0]);
    }
}

struct EdgeInter {
    spt::vec2d p;
    Edge edge;
};

std::optional<std::array<EdgeInter, 2>> triangle_horizline_intersections(
    const Mesh& mesh, const Face& face, double y
) {
    std::array<spt::vec2d, 2> pres;
    std::array<Edge, 2> eres;
    std::size_t res_i = 0;
    std::size_t prev = face.verts[2];
    auto prevpos = spt::vec2d(mesh.verts[prev]);
    for (int i = 0; i < 3; ++i) {
        std::size_t cur = face.verts[i];
        auto curpos = spt::vec2d(mesh.verts[cur]);
        auto ointer = segm_horizline_intersection({prevpos, curpos}, y);
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
                    if (++res_i == 2) {
                        break;
                    }
                }
            } else {
                pres[res_i] = ointer.value();
                eres[res_i] = { prev, cur };
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
    }
    return std::array{
        EdgeInter{ pres[0], eres[0] },
        EdgeInter{ pres[1], eres[1] }
    };
}

void render_gouraud(Viewport& viewport, ViewTransformer vtran, const Scene& scene) {
    LocalScene locscene(scene);
    locscene.backface_cull();
    locscene.occlusion_cull(vtran);

    auto& mesh = locscene.mesh;
    auto& light = locscene.light;
    std::vector<double> verts_intens;
    verts_intens.reserve(mesh.verts.size());
    for (std::size_t i = 0; i < mesh.verts.size(); ++i) {
        double inten = light.compute_intensity(mesh.verts[i], mesh.normals[i]);
        verts_intens.push_back(inten);
    }

    int min_y = vtran.to_viewport(std::array{0.0, mesh.max_coor(1)})[1];
    int max_y = vtran.to_viewport(std::array{0.0, mesh.min_coor(1)})[1] + 1;
    for (int ay = min_y; ay < max_y; ++ay) {
        for (auto& face : mesh.surface) {
            double world_y = vtran.to_localworld(spt::vec2d({0.0, static_cast<double>(ay)}))[1];
            auto oxinters = triangle_horizline_intersections(mesh, face, world_y);
            if (!oxinters.has_value()) {
                continue;
            }
            auto xinters = oxinters.value();

            std::array<double, 2> intens;
            for (std::size_t i = 0; i < 2; ++i) {
                auto edge = xinters[i].edge;
                double inten_v0 = verts_intens[edge[0]];
                double inten_v1 = verts_intens[edge[1]];

                auto p = xinters[i].p;
                auto p0 = spt::vec2d(mesh.verts[edge[0]]);
                auto p1 = spt::vec2d(mesh.verts[edge[1]]);

                double t = std::sqrt((p - p0).magnitude2() / (p1 - p0).magnitude2());
                intens[i] = inten_v0 + t * (inten_v1 - inten_v0);
            }

            int xbeg = std::max(static_cast<int>(vtran.to_viewport(xinters[0].p)[0]), 0);
            int xend = std::min(static_cast<int>(vtran.to_viewport(xinters[1].p)[0]) + 1, viewport.width());
            double delta_inten = (intens[1] - intens[0]) / xend;
            for (int ax = xbeg; ax < xend; ++ax) {
                viewport.draw_point_grayscale({ax, ay}, intens[0] + ax * delta_inten);
            }
        }
    }
}

spt::mat3d direct_z_along_vec(spt::mat3d orient, spt::vec3d v) {
    auto y = spt::cross(v, orient[0]).normalize();
    auto x = spt::cross(y, v).normalize();
    v.normalize();
    return spt::mat3d(x, y, v);
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
    Camera cam;
    cam.pos = spt::vec3d({
        ui->campos_x_sbox->value(),
        ui->campos_y_sbox->value(),
        ui->campos_z_sbox->value()
    });
    cam.orient = spt::mat3d(direct_z_along_vec(
        spt::mat3d::identity(), -spt::vec3d({
            ui->camdir_x_sbox->value(),
            ui->camdir_y_sbox->value(),
            ui->camdir_z_sbox->value()
    })));

    spt::vec3d lightpos({
        ui->light_x_sbox->value(),
        ui->light_y_sbox->value(),
        ui->light_z_sbox->value()
    });
    AmbientLight ambient;
    ambient.ia = 1.0;
    ambient.ka = 0.15;
    DiffuseLight diffuse;
    diffuse.il = 10.0;
    diffuse.kd = ambient.ka;
    PointLight light(lightpos, SimpleIllum{ambient, diffuse});

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

    double pixelsize = 100.0;
    auto render_and_show = [&scene, pixelsize]
        (std::function<void(Viewport&, ViewTransformer, const Scene&)> render_fn,
         QGraphicsView* gview, QGraphicsScene *gscene
        ) {
        QImage image(gview->size(), QImage::Format_RGB32);
        image.fill(Qt::white);
        Viewport viewport(image);
        ViewTransformer vtran(viewport, pixelsize);
        render_fn(viewport, vtran, scene);

        gscene->clear();
        gscene->setSceneRect(gview->rect());
        gscene->addPixmap(QPixmap::fromImage(image))->setPos(0, 0);
    };

    render_and_show(render_full, ui->gview0, gscenes[0]);
    render_and_show(render_simple, ui->gview1, gscenes[1]);
    render_and_show(render_gouraud, ui->gview2, gscenes[2]);
}
