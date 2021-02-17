#include "mainwindow.h"

#include "sptalgs.h"
#include <QVector2D>
#include <QGraphicsPixmapItem>
#include <algorithm>

#include "ui_mainwindow.h"

int det(QPoint col0, QPoint col1) {
    return col0.x() * col1.y() - col0.y() * col1.x();
}

std::optional<QPointF> segm_line_intersection(QLine s, QLine l) {
    QPoint p0 = s.p1();
    QPoint p1 = s.p2();
    QPoint q0 = l.p1();
    QPoint q1 = l.p2();
    QPoint p1_p0 = p1 - p0;
    QPoint q1_q0 = q1 - q0;
    int d = det(q1_q0, -p1_p0);
    if (d == 0) {
        return std::nullopt;
    }

    QPoint p0_q0 = p0 - q0;
    int d1 = det(q1_q0, p0_q0);
    if (d * d1 < 0 || std::abs(d1) > std::abs(d)) {
        return std::nullopt;
    }

    double t = static_cast<double>(d1) / d;
    return QPointF(p0) + t * QPointF(p1_p0);
}

QLine horiz_line(int y) {
    return {QPoint(0, y), QPoint(1, y)};
}

std::optional<QPointF> segm_horizline_intersection(QLine s, int y) {
    int p0y = s.p1().y();
    int p1y = s.p2().y();
    if (p0y == p1y) {
        return std::nullopt;
    }
    double t = static_cast<double>(y - p0y) / (p1y - p0y);
    if (t < 0.0 || t > 1) {
        return std::nullopt;
    } else {
        return QPointF(s.p1()) + t * QPointF(s.p2() - s.p1());
    }
}

QList<double> polygon_horiz_intersections(const QList<QLine> &edges, int y) {
//    auto horizlile = horiz_line(y);
    QList<double> res;
    res.reserve(2);
    auto prev_edge = edges.last();
    for (int i = 0; i < edges.size(); ++i) {
        auto cur_edge = edges[i];
//        auto ointer = segm_line_intersection(cur_edge, horizlile);
        auto ointer = segm_horizline_intersection(cur_edge, y);
        if (ointer.has_value()) {
            if (cur_edge.p1().y() == y) {
                auto p0 = prev_edge.p1();
                auto p1 = cur_edge.p1();
                auto p2 = cur_edge.p2();
                if ((p1.y() - p0.y()) * (p2.y() - p1.y()) <= 0) {
                    res.push_back(ointer.value().x());
                }
            } else {
                res.push_back(ointer.value().x());
            }
        }
        prev_edge = cur_edge;
    }
    std::sort(res.begin(), res.end());
    return res;
}

void draw_polygon(
    QImage &image, const QPolygon &poly, std::function<QRgb()> rgb) {

    QList<QLine> edges;
    edges.reserve(poly.size());
    edges.push_back(QLine(poly.last(), poly.first()));
    for (int i = 1; i < poly.size(); ++i) {
        edges.push_back(QLine(poly[i - 1], poly[i]));
    }

    QRgb *bits = reinterpret_cast<QRgb *>(image.bits());
    QRect brect = poly.boundingRect();
    for (int y = brect.top(); y < brect.bottom(); ++y) {
        auto xinters = polygon_horiz_intersections(edges, y);
        for (int i = 0; i < xinters.size(); i += 2) {
            int xbeg = std::max(static_cast<int>(xinters[i]), 0);
            int xend = std::min(
                static_cast<int>(xinters[i + 1]) + 1,
                image.width()
            );
            int start = y * image.width();
            for (int x = xbeg; x < xend; ++x) {
                bits[start + x] = rgb();
            }
        }
    }
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

struct IllumParams {
    double ia, ka, il, kd;
};

class SimpleIllum {
  public:
    double compute_intensity(spt::vec3d light, spt::vec3d normal) const {
        return m_diff_light +
               m_diff_refl_koef * std::max(0.0, -spt::dot(normal, light));
    }

    void set_params(IllumParams illum) {
        m_illum = illum;
        update_diff_light();
        update_diff_refl_koef();
    }

    void set_ia(double ia) {
        m_illum.ia = ia;
        update_diff_light();
    }

    void set_ka(double ka) {
        m_illum.ka = ka;
        update_diff_light();
    }

    void set_il(double il) {
        m_illum.il = il;
        update_diff_refl_koef();
    }

    void set_kd(double kd) {
        m_illum.kd = kd;
        update_diff_refl_koef();
    }

    SimpleIllum() {}

    SimpleIllum(IllumParams illum) : m_illum(illum) {
        update_diff_light();
        update_diff_refl_koef();
    }

  private:
    IllumParams m_illum;
    double m_diff_light;
    double m_diff_refl_koef;

    void update_diff_light() {
        m_diff_light = m_illum.ia * m_illum.ka;
    }

    void update_diff_refl_koef() {
        m_diff_refl_koef = m_illum.il * m_illum.kd;
    }
};

struct PointLight {
    spt::vec3d pos;
    SimpleIllum illum;

    double compute_intensity(spt::vec3d at, spt::vec3d normal) const {
        return illum.compute_intensity((at - this->pos).normalize(), normal);
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

struct Face {
    std::array<std::size_t, 3> verts;

    std::array<std::size_t, 2> edge(std::size_t i) const {
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

    bool contains_any_vert(const std::vector<std::size_t>& verts_ids) const {
        for (std::size_t id : verts_ids) {
            if (contains_vert(id)) {
                return true;
            }
        }
        return false;
    }
};

struct Mesh {
    std::vector<Vert> verts;
    std::vector<Face> faces;
    std::vector<spt::vec3d> normals; // vertex normals

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
        std::vector<std::size_t> counts(verts.size(), 0);
        for (auto& face : faces) {
            auto normal = face_normal(face);
            for (std::size_t vid : face.verts) {
                normals[vid] += normal;
                ++counts[vid]; // maybe just normalize at the end instead of averaging?
            }
        }
        for (std::size_t i = 0; i < normals.size(); ++i) {
            double inv_count = 1.0 / counts[i];
            normals[i] *= inv_count;
        }
    }

    Mesh() {}
    Mesh(const std::vector<Vert>& verts, const std::vector<Face>& faces)
        : verts(verts), faces(faces), normals(verts.size(), spt::vec3d()) {
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
    for (std::size_t i = 0; i < mesh.faces.size(); ++i) {
        auto face = mesh.faces[i];
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
    for (std::size_t i = 0; i < mesh.faces.size(); ++i) {
        auto face = mesh.faces[i];
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

    for (std::size_t i = 0; i < mesh.faces.size(); ++i) {
        auto face = mesh.faces[i];
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

    void backface_cull() {
        std::vector<Face> filtered;
        for (auto& face : mesh.faces) {
            if (mesh.face_normal(face)[2] > 0) {
                filtered.push_back(face);
            }
        }
        mesh.faces = std::move(filtered);
    }

    // does not view frustum
    void occlusion_cull() {
        std::vector<std::size_t> invis_verts;
        spt::vec3d backward({0.0, 0.0, 1.0});
        for (std::size_t i = 0; i < mesh.verts.size(); ++i) {
            if (vertray_intersect_mesh(i, backward, mesh)) {
                invis_verts.push_back(i);
            }
        }
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
        for (auto& face : mesh.faces) {
            if (face_visible(face)) {
                vis_faces.push_back(face);
            }
        }

        mesh.verts = std::move(vis_verts);
        mesh.faces = std::move(vis_faces);
    }

//    void view_frustum(ViewTransformer vtran) { }

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
    for (int ay = 0; ay < viewport.height(); ++ay) {
        for (int ax = 0; ax < viewport.width(); ++ax) {
            spt::vec2i absolute({ax, ay});
            auto origin = vtran.to_localworld(absolute);
            auto dir = spt::vec3d({0.0, 0.0, -1.0});
            auto ointer = ray_mesh_nearest_intersection(origin, dir, locscene.mesh);
            if (ointer.has_value()) {
                auto inter = ointer.value();
                double intensity = locscene.light.compute_intensity(inter.at, inter.normal);
                viewport.draw_point_grayscale({ax, ay}, intensity);
            }
        }
    }
}

void render_simple(Viewport& viewport, ViewTransformer vtran, const Scene& scene) {
    LocalScene locscene(scene);
    auto& mesh = locscene.mesh;
    auto& light = locscene.light;
    std::vector<double> intesities;
    intesities.reserve(mesh.faces.size());
    for (auto& face : mesh.faces) {
        auto center = mesh.face_center(face);
        auto normal = mesh.face_normal(face);
        double intensity = light.compute_intensity(center, normal);
        intesities.push_back(intensity);
    }

    for (int ay = 0; ay < viewport.height(); ++ay) {
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
    IllumParams illpars;
    illpars.ia = 1.0;
    illpars.il = 10.0;
    illpars.ka = 0.15;
    illpars.kd = illpars.ka;
    PointLight light(lightpos, SimpleIllum(illpars));

    std::vector<Vert> verts{
        std::array{-1.0, -1.0, 1.0},
        std::array{1.0, -1.0, 1.0},
        std::array{1.0, 1.0, 1.0},
        std::array{-1.0, 1.0, 1.0},
        std::array{0.0, 0.0, 1.0}
    };
    std::vector<Face> faces{
        {0, 1, 4},
        {1, 2, 4},
        {2, 3, 4},
        {3, 0, 4}
    };
    Mesh mesh(verts, faces);

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
}
