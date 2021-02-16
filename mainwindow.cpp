#include "mainwindow.h"

#include "sptalgs.h"
#include <QVector2D>
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
            int xbeg = std::max(static_cast<int>(xinters[i] + 0.5), 0);
            int xend = std::min(
                static_cast<int>(xinters[i + 1] + 1.5),
                image.width()
            );
            int start = y * image.width();
            for (int x = xbeg; x < xend; ++x) {
                bits[start + x] = rgb();
            }
        }
    }
}

// 0 <= intens <= 1
void draw_point(QImage &image, QPoint p, double intens) {
    int c = static_cast<int>(intens * 255.0);
    image.setPixel(p, qRgb(c, c, c));
}

struct IllumParams {
    double ia, ka, il, kd;
};

class SimpleIllum {
  public:
    double compute_intensity(spt::vec3d light, spt::vec3d normal) const {
        return m_diff_light +
               m_diff_refl_koef * std::abs(-spt::dot(normal, light));
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
        return illum.compute_intensity(at - this->pos, normal);
    }

    PointLight() {}

    PointLight(spt::vec3d pos, SimpleIllum illum)
        : pos(pos), illum(illum) {}
};

struct Camera {
    spt::vec3d pos;
    spt::mat3d orient;
    double scale;
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

    void update_normals() {
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

struct Scene {
    Camera cam;
    PointLight light;
    Mesh mesh;
};

struct LocalScene {
    PointLight light;
    Mesh mesh;

    LocalScene(Scene scene) : light(scene.light), mesh(scene.mesh) {
        spt::mat3d rot = scene.cam.orient.transpose().inversed();
        light.pos = spt::dot(rot, light.pos);
        for (auto& vert : mesh.verts) {
            vert = spt::dot(rot, vert);
        }
        for (auto& normal : mesh.normals) {
            normal = spt::dot(rot, normal);
        }
    }
};

spt::vec2d pixel_to_actual(const QImage &image, spt::vec2i pixel, double delta) {
    spt::vec2d fpix(pixel);
    spt::vec2d center({image.width() * 0.5, image.height() * 0.5});
    spt::vec2d centered = spt::vec2d({fpix[0], fpix[1]}) - center;
    return centered * delta;
}

struct Inter {
    spt::vec3d at;
    spt::vec3d normal;
};

std::optional<Inter> ray_mesh_nearest_intersection(
    spt::vec3d origin, spt::vec3d dir, const Mesh& mesh) {

    spt::vec3d normal;
    spt::vec3d at;
    double max_z = std::numeric_limits<double>::min();
    bool no_inters = true;
    for (auto& face : mesh.faces) {
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
        }
    }

    if (no_inters) {
        return std::nullopt;
    } else {
        return Inter{at, normal};
    }
}

void render_full(QImage& image, const Scene& scene) {
    double delta = 1.0 / scene.cam.scale;
    LocalScene locscene(scene);
    for (int ay = 0; ay < image.height(); ++ay) {
        for (int ax = 0; ax < image.width(); ++ax) {
            spt::vec2i absolute({ax, ay});
            auto p_xy = pixel_to_actual(image, absolute, delta);
            auto origin = spt::vec3d({p_xy[0], p_xy[1], 0.0});
            auto dir = spt::vec3d({0.0, 0.0, -1.0});
            auto ointer = ray_mesh_nearest_intersection(origin, dir, locscene.mesh);
            if (ointer.has_value()) {
                auto inter = ointer.value();
                double intensity = locscene.light.compute_intensity(inter.at, inter.normal);
                int inten_c = std::min(static_cast<int>(intensity), 255);
                auto inten_rgb = qRgb(inten_c, inten_c, inten_c);
                image.setPixel(ax, ay, inten_rgb);
            }
        }
    }
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {
    ui->setupUi(this);
    for (int i = 0; i < 4; ++i) {
        scenes[i] = new QGraphicsScene();
    }

    auto set_scene = [](QGraphicsView *view, QGraphicsScene *scene) {
        view->setScene(scene);
        view->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        view->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    };

    set_scene(ui->gview0, scenes[0]);
    set_scene(ui->gview1, scenes[1]);
    set_scene(ui->gview2, scenes[2]);
    set_scene(ui->gview3, scenes[3]);
}

MainWindow::~MainWindow() { delete ui; }
