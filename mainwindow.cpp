#include "mainwindow.h"

#include <QVector2D>
#include <QGraphicsPixmapItem>
#include <algorithm>
#include <memory>
#include "sptalgs.h"
#include "render.h"
#include "scene.h"

#include "ui_mainwindow.h"

template <typename T>
QList<typename T::value_type> qlist_from_vec(T vec) {
    return QList(vec.x.begin(), vec.x.end());
}

spt::mat3d direct_z_along_vec(spt::mat3d orient, spt::vec3d v) {
    auto y = spt::cross(v, orient[0]).normalize();
    auto x = spt::cross(y, v).normalize();
    v.normalize();
    return spt::mat3d(x, y, v);
}

LocalScene prepare_scene(Ui::MainWindow* ui, Camera* pcam = nullptr) {
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

    LocalScene scene;
    scene.light = light;
    scene.mesh = mesh;
    scene.transform_camera(cam);
    if (pcam) {
        *pcam = cam;
    }
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

MainWindow::~MainWindow() {
    delete ui;
}

void MainWindow::on_render_button_clicked() {
    Camera actual_cam;
    LocalScene scene = prepare_scene(ui, &actual_cam);

    double pixelsize = 100.0;
    auto render_and_show = [&scene, &actual_cam, pixelsize]
        (auto render_fn, QGraphicsView* gview, QGraphicsScene *gscene) {

        QImage image(gview->size(), QImage::Format_RGB32);
        image.fill(Qt::white);
        Viewport viewport(image);
        ViewTransformer vtran(viewport, pixelsize);
        scene.backface_cull().occlusion_cull(vtran);
        Camera ghost_cam;
        ghost_cam.pos = spt::vec3d({0.0, 0.0, 2.0});
        ghost_cam.orient = spt::mat3d::identity();
        GhostScene ghost(scene, actual_cam, ghost_cam);
        ghost.backface_cull().occlusion_cull(vtran);
        render_fn(viewport, vtran, ghost);

        gscene->clear();
        gscene->setSceneRect(gview->rect());
        gscene->addPixmap(QPixmap::fromImage(image))->setPos(0, 0);
    };

    using TScene = GhostScene;
    render_and_show(render_full<TScene>, ui->gview0, gscenes[0]);
    render_and_show(render_simple<TScene>, ui->gview1, gscenes[1]);
    render_and_show(render_gouraud<TScene>, ui->gview2, gscenes[2]);
    render_and_show(render_phong<TScene>, ui->gview3, gscenes[3]);
}
