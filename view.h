#pragma once

#include <QGraphicsPixmapItem>
#include "vec.h"

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
