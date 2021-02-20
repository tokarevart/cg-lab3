#pragma once

#include "sptops.h"

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
