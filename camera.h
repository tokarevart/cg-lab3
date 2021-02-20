#pragma once

#include "vec.h"
#include "mat.h"

struct Camera {
    spt::vec3d pos;
    spt::mat3d orient;
};
