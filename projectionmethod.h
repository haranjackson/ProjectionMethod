#ifndef PROJECTIONMETHOD_H
#define PROJECTIONMETHOD_H

#include <vector>

#include "lib/Eigen327/Dense"

#include "matrixops.h"


Mat avg(Mat A);

std::vector<Mat> simulate(double Re, double tf, double uLid, int nx, int ny,
                          double dt, bool steadyState);


#endif // PROJECTIONMETHOD_H
