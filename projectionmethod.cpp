#include <vector>

#include "lib/Eigen327/Dense"
#include "lib/Eigen327/Sparse"
#include "lib/Eigen327/Eigenvalues"

#include "matrixops.h"


double TOL = 1e-6;      // For calculating steady state


SpMat diff_mat(int n, double h, double c)
{
    // c: Neumann=1, Dirichlet=2, Dirichlet mid=3;

    SpMat ret(n,n);
    std::vector<T> tripletList;
    tripletList.reserve(3*n-2);

    tripletList.push_back(T(0, 0, c));
    tripletList.push_back(T(n-1, n-1, c));
    tripletList.push_back(T(0, 1, -1));
    tripletList.push_back(T(1, 0, -1));
    for (int i=1; i<n-1; i++)
    {
        tripletList.push_back(T(i, i+1, -1));
        tripletList.push_back(T(i+1 ,i, -1));
        tripletList.push_back(T(i, i, 2));
    }

    ret.setFromTriplets(tripletList.begin(), tripletList.end());
    return ret / (h*h);
}


std::vector<Mat> simulate(double Re, double tf, double uLid, int nx, int ny,
                          double dt, bool steadyState)
{
    int nt = (int)ceil(tf/dt);
    dt = tf / nt;
    double hx = 1./nx;
    double hy = 1./ny;

    // initial conditions
    Mat U = Mat::Zero(nx-1, ny);
    Mat V = Mat::Zero(nx, ny-1);
    Mat P;

    // boundary conditions
    Vec uN = uLid * Vec::Ones(nx+1);
    Vec uS = Vec::Zero(nx+1);
    Vec uE = Vec::Zero(ny);
    Vec uW = Vec::Zero(ny);
    Vec vN = Vec::Zero(nx);
    Vec vS = Vec::Zero(nx);
    Vec vE = Vec::Zero(ny+1);
    Vec vW = Vec::Zero(ny+1);

    Mat Ubc(nx-1,ny);
    Ubc << Mat::Zero(nx-1,ny-1), 2*uN.segment(1,nx-1);
    Ubc *= dt/(Re*hx*hx);

    SpMat Ap = kron(speye(ny),diff_mat(nx,hx,1)) +
               kron(diff_mat(ny,hy,1),speye(nx));
    Ap.coeffRef(0,0) *= 3./2;

    SpMat Au = speye((nx-1)*ny) +
               dt/Re * (kron(speye(ny), diff_mat(nx-1,hx,2)) +
                        kron(diff_mat(ny,hy,3), speye(nx-1)));

    SpMat Av = speye(nx*(ny-1)) +
               dt/Re * (kron(speye(ny-1), diff_mat(nx,hx,3)) +
                        kron(diff_mat(ny-1,hy,2), speye(nx)));

    SpMat As = kron(speye(ny-1),diff_mat(nx-1,hx,2)) +
               kron(diff_mat(ny-1,hy,2),speye(nx-1));

    Eigen::SimplicialLDLT<SpMat> Lp(Ap);
    Eigen::SimplicialLDLT<SpMat> Lu(Au);
    Eigen::SimplicialLDLT<SpMat> Lv(Av);
    Eigen::SimplicialLDLT<SpMat> Ls(As);

    int k = 1;

    while(true)
    {
        Mat Uprev = U;
        Mat Vprev = V;

        // calculate gamma
        double y1 = U.lpNorm<Eigen::Infinity>() / hx;
        double y2 = V.lpNorm<Eigen::Infinity>() / hy;
        double y0 = 1.2 * dt * std::max(y1,y2);
        double gamma = std::min(y0, 1.);

        // nonlinear terms
        Mat Ue0 = vconcat(uW.transpose(), U, uE.transpose());
        Mat Ue = hconcat(-Ue0.col(0), Ue0, 2*uN-Ue0.col(ny-1));
        Mat Ve0 = hconcat(vS, V, vN);
        Mat Ve = vconcat(-Ve0.row(0), Ve0, -Ve0.row(nx-1));

        Mat Ua = avg(Ue.transpose()).transpose();
        Mat Ud = diff(Ue.transpose()).transpose() / 2;
        Mat Va = avg(Ve);
        Mat Vd = diff(Ve) / 2;
        Mat tmp1 = Ua.cwiseProduct(Va) - gamma * Ua.cwiseAbs().cwiseProduct(Vd);
        Mat UVx = diff(tmp1) / hx;
        Mat tmp2 = Ua.cwiseProduct(Va) - gamma * Ud.cwiseProduct(Va.cwiseAbs());
        Mat UVy = diff(tmp2.transpose()).transpose() / hy;

        Ua = avg(Ue.block(0,1,nx+1,ny));
        Ud = diff(Ue.block(0,1,nx+1,ny)) / 2;
        Va = avg(Ve.block(1,0,nx,ny+1).transpose()).transpose();
        Vd = diff(Ve.block(1,0,nx,ny+1).transpose()).transpose() / 2;
        tmp1 = Ua.cwiseProduct(Ua) - gamma * Ua.cwiseAbs().cwiseProduct(Ud);
        Mat U2x = diff(tmp1) / hx;
        tmp2 = Va.cwiseProduct(Va) - gamma * Va.cwiseAbs().cwiseProduct(Vd);
        Mat V2y = diff(tmp2.transpose()).transpose() / hy;

        U -= dt * (UVy.block(1, 0, nx-1, ny) + U2x);
        V -= dt * (UVx.block(0, 1, nx, ny-1) + V2y);

        // implicit viscosity
        Vec rhs = reshape(U+Ubc,0,1);
        Vec u = Lu.solve(rhs);
        U = reshape(u,nx-1,ny);
        rhs = reshape(V,0,1);
        Vec v = Lv.solve(rhs);
        V = reshape(v,nx,ny-1);

        // pressure
        Mat tmp = diff(vconcat(uW.transpose(), U, uE.transpose())) / hx +
                  diff(hconcat(vS, V, vN).transpose()).transpose() / hy;
        rhs = reshape(tmp,0,1);
        Vec p = -Lp.solve(rhs);
        P = reshape(p,nx,ny);
        U -= diff(P) / hx;
        V -= diff(P.transpose()).transpose() / hy;

        k += 1;

        if (steadyState)
        {
            if ((U-Uprev).lpNorm<Eigen::Infinity>() < TOL &&
                (V-Vprev).lpNorm<Eigen::Infinity>() < TOL)
                break;
        }
        else if (k>nt)
            break;

        int N = nx/2;
        assert(!isnan(P(N,N)));
    }

    std::vector<Mat> ret(5);

    Mat tmp1 = vconcat(uW.transpose(), U, uE.transpose());
    Mat tmp2 = avg(tmp1.transpose()).transpose();
    ret[0] = hconcat(uS, tmp2, uN);

    tmp1 = hconcat(vS, V, vN);
    tmp2 = avg(tmp1);
    ret[1] = vconcat(vW.transpose(), tmp2, vE.transpose());

    ret[2] = P;

    Mat W0 = diff(U.transpose()).transpose()/hy - diff(V)/hx;
    Vec rhs = reshape(W0, 0, 1);
    Vec s = Ls.solve(rhs);
    Mat S = Mat::Zero(nx+1, ny+1);
    Mat W = Mat::Zero(nx+1, ny+1);
    S.block(1,1,nx-1,ny-1) = reshape(s,nx-1,ny-1);
    W.block(1,1,nx-1,ny-1) = W0;
    ret[3] = S;
    ret[4] = W;

    return ret;
}
