#include <iostream>
#include <vector>

#include "lib/Eigen327/Dense"
#include "lib/Eigen327/Sparse"


typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vec;
typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;


void print(Mat A)
{
    // Prints dense matrix A

    int m = A.rows();
    int n = A.cols();
    for (int i=0; i<m; i++)
        for (int j=0; j<n; j++)
            std::cout << i+1 << j+1 << A(i,j);
}


void print (SpMat A)
{
    // Prints sparse matrix A

    for (int k=0; k<A.outerSize(); ++k)
        for (SpMat::InnerIterator it(A,k); it; ++it)
            std::cout << it.row()+1 << it.col()+1 << it.value();
}


Mat vconcat(Mat A, Mat B, Mat C)
{
    // Concatenates A,B,C vertically

    int n = A.cols();
    assert(B.cols() == n);
    assert(C.cols() == n);
    int mA = A.rows();
    int mB = B.rows();
    int mC = C.rows();
    Mat ret(mA+mB+mC, n);
    ret << A,
           B,
           C;
    return ret;
}


Mat hconcat(Mat A, Mat B, Mat C)
{
    // Concatenates A,B,C horizontally

    int m = A.rows();
    assert(B.rows() == m);
    assert(C.rows() == m);
    int nA = A.cols();
    int nB = B.cols();
    int nC = C.cols();
    Mat ret(m, nA+nB+nC);
    ret << A, B, C;
    return ret;
}


Mat avg(Mat A)
{
    // If A is a row vector, returns row vector B with B(i) = (A(i)+A(i+1))/2
    // If A is not a row vector, B(i,j) = (A(i,j)+A(i+1,j))/2

    if (A.rows()==1)  A.transposeInPlace();
    int m = A.rows();
    int n = A.cols();
    Mat B = (A.block(0,0,m-1,n) + A.block(1,0,m-1,n)) / 2;
    if (n==1)  B.transposeInPlace();
    return B;
}


SpMat kron(SpMat A, SpMat B)
{
    // Returns the kronecker product of A and B

    int mA = A.rows();
    int nA = A.cols();
    int mB = B.rows();
    int nB = B.cols();
    int NA = A.nonZeros();
    int NB = B.nonZeros();

    SpMat ret(mA*mB,nA*nB);
    std::vector<T> tripletList;
    tripletList.reserve(NA*NB);

    for (int j=0; j<A.outerSize(); ++j)
        for (SpMat::InnerIterator itA(A,j); itA; ++itA)
            for (int k=0; k<B.outerSize(); ++k)
                for (SpMat::InnerIterator itB(B,k); itB; ++itB)
                {
                    int row = itA.row() * mB + itB.row();
                    int col = itA.col() * nB + itB.col();
                    double val = itA.value() * itB.value();
                    tripletList.push_back(T(row,col,val));
                }
    ret.setFromTriplets(tripletList.begin(), tripletList.end());
    return ret;
}


SpMat speye(int n)
{
    // Returns sparse identity matrix of size n

    SpMat ret(n,n);
    ret.setIdentity();
    return ret;
}


Mat diff(Mat X)
{
    // Calculates differences between adjacent elements of X along the first
    // array dimension whose size does not equal 1

    int m = X.rows();
    int n = X.cols();
    Mat ret;
    if (m==1)
    {
        ret.resize(1,n-1);
        for (int i=0; i<n-1; i++)
            ret(i) = X(i+1) - X(i);
    }
    else
    {
        ret.resize(m-1,n);
        for (int i=0; i<m-1; i++)
            for (int j=0; j<n; j++)
                ret(i,j) = X(i+1,j) - X(i,j);
    }
    return ret;
}


Mat reshape(Mat A, int m, int n)
{
    // Reshapes matrix A to an mxn matrix.
    // If m=0, m is determined from n. Similarly for n.

    int mA = A.rows();
    int nA = A.cols();
    int mnA = mA*nA;
    if (m==0)  m = mnA / n;
    if (n==0)  n = mnA / m;
    assert(mnA == m*n);
    Mat ret(m,n);
    for (int j=0; j<nA; j++)
        for (int i=0; i<mA; i++)
        {
            int k = j*mA+i;
            div_t divresult = div(k,m);
            int jj = divresult.quot;
            int ii = divresult.rem;
            ret(ii,jj) = A(i,j);
        }
    return ret;
}


double sect(std::vector<double> f, std::vector<double> h, int p1, int p2)
{
    return (h[p2]*f[p1] - h[p1]*f[p2]) / (h[p2]-h[p1]);
}


std::vector<std::vector<double> > conrec(Mat & Q, Vec x, Vec y,
                                         std::vector<double> z)
{
    // Given levels  z[0],z[1],...  conrec returns a vector of quintuples
    // (x1,y1,x2,y2,z[k])  where each (x1,y1)->(x2,y2) is a line segment of the
    // contour at level z[k]

    int nc = z.size();
    int jub = Q.rows()-1;
    int iub = Q.cols()-1;
    std::vector<double> h(5), xh(5), yh(5);
    std::vector<int> sh(5);
    std::vector<std::vector<double> > ret;

    int im[4] = {0, 1, 1, 0}, jm[4] = {0, 0, 1, 1};

    int castab[3][3][3] =
    {
        {
            {0, 0, 8}, {0, 2, 5}, {7, 6, 9}
        },
        {
            {0, 3, 4}, {1, 3, 1}, {4, 3, 0}
        },
        {
            {9, 6, 7}, {5, 2, 0}, {8, 0, 0}
        }
    };
    for (int j=jub-1; j>=0; j--)
    {
        for (int i=0; i<iub; i++)
        {
            double temp1, temp2;
            temp1 = std::min(Q(i,j), Q(i,j+1));
            temp2 = std::min(Q(i+1,j), Q(i+1,j+1));
            double dmin = std::min(temp1, temp2);
            temp1 = std::max(Q(i,j), Q(i,j+1));
            temp2 = std::max(Q(i+1,j), Q(i+1,j+1));
            double dmax = std::max(temp1, temp2);
            if (dmax>=z[0] && dmin<=z[nc-1])
            {
                for (int k=0; k<nc; k++)
                {
                    if (z[k]>=dmin && z[k]<=dmax)
                    {
                        for (int m=4; m>=0; m--)
                        {
                            if (m>0)
                            {
                                h[m] = Q(i+im[m-1], j+jm[m-1]) - z[k];
                                xh[m] = x[i+im[m-1]];
                                yh[m] = y[j+jm[m-1]];
                            }
                            else
                            {
                                h[0] = 0.25 * (h[1]+h[2]+h[3]+h[4]);
                                xh[0] = 0.5 * (x[i]+x[i+1]);
                                yh[0] = 0.5 * (y[j]+y[j+1]);
                            }
                            if (h[m] > 0.)
                                sh[m] = 1;
                            else if (h[m] < 0.)
                                sh[m] = -1;
                            else
                                sh[m] = 0;
                        }

// Note: at this stage the relative heights of the corners and the
// centre are in the h array, and the corresponding coordinates are
// in the xh and yh arrays. The centre of the box is indexed by 0
// and the 4 corners by 1 to 4 as shown below.
// Each triangle is then indexed by the parameter m, and the 3
// vertices of each triangle are indexed by parameters m1,m2,and
// m3.
// It is assumed that the centre of the box is always vertex 2
// though this isimportant only when all 3 vertices lie exactly on
// the same contour level, in which case only the side of the box
// is drawn.
//
//      vertex 4 +-------------------+ vertex 3
//               | \               / |
//               |   \    m-3    /   |
//               |     \       /     |
//               |       \   /       |
//               |  m=2    X   m=2   |       the centre is vertex 0
//               |       /   \       |
//               |     /       \     |
//               |   /    m=1    \   |
//               | /               \ |
//      vertex 1 +-------------------+ vertex 2

                        for (int m=1; m<=4; m++)
                        {
                            int m1 = m;
                            int m2 = 0;
                            int m3 = (m!=4) ? m+1 : 1;
                            int case_value = castab[sh[m1]+1][sh[m2]+1][sh[m3]+1];
                            if (case_value != 0)
                            {
                                double x1, x2, y1, y2;
                                switch (case_value)
                                {
                                    // Case 1:Line between vertices 1 and 2
                                    case 1:
                                        x1 = xh[m1];
                                        y1 = yh[m1];
                                        x2 = xh[m2];
                                        y2 = yh[m2];
                                        break;
                                    // Case 2:Line between vertices 2 and 3
                                    case 2:
                                        x1 = xh[m2];
                                        y1 = yh[m2];
                                        x2 = xh[m3];
                                        y2 = yh[m3];
                                        break;
                                    // Case 3:Line between vertices 3 and 1
                                    case 3:
                                        x1 = xh[m3];
                                        y1 = yh[m3];
                                        x2 = xh[m1];
                                        y2 = yh[m1];
                                        break;
                                    // Case 4:Line between vertex 1 and side 2-3
                                    case 4:
                                        x1 = xh[m1];
                                        y1 = yh[m1];
                                        x2 = sect(xh, h, m2, m3);
                                        y2 = sect(yh, h, m2, m3);
                                        break;
                                    // Case 5:Line between vertex 2 and side 3-1
                                    case 5:
                                        x1 = xh[m2];
                                        y1 = yh[m2];
                                        x2 = sect(xh, h, m3, m1);
                                        y2 = sect(yh, h, m3, m1);
                                        break;
                                    // Case 6:Line between vertex 3 and side 1-2
                                    case 6:
                                        x1 = xh[m3];
                                        y1 = yh[m3];
                                        x2 = sect(xh, h, m1, m2);
                                        y2 = sect(yh, h, m1, m2);
                                        break;
                                    // Case 7:Line between sides 1-2 and 2-3
                                    case 7:
                                        x1 = sect(xh, h, m1, m2);
                                        y1 = sect(yh, h, m1, m2);
                                        x2 = sect(xh, h, m2, m3);
                                        y2 = sect(yh, h, m2, m3);
                                        break;
                                    // Case 8:Line between sides 2-3 and 3-1
                                    case 8:
                                        x1 = sect(xh, h, m2, m3);
                                        y1 = sect(yh, h, m2, m3);
                                        x2 = sect(xh, h, m3, m1);
                                        y2 = sect(yh, h, m3, m1);
                                        break;
                                    // Case 9:Line between sides 3-1 and 1-2
                                    case 9:
                                        x1 = sect(xh, h, m3, m1);
                                        y1 = sect(yh, h, m3, m1);
                                        x2 = sect(xh, h, m1, m2);
                                        y2 = sect(yh, h, m1, m2);
                                        break;
                                    default:
                                        break;
                                }
                                std::vector<double> temp(5);
                                temp[0] = x1;
                                temp[1] = y1;
                                temp[2] = x2;
                                temp[3] = y2;
                                temp[4] = z[k];
                                ret.push_back(temp);
                            }
                        }
                    }
                }
            }
        }
    }
    return ret;
}
