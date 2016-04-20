#include <iostream>
#include <vector>

#include "projectionmethod.h"


int main()
{
    int n = 10;

    std::vector<Mat> out = simulate(100, 1, 1, n, n, 0.001, 0);

    std::cout << "\n\n/// U Matrix /// \n\n";
    for (int i=0; i<n+1; i++)
    {
        for (int j=0; j<n+1; j++)
        {
            std::cout << out[0](j,n-i) << "\t";
        }
        std::cout << "\n";
    }

    std::cout << "\n\n/// V Matrix /// \n\n";
    for (int i=0; i<n+1; i++)
    {
        for (int j=0; j<n+1; j++)
        {
            std::cout << out[1](j,n-i) << "\t";
        }
        std::cout << "\n";
    }

    std::cout << "\n\n/// P Matrix /// \n\n";
    for (int i=0; i<n; i++)
    {
        for (int j=0; j<n; j++)
        {
            std::cout << out[2](j,n-i-1) << "\t";
        }
        std::cout << "\n";
    }

    return 0;
}



