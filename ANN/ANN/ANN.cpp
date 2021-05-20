#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <vector>
#include <array>
#include <cmath>

using namespace std;

double scalarMult(array<double, 5>, array<double, 5>);
double scalarMult(array<double, 2>, array<double, 2>);
double sigmoid(array<double, 5>, array<double, 5>);
double sigmoid(array<double, 2>, array<double, 2>);
double sigmoid(double x);
double resultFunction(array<double, 5>, array< array<double, 5>, 2 >, array< array<double, 2>, 2 >, array< double, 2 >);
double error(array<double, 5>, array< array<double, 5>, 2 >, array< array<double, 2>, 2 >, array< double, 2 >, double);
double loss(vector< array<double, 5> >, vector<double>, array< array<double, 5>, 2 >, array< array<double, 2>, 2 >, array< double, 2 >);
double dEdW1ij(vector< array<double, 5> >, array< array<double, 5>, 2 >, array< array<double, 2>, 2 >, array< double, 2 >, vector<double>, int, int);
double dEdW2ij(vector< array<double, 5> >, array< array<double, 5>, 2 >, array< array<double, 2>, 2 >, array< double, 2 >, vector<double>, int, int);
double dEdW3i(vector< array<double, 5> >, array< array<double, 5>, 2 >, array< array<double, 2>, 2 >, array< double, 2 >, vector<double>, int);

int main()
{
    fstream XFile, yFile; // files with data: X has parameters values, y has results
    int i = 0, j = 0, k = 0;
    int iters = 0;

    XFile.open("X.txt", ios_base::binary | ios_base::in);
    yFile.open("y.txt", ios_base::binary | ios_base::in);

    if (XFile.is_open() && yFile.is_open())
    {
        int i = 0, j = 0;

        vector< array<double, 5> > X; // matrix of values of parameters
        vector<double> y; // vector of values
        array<double, 5> object; // vector of parameter values for object
        string value; // a some value
        array< array<double, 5>, 2 > W1;
        array< array<double, 2>, 2 > W2;
        array< double, 2 > W3;
        
        //reading data from files
        cout << "Files are opened" << endl;
        while (XFile >> value)
        {
            if (j != 5)
            {
                object[j] = stod(value);
                ++j;
            }
            else
            {
                X.push_back(object);
                j = 0;
            }
        }
        while (yFile >> value)
        {
            y.push_back(stod(value));
        }

        if (X.size() > y.size())
        {
            while (X.size() - y.size())
            {
                X.pop_back();
            }
        }
        if (y.size() > X.size())
        {
            while (y.size() - X.size())
            {
                y.pop_back();
            }
        }

        for (i = 0; i < 2; ++i)
        {
            for (j = 0; j < 5; ++j)
            {
                W1[i][j] = 1;
            }
            for (j = 0; j < 2; ++j)
            {
                W2[i][j] = 1;
            }
            W3[i] = 1;
        }

        cout << "Data is read\n\nEnter number of iterations:" << endl;
        cin >> iters;
        //studying of model
        for (; k < iters; ++k)
        {
            for (i = 0; i < 2; ++i)
            {
                for (j = 0; j < 5; j++)
                {
                    W1[i][j] -= dEdW1ij(X, W1, W2, W3, y, i, j);
                }
                for (j = 0; j < 2; j++)
                {
                    W2[i][j] -= dEdW2ij(X, W1, W2, W3, y, i, j);
                }
                for (j = 0; j < 2; ++j)
                {
                    W3[j] -= dEdW3i(X, W1, W2, W3, y, j);
                }
            }
            
        }

        cout << "\n\n\nNeural network is studied." << endl;
        cout << "Error on data: " << loss(X, y, W1, W2, W3) << endl;
    }
    else
    {
        cout << "Data wasn't read." << endl;
    }


    XFile.close();
    yFile.close();

    system("pause");
}

double scalarMult(array<double, 5> W, array<double, 5> X)
{
    double S = 0;

    for (int i = 0; i < 5; ++i)
    {
        S += W[i] * X[i];
    }

    return S;
}
double scalarMult(array<double, 2> W, array<double, 2> X)
{
    double S = 0;

    for (int i = 0; i < 2; ++i)
    {
        S += W[i] * X[i];
    }

    return S;
}

double sigmoid(array<double, 5> W, array<double, 5> X)
{
    return 1 / (1 + exp(-scalarMult(W, X)));
}
double sigmoid(array<double, 2> W, array<double, 2> X)
{
    return 1 / (1 + exp(-scalarMult(W, X)));
}
double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}


double resultFunction(array<double, 5> X, array< array<double, 5>, 2 > W1, array< array<double, 2>, 2 > W2, array< double, 2 > W3)
{
    array<double, 2> h1 = { sigmoid(W1[0], X),  sigmoid(W1[1], X) },
                     h2 = { sigmoid(W2[0], h1), sigmoid(W2[1], h1) };

    return sigmoid(W3, h2);
}


double error(array<double, 5> X, array< array<double, 5>, 2 > W1, array< array<double, 2>, 2 > W2, array< double, 2 > W3, double y)
{
    return fabs(resultFunction(X, W1, W2, W3)-y);
}

double loss(vector< array<double, 5> > X, vector<double> y, array< array<double, 5>, 2 > W1, array< array<double, 2>, 2 > W2, array< double, 2 > W3)
{
    double S = 0;

    for (int i = 0; i < X.size(); ++i)
    {
        S += pow( error(X[i], W1, W2, W3, y[i]), 2 );
    }

    return S / X.size();
}

double dEdW1ij(vector< array<double, 5> > X, array< array<double, 5>, 2 > W1, array< array<double, 2>, 2 > W2, array< double, 2 > W3, vector<double> y, int i, int j)
{
    array< array<double, 5>, 2 > newW1 = W1;
    double h = 0.001;
    newW1[i][j] += h;

    return ( loss(X, y, newW1, W2, W3) - loss(X, y, W1, W2, W3)) / h;
}
double dEdW2ij(vector< array<double, 5> > X, array< array<double, 5>, 2 > W1, array< array<double, 2>, 2 > W2, array< double, 2 > W3, vector<double> y, int i, int j)
{
    array< array<double, 2>, 2 > newW2 = W2;
    double h = 0.001;
    newW2[i][j] += h;

    return (loss(X, y, W1, newW2, W3) - loss(X, y, W1, W2, W3)) / h;
}
double dEdW3i(vector< array<double, 5> > X, array< array<double, 5>, 2 > W1, array< array<double, 2>, 2 > W2, array< double, 2 > W3, vector<double> y, int i)
{
    array<double, 2> newW3 = W3;
    double h = 0.001;
    newW3[i] += h;

    return (loss(X, y, W1, W2, newW3) - loss(X, y, W1, W2, W3)) / h;
}