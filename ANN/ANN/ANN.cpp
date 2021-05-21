#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <ctime>

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

double transformResult(array<double, 5>, array< array<double, 5>, 2 >, array< array<double, 2>, 2 >, array< double, 2 >);
double accuracy(vector< array<double, 5> >, array< array<double, 5>, 2 >, array< array<double, 2>, 2 >, array< double, 2 >, vector<double>);
double metrics(vector< array<double, 5> >, array< array<double, 5>, 2 >, array< array<double, 2>, 2 >, array< double, 2 >, vector<double>);

void shake(vector< array<double, 5> >&, vector<double>&);

int main()
{
    fstream XFile, yFile; // files with data: X has parameters values, y has results
    double precision = 0;
    int i = 0, j = 0, k = 0;
    int ages = 1;

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

        cout << "Data is read\n\nEnter precision of metrics: "; cin >> precision;
        cout << "Enter number of ages: "; cin >> ages;
        //studying of model
        for (k = 0; k < ages; ++k)
        {
            shake(X, y);
            while (metrics(X, W1, W2, W3, y) > precision)
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
        }

        cout << "\n\nNeural network is studied." << endl;
        cout << "Error on data: " << loss(X, y, W1, W2, W3) << endl;
        cout << "Accuracy: " << accuracy(X, W1, W2, W3, y) << endl;
        cout << "Metrics L22: " << metrics(X, W1, W2, W3, y) << endl;
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

double transformResult(array<double, 5> X, array< array<double, 5>, 2 > W1, array< array<double, 2>, 2 > W2, array< double, 2 > W3)
{
    return resultFunction(X, W1, W2, W3) > 0.5 ? 1 : 0;
}
double accuracy(vector< array<double, 5> > X, array< array<double, 5>, 2 > W1, array< array<double, 2>, 2 > W2, array< double, 2 > W3, vector<double> y)
{
    double acc = 0;
    int i = 0;

    for (; i < X.size(); ++i)
    {
        acc += (float)(transformResult(X[i], W1, W2, W3)==y[i]);
    }
    acc /= X.size();

    return acc;
}

double metrics(vector< array<double, 5> > X, array< array<double, 5>, 2 > W1, array< array<double, 2>, 2 > W2, array< double, 2 > W3, vector<double> y)
{
    double S = 0;
    int i = 0, j = 0;

    for (; i < 2; ++i)
    {
        for (j = 0; j < 5; ++j)
        {
            S += pow(dEdW1ij(X, W1, W2, W3, y, i, j), 2);
        }
        for (j = 0; j < 2; ++j)
        {
            S += pow(dEdW2ij(X, W1, W2, W3, y, i, j), 2);
        }
    }
    for (i = 0; i < 2; ++i)
    {
        S += pow(dEdW3i(X, W1, W2, W3, y, i), 2);
    }

    return sqrt(S);
}

void shake(vector< array<double, 5> >& X, vector<double>& y)
{
    int i = 0, c = 0;
    for (i = X.size() - 1; i>2; --i)
    {
        srand(time(0));

        c = rand() % i;
        swap(X[i], X[c]);
        swap(y[i], y[c]);
    }
}