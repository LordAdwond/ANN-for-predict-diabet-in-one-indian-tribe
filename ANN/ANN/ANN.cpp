#include <algorithm>
#include <iostream>
//#include <functional>
//#include <algorithm>
#include <fstream>
#include <cstdlib>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <ctime>

using namespace std;

double scalarMult(array<double, 5>&, array<double, 5>&);
double scalarMult(array<double, 10>&, array<double, 10>&);
double sigmoid(array<double, 5>&, array<double, 5>&);
double sigmoid(double x);
double f(array<double, 5>&, array< array<double, 5>, 10 >&, array< double, 10 >&);
double loss(vector< array<double, 5> >&, array< array<double, 5>, 10 >&, array< double, 10 >&, vector<double>&);

double dLOSSdW(vector< array<double, 5> >&, array< array<double, 5>, 10 >&, array< double, 10 >&, vector<double>, int, int, int);
void updateWeights(vector< array<double, 5> >&, array< array<double, 5>, 10 >&, array< double, 10 >&, vector<double>, int, int, int);

double transformResult(array<double, 5>&, array< array<double, 5>, 10 >&, array< double, 10 >&);
double accuracy(vector< array<double, 5> >&, array< array<double, 5>, 10 >&, array< double, 10 >&, vector<double>&);

int main()
{
    fstream XFile, yFile; // files with data: X has parameters values, y has results
    int numberOfUpdates = 0; // number of updates
    double m1 = 0, m2 = 0; // variables for saving of differences metrics
    int i = 0, j = 0, k = 0, updates = 0;
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
        array< array<double, 5>, 10 > W1; // first weight matrix
        array< double, 10 > W2; // second weight matrix
        char toPredict = 'n';
        
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

        for (i = 0; i < 10; ++i)
        {
            for (j = 0; j < 5; ++j)
            {
                W1[i][j] = 1;
            }
            W2[i] = 1;
        }

        cout << "Data is read\n\nEnter number of updates per epoch: "; cin >> numberOfUpdates;
        cout << "Enter number of epochs: "; cin >> ages;
        //studying of model
        for (k = 0; k < ages; ++k)
        {
            for(updates=0; updates< numberOfUpdates; ++updates)
            {

                for (i = 0; i < 10; ++i)
                {
                    for (j = 0; j < 5; ++j)
                    {
                        updateWeights(X, W1, W2, y, i, j, 1);
                    }
                    updateWeights(X, W1, W2, y, i, 0, 2);
                }

            }
            
        }

        cout << "\n\nNeural network is studied." << endl;
        cout << "Accuracy: " << accuracy(X, W1, W2, y) << endl;

        cout << "Do you want to predict outcome? (y/n) "; cin >> toPredict;
        while (toPredict == 'y')
        {
            cout << "Blood Pressure: "; cin >> object[0];
            cout << "Insulin: "; cin >> object[1];
            cout << "BMI: "; cin >> object[2];
            cout << "Diabetes Pedigree Function: "; cin >> object[3];
            cout << "Age: "; cin >> object[4];

            cout << "Have diabet? ";
            if (transformResult(object, W1, W2))
            {
                cout << "Yes";
            }
            else
            {
                cout << "No";
            }
            cout << "\nAgain? (y/n) "; cin >> toPredict;
        }
    }
    else
    {
        cout << "Data wasn't read." << endl;
    }


    XFile.close();
    yFile.close();

    system("pause");
}

double scalarMult(array<double, 5>& W, array<double, 5>& X)
{
    double S = 0;

    for (int i = 0; i < 5; ++i)
    {
        S += W[i] * X[i];
    }

    return S;
}
double scalarMult(array<double, 10>& W, array<double, 10>& X)
{
    double S = 0;

    for (int i = 0; i < 10; ++i)
    {
        S += W[i] * X[i];
    }

    return S;
}

double sigmoid(array<double, 5>& x, array<double, 5>& w)
{
    return 1 / (1 + exp(-scalarMult(x, w)));
}
double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double f(array<double, 5>& X, array< array<double, 5>, 10 >& W1, array< double, 10 >& W2)
{
    double S = 0;
    int i = 0;

    for (i = 0; i < 10; ++i)
    {
        S += W2[i] * scalarMult(W1[i], X);
    }

    return S;
}

double loss(vector< array<double, 5> >& X, array< array<double, 5>, 10 >& W1, array< double, 10 >& W2, vector<double>& y)
{
    double S = 0;
    int i = 0;

    for (i = 0; i < (int)(X.size()); ++i)
    {
        S += pow(f(X[i], W1, W2), 2);
    }

    return S / X.size();
}

double dLOSSdW(vector< array<double, 5> >& X, array< array<double, 5>, 10 >& W1, array< double, 10 >& W2, vector<double> y, int i, int j, int matrix)
{
    double h = 0.0001;

    if (matrix == 1)
    {
        if ((i >= 0 && i < 10) && (j >= 0 && j < 5))
        {
            array< array<double, 5>, 10 > newW1 = W1;
            newW1[i][j] += h;

            return (loss(X, newW1, W2, y) - loss(X, W1, W2, y)) / h;
        }
        else
        {
            return 0;
        }
    }
    else if (matrix == 2)
    {
        if (i >= 0 && i < 10)
        {
            array< double, 10 > newW2 = W2;
            newW2[i] += h;

            return (loss(X, W1, newW2, y) - loss(X, W1, W2, y)) / h;
        }
        else
        {
            return 0;
        }
    }
    else
    {
        return 0;
    }
}

void updateWeights(vector< array<double, 5> >& X, array< array<double, 5>, 10 >& W1, array< double, 10 >& W2, vector<double> y, int i, int j, int matrix)
{
    int m = 0;

    for (m = 1; m <= 2; ++m)
    {
        if (m == 1) 
        {
            W1[i][j] -= dLOSSdW(X, W1, W2, y, i, j, m);
        }
        else
        {
            W2[i] -= dLOSSdW(X, W1, W2, y, i, j, m);
        }
    }
}

double transformResult(array<double, 5>& X, array< array<double, 5>, 10 >& W1, array< double, 10 >& W2)
{
    return f(X, W1, W2) <= 0.5 ? 0 : 1;
}
double accuracy(vector< array<double, 5> >& X, array< array<double, 5>, 10 >& W1, array< double, 10 >& W2, vector<double>& y)
{
    double k = 0;
    int i = 0;

    for (i = 0; i < (int)(X.size()); ++i)
    {
        if (transformResult(X[i], W1, W2) == y[i])
        {
            ++k;
        }
    }

    return k / X.size();
}