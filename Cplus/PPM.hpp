#ifndef Experiments_PPMWriter_h
#define Experiments_PPMWriter_h

#include "Library.hpp"

class PPM
{
public:
    static void write(const MatrixXd& md, const string filename)
    {
        ofstream fout(filename.c_str());
        fout<<"P3"<<endl;
        fout<<md.rows()<<' '<<md.cols()<<endl;
        fout<<255<<endl;
        for(auto i=0; i<md.rows(); ++i)
        {
            for(auto j=0; j<md.cols(); ++j)
            {
                if (md(i,j) < 0)
                {
                    fout <<255 << ' ' << 255 << ' ' << 255 << ' ';
                }
                else
                {
                    fout <<0 <<' ' <<0 <<' ' <<0 <<' ';
                }
            }
            fout<<endl;
        }
        fout.close();
    }
};

#endif
