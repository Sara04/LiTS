
#include "liver_tumor_segmentation.h"

LiTS_segmentator::LiTS_segmentator()
{
    W_lf = new double[10];
    b_lf = new double[1];

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,0.1);
    for(unsigned int i = 0; i < 10; i++)
        W_lf[i] = distribution(generator);
    b_lf[0] = distribution(generator);
}

LiTS_segmentator::~LiTS_segmentator()
{
    delete [] W_lf;
    delete [] b_lf;
}

void LiTS_segmentator::development(std::list<LiTS_scan> development_set)
{
    unsigned long n = 0;
    unsigned int *lenghts = new unsigned int[development_set.size()];
    unsigned long int *acc = new unsigned long[development_set.size()];
    unsigned int *sizes = new unsigned int[3 * development_set.size()];
    float *voxel_sizes = new float[3 * development_set.size()];

    unsigned int c = 0;
    for(std::list<LiTS_scan>::iterator it = development_set.begin();
        it != development_set.end(); it++)
    {
        lenghts[c] = it->get_width() * it->get_height() * it->get_depth();

        sizes[3 * c] = it->get_width();
        sizes[3 * c + 1] = it->get_height();
        sizes[3 * c + 2] = it->get_depth();

        voxel_sizes[3 * c] = it->get_voxel_width();
        voxel_sizes[3 * c + 1] = it->get_voxel_height();
        voxel_sizes[3 * c + 2] = it->get_voxel_depth();

        c++;
        n += (it->get_width() * it->get_height() * it->get_depth());
        acc[c-1] = n;
    }

    float * volumes = new float[n];
    unsigned char * ground_truths = new unsigned char[n];

    c = 0;
    unsigned long shift = 0;
    for(std::list<LiTS_scan>::iterator it = development_set.begin();
        it != development_set.end(); it++)
    {
        if(c > 0)
            shift = acc[c - 1];
        memcpy(&volumes[shift],
               it->get_volume()->GetBufferPointer(),
               lenghts[c] * sizeof(float));
        memcpy(&ground_truths[shift],
               it->get_segmentation()->GetBufferPointer(),
               lenghts[c] * sizeof(unsigned char));
        c++;
    }

    develop(W_lf, b_lf,
            volumes, ground_truths,
            acc, lenghts, sizes, voxel_sizes,
            development_set.size());

    delete [] lenghts;
    delete [] sizes;
    delete [] voxel_sizes;
    delete [] acc;
    delete [] volumes;
    delete [] ground_truths;
}
