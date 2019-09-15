#include<iostream>
#include<vector>
#include<utility>
#include<pcl/visualization/pcl_plotter.h>
#include<pcl/features/esf.h>
#include<pcl/features/normal_3d.h>
#include<pcl/features/cvfh.h>
#include<pcl/features/gfpfh.h>
#include<pcl/io/pcd_io.h>
#include<pcl/io/ply_io.h>
#include<pcl/point_types.h>
#include<pcl/visualization/pcl_visualizer.h>
#include<fstream>
#include<string.h>
#include<stdlib.h>
#include<dirent.h>
#include<stdio.h>
#include<unistd.h>
#include<sys/stat.h>
#include<sys/types.h>


using namespace std;

// the main can take in command line arguments
int main(int argc, char** argv)
{

    ofstream fi;
    DIR *dp;
    struct dirent *dirp;
    struct stat filestat;
    string filepath, folder;

    fi.open("/home/nithin/Desktop/Macgyver-Tool-Substitution/Descriptors_Shape/Descriptors/esf_cvfh_descriptors_cut_final.csv");
    fi<<"Object"<<",";
    for(int a = 1; a<=964; a++)
    {
        fi<<"Feature"<<a<<",";
    }
    fi<<"\n";
    pcl::visualization::PCLPlotter plotter ;
    // create a pointer for pointcloud data called cloud and initialize it
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    // pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_l (new pcl::PointCloud<pcl::PointXYZL>);

    // for (size_t i = 0; i < cloud_l->points.size(); ++i)
    // {
    //     cloud_l->points[i].label = 1 + i % 4;
    // }


    folder = "/home/nithin/Desktop/Macgyver-Tool-Substitution/Point_Clouds/CUT";
    dp = opendir(folder.c_str());
    // if(dp == NULL)
    //     {
    //         return -1;
    //     }
    while(dirp = readdir(dp))
    {
        filepath = folder + "/" + dirp->d_name;
        cout<<"\n"<<filepath;
        if (stat( filepath.c_str(), &filestat )) continue;
        if (S_ISDIR( filestat.st_mode ))         continue;
        fi<<dirp->d_name<<",";
        if (pcl::io::loadPLYFile<pcl::PointXYZ> (filepath, *cloud)==-1)
        {
            PCL_ERROR("couldn't read file\n");
            return(-1);
        }

        // if (pcl::io::loadPLYFile<pcl::PointXYZL> (filepath, *cloud_l)==-1)
        // {
        //     PCL_ERROR("couldn't read file\n");
        //     return(-1);
        // }
        // Create the ESF estimation class 
        pcl::ESFEstimation<pcl::PointXYZ, pcl::ESFSignature640> esf;

        // Create the Normal estimation class 
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;

        // Create the CVFH estimation class 
        pcl::CVFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> cvfh;

        // Create the GFPFH estimation class 
        pcl::GFPFHEstimation<pcl::PointXYZL, pcl::PointXYZL, pcl::GFPFHSignature16> gfpfh;


        // descriptors
        pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<pcl::VFHSignature308>::Ptr cvfhSignature (new pcl::PointCloud<pcl::VFHSignature308>);
        pcl::PointCloud<pcl::ESFSignature640>::Ptr esfSignature (new pcl::PointCloud<pcl::ESFSignature640>);
        // pcl::PointCloud<pcl::GFPFHSignature16>::Ptr gfpfhSignature (new pcl::PointCloud<pcl::GFPFHSignature16>);

        // Input the dataset
        esf.setInputCloud (cloud);

        normalEstimation.setInputCloud(cloud);
        normalEstimation.setRadiusSearch(0.03);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
        normalEstimation.setSearchMethod(kdtree);
        normalEstimation.compute(*normals);

        cvfh.setInputCloud(cloud);
        cvfh.setInputNormals(normals);
        cvfh.setSearchMethod(kdtree);
        cvfh.setEPSAngleThreshold(5.0 / 180.0 * M_PI);
        cvfh.setCurvatureThreshold(0.7);
        cvfh.setNormalizeBins(true);

        // gfpfh.setInputCloud(cloud_l);
        // gfpfh.setInputLabels(cloud_l);
        // gfpfh.setOctreeLeafSize(0.01);
        // gfpfh.setNumberOfClasses(4);

        // compute esf features
        esf.compute(*esfSignature);
        cvfh.compute(*cvfhSignature);
        // gfpfh.compute(*gfpfhSignature);
        for(int i=0;i<640;i++)
        {
            fi<<float(esfSignature->points[0].histogram[i])<<",";
        }
        for(int i=0;i<308;i++)
        {
            fi<<float(cvfhSignature->points[0].histogram[i])<<",";
        }
        // for(int i=0;i<16;i++)
        // {
        //     fi<<float(gfpfhSignature->points[0].histogram[i])<<",";
        // }
        fi<<"\n";
    }

    closedir(dp);
    fi.close();
return 0;
}
