#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <boost/filesystem.hpp>
#include "lits_db/lits_database.h"
#include "lits_scan/lits_scan.h"


int main(int argc, char **argv)
{
	if(argc != 2)
	{
		std::cout<<"Invalid number of input arguments."<<std::endl;
		std::cout<<"First argument: directory path where the Train Batch 1 and Train Batch 2 are placed."<<std::endl;
		exit(EXIT_FAILURE);
	}

	if(!strcmp(argv[1], "-h") or !strcmp(argv[1], "-help") or !strcmp(argv[1], "help") or !strcmp(argv[1], "h"))
	{
		std::cout<<"comparison:"<<strcmp(argv[1], "-h")<<std::endl;
		std::cout<<"First argument: directory path where the Train Batch 1 and Train Batch 2 are placed."<<std::endl;
		exit(EXIT_SUCCESS);
	}

	boost::filesystem::path db_path(argv[1]);

	if(boost::filesystem::is_directory(db_path))
	{
		std::string db_path_str(argv[1]);
		boost::filesystem::path db_path_batch((db_path_str + "/Training Batch 1").c_str());
		if(!boost::filesystem::is_directory(db_path_batch))
		{
			std::cout<<"Training Batch 1 does not exists."<<std::endl;
			exit(EXIT_FAILURE);
		}

		db_path_batch = db_path_str + "/Training Batch 2";
		if(!boost::filesystem::is_directory(db_path_batch))
		{
			std::cout<<"Training Batch 2 does not exists."<<std::endl;
			exit(EXIT_FAILURE);
		}

	}
	else
	{
		std::cout<<"Invalid database path!"<<std::endl;
		exit(EXIT_FAILURE);
	}

	LiTS_db db(db_path.string());
	db.load_subject_names();
	db.data_split(0, 1);

	for(unsigned int i = 0; i < 5; i++)
	{
		std::string volume_path;
		std::string segmentation_path;
		std::string scan_name;

		scan_name = db.get_train_scan_name(i);
		db.get_scan_paths(scan_name, volume_path, segmentation_path);

		LiTS_scan ls(volume_path, segmentation_path);
		ls.load_volume();
		ls.load_segmentation();
		ls.load_info();
	}

	return 0;
}
