#pragma once
#include "common.h"

namespace simple_nn
{
	int ReverseInt(int i)
	{
		unsigned char ch1, ch2, ch3, ch4;
		ch1 = i & 255;
		ch2 = (i >> 8) & 255;
		ch3 = (i >> 16) & 255;
		ch4 = (i >> 24) & 255;
		return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
	}

	MatXf read_mnist(string data_dir, string fname, int n_imgs, bool train = true)
	{
		MatXf img;

		string path = data_dir + "/" + fname;
		ifstream fin(path, ios::binary);
		if (fin.is_open()) {
			int magic_number = 0;
			int number_of_images = 0;
			int n_rows = 0;
			int n_cols = 0;

			fin.read((char*)&magic_number, sizeof(magic_number));
			magic_number = ReverseInt(magic_number);
			fin.read((char*)&number_of_images, sizeof(number_of_images));
			number_of_images = ReverseInt(number_of_images);
			fin.read((char*)&n_rows, sizeof(n_rows));
			n_rows = ReverseInt(n_rows);
			fin.read((char*)&n_cols, sizeof(n_cols));
			n_cols = ReverseInt(n_cols);

			img.resize(n_imgs, n_rows * n_cols);

			float m = 0.1306604762738431f;
			float s = 0.3081078038564622f;

			if (!train) {
				m = 0.13251460584233699f;
				s = 0.3104802479305348f;
			}

			for (int n = 0; n < n_imgs; n++) {
				for (int i = 0; i < n_rows; i++) {
					for (int j = 0; j < n_cols; j++) {
						unsigned char temp = 0;
						fin.read((char*)&temp, sizeof(temp));
						img(n, j + n_cols * i) = (temp / 255.f - m) / s;
					}
				}
			}
		}
		else {
			cout << "The file(" << path << ") does not exist." << endl;
			exit(1);
		}

		return img;
	}

	VecXi read_mnist_label(string data_dir, string fname, int n_imgs)
	{
		VecXi label(n_imgs);

		string path = data_dir + "/" + fname;
		ifstream fin(path);
		if (fin.is_open()) {
			for (int i = 0; i < n_imgs + 8; ++i) {
				unsigned char temp = 0;
				fin.read((char*)&temp, sizeof(temp));
				if (i > 7) {
					label[i - 8] = (int)temp;
				}
			}
		}
		else {
			cout << "The file(" << path << ") does not exist." << endl;
			exit(1);
		}

		return label;
	}


MatXf read_dummy_images(int n_imgs, int channel, int height, int width) {
    int image_size = channel * height * width;
    return MatXf::Zero(n_imgs, image_size);
}


VecXi read_dummy_labels(int n_imgs) {
    return VecXi::Zero(n_imgs);
}


MatXf read_custom_images(const string& filename, int n_imgs, int channel, int height, int width) {
    ifstream file(filename, ios::binary);

    if (!file.is_open()) {
        cout << "Unable to open file, defining dummy images instead: " << filename << endl;
        return read_dummy_images(n_imgs, channel, height, width);
    }

    int image_size = channel * height * width;
    MatXf images(n_imgs, image_size);

    /* for(int k = 0; k < process_offset; k++) { */
    /*     for (int img = 0; img < n_imgs; ++img) { */
    /*         for (int i = 0; i < image_size; ++i) { */
    /*             /1* file.ignore(sizeof(float)); *1/ */
    /*         float pixel; */
    /*         file.read(reinterpret_cast<char*>(&pixel), sizeof(float)); */
    /*         images(img,i) = pixel; */
            
                
    /* }}} */ 
    /* for(int k = 0; k < process_offset+1; k++) { */
    for (int img = 0; img < n_imgs; ++img) {
        for (int i = 0; i < image_size; ++i) {
            float pixel;
            file.read(reinterpret_cast<char*>(&pixel), sizeof(float));
            images(img,i) = pixel;
        }
    }
    /* } */
    file.close();
    return images;
}


VecXi read_custom_labels(const string& filename, int n_imgs) {
    ifstream file(filename, ios::binary);

    if (!file.is_open()) {
        cout << "Unable to open file, defining dummy labels instead: " << filename << endl;
        return read_dummy_labels(n_imgs);
    }
    
    VecXi labels(n_imgs);

    /* for(int i = 0; i < process_offset+1; i++) { */
    /*     for (int img = 0; img < n_imgs; ++img) { */
    /*         /1* file.ignore(sizeof(uint32_t)); *1/ */
    /* uint32_t label; */
    /* file.read(reinterpret_cast<char*>(&label), sizeof(uint32_t)); */
    /* labels[i] = label; */
    /* }} */
    /* for(int k = 0; k < process_offset+1; k++) { */
    for (int i = 0; i < n_imgs; ++i) {
    uint32_t label;
    file.read(reinterpret_cast<char*>(&label), sizeof(uint32_t));
    labels[i] = label;
        /* std::cout << label << std::endl; */
    }
    /* } */
    file.close();
    return labels;
}







}
